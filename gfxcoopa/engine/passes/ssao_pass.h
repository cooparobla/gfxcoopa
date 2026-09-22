/**
 * @file ssao_pass.h
 * @brief Screen-Space Ambient Occlusion: horizon-based estimation (GTAO) over the SSR
 * chain's Hi-Z depth pyramid, temporal resolve, and a depth/normal-aware bilateral blur.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SSAO_PASS_H
#define GFXCOOPA_ENGINE_PASSES_SSAO_PASS_H

#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/memory/image_upload.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/passes/fullscreen_stage.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

/// Bundles three sub-stages behind one object -- raw horizon-based estimate (GTAO), temporal
/// resolve, and a bilateral blur -- the same multi-stage-bundling shape SsrPass uses for its raymarch/resolve/
/// composite trio. Callers only ever touch update_descriptors()/execute()/output_view().
///
/// Pipeline order is raw -> resolve -> blur (temporal first, spatial second), matching SsrPass's
/// own raymarch -> resolve -> composite ordering and the standard denoiser convention: a spatial
/// filter over an already temporally-accumulated signal converges faster and doesn't have to
/// fight the same per-frame noise the temporal stage exists to remove.
class SsaoPass {
public:
    /// Raw-pass GPU push constants. mat4 first (16-byte aligned) then plain scalars,
    /// matching ResolvePushConstants' layout rule below.
    struct PushConstants {
        glm::mat4 inv_proj  = glm::mat4(1.0f);  // offset 0 -- view reconstruction of Hi-Z samples
        float radius        = 0.5f;             // offset 64
        float bias          = 0.025f;           // offset 68
        float power         = 1.5f;             // offset 72
        int   slices        = 2;                // offset 76
        int   steps         = 8;                // offset 80
        float resolution_x  = 1.0f;             // offset 84
        float resolution_y  = 1.0f;             // offset 88
        // 0 when temporal resolve is off -- see ssao.frag. Must match Params::temporal_enabled;
        // execute() derives this, callers don't set it directly.
        int   noise_rotation = 0;               // offset 92
        int   max_mip        = 5;               // offset 96
        float max_radius_px  = 80.0f;           // offset 100 -- upper clamp on the march extent
    };
    static_assert(sizeof(PushConstants) == 104,
                  "ssao.frag's SsaoPushConstants block must match this layout byte-for-byte");

    /// Format of the temporal resolve's target and of the history image copied from it. The
    /// resolve carries an eye distance (world units, see ssao_resolve.frag's disocclusion
    /// test) and a per-pixel accumulation count alongside the occlusion value, so the target
    /// needs float range beyond two channels.
    static constexpr VkFormat kResolveFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    /// Resolve-pass GPU push constants. mat4 first (16-byte aligned), then plain scalars --
    /// vec2 is deliberately flattened to two floats, matching PushConstants' house rule, so nothing
    /// here depends on std430 vec2 base-alignment (8 bytes) lining up with this C++ struct's layout.
    struct ResolvePushConstants {
        glm::mat4 prev_view_proj = glm::mat4(1.0f); // offset 0
        float     resolution_x   = 1.0f;            // offset 64
        float     resolution_y   = 1.0f;            // offset 68
        // Accumulation-count cap: each pixel blends the new frame at 1/(count+1) until its
        // count reaches this, then keeps averaging at that fixed rate. 0 degenerates the
        // resolve into a passthrough of the current frame (how temporal_enabled=false works).
        float     max_accum      = 32.0f;            // offset 72
        int       history_valid  = 0;                // offset 76
        // Flattened vec3s, same house rule as the vec2 above: the eye that produced this
        // frame's G-buffer, and the eye the history image was resolved from. The resolve
        // shader needs both to turn its stored distance channel into a surface-identity test.
        float     camera_pos_x      = 0.0f;          // offset 80
        float     camera_pos_y      = 0.0f;          // offset 84
        float     camera_pos_z      = 0.0f;          // offset 88
        float     prev_camera_pos_x = 0.0f;          // offset 92
        float     prev_camera_pos_y = 0.0f;          // offset 96
        float     prev_camera_pos_z = 0.0f;          // offset 100
        // Nonzero once the camera has been still long enough for the accumulated average
        // to top up -- the resolve then holds accepted-history pixels verbatim, which is
        // what keeps a still image byte-static (see ssao_resolve.frag).
        int       frozen            = 0;             // offset 104
    };
    static_assert(sizeof(ResolvePushConstants) == 108, "ssao_resolve.frag's PushConstants block must match this layout byte-for-byte");

    /// Blur-pass GPU push constants.
    struct BlurPushConstants {
        float plane_sigma = 0.375f; // world-space plane-distance tolerance of the bilateral
                                    // weight, per spacing unit (see ssao_blur.frag's doc on
                                    // why it is not derived from the gather radius)
        float max_accum   = 32.0f;  // resolve accumulation cap; drives the count-adaptive dilation
        float motion_px   = 0.0f;   // camera sweep speed in pixels/frame; widens the kernel in motion
    };

    /// execute() parameters. Mirrors SsrPass::Params' shape: raw-pass tunables plus temporal
    /// resolve controls, all bundled so pbr_render_pipeline.h only builds one struct per frame.
    struct Params {
        float radius        = 0.5f;
        float bias          = 0.025f;
        float power         = 1.5f;
        int   slices        = 2;   // horizon slices per pixel
        int   steps         = 8;   // march steps per slice direction
        // Upper clamp on the horizon march's screen-space extent, in render-target pixels
        // (Unity HDRP's "Maximum Radius in Pixels").
        float max_radius_px = 80.0f;
        // Blur pass: world-space plane-distance tolerance of the bilateral weight, per
        // spacing unit -- tracks the geometry's step scale, independent of `radius`.
        float blur_plane_sigma = 0.375f;
        // Inverse of the projection that rendered the depth buffer the Hi-Z pyramid was
        // downsampled from (the TAA-jittered one, when jitter is active).
        glm::mat4 inv_proj  = glm::mat4(1.0f);
        int   max_mip       = 5;   // top usable Hi-Z mip (HiZPass::mip_levels() - 1, capped)
        // Caller-driven frame counter (e.g. ssao_frame_index_ & 0x7 in pbr_render_pipeline.h).
        // Only takes effect when temporal_enabled is true -- see execute()'s raw_pc.noise_rotation.
        int   noise_rotation = 0;

        bool  temporal_enabled = true;
        // Accumulation depth: how many frames a pixel averages before the running mean turns
        // into a fixed-rate EMA. Mapped straight into ResolvePushConstants::max_accum.
        int   temporal_frames  = 32;
        // Reprojection: previous frame's JITTERED proj * view, and whether it (and the history
        // image) are actually valid yet -- see execute()'s history_valid derivation.
        glm::mat4 prev_view_proj       = glm::mat4(1.0f);
        bool      prev_view_proj_valid = false;
        // This frame's eye position. The pass remembers the previous frame's itself, so callers
        // only ever supply the current one.
        glm::vec3 camera_pos           = glm::vec3(0.0f);
        // True once the camera has been still long enough to freeze the accumulated image
        // (the caller counts still frames -- see PixelRenderPipeline's ssao_frames_still_).
        bool      frozen               = false;
        // Camera rotation between consecutive frames, in screen-centre pixels -- drives the
        // blur's velocity widening (0 at rest keeps the kernel bit-identical to the resting
        // one, which the byte-static contracts depend on).
        float     motion_px            = 0.0f;
    };

    SsaoPass(coopa::gfx::core::Device& device,
             coopa::gfx::memory::Allocator& allocator,
             coopa::gfx::command::CommandPool& cmd_pool,
             const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
             const std::string& vert_spv,
             const std::string& raw_frag_spv,
             const std::string& resolve_frag_spv,
             const std::string& blur_frag_spv)
        : device_(device), allocator_(allocator)
    {
        create_neutral_texture_(cmd_pool);

        // All three targets hold an occlusion mask, not colour to be filtered -- NEAREST
        // throughout (the resolve pass's history read uses the caller's linear_sampler instead,
        // since it samples at a reprojected, non-texel-aligned UV).
        coopa::gfx::SamplerDesc nearest_clamp;
        nearest_clamp.min = nearest_clamp.mag = coopa::gfx::Filter::Nearest;
        nearest_clamp.mipmap  = coopa::gfx::MipmapMode::Nearest;
        nearest_clamp.address = coopa::gfx::AddressMode::ClampToEdge;
        raw_sampler_  = std::make_unique<util::Sampler>(device, nearest_clamp);
        blur_sampler_ = std::make_unique<util::Sampler>(device, nearest_clamp);


        // Single colour attachment, no depth -- same shape as HiZPass/SceneColorMipPass, just
        // one mip instead of a chain. All three sub-stages share that shape; they differ only in
        // format (see kResolveFormat).
        raw_render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device, VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        // RG16F, not R8: the resolve carries a per-pixel eye distance in .g alongside the
        // occlusion in .r (see ssao_resolve.frag), and that distance needs real range and
        // precision rather than a normalized byte. The raw and blur targets below stay R8.
        resolve_render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device, kResolveFormat, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        blur_render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device, VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        // One FullscreenStage per stage.
        using coopa::gfx::DescriptorType;
        using coopa::gfx::ShaderStage;

        auto sampled = [](uint32_t n) {
            std::vector<coopa::gfx::pipeline::DescriptorBinding> b;
            for (uint32_t i = 0; i < n; ++i) {
                b.push_back({i, DescriptorType::CombinedImageSampler, ShaderStage::Fragment, 1});
            }
            return b;
        };

        // Raw pass. Set 0 is the caller's camera layout; this stage's own set 1 holds
        // G1 normal, G2 position and the caller's Hi-Z pyramid -- all three bound via
        // update_descriptors(), the same reason DeferredLightingPass::set_gbuffer_images() is
        // called after every recreate() in pbr_render_pipeline.h.
        FullscreenStageDesc raw_sd;
        raw_sd.vert_spv = vert_spv;
        raw_sd.frag_spv = raw_frag_spv;
        raw_sd.leading_layouts = {&camera_layout};
        raw_sd.owned_sets = {sampled(3)};
        raw_sd.push_constants = {{ShaderStage::Fragment, 0, sizeof(PushConstants)}};
        raw_ = std::make_unique<FullscreenStage>(device, *raw_render_pass_, raw_sd);

        // Resolve pass. No camera set -- reprojection uses the prev_view_proj push constant,
        // matching SsrPass's resolve set: current raw AO, history AO, G2 position, G1 normal.
        FullscreenStageDesc resolve_sd;
        resolve_sd.vert_spv = vert_spv;
        resolve_sd.frag_spv = resolve_frag_spv;
        resolve_sd.owned_sets = {sampled(4)};
        resolve_sd.push_constants = {{ShaderStage::Fragment, 0, sizeof(ResolvePushConstants)}};
        resolve_ = std::make_unique<FullscreenStage>(device, *resolve_render_pass_, resolve_sd);

        // Blur pass: resolved AO, G1 normal, G2 position (bilateral weights).
        FullscreenStageDesc blur_sd;
        blur_sd.vert_spv = vert_spv;
        blur_sd.frag_spv = blur_frag_spv;
        blur_sd.owned_sets = {sampled(3)};
        blur_sd.push_constants = {{ShaderStage::Fragment, 0, sizeof(BlurPushConstants)}};
        blur_ = std::make_unique<FullscreenStage>(device, *blur_render_pass_, blur_sd);
    }

    ~SsaoPass() {
        destroy_resources_();
        // neutral_image_ (a memory::Image) cleans up its own view + allocation.
    }

    SsaoPass(const SsaoPass&) = delete;
    SsaoPass& operator=(const SsaoPass&) = delete;

    void recreate(uint32_t width, uint32_t height) {
        destroy_resources_();
        width_  = width;
        height_ = height;

        create_target_(raw_image_, raw_allocation_, raw_view_, raw_framebuffer_, *raw_render_pass_,
                       VK_FORMAT_R8_UNORM, 0);
        // resolved_image_ is the source of the vkCmdCopyImage in update_ssao_history_() --
        // needs TRANSFER_SRC on top of the usual COLOR_ATTACHMENT | SAMPLED, or that copy is a
        // VUID-vkCmdCopyImage-srcImage-00126 validation error.
        create_target_(resolved_image_, resolved_allocation_, resolved_view_, resolved_framebuffer_,
                       *resolve_render_pass_, kResolveFormat, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        create_target_(blur_image_, blur_allocation_, blur_view_, blur_framebuffer_, *blur_render_pass_,
                       VK_FORMAT_R8_UNORM, 0);

        // History is a plain persistent image (not one of the render-target triples above): it's
        // never rendered into via a render pass, only vkCmdCopyImage'd into at the end of
        // execute(). Allocating a fresh image on resize means history from the previous
        // resolution can never leak into the new one, and history_initialized_ = false keeps
        // the resolve pass from reading it until update_ssao_history_() populates it.
        // kResolveFormat, matching resolved_image_: update_ssao_history_() connects the two with
        // a vkCmdCopyImage, which requires compatible formats on both sides.
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width_, height_, kResolveFormat,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        history_initialized_ = false;

        // raw_view_/resolved_view_ stay raw VkImageView (part of the render-target-triple
        // machinery with no sealed equivalent -- see create_target_()'s doc), wrapped per-use
        // via detail::wrap() since this is gfxcoopa's own internal code.
        resolve_->set().bind_image(0, coopa::gfx::detail::wrap(raw_view_), *raw_sampler_);
        blur_->set().bind_image(0, coopa::gfx::detail::wrap(resolved_view_), *raw_sampler_);
    }

    /// g1_view/g2_view: the live G-Buffer normal/position views (rebound every resize).
    /// hiz_view/hiz_sampler: the SSR chain's Hi-Z min-depth pyramid and its every-mip-reachable
    /// sampler (HiZPass::full_hiz_view_typed()/sampler()) -- the raw pass marches it.
    /// linear_sampler: used only for the resolve pass's history read, which samples at a
    /// reprojected (non-texel-aligned) UV -- everything else here stays NEAREST.
    void update_descriptors(coopa::gfx::TextureView g1_view, coopa::gfx::TextureView g2_view,
                            coopa::gfx::TextureView hiz_view, const util::Sampler& hiz_sampler,
                            const util::Sampler& linear_sampler) {
        raw_->set().bind_image(0, g1_view, linear_sampler);
        raw_->set().bind_image(1, g2_view, linear_sampler);
        raw_->set().bind_image(2, hiz_view, hiz_sampler);

        resolve_->set().bind_image(1, history_image_->view_typed(), linear_sampler);
        resolve_->set().bind_image(2, g2_view, *raw_sampler_);
        resolve_->set().bind_image(3, g1_view, *raw_sampler_);

        blur_->set().bind_image(1, g1_view, *raw_sampler_);
        blur_->set().bind_image(2, g2_view, *raw_sampler_);
    }

    void execute(coopa::gfx::command::CommandBuffer& cmd,
                 const coopa::gfx::pipeline::DescriptorSet& camera_set,
                 const Params& params)
    {
        // On the very first frame (or right after a resize), history_image_ is UNDEFINED --
        // transition it once before it's bound as a sampled image in the resolve pass below. The
        // resolve shader doesn't read it in that case (history_valid = 0 in the push constant),
        // but the descriptor binding still needs a valid layout at draw time regardless of the
        // runtime branch. Identical reasoning/shape to SsrPass::execute()'s equivalent check.
        if (!history_initialized_) {
            VkImageMemoryBarrier barrier{};
            barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.image                           = history_image_->handle();
            barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel   = 0;
            barrier.subresourceRange.levelCount     = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount     = 1;
            barrier.srcAccessMask                   = 0;
            barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        // --- 1. Raw AO pass ---
        VkClearValue raw_clear{};
        raw_clear.color = {{1.0f, 0.0f, 0.0f, 0.0f}};

        VkRenderPassBeginInfo raw_rp_info{};
        raw_rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        raw_rp_info.renderPass        = raw_render_pass_->handle();
        raw_rp_info.framebuffer       = raw_framebuffer_;
        raw_rp_info.renderArea.offset = {0, 0};
        raw_rp_info.renderArea.extent = {width_, height_};
        raw_rp_info.clearValueCount   = 1;
        raw_rp_info.pClearValues      = &raw_clear;

        vkCmdBeginRenderPass(cmd.handle(), &raw_rp_info, VK_SUBPASS_CONTENTS_INLINE);

        raw_->bind(cmd, width_, height_);
        cmd.bind_descriptor_set(camera_set, 0);

        PushConstants raw_pc{};
        raw_pc.inv_proj       = params.inv_proj;
        raw_pc.radius         = params.radius;
        raw_pc.bias           = params.bias;
        raw_pc.power          = params.power;
        raw_pc.slices         = params.slices;
        raw_pc.steps          = params.steps;
        raw_pc.resolution_x   = static_cast<float>(width_);
        raw_pc.resolution_y   = static_cast<float>(height_);
        raw_pc.noise_rotation = params.temporal_enabled ? params.noise_rotation : 0;
        raw_pc.max_mip        = params.max_mip;
        raw_pc.max_radius_px  = params.max_radius_px;
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, raw_pc);
        raw_->draw(cmd);

        cmd.end_render_pass();

        // Availability/visibility barrier before the resolve pass reads it -- the shared
        // RenderPass's only subpass dependency (EXTERNAL -> 0, srcAccessMask = 0) does not make
        // the raw target's colour write visible to a later fragment-shader read of it. Identical
        // reasoning to the per-mip barrier in HiZPass::execute().
        image_barrier_(cmd, raw_image_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        // --- 2. Temporal resolve pass: blend raw_view_ (this frame) with history_image_ (last
        // frame's resolved output) into resolved_view_, which the blur pass below reads instead
        // of raw_view_ directly. When temporal_enabled is false, max_accum = 0 degenerates
        // this into a pure passthrough of the current frame -- one code path, no branching
        // pipeline structure, matching SsrPass's resolve.
        VkClearValue resolve_clear{};
        resolve_clear.color = {{1.0f, 0.0f, 0.0f, 0.0f}};

        VkRenderPassBeginInfo resolve_rp_info{};
        resolve_rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        resolve_rp_info.renderPass        = resolve_render_pass_->handle();
        resolve_rp_info.framebuffer       = resolved_framebuffer_;
        resolve_rp_info.renderArea.offset = {0, 0};
        resolve_rp_info.renderArea.extent = {width_, height_};
        resolve_rp_info.clearValueCount   = 1;
        resolve_rp_info.pClearValues      = &resolve_clear;

        vkCmdBeginRenderPass(cmd.handle(), &resolve_rp_info, VK_SUBPASS_CONTENTS_INLINE);

        resolve_->bind(cmd, width_, height_);

        ResolvePushConstants resolve_pc{};
        resolve_pc.prev_view_proj = params.prev_view_proj;
        resolve_pc.resolution_x   = static_cast<float>(width_);
        resolve_pc.resolution_y   = static_cast<float>(height_);
        resolve_pc.max_accum      = params.temporal_enabled
            ? static_cast<float>(params.temporal_frames) : 0.0f;
        // Both a history IMAGE (history_initialized_) and a previous view-projection MATRIX
        // (prev_view_proj_valid) must exist -- the matrix lags the image by a frame on a
        // fresh-start/resize, so ANDing avoids reprojecting with a stale/identity matrix.
        resolve_pc.history_valid = (history_initialized_ && params.prev_view_proj_valid) ? 1 : 0;
        resolve_pc.camera_pos_x      = params.camera_pos.x;
        resolve_pc.camera_pos_y      = params.camera_pos.y;
        resolve_pc.camera_pos_z      = params.camera_pos.z;
        // prev_camera_pos_ is only read by the shader when history_valid is 1, which already
        // requires a history image AND a previous view-projection -- the same frame of lag this
        // eye position has, so it needs no validity flag of its own.
        resolve_pc.prev_camera_pos_x = prev_camera_pos_.x;
        resolve_pc.prev_camera_pos_y = prev_camera_pos_.y;
        resolve_pc.prev_camera_pos_z = prev_camera_pos_.z;
        resolve_pc.frozen            = params.frozen ? 1 : 0;
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, resolve_pc);
        resolve_->draw(cmd);

        cmd.end_render_pass();

        image_barrier_(cmd, resolved_image_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        // --- 3. Bilateral blur pass ---
        VkClearValue blur_clear{};
        blur_clear.color = {{1.0f, 0.0f, 0.0f, 0.0f}};

        VkRenderPassBeginInfo blur_rp_info{};
        blur_rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        blur_rp_info.renderPass        = blur_render_pass_->handle();
        blur_rp_info.framebuffer       = blur_framebuffer_;
        blur_rp_info.renderArea.offset = {0, 0};
        blur_rp_info.renderArea.extent = {width_, height_};
        blur_rp_info.clearValueCount   = 1;
        blur_rp_info.pClearValues      = &blur_clear;

        vkCmdBeginRenderPass(cmd.handle(), &blur_rp_info, VK_SUBPASS_CONTENTS_INLINE);

        blur_->bind(cmd, width_, height_);

        BlurPushConstants blur_pc{};
        blur_pc.plane_sigma = params.blur_plane_sigma;
        blur_pc.max_accum = params.temporal_enabled
            ? static_cast<float>(params.temporal_frames) : 0.0f;
        blur_pc.motion_px = params.motion_px;
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, blur_pc);
        blur_->draw(cmd);

        cmd.end_render_pass();

        // --- 4. Copy resolved_view_ into history_image_ for next frame's resolve pass.
        update_ssao_history_(cmd);
        prev_camera_pos_ = params.camera_pos;
    }

    VkImageView output_view() const { return blur_view_; }
    coopa::gfx::TextureView output_view_typed() const { return coopa::gfx::detail::wrap(blur_view_); }
    const util::Sampler& sampler() const { return *blur_sampler_; }

    /// A permanent 1x1 texel = 255 (fully unoccluded) texture, valid from construction and never
    /// touched again. Bind this instead of output_view() on frames execute() doesn't run (SSAO
    /// disabled) -- blur_view_ is only transitioned to SHADER_READ_ONLY_OPTIMAL by execute()
    /// actually running, so binding it unconditionally would read an image still sitting in
    /// VK_IMAGE_LAYOUT_UNDEFINED on the very first disabled frame.
    VkImageView neutral_view() const { return neutral_image_->view(); }
    coopa::gfx::TextureView neutral_view_typed() const { return neutral_image_->view_typed(); }

    /// Drops the accumulated temporal history. Call when execute() is skipped for a frame (SSAO
    /// disabled) -- otherwise the next re-enabled frame's resolve pass blends against AO captured
    /// under a camera pose from an arbitrary number of frames ago. Safe to call repeatedly.
    void invalidate_history() { history_initialized_ = false; }

private:
    void image_barrier_(coopa::gfx::command::CommandBuffer& cmd, VkImage image,
                        VkImageLayout old_layout, VkImageLayout new_layout,
                        VkAccessFlags src_access, VkAccessFlags dst_access,
                        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = old_layout;
        barrier.newLayout                       = new_layout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = src_access;
        barrier.dstAccessMask                   = dst_access;

        vkCmdPipelineBarrier(cmd.handle(), src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // Copies resolved_image_ into history_image_ (same barrier/copy/barrier pattern as
    // SsrPass::update_ssr_history_ / TaaPass::update_history), so the next frame's resolve pass
    // has something to blend against.
    void update_ssao_history_(coopa::gfx::command::CommandBuffer& cmd) {
        VkImage src_image = resolved_image_;
        VkImage dst_image = history_image_->handle();

        VkImageMemoryBarrier barriers[2]{};

        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src_image;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;
        barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // Already in SHADER_READ_ONLY_OPTIMAL: either from the one-time UNDEFINED transition at
        // the top of execute() (frame 0) or from the end of this same function last frame.
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image = dst_image;
        barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[1].subresourceRange.baseMipLevel = 0;
        barriers[1].subresourceRange.levelCount = 1;
        barriers[1].subresourceRange.baseArrayLayer = 0;
        barriers[1].subresourceRange.layerCount = 1;
        barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, barriers);

        VkImageCopy copy_region{};
        copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.srcSubresource.layerCount = 1;
        copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.dstSubresource.layerCount = 1;
        copy_region.extent = { width_, height_, 1 };

        vkCmdCopyImage(cmd.handle(), src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, barriers);

        history_initialized_ = true;
    }

    void create_neutral_texture_(coopa::gfx::command::CommandPool& cmd_pool) {
        uint8_t white = 255;
        neutral_image_ = coopa::gfx::memory::upload_image_2d(
            device_, allocator_, cmd_pool, &white, 1, 1, coopa::gfx::Format::R8_Unorm, 1);
    }

    void create_target_(VkImage& image, VmaAllocation& allocation, VkImageView& view,
                        VkFramebuffer& framebuffer, coopa::gfx::pipeline::RenderPass& render_pass,
                        VkFormat format, VkImageUsageFlags extra_usage)
    {
        VkImageCreateInfo img_info{};
        img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType     = VK_IMAGE_TYPE_2D;
        img_info.format        = format;
        img_info.extent        = {width_, height_, 1};
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | extra_usage;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

        GFX_VK_CHECK(vmaCreateImage(allocator_.handle(), &img_info, &alloc_info, &image, &allocation, nullptr));

        VkImageViewCreateInfo view_info{};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = image;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format                          = format;
        view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel   = 0;
        view_info.subresourceRange.levelCount     = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount     = 1;

        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view));

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass.handle();
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = &view;
        fb_info.width           = width_;
        fb_info.height          = height_;
        fb_info.layers          = 1;

        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &framebuffer));
    }

    void destroy_resources_() {
        destroy_target_(raw_view_, raw_framebuffer_, raw_image_, raw_allocation_);
        destroy_target_(resolved_view_, resolved_framebuffer_, resolved_image_, resolved_allocation_);
        destroy_target_(blur_view_, blur_framebuffer_, blur_image_, blur_allocation_);
        history_image_.reset();
    }

    void destroy_target_(VkImageView& view, VkFramebuffer& framebuffer, VkImage& image, VmaAllocation& allocation) {
        if (framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), framebuffer, nullptr);
            framebuffer = VK_NULL_HANDLE;
        }
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_.handle(), view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (image != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_.handle(), image, allocation);
            image      = VK_NULL_HANDLE;
            allocation = VK_NULL_HANDLE;
        }
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    uint32_t width_  = 0;
    uint32_t height_ = 0;

    VkImage       raw_image_       = VK_NULL_HANDLE;
    VmaAllocation raw_allocation_  = VK_NULL_HANDLE;
    VkImageView   raw_view_        = VK_NULL_HANDLE;
    VkFramebuffer raw_framebuffer_ = VK_NULL_HANDLE;

    VkImage       resolved_image_       = VK_NULL_HANDLE;
    VmaAllocation resolved_allocation_  = VK_NULL_HANDLE;
    VkImageView   resolved_view_        = VK_NULL_HANDLE;
    VkFramebuffer resolved_framebuffer_ = VK_NULL_HANDLE;

    VkImage       blur_image_       = VK_NULL_HANDLE;
    VmaAllocation blur_allocation_  = VK_NULL_HANDLE;
    VkImageView   blur_view_        = VK_NULL_HANDLE;
    VkFramebuffer blur_framebuffer_ = VK_NULL_HANDLE;

    // Persistent across resizes only via recreate() rebuilding it -- not one of the render-target
    // triples above, since it's never a render pass attachment, only a vkCmdCopyImage destination.
    std::unique_ptr<coopa::gfx::memory::Image> history_image_;
    bool history_initialized_ = false;
    // The eye position history_image_ was resolved from, i.e. the origin its stored distance
    // channel is measured against. Written at the end of execute(); read only on the next call,
    // and only when history_valid says that history exists.
    glm::vec3 prev_camera_pos_{0.0f};

    std::unique_ptr<coopa::gfx::memory::Image> neutral_image_;

    std::unique_ptr<util::Sampler>          raw_sampler_;
    std::unique_ptr<util::Sampler>          blur_sampler_;

    std::unique_ptr<coopa::gfx::pipeline::RenderPass> raw_render_pass_;
    std::unique_ptr<coopa::gfx::pipeline::RenderPass> resolve_render_pass_;
    std::unique_ptr<coopa::gfx::pipeline::RenderPass> blur_render_pass_;
    std::unique_ptr<FullscreenStage> raw_;      ///< Stage 1: Hi-Z horizon-march AO (GTAO).
    std::unique_ptr<FullscreenStage> resolve_;  ///< Stage 2: temporal reprojection.
    std::unique_ptr<FullscreenStage> blur_;     ///< Stage 3: bilateral blur.
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SSAO_PASS_H
