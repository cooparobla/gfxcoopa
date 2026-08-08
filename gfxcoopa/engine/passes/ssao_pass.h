#include <gfxcoopa/engine/util/fullscreen_quad.h>
#include <gfxcoopa/engine/util/ssao_kernel.h>
/**
 * @file ssao_pass.h
 * @brief Screen-Space Ambient Occlusion: world-space hemisphere sampling, temporal resolve, and
 * a depth/normal-aware bilateral blur.
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
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/engine/util/ssao_kernel.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



/// Bundles three sub-stages behind one object -- raw hemisphere sample, temporal resolve, and a
/// bilateral blur -- the same multi-stage-bundling shape SsrPass uses for its raymarch/resolve/
/// composite trio. Callers only ever touch update_descriptors()/execute()/output_view().
///
/// Pipeline order is raw -> resolve -> blur (temporal first, spatial second), matching SsrPass's
/// own raymarch -> resolve -> composite ordering and the standard denoiser convention: a spatial
/// filter over an already temporally-accumulated signal converges faster and doesn't have to
/// fight the same per-frame noise the temporal stage exists to remove.
class SsaoPass {
public:
    /// Raw-pass GPU push constants.
    struct PushConstants {
        float radius        = 0.5f;
        float bias          = 0.025f;
        float power         = 1.5f;
        int   kernel_size   = 24;
        float noise_scale_x = 1.0f;
        float noise_scale_y = 1.0f;
        // 0 when temporal resolve is off -- see ssao.frag. Must match Params::temporal_enabled;
        // execute() derives this, callers don't set it directly.
        int   noise_rotation = 0;
    };

    /// Resolve-pass GPU push constants. mat4 first (16-byte aligned), then plain scalars --
    /// vec2 is deliberately flattened to two floats, matching PushConstants' house rule, so nothing
    /// here depends on std430 vec2 base-alignment (8 bytes) lining up with this C++ struct's layout.
    struct ResolvePushConstants {
        glm::mat4 prev_view_proj = glm::mat4(1.0f); // offset 0
        float     resolution_x   = 1.0f;            // offset 64
        float     resolution_y   = 1.0f;            // offset 68
        float     blend_factor   = 0.0f;             // offset 72
        int       history_valid  = 0;                // offset 76
    };
    static_assert(sizeof(ResolvePushConstants) == 80, "ssao_resolve.frag's PushConstants block must match this layout byte-for-byte");

    /// Blur-pass GPU push constants.
    struct BlurPushConstants {
        float radius = 0.5f; // ssao_radius -- sigma for the plane-distance weight scales with it
    };

    /// execute() parameters. Mirrors SsrPass::Params' shape: raw-pass tunables plus temporal
    /// resolve controls, all bundled so pbr_render_pipeline.h only builds one struct per frame.
    struct Params {
        float radius        = 0.5f;
        float bias          = 0.025f;
        float power         = 1.5f;
        int   kernel_size   = 24;
        float noise_scale_x = 1.0f;
        float noise_scale_y = 1.0f;
        // Caller-driven frame counter (e.g. ssao_frame_index_ & 0x7 in pbr_render_pipeline.h).
        // Only takes effect when temporal_enabled is true -- see execute()'s raw_pc.noise_rotation.
        int   noise_rotation = 0;

        bool  temporal_enabled = true;
        float temporal_blend   = 0.85f;
        // Reprojection: previous frame's JITTERED proj * view, and whether it (and the history
        // image) are actually valid yet -- see execute()'s history_valid derivation.
        glm::mat4 prev_view_proj       = glm::mat4(1.0f);
        bool      prev_view_proj_valid = false;
    };

    SsaoPass(coopa::gfx::core::Device& device,
             coopa::gfx::memory::Allocator& allocator,
             coopa::gfx::command::CommandPool& cmd_pool,
             VkDescriptorSetLayout camera_layout,
             const std::string& vert_spv,
             const std::string& raw_frag_spv,
             const std::string& resolve_frag_spv,
             const std::string& blur_frag_spv)
        : device_(device), allocator_(allocator)
    {
        kernel_ = std::make_unique<util::SsaoKernel>(device, allocator);
        noise_  = std::make_unique<util::SsaoNoiseTexture>(device, allocator, cmd_pool);
        create_neutral_texture_(cmd_pool);

        // All three targets hold an occlusion mask, not colour to be filtered -- NEAREST
        // throughout (the resolve pass's history read uses the caller's linear_sampler instead,
        // since it samples at a reprojected, non-texel-aligned UV).
        raw_sampler_  = std::make_unique<util::Sampler>(device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        blur_sampler_ = std::make_unique<util::Sampler>(device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        vert_shader_       = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        raw_frag_shader_   = std::make_unique<coopa::gfx::pipeline::Shader>(device, raw_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        resolve_frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, resolve_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        blur_frag_shader_  = std::make_unique<coopa::gfx::pipeline::Shader>(device, blur_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Single R8_UNORM attachment, no depth -- same shape as HiZPass/SceneColorMipPass, just
        // one mip instead of a chain. All three sub-stages share this shape.
        raw_render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device, VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        resolve_render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device, VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        blur_render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device, VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        // Set 1 for the raw pass (Set 0 is the caller's reused camera layout): G1 normal,
        // G2 position, noise texture, kernel UBO.
        std::vector<VkDescriptorSetLayoutBinding> raw_bindings = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        raw_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, raw_bindings);
        raw_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1,
            std::vector<VkDescriptorPoolSize>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
            }
        );
        raw_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *raw_desc_pool_, *raw_desc_layout_);
        raw_desc_set_->bind_buffer(3, kernel_->buffer());
        raw_desc_set_->bind_image(2, noise_->view(), noise_->sampler().handle());
        // Bindings 0/1 (G1/G2) are rebound every resize via update_descriptors(), the same
        // reason DeferredLightingPass::set_gbuffer_images() is called after every recreate() in
        // pbr_render_pipeline.h.

        // Set 0 for the resolve pass (no camera set needed -- reprojection uses the prev_view_proj
        // push constant, matching SsrPass's resolve set): current raw AO, history AO, G2 position,
        // G1 normal (background early-out).
        std::vector<VkDescriptorSetLayoutBinding> resolve_bindings = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        resolve_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, resolve_bindings);
        resolve_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}}
        );
        resolve_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *resolve_desc_pool_, *resolve_desc_layout_);

        // Set 0 for the blur pass: resolved AO, G1 normal, G2 position (bilateral weights).
        std::vector<VkDescriptorSetLayoutBinding> blur_bindings = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        blur_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, blur_bindings);
        blur_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}}
        );
        blur_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *blur_desc_pool_, *blur_desc_layout_);

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange raw_pc_range{};
        raw_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        raw_pc_range.offset     = 0;
        raw_pc_range.size       = sizeof(PushConstants);

        raw_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, *raw_render_pass_,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), raw_frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{camera_layout, raw_desc_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{raw_pc_range}
        );

        VkPushConstantRange resolve_pc_range{};
        resolve_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        resolve_pc_range.offset     = 0;
        resolve_pc_range.size       = sizeof(ResolvePushConstants);

        resolve_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, *resolve_render_pass_,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), resolve_frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{resolve_desc_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{resolve_pc_range}
        );

        VkPushConstantRange blur_pc_range{};
        blur_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        blur_pc_range.offset     = 0;
        blur_pc_range.size       = sizeof(BlurPushConstants);

        blur_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, *blur_render_pass_,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), blur_frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{blur_desc_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{blur_pc_range}
        );
    }

    ~SsaoPass() {
        destroy_resources_();
        if (neutral_view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_.handle(), neutral_view_, nullptr);
        }
        if (neutral_image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_.handle(), neutral_image_, neutral_allocation_);
        }
    }

    SsaoPass(const SsaoPass&) = delete;
    SsaoPass& operator=(const SsaoPass&) = delete;

    void recreate(uint32_t width, uint32_t height) {
        destroy_resources_();
        width_  = width;
        height_ = height;

        create_target_(raw_image_, raw_allocation_, raw_view_, raw_framebuffer_, *raw_render_pass_, 0);
        // resolved_image_ is the source of the vkCmdCopyImage in update_ssao_history_() --
        // needs TRANSFER_SRC on top of the usual COLOR_ATTACHMENT | SAMPLED, or that copy is a
        // VUID-vkCmdCopyImage-srcImage-00126 validation error.
        create_target_(resolved_image_, resolved_allocation_, resolved_view_, resolved_framebuffer_,
                       *resolve_render_pass_, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        create_target_(blur_image_, blur_allocation_, blur_view_, blur_framebuffer_, *blur_render_pass_, 0);

        // History is a plain persistent image (not one of the render-target triples above): it's
        // never rendered into via a render pass, only vkCmdCopyImage'd into at the end of
        // execute(). A fresh image on every resize means stale history from the old resolution
        // can never leak into the new one -- history_initialized_ = false forces the resolve
        // pass to ignore it until update_ssao_history_() actually populates it again.
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width_, height_, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        history_initialized_ = false;

        resolve_desc_set_->bind_image(0, raw_view_, raw_sampler_->handle());
        blur_desc_set_->bind_image(0, resolved_view_, raw_sampler_->handle());
    }

    /// g1_view/g2_view: the live G-Buffer normal/position views (rebound every resize).
    /// linear_sampler: used only for the resolve pass's history read, which samples at a
    /// reprojected (non-texel-aligned) UV -- everything else here stays NEAREST.
    void update_descriptors(VkImageView g1_view, VkImageView g2_view, const util::Sampler& linear_sampler) {
        raw_desc_set_->bind_image(0, g1_view, linear_sampler.handle());
        raw_desc_set_->bind_image(1, g2_view, linear_sampler.handle());

        resolve_desc_set_->bind_image(1, history_image_->view(), linear_sampler.handle());
        resolve_desc_set_->bind_image(2, g2_view, raw_sampler_->handle());
        resolve_desc_set_->bind_image(3, g1_view, raw_sampler_->handle());

        blur_desc_set_->bind_image(1, g1_view, raw_sampler_->handle());
        blur_desc_set_->bind_image(2, g2_view, raw_sampler_->handle());
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

        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);
        cmd.bind_pipeline(*raw_pipeline_);
        cmd.bind_descriptor_set(raw_pipeline_->layout(), camera_set, 0);
        cmd.bind_descriptor_set(raw_pipeline_->layout(), *raw_desc_set_, 1);

        PushConstants raw_pc{};
        raw_pc.radius         = params.radius;
        raw_pc.bias           = params.bias;
        raw_pc.power          = params.power;
        raw_pc.kernel_size    = params.kernel_size;
        raw_pc.noise_scale_x  = params.noise_scale_x;
        raw_pc.noise_scale_y  = params.noise_scale_y;
        raw_pc.noise_rotation = params.temporal_enabled ? params.noise_rotation : 0;
        cmd.push_constants(raw_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &raw_pc);
        cmd.draw(3);

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
        // of raw_view_ directly. When temporal_enabled is false, blend_factor = 0 degenerates
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

        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);
        cmd.bind_pipeline(*resolve_pipeline_);
        cmd.bind_descriptor_set(resolve_pipeline_->layout(), *resolve_desc_set_, 0);

        ResolvePushConstants resolve_pc{};
        resolve_pc.prev_view_proj = params.prev_view_proj;
        resolve_pc.resolution_x   = static_cast<float>(width_);
        resolve_pc.resolution_y   = static_cast<float>(height_);
        resolve_pc.blend_factor   = params.temporal_enabled ? params.temporal_blend : 0.0f;
        // Both a history IMAGE (history_initialized_) and a previous view-projection MATRIX
        // (prev_view_proj_valid) must exist -- the matrix lags the image by a frame on a
        // fresh-start/resize, so ANDing avoids reprojecting with a stale/identity matrix.
        resolve_pc.history_valid = (history_initialized_ && params.prev_view_proj_valid) ? 1 : 0;
        cmd.push_constants(resolve_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ResolvePushConstants), &resolve_pc);
        cmd.draw(3);

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

        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);
        cmd.bind_pipeline(*blur_pipeline_);
        cmd.bind_descriptor_set(blur_pipeline_->layout(), *blur_desc_set_, 0);

        BlurPushConstants blur_pc{};
        blur_pc.radius = params.radius;
        cmd.push_constants(blur_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BlurPushConstants), &blur_pc);
        cmd.draw(3);

        cmd.end_render_pass();

        // --- 4. Copy resolved_view_ into history_image_ for next frame's resolve pass.
        update_ssao_history_(cmd);
    }

    VkImageView output_view() const { return blur_view_; }
    const util::Sampler& sampler() const { return *blur_sampler_; }

    /// A permanent 1x1 texel = 255 (fully unoccluded) texture, valid from construction and never
    /// touched again. Bind this instead of output_view() on frames execute() doesn't run (SSAO
    /// disabled) -- blur_view_ is only transitioned to SHADER_READ_ONLY_OPTIMAL by execute()
    /// actually running, so binding it unconditionally would read an image still sitting in
    /// VK_IMAGE_LAYOUT_UNDEFINED on the very first disabled frame.
    VkImageView neutral_view() const { return neutral_view_; }

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

        coopa::gfx::memory::Buffer staging(
            device_, allocator_, 1,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
        );
        staging.upload(&white, 1);

        VkImageCreateInfo img_info{};
        img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType     = VK_IMAGE_TYPE_2D;
        img_info.format        = VK_FORMAT_R8_UNORM;
        img_info.extent        = {1, 1, 1};
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

        GFX_VK_CHECK(vmaCreateImage(allocator_.handle(), &img_info, &alloc_info, &neutral_image_, &neutral_allocation_, nullptr));

        VkImageViewCreateInfo view_info{};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = neutral_image_;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format                          = VK_FORMAT_R8_UNORM;
        view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel   = 0;
        view_info.subresourceRange.levelCount     = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount     = 1;

        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &neutral_view_));

        VkCommandBufferAllocateInfo alloc_cmd_info{};
        alloc_cmd_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_cmd_info.commandPool        = cmd_pool.handle();
        alloc_cmd_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_cmd_info.commandBufferCount = 1;

        VkCommandBuffer cmd_handle = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_.handle(), &alloc_cmd_info, &cmd_handle);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_handle, &begin_info);

        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = neutral_image_;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = 0;
        barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd_handle, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset      = 0;
        copy_region.bufferRowLength   = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel       = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount     = 1;
        copy_region.imageOffset                     = {0, 0, 0};
        copy_region.imageExtent                     = {1, 1, 1};

        vkCmdCopyBufferToImage(cmd_handle, staging.handle(), neutral_image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd_handle, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        vkEndCommandBuffer(cmd_handle);

        VkSubmitInfo submit{};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &cmd_handle;
        vkQueueSubmit(device_.graphics_queue(), 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(device_.graphics_queue());

        vkFreeCommandBuffers(device_.handle(), cmd_pool.handle(), 1, &cmd_handle);
    }

    void create_target_(VkImage& image, VmaAllocation& allocation, VkImageView& view,
                        VkFramebuffer& framebuffer, coopa::gfx::pipeline::RenderPass& render_pass,
                        VkImageUsageFlags extra_usage)
    {
        VkImageCreateInfo img_info{};
        img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType     = VK_IMAGE_TYPE_2D;
        img_info.format        = VK_FORMAT_R8_UNORM;
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
        view_info.format                          = VK_FORMAT_R8_UNORM;
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

    VkImage       neutral_image_      = VK_NULL_HANDLE;
    VmaAllocation neutral_allocation_ = VK_NULL_HANDLE;
    VkImageView   neutral_view_       = VK_NULL_HANDLE;

    std::unique_ptr<util::SsaoKernel>       kernel_;
    std::unique_ptr<util::SsaoNoiseTexture> noise_;
    std::unique_ptr<util::Sampler>          raw_sampler_;
    std::unique_ptr<util::Sampler>          blur_sampler_;

    std::unique_ptr<coopa::gfx::pipeline::RenderPass> raw_render_pass_;
    std::unique_ptr<coopa::gfx::pipeline::RenderPass> resolve_render_pass_;
    std::unique_ptr<coopa::gfx::pipeline::RenderPass> blur_render_pass_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>      vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>      raw_frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>      resolve_frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>      blur_frag_shader_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> raw_desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      raw_desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       raw_desc_set_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> resolve_desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      resolve_desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       resolve_desc_set_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> blur_desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      blur_desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       blur_desc_set_;

    std::unique_ptr<coopa::gfx::pipeline::Pipeline> raw_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> resolve_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> blur_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SSAO_PASS_H
