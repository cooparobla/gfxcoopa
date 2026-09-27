/**
 * @file temporal_history_pass.h
 * @brief Shared per-pixel history-validity / sample-count buffer for the temporally
 *        accumulated screen-space effects.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_TEMPORAL_HISTORY_PASS_H
#define GFXCOOPA_ENGINE_PASSES_TEMPORAL_HISTORY_PASS_H

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
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

/**
 * @class TemporalHistoryPass
 * @brief Answers "is this pixel's history the same surface it was last frame?" once per frame
 *        and publishes the resulting per-pixel accumulation count.
 *
 * Unity HDRP's `_HistoryValidityBuffer` plays this role for its ray-tracing denoisers: the
 * disocclusion test depends only on depth and camera motion, so every consumer that
 * accumulates a stochastic screen-space signal — reflections, the diffuse bounce, contact
 * shadows — can share one answer. Holding a *count* rather than a bare valid/invalid flag is
 * what lets those consumers run a converging running average (weight `1/(n+1)`) instead of a
 * fixed-rate exponential blend, which never converges when its input is re-jittered every
 * frame.
 *
 * Output is `kFormat` (RG16F): R carries the pixel's linear view distance (next frame's
 * surface-identity reference), G the count. See `temporal_history.frag` for the test itself.
 *
 * One render target plus one history image, copied at the end of every execute() the way
 * SsrPass manages its own reflection history. A ping-pong would save the copy (SsaoPass does
 * that for its resolve), but it costs every consumer a stable view to bind: descriptors here are
 * bound once at setup, because rebinding per frame is unsafe under an overlapped-frame pipeline,
 * and a consumer pinned to one of two alternating images would read a frame-stale count on half
 * the frames. A full-resolution RG16F copy is 4 MB of transfer; a correct, never-stale count is
 * worth more than that.
 */
class TemporalHistoryPass {
public:
    /// Format of the two ping-pong targets. Two channels of real float range: a linear view
    /// distance needs more than a normalized byte, and the count must stay an exact integer
    /// (half-float represents integers exactly well past any usable accumulation cap).
    static constexpr VkFormat kFormat = VK_FORMAT_R16G16_SFLOAT;

    /// Matches temporal_history.frag's push-constant block byte for byte. mat4 first (16-byte
    /// aligned), then plain scalars with vec2 flattened — the house rule SsaoPass follows.
    struct PushConstants {
        glm::mat4 reproject      = glm::mat4(1.0f);  // offset 0
        float     resolution_x   = 1.0f;             // offset 64
        float     resolution_y   = 1.0f;             // offset 68
        float     max_accum      = 32.0f;            // offset 72
        float     near_z         = 0.1f;             // offset 76
        float     far_z          = 1000.0f;          // offset 80
        float     is_perspective = 1.0f;             // offset 84
        int       history_valid  = 0;                // offset 88
    };
    static_assert(sizeof(PushConstants) == 92,
                  "temporal_history.frag's PushConstants block must match this layout byte-for-byte");

    /// execute() parameters.
    struct Params {
        /// Current clip space -> previous frame's clip space, composed in double precision by the
        /// caller (`glm::dmat4(prev_view_proj) * glm::inverse(glm::dmat4(proj) * glm::dmat4(view))`)
        /// before truncating to float. The world-scale magnitudes inside the two view-projections
        /// cancel in the double product; a float composition is off by whole pixels at
        /// scene scales of a few hundred units. Same scheme TaaPass and SsaoPass use.
        glm::mat4 reproject       = glm::mat4(1.0f);
        /// Whether that matrix (and a rendered history target) exist yet — false on the first
        /// frames and right after a resize.
        bool      reproject_valid = false;
        /// Accumulation cap: how many frames a pixel averages before its consumers' running mean
        /// becomes a fixed-rate blend. 0 pins every count to 1, which makes every consumer a
        /// passthrough of its current frame.
        int       max_accum       = 32;
        float     near_z          = 0.1f;
        float     far_z           = 1000.0f;
        /// False for an orthographic projection — depth is already linear in the raw value there.
        bool      perspective     = true;
    };

    /**
     * @brief Builds the pass. Call recreate() before the first execute().
     *
     * @param device    Vulkan logical device.
     * @param allocator VMA allocator for the two ping-pong targets.
     * @param cmd_pool  One-shot pool for the permanent neutral texture.
     * @param vert_spv  Fullscreen-triangle vertex shader (fullscreen.vert).
     * @param frag_spv  temporal_history.frag.spv.
     */
    TemporalHistoryPass(core::Device& device,
                        memory::Allocator& allocator,
                        command::CommandPool& cmd_pool,
                        const std::string& vert_spv,
                        const std::string& frag_spv)
        : device_(device), allocator_(allocator)
    {
        create_neutral_texture_(cmd_pool);

        // The count is discrete per-pixel data, never something to filter: NEAREST for every
        // read of this buffer by a consumer, and by this pass's own depth tap. The history read
        // below is the one exception — it samples at a reprojected, non-texel-aligned UV.
        coopa::gfx::SamplerDesc nearest_clamp;
        nearest_clamp.min = nearest_clamp.mag = coopa::gfx::Filter::Nearest;
        nearest_clamp.mipmap  = coopa::gfx::MipmapMode::Nearest;
        nearest_clamp.address = coopa::gfx::AddressMode::ClampToEdge;
        nearest_sampler_ = std::make_unique<util::Sampler>(device, nearest_clamp);

        coopa::gfx::SamplerDesc linear_clamp = nearest_clamp;
        linear_clamp.min = linear_clamp.mag = coopa::gfx::Filter::Linear;
        linear_sampler_  = std::make_unique<util::Sampler>(device, linear_clamp);

        render_pass_ = std::make_unique<pipeline::RenderPass>(
            device, kFormat, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        FullscreenStageDesc sd;
        sd.vert_spv = vert_spv;
        sd.frag_spv = frag_spv;
        sd.owned_sets = {{
            {0, coopa::gfx::DescriptorType::CombinedImageSampler, coopa::gfx::ShaderStage::Fragment, 1},
            {1, coopa::gfx::DescriptorType::CombinedImageSampler, coopa::gfx::ShaderStage::Fragment, 1},
        }};
        sd.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(PushConstants)}};
        stage_ = std::make_unique<FullscreenStage>(device, *render_pass_, sd);
    }

    ~TemporalHistoryPass() { destroy_resources_(); }

    TemporalHistoryPass(const TemporalHistoryPass&) = delete;
    TemporalHistoryPass& operator=(const TemporalHistoryPass&) = delete;

    /// Allocates (or reallocates) the target and its history image at the given resolution, and
    /// binds the history read. A fresh pair on resize means counts accumulated at the old
    /// resolution can never leak into the new one. The depth binding comes from
    /// set_depth_image(), which every caller runs after this.
    void recreate(uint32_t width, uint32_t height) {
        destroy_resources_();
        width_  = width;
        height_ = height;

        create_target_();
        history_image_ = std::make_unique<memory::Image>(
            device_, allocator_, width_, height_, kFormat,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO);
        history_initialized_ = false;

        // LINEAR: the history is read at a reprojected, non-texel-aligned UV. Filtering the
        // stored distance across a silhouette is not a flaw here -- it makes the depth test
        // reject exactly at the edge, which is where blended history would be wrong anyway.
        stage_->set().bind_image(1, history_image_->view_typed(), *linear_sampler_);
    }

    /// Points the pass at the live scene depth image (rebind after every resize). Depth must
    /// already be in SHADER_READ_ONLY_OPTIMAL by the time execute() records.
    void set_depth_image(coopa::gfx::TextureView depth_view) {
        stage_->set().bind_image(0, depth_view, *nearest_sampler_);
    }

    void execute(command::CommandBuffer& cmd, const Params& params) {
        // On the first frame (or right after a resize) the history image has never been written
        // and sits in UNDEFINED; transition it before it is bound as a sampled image. The shader
        // doesn't read it in that case (history_valid = 0), but the descriptor still needs a valid
        // layout at draw time regardless of the runtime branch — identical reasoning to
        // SsaoPass::execute()'s own check.
        if (!history_initialized_) {
            VkImageMemoryBarrier barrier{};
            barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.image                           = history_image_->handle();
            barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount     = 1;
            barrier.subresourceRange.layerCount     = 1;
            barrier.srcAccessMask                   = 0;
            barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        VkClearValue clear{};
        clear.color = {{0.0f, 1.0f, 0.0f, 0.0f}};

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = render_pass_->handle();
        rp_info.framebuffer       = framebuffer_;
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {width_, height_};
        rp_info.clearValueCount   = 1;
        rp_info.pClearValues      = &clear;

        vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        stage_->bind(cmd, width_, height_);

        PushConstants pc{};
        pc.reproject      = params.reproject;
        pc.resolution_x   = static_cast<float>(width_);
        pc.resolution_y   = static_cast<float>(height_);
        pc.max_accum      = static_cast<float>(params.max_accum);
        pc.near_z         = params.near_z;
        pc.far_z          = params.far_z;
        pc.is_perspective = params.perspective ? 1.0f : 0.0f;
        // Both a history IMAGE and a reprojection MATRIX must exist — the matrix lags the image
        // by a frame on a fresh start, so ANDing them avoids reprojecting with an identity.
        pc.history_valid  = (history_initialized_ && params.reproject_valid) ? 1 : 0;
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);
        stage_->draw(cmd);

        cmd.end_render_pass();

        // The shared RenderPass's only subpass dependency (EXTERNAL -> 0, srcAccessMask = 0)
        // does not make this colour write visible to a later fragment-shader read of it.
        // Same reasoning as the per-stage barriers in SsaoPass::execute()/HiZPass::execute().
        VkImageMemoryBarrier barrier{};
        barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout                   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                       = image_;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask               = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask               = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        copy_history_(cmd);
        history_initialized_ = true;
    }

    /// @brief The count buffer consumers sample. One image for the pass's whole lifetime (until
    /// a recreate()), so a consumer binds it once at setup and always reads this frame's counts.
    coopa::gfx::TextureView count_view_typed() const { return coopa::gfx::detail::wrap(view_); }

    /// @brief NEAREST/ClampToEdge sampler every consumer should read this buffer through — the
    /// count is discrete per-pixel data and must never be filtered.
    const util::Sampler& sampler() const { return *nearest_sampler_; }

    /// @brief A permanent 1x1 (distance 0, count 1) texture, valid from construction. Bind this
    /// in place of count_view_typed() on a frame execute() never runs: the real targets only
    /// leave VK_IMAGE_LAYOUT_UNDEFINED once it has. Count 1 makes any consumer reading it take
    /// its current frame whole, which is the safe answer.
    coopa::gfx::TextureView neutral_view_typed() const { return neutral_image_->view_typed(); }

    /// Drops the accumulated counts. Call when execute() is skipped for a frame, so the next
    /// re-enabled frame doesn't credit consumers with accumulation that never happened.
    void invalidate_history() { history_initialized_ = false; }

private:
    void create_neutral_texture_(command::CommandPool& cmd_pool) {
        // Half-float (0.0, 1.0): distance 0 never matches any real surface, count 1 means
        // "one sample", so a consumer weights its current frame at 1/1.
        const uint16_t texel[2] = {0x0000, 0x3C00};
        neutral_image_ = memory::upload_image_2d(
            device_, allocator_, cmd_pool, texel, 1, 1, coopa::gfx::Format::RG16_Sfloat, 4);
    }

    void create_target_() {
        VkImageCreateInfo img_info{};
        img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType     = VK_IMAGE_TYPE_2D;
        img_info.format        = kFormat;
        img_info.extent        = {width_, height_, 1};
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_SRC too: copy_history_() reads this image into history_image_ every frame.
        img_info.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                               | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

        GFX_VK_CHECK(vmaCreateImage(allocator_.handle(), &img_info, &alloc_info, &image_,
                                    &allocation_, nullptr));

        VkImageViewCreateInfo view_info{};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = image_;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format                          = kFormat;
        view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.levelCount     = 1;
        view_info.subresourceRange.layerCount     = 1;

        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view_));

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass_->handle();
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = &view_;
        fb_info.width           = width_;
        fb_info.height          = height_;
        fb_info.layers          = 1;

        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &framebuffer_));
    }

    /// Copies this frame's counts into the history image, so the next execute() has something to
    /// reproject against. Structurally identical to SsrPass::copy_history_(), over one image pair.
    void copy_history_(command::CommandBuffer& cmd) {
        VkImageMemoryBarrier barriers[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            barriers[i].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            barriers[i].subresourceRange.levelCount     = 1;
            barriers[i].subresourceRange.layerCount     = 1;
        }
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].image         = image_;
        barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        // Already SHADER_READ_ONLY_OPTIMAL: either from the one-time UNDEFINED transition at the
        // top of execute() (first frame) or from the end of this same function last frame.
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].image         = history_image_->handle();
        barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);

        VkImageCopy copy{};
        copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.srcSubresource.layerCount = 1;
        copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.dstSubresource.layerCount = 1;
        copy.extent                    = {width_, height_, 1};
        vkCmdCopyImage(cmd.handle(), image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       history_image_->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    }

    void destroy_resources_() {
        if (framebuffer_ != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), framebuffer_, nullptr);
            framebuffer_ = VK_NULL_HANDLE;
        }
        if (view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_.handle(), view_, nullptr);
            view_ = VK_NULL_HANDLE;
        }
        if (image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_.handle(), image_, allocation_);
            image_      = VK_NULL_HANDLE;
            allocation_ = VK_NULL_HANDLE;
        }
        // history_image_ (a memory::Image) releases its own view and allocation.
        history_image_.reset();
    }

    core::Device&      device_;
    memory::Allocator& allocator_;

    uint32_t width_  = 0;
    uint32_t height_ = 0;

    VkImage       image_       = VK_NULL_HANDLE;
    VmaAllocation allocation_  = VK_NULL_HANDLE;
    VkImageView   view_        = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;

    /// Last frame's copy of the target, the surface-identity reference the disocclusion test
    /// reprojects onto.
    std::unique_ptr<memory::Image> history_image_;
    bool history_initialized_ = false;

    std::unique_ptr<memory::Image>  neutral_image_;
    std::unique_ptr<util::Sampler>  nearest_sampler_;
    std::unique_ptr<util::Sampler>  linear_sampler_;

    std::unique_ptr<pipeline::RenderPass> render_pass_;
    std::unique_ptr<FullscreenStage>      stage_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_TEMPORAL_HISTORY_PASS_H
