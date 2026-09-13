/**
 * @file hiz_pass.h
 * @brief Hierarchical Z-Buffer (Hi-Z) Depth Pyramid Generation Pass.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_HIZ_PASS_H
#define GFXCOOPA_ENGINE_PASSES_HIZ_PASS_H

#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
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

class HiZPass {
public:
    struct PushConstants {
        glm::ivec2 src_size;
        int is_first_pass;
    };

    HiZPass(coopa::gfx::core::Device& device,
            coopa::gfx::memory::Allocator& allocator,
            const std::string& vert_spv,
            const std::string& frag_spv)
        : device_(device), allocator_(allocator)
    {
        // 2. Sampler (Nearest filtering for exact depth reads). max_lod is set in recreate()
        // once mip_levels_ is known, so every mip of the Hi-Z pyramid is actually reachable by
        // textureLod() in the SSR raymarch shader (a sampler with maxLod = 0 silently clamps
        // every explicit-LOD read back to mip 0, defeating the whole hierarchy).
        sampler_ = std::make_unique<util::Sampler>(
            device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f
        );

        // 3. Render Pass for R32_SFLOAT attachments
        render_pass_ = std::make_unique<coopa::gfx::pipeline::RenderPass>(
            device,
            VK_FORMAT_R32_SFLOAT,
            VK_FORMAT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        // One sampled image at binding 0 (the previous mip), plus the fragment push
        // constants. One set per mip level, reallocated by rebuild_sets() whenever the
        // chain length changes; the pipeline is built once and stays compatible.
        FullscreenStageDesc sd;
        sd.vert_spv = vert_spv;
        sd.frag_spv = frag_spv;
        sd.owned_sets = {{{0, coopa::gfx::DescriptorType::CombinedImageSampler,
                           coopa::gfx::ShaderStage::Fragment, 1}}};
        sd.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(PushConstants)}};
        stage_ = std::make_unique<FullscreenStage>(device, *render_pass_, sd);
    }

    ~HiZPass() {
        destroy_resources_();
    }

    HiZPass(const HiZPass&) = delete;
    HiZPass& operator=(const HiZPass&) = delete;

    void recreate(uint32_t width, uint32_t height) {
        destroy_resources_();
        width_  = width;
        height_ = height;
        mip_levels_ = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

        // Rebuild the sampler with maxLod = mip_levels_ so textureLod() in ssr.frag can
        // actually reach every coarse mip of the pyramid (see constructor comment above).
        sampler_ = std::make_unique<util::Sampler>(
            device_, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            static_cast<float>(mip_levels_)
        );

        // Allocate image
        VkImageCreateInfo img_info{};
        img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType     = VK_IMAGE_TYPE_2D;
        img_info.format        = VK_FORMAT_R32_SFLOAT;
        img_info.extent        = {width_, height_, 1};
        img_info.mipLevels     = mip_levels_;
        img_info.arrayLayers   = 1;
        img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

        GFX_VK_CHECK(vmaCreateImage(allocator_.handle(), &img_info, &alloc_info, &hiz_image_, &allocation_, nullptr));

        // Create full image view for all mips
        VkImageViewCreateInfo full_view_info{};
        full_view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        full_view_info.image                           = hiz_image_;
        full_view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        full_view_info.format                          = VK_FORMAT_R32_SFLOAT;
        full_view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        full_view_info.subresourceRange.baseMipLevel   = 0;
        full_view_info.subresourceRange.levelCount     = mip_levels_;
        full_view_info.subresourceRange.baseArrayLayer = 0;
        full_view_info.subresourceRange.layerCount     = 1;

        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &full_view_info, nullptr, &full_view_));

        // Create per-mip views & framebuffers
        mip_views_.resize(mip_levels_);
        mip_framebuffers_.resize(mip_levels_);

        for (uint32_t m = 0; m < mip_levels_; ++m) {
            uint32_t mw = std::max(1u, width_ >> m);
            uint32_t mh = std::max(1u, height_ >> m);

            VkImageViewCreateInfo view_info{};
            view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image                           = hiz_image_;
            view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format                          = VK_FORMAT_R32_SFLOAT;
            view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel   = m;
            view_info.subresourceRange.levelCount     = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount     = 1;

            GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &mip_views_[m]));

            VkFramebufferCreateInfo fb_info{};
            fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb_info.renderPass      = render_pass_->handle();
            fb_info.attachmentCount = 1;
            fb_info.pAttachments    = &mip_views_[m];
            fb_info.width           = mw;
            fb_info.height          = mh;
            fb_info.layers          = 1;

            GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &mip_framebuffers_[m]));
        }

        stage_->rebuild_sets(mip_levels_);
    }

    void update_descriptors(coopa::gfx::TextureView gbuffer_depth_view) {
        if (mip_levels_ == 0) return;
        // Level 0 samples G-Buffer depth
        stage_->set(0, 0).bind_image(0, gbuffer_depth_view, *sampler_);

        // Level m >= 1 samples mip_views_[m-1] -- mip_views_ itself stays a raw
        // std::vector<VkImageView> (part of the mip-chain machinery with no sealed
        // equivalent; see the class doc), wrapped per-use via detail::wrap() since this
        // is gfxcoopa's own internal code (the leak gate only scopes consumer repos).
        for (uint32_t m = 1; m < mip_levels_; ++m) {
            stage_->set(0, m).bind_image(0, coopa::gfx::detail::wrap(mip_views_[m - 1]), *sampler_);
        }
    }

    void execute(coopa::gfx::command::CommandBuffer& cmd,
                 VkImage gbuffer_depth_image,
                 coopa::gfx::TextureView gbuffer_depth_view)
    {
        update_descriptors(gbuffer_depth_view);

        // 1. Transition G-Buffer depth buffer for shader read
        VkImageMemoryBarrier depth_barrier{};
        depth_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depth_barrier.oldLayout                       = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depth_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        depth_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        depth_barrier.image                           = gbuffer_depth_image;
        depth_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        depth_barrier.subresourceRange.baseMipLevel   = 0;
        depth_barrier.subresourceRange.levelCount     = 1;
        depth_barrier.subresourceRange.baseArrayLayer = 0;
        depth_barrier.subresourceRange.layerCount     = 1;
        depth_barrier.srcAccessMask                   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        depth_barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd.handle(),
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &depth_barrier);

        cmd.bind_pipeline(stage_->pipeline());

        // 2. Loop through all mip levels
        for (uint32_t m = 0; m < mip_levels_; ++m) {
            uint32_t mw = std::max(1u, width_ >> m);
            uint32_t mh = std::max(1u, height_ >> m);

            PushConstants pc{};
            pc.src_size      = (m == 0) ? glm::ivec2(width_, height_) : glm::ivec2(std::max(1u, width_ >> (m - 1)), std::max(1u, height_ >> (m - 1)));
            pc.is_first_pass = (m == 0) ? 1 : 0;

            cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);

            VkClearValue clear_val{};
            clear_val.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

            VkRenderPassBeginInfo rp_info{};
            rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp_info.renderPass        = render_pass_->handle();
            rp_info.framebuffer       = mip_framebuffers_[m];
            rp_info.renderArea.offset = {0, 0};
            rp_info.renderArea.extent = {mw, mh};
            rp_info.clearValueCount   = 1;
            rp_info.pClearValues      = &clear_val;

            vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);

            cmd.set_viewport(0.0f, 0.0f, static_cast<float>(mw), static_cast<float>(mh));
            cmd.set_scissor(0, 0, mw, mh);

            cmd.bind_descriptor_set(stage_->set(0, m), 0);

            stage_->draw(cmd);

            cmd.end_render_pass();

            // The shared RenderPass's only subpass dependency handles entry synchronization
            // (EXTERNAL -> 0) with srcAccessMask = 0, which is not sufficient to make this
            // mip's color write visible to the next mip's (or, on the last iteration, SSR's)
            // fragment-shader read of it -- an unsynchronized read-after-write hazard across
            // the whole pyramid. Layout doesn't change (render pass finalLayout is already
            // SHADER_READ_ONLY_OPTIMAL, matching the read), this is purely an availability/
            // visibility barrier scoped to the single mip level just written.
            VkImageMemoryBarrier mip_barrier{};
            mip_barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            mip_barrier.oldLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            mip_barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            mip_barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            mip_barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            mip_barrier.image                           = hiz_image_;
            mip_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            mip_barrier.subresourceRange.baseMipLevel   = m;
            mip_barrier.subresourceRange.levelCount     = 1;
            mip_barrier.subresourceRange.baseArrayLayer = 0;
            mip_barrier.subresourceRange.layerCount     = 1;
            mip_barrier.srcAccessMask                   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            mip_barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd.handle(),
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &mip_barrier);
        }
    }

    VkImageView full_hiz_view() const { return full_view_; }
    coopa::gfx::TextureView full_hiz_view_typed() const { return coopa::gfx::detail::wrap(full_view_); }
    uint32_t max_mip_level() const { return mip_levels_ > 0 ? mip_levels_ - 1 : 0; }
    const util::Sampler& sampler() const { return *sampler_; }

private:
    void destroy_resources_() {
        if (full_view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_.handle(), full_view_, nullptr);
            full_view_ = VK_NULL_HANDLE;
        }
        for (auto v : mip_views_) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device_.handle(), v, nullptr);
        }
        mip_views_.clear();

        for (auto fb : mip_framebuffers_) {
            if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device_.handle(), fb, nullptr);
        }
        mip_framebuffers_.clear();

        if (hiz_image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_.handle(), hiz_image_, allocation_);
            hiz_image_ = VK_NULL_HANDLE;
            allocation_ = VK_NULL_HANDLE;
        }
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    uint32_t width_      = 0;
    uint32_t height_     = 0;
    uint32_t mip_levels_ = 0;

    VkImage       hiz_image_  = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView   full_view_  = VK_NULL_HANDLE;

    std::vector<VkImageView>   mip_views_;
    std::vector<VkFramebuffer> mip_framebuffers_;

    std::unique_ptr<coopa::gfx::pipeline::RenderPass>          render_pass_;
    std::unique_ptr<util::Sampler>   sampler_;
    std::unique_ptr<FullscreenStage> stage_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_HIZ_PASS_H
