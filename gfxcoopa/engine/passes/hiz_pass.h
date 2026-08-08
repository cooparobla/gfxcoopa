#include <gfxcoopa/engine/util/fullscreen_quad.h>
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
#include <gfxcoopa/pipeline/pipeline.h>
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
        // 1. Shaders
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

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

        // 4. Descriptor Set Layout (Set 0: Binding 0 input depth)
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, bindings);

        // 5. Pipeline Configuration
        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, *render_pass_,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{desc_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
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

        // Create Descriptor Pool & Descriptor Sets
        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device_, mip_levels_,
            std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mip_levels_}}
        );

        desc_sets_.resize(mip_levels_);
        for (uint32_t m = 0; m < mip_levels_; ++m) {
            desc_sets_[m] = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
                device_, *desc_pool_, *desc_layout_
            );
        }
    }

    void update_descriptors(VkImageView gbuffer_depth_view) {
        if (mip_levels_ == 0) return;
        // Level 0 samples G-Buffer depth
        desc_sets_[0]->bind_image(0, gbuffer_depth_view, sampler_->handle());

        // Level m >= 1 samples mip_views_[m-1]
        for (uint32_t m = 1; m < mip_levels_; ++m) {
            desc_sets_[m]->bind_image(0, mip_views_[m - 1], sampler_->handle());
        }
    }

    void execute(coopa::gfx::command::CommandBuffer& cmd,
                 VkImage gbuffer_depth_image,
                 VkImageView gbuffer_depth_view)
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

        cmd.bind_pipeline(*pipeline_);

        // 2. Loop through all mip levels
        for (uint32_t m = 0; m < mip_levels_; ++m) {
            uint32_t mw = std::max(1u, width_ >> m);
            uint32_t mh = std::max(1u, height_ >> m);

            PushConstants pc{};
            pc.src_size      = (m == 0) ? glm::ivec2(width_, height_) : glm::ivec2(std::max(1u, width_ >> (m - 1)), std::max(1u, height_ >> (m - 1)));
            pc.is_first_pass = (m == 0) ? 1 : 0;

            cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);

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

            cmd.bind_descriptor_set(pipeline_->layout(), *desc_sets_[m], 0);

            cmd.draw(3);

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
        desc_sets_.clear();
        desc_pool_.reset();
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
    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<util::Sampler>               sampler_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::vector<std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>> desc_sets_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_HIZ_PASS_H
