#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file taa_pass.h
 * @brief Temporal Anti-Aliasing (TAA) pass.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_TAA_PASS_H
#define GFXCOOPA_ENGINE_PASSES_TAA_PASS_H

#include <volk/volk.h>
#include <string>
#include <vector>
#include <memory>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/memory/image.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class TaaPass {
public:
    struct PushConstants {
        glm::vec2 resolution;
        float blend_factor;
        float weight_scale;
    };

    TaaPass(coopa::gfx::core::Device& device,
            coopa::gfx::memory::Allocator& allocator,
            coopa::gfx::pipeline::RenderPass& render_pass,
            util::Sampler& sampler,
            uint32_t width,
            uint32_t height,
            const std::string& vert_path,
            const std::string& frag_path)
        : device_(device), allocator_(allocator), sampler_(sampler), width_(width), height_(height)
    {
        // History image with TRANSFER_DST so we can copy into it
        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device, allocator, width, height,
            VK_FORMAT_R8G8B8A8_SRGB, usage,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        history_initialized_ = false;

        // Clear history immediately? Not strictly necessary as it will be initialized soon,
        // but it's good practice. (We assume it's just zeroed or we ignore the first frame).
        
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        
        layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, bindings);

        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}}
        );

        set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *layout_);

        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_path, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_path, VK_SHADER_STAGE_FRAGMENT_BIT);

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, render_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
    }

    void recreate(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        
        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width, height,
            VK_FORMAT_R8G8B8A8_SRGB, usage,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        history_initialized_ = false;

        // Need to re-bind because views changed
        if (last_scene_view_ != VK_NULL_HANDLE) {
            set_source_image(last_scene_view_);
        }
    }

    void set_source_image(VkImageView scene_view) {
        last_scene_view_ = scene_view;
        set_->bind_image(0, scene_view, sampler_.handle());
        set_->bind_image(1, history_image_->view(), sampler_.handle());
    }

    void set_taa_config(float blend_factor, float weight_scale) {
        blend_factor_ = blend_factor;
        weight_scale_ = weight_scale;
    }

    void prepare_history(coopa::gfx::command::CommandBuffer& cmd) {
        if (!history_initialized_) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = history_image_->handle();
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t width, uint32_t height) {

        PushConstants pc{};
        pc.resolution = glm::vec2(static_cast<float>(width), static_cast<float>(height));
        pc.blend_factor = blend_factor_;
        pc.weight_scale = weight_scale_;

        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
        cmd.set_scissor(0, 0, width, height);
        
        cmd.bind_descriptor_set(pipeline_->layout(), *set_, 0);
        cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);

        cmd.draw(3);
    }
    
    // Copy the rendered result back into the history buffer
    void update_history(coopa::gfx::command::CommandBuffer& cmd, targets::OffscreenTarget& post_process_target) {
        VkImage src_image = post_process_target.color_image_object()->handle();
        VkImage dst_image = history_image_->handle();
        
        // Transition both to TRANSFER_SRC / TRANSFER_DST
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

        // Transition back to SHADER_READ_ONLY
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

private:
    coopa::gfx::core::Device& device_;
    coopa::gfx::memory::Allocator& allocator_;
    util::Sampler& sampler_;
    uint32_t width_;
    uint32_t height_;
    bool history_initialized_ = false;
    
    float blend_factor_ = 0.9f;
    float weight_scale_ = 30.0f;
    
    VkImageView last_scene_view_ = VK_NULL_HANDLE;

    std::unique_ptr<coopa::gfx::memory::Image>                 history_image_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       set_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>            pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_TAA_PASS_H
