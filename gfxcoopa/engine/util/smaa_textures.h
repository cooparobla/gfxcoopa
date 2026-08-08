/**
 * @file smaa_textures.h
 * @brief Official Jimenez 2013 SMAA 1x Area and Search textures loader.
 */

#ifndef GFXCOOPA_ENGINE_UTIL_SMAA_TEXTURES_H
#define GFXCOOPA_ENGINE_UTIL_SMAA_TEXTURES_H

#include <volk/volk.h>
#include <vector>
#include <memory>
#include <cstdint>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/util/sampler.h>

// Include official Jimenez SMAA texture headers
#include "/home/coopa/third-party/smaa/Textures/SearchTex.h"
#include "/home/coopa/third-party/smaa/Textures/AreaTex.h"

namespace coopa {
namespace gfx {
namespace engine {
namespace util {

class SmaaTextures {
public:
    SmaaTextures(coopa::gfx::core::Device& device,
                 coopa::gfx::memory::Allocator& allocator,
                 coopa::gfx::command::CommandPool& cmd_pool)
        : device_(device), allocator_(allocator)
    {
        create_search_texture_(cmd_pool);
        create_area_texture_(cmd_pool);

        sampler_ = std::make_unique<Sampler>(
            device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
        );
    }

    VkImageView search_view() const { return search_image_->view(); }
    VkImageView area_view()   const { return area_image_->view(); }
    const Sampler& sampler() const { return *sampler_; }

private:
    void create_search_texture_(coopa::gfx::command::CommandPool& cmd_pool) {
        upload_texture_(cmd_pool, searchTexBytes, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT,
                        VK_FORMAT_R8_UNORM, 1, search_image_);
    }

    void create_area_texture_(coopa::gfx::command::CommandPool& cmd_pool) {
        upload_texture_(cmd_pool, areaTexBytes, AREATEX_WIDTH, AREATEX_HEIGHT,
                        VK_FORMAT_R8G8_UNORM, 2, area_image_);
    }

    void upload_texture_(coopa::gfx::command::CommandPool& cmd_pool,
                         const void* data, uint32_t width, uint32_t height,
                         VkFormat format, uint32_t bytes_per_pixel,
                         std::unique_ptr<coopa::gfx::memory::Image>& dst_image)
    {
        VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * bytes_per_pixel;

        coopa::gfx::memory::Buffer staging(
            device_, allocator_, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
        );
        staging.upload(data, size);

        dst_image = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width, height, format,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VK_SAMPLE_COUNT_1_BIT
        );

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool        = cmd_pool.handle();
        alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer cmd_handle = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_.handle(), &alloc_info, &cmd_handle);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_handle, &begin_info);

        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = dst_image->handle();
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

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
        copy_region.imageExtent                     = {width, height, 1};

        vkCmdCopyBufferToImage(cmd_handle, staging.handle(), dst_image->handle(),
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

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    std::unique_ptr<coopa::gfx::memory::Image>   search_image_;
    std::unique_ptr<coopa::gfx::memory::Image>   area_image_;
    std::unique_ptr<Sampler> sampler_;
};

} // namespace util
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_UTIL_SMAA_TEXTURES_H
