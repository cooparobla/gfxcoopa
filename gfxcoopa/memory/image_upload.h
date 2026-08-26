/**
 * @file image_upload.h
 * @brief Free helper that uploads raw pixel data into a new 2D Image via a staging buffer.
 *
 * Consolidates the staging-buffer -> layout transition -> copy -> layout
 * transition recipe that used to be independently hand-rolled in three
 * places (engine::util::SmaaTextures::upload_texture_, engine::util::
 * SsaoNoiseTexture::upload_texture_, engine::passes::SsaoPass::
 * create_neutral_texture_) plus a fourth copy in uicoopa's Texture — all
 * byte-for-byte identical apart from the source data/dimensions/format.
 */

#ifndef COOPA_GFX_MEMORY_IMAGE_UPLOAD_H
#define COOPA_GFX_MEMORY_IMAGE_UPLOAD_H

#include <volk/volk.h>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/types/enums.h>

#include <memory>
#include <cstdint>

namespace coopa {
namespace gfx {
namespace memory {

/**
 * @brief Uploads tightly-packed, row-major pixel data into a newly created 2D Image.
 *
 * Records and submits a one-shot command buffer that transitions the image
 * UNDEFINED -> TRANSFER_DST_OPTIMAL, copies the staging buffer into it, then
 * transitions TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL, and blocks
 * until the transfer queue is idle before returning — matching the
 * synchronous, "ready to sample immediately" contract every call site this
 * replaces already relied on.
 *
 * @param device          Logical device.
 * @param allocator       VMA allocator.
 * @param cmd_pool        Command pool the one-shot upload command buffer is allocated from.
 * @param data            Raw pixel bytes, tightly packed, row-major, `width * height * bytes_per_pixel` long.
 * @param width           Image width in pixels.
 * @param height          Image height in pixels.
 * @param format          Pixel format; must match bytes_per_pixel.
 * @param bytes_per_pixel Bytes per texel (e.g. 1 for R8_UNORM, 4 for R8G8B8A8_UNORM).
 * @return A new Image, already in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
 */
inline std::unique_ptr<Image> upload_image_2d(core::Device&         device,
                                              Allocator&            allocator,
                                              command::CommandPool& cmd_pool,
                                              const void*           data,
                                              uint32_t              width,
                                              uint32_t              height,
                                              VkFormat              format,
                                              uint32_t              bytes_per_pixel)
{
    VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * bytes_per_pixel;

    Buffer staging(
        device, allocator, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
    );
    staging.upload(data, size);

    auto image = std::make_unique<Image>(
        device, allocator, width, height, format,
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
    vkAllocateCommandBuffers(device.handle(), &alloc_info, &cmd_handle);

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
    barrier.image               = image->handle();
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

    vkCmdCopyBufferToImage(cmd_handle, staging.handle(), image->handle(),
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
    vkQueueSubmit(device.graphics_queue(), 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(device.graphics_queue());

    vkFreeCommandBuffers(device.handle(), cmd_pool.handle(), 1, &cmd_handle);

    // This function predates command::CommandBuffer::transition() and
    // records its own raw barriers above rather than using it, but the
    // image it returns should still report its true usage to later
    // transition() calls -- see Image::mark_transitioned()'s docs.
    image->mark_transitioned(TextureUsage::ShaderRead);

    return image;
}

} // namespace memory
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_MEMORY_IMAGE_UPLOAD_H
