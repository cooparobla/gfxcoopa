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

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/types/enums.h>
#include <gfxcoopa/types/format.h>
#include <gfxcoopa/types/clear.h>
#include <gfxcoopa/detail/vk_convert.h>

#include <memory>
#include <cstdint>

namespace coopa {
namespace gfx {
namespace memory {

/**
 * @brief Uploads tightly-packed, row-major pixel data into a newly created 2D Image.
 *
 * Records and submits a one-shot command buffer (via CommandPool::submit_once()) that
 * transitions the image Undefined -> TransferDst, copies the staging buffer into it, then
 * transitions TransferDst -> ShaderRead, and blocks until the transfer queue is idle before
 * returning — matching the synchronous, "ready to sample immediately" contract every call
 * site this replaces already relied on.
 *
 * @param device          Logical device.
 * @param allocator       VMA allocator.
 * @param cmd_pool        Command pool the one-shot upload command buffer is allocated from.
 * @param data            Raw pixel bytes, tightly packed, row-major, `width * height * bytes_per_pixel` long.
 * @param width           Image width in pixels.
 * @param height          Image height in pixels.
 * @param format          Pixel format; must match bytes_per_pixel.
 * @param bytes_per_pixel Bytes per texel (e.g. 1 for R8_Unorm, 4 for RGBA8_Unorm).
 * @return A new Image, already transitioned to TextureUsage::ShaderRead.
 */
inline std::unique_ptr<Image> upload_image_2d(core::Device&         device,
                                              Allocator&            allocator,
                                              command::CommandPool& cmd_pool,
                                              const void*           data,
                                              uint32_t              width,
                                              uint32_t              height,
                                              Format                format,
                                              uint32_t              bytes_per_pixel)
{
    VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * bytes_per_pixel;

    Buffer staging = Buffer::staging(device, allocator, size);
    staging.upload(data, size);

    auto image = std::make_unique<Image>(
        device, allocator, width, height, format,
        ImageUsage::TransferDst | ImageUsage::Sampled
    );

    cmd_pool.submit_once([&](command::CommandBuffer& cmd) {
        cmd.transition(*image, TextureUsage::TransferDst);
        cmd.copy_buffer_to_image(staging, *image, Extent2D{width, height});
        cmd.transition(*image, TextureUsage::ShaderRead);
    });

    return image;
}

/**
 * @brief Raw-`VkFormat` overload of the above, for the small number of internal callers not
 * yet migrated to the sealed `Format` enum. Prefer the `Format`-taking overload in new code.
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
    return upload_image_2d(device, allocator, cmd_pool, data, width, height,
                           detail::from_vk(format), bytes_per_pixel);
}

} // namespace memory
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_MEMORY_IMAGE_UPLOAD_H
