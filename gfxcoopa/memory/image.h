/**
 * @file image.h
 * @brief 2D Vulkan image and image view management via VMA.
 *
 * Provides a 2D Image backed by a VMA allocation, automatically creating a
 * matching VkImageView. Supports layout transitions and pixel data upload
 * via a staging buffer. Analogous to OpenGL's texture objects.
 */

#ifndef COOPA_GFX_MEMORY_IMAGE_H
#define COOPA_GFX_MEMORY_IMAGE_H

#include <volk/volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vma/vk_mem_alloc.h>

#include <cstring>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/util/error.h>
#include <gfxcoopa/util/format.h>

namespace coopa {
namespace gfx {
namespace memory {

/**
 * @class Image
 * @brief RAII 2D Vulkan image with an auto-created VkImageView.
 *
 * Wraps VkImage + VmaAllocation + VkImageView into a single object.
 * Provides transition_layout() for pipeline barrier-based layout changes
 * and upload() for texel data upload (like glTexSubImage2D).
 */
class Image {
public:
    /**
     * @brief Creates a 2D VkImage with a VMA-managed allocation and a VkImageView.
     *
     * @param device        The logical device.
     * @param allocator     The VMA allocator.
     * @param width         Image width in pixels.
     * @param height        Image height in pixels.
     * @param format        Pixel format.
     * @param usage         VkImageUsageFlags (sampled, color attachment, transfer dst, etc.).
     * @param aspect_mask   Which aspects to expose in the image view (color or depth).
     * @param memory_usage  VMA memory usage hint.
     */
    Image(core::Device&      device,
          Allocator&         allocator,
          uint32_t           width,
          uint32_t           height,
          VkFormat           format,
          VkImageUsageFlags  usage,
          VkImageAspectFlags aspect_mask  = VK_IMAGE_ASPECT_COLOR_BIT,
          VmaMemoryUsage     memory_usage = VMA_MEMORY_USAGE_AUTO)
        : device_(device), allocator_(allocator.handle()),
          width_(width), height_(height), format_(format)
    {
        create_image(usage, memory_usage);
        create_view(aspect_mask);
    }

    /**
     * @brief Destroys the image view, then the image and its allocation.
     */
    ~Image() {
        if (view_  != VK_NULL_HANDLE) vkDestroyImageView(device_.handle(), view_,  nullptr);
        if (image_ != VK_NULL_HANDLE) vmaDestroyImage(allocator_, image_, allocation_);
    }

    /// @brief Non-copyable.
    Image(const Image&) = delete;
    /// @brief Non-copyable.
    Image& operator=(const Image&) = delete;

    // --- Accessors ---

    /**
     * @brief Returns the underlying VkImage handle.
     * @return Raw VkImage.
     */
    VkImage handle() const { return image_; }

    /**
     * @brief Returns the VkImageView covering the full image.
     * @return Raw VkImageView.
     */
    VkImageView view() const { return view_; }

    /**
     * @brief Returns the image format.
     * @return VkFormat of this image.
     */
    VkFormat format() const { return format_; }

    /**
     * @brief Returns the image width in pixels.
     * @return Width in pixels.
     */
    uint32_t width()  const { return width_; }

    /**
     * @brief Returns the image height in pixels.
     * @return Height in pixels.
     */
    uint32_t height() const { return height_; }

    // --- Layout transition ---

    /**
     * @brief Inserts a pipeline barrier to transition this image's layout.
     *
     * Analogous to ensuring a texture is in the correct mip state before
     * sampling. The src/dst stage and access masks are inferred from the
     * old and new layouts.
     *
     * @param cmd        Command buffer to record the barrier into.
     * @param old_layout Current layout.
     * @param new_layout Target layout.
     */
    void transition_layout(VkCommandBuffer cmd,
                           VkImageLayout   old_layout,
                           VkImageLayout   new_layout) const
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = old_layout;
        barrier.newLayout           = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image_;

        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        // Infer access masks from well-known layout pairs.
        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                 new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            // Generic fallback: full pipeline stall.
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        }

        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0,
                             0, nullptr, 0, nullptr,
                             1, &barrier);
    }

private:
    /**
     * @brief Creates the VkImage and VMA allocation.
     * @param usage        Image usage flags.
     * @param memory_usage VMA memory usage hint.
     */
    void create_image(VkImageUsageFlags usage, VmaMemoryUsage memory_usage) {
        VkImageCreateInfo image_info{};
        image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType     = VK_IMAGE_TYPE_2D;
        image_info.extent.width  = width_;
        image_info.extent.height = height_;
        image_info.extent.depth  = 1;
        image_info.mipLevels     = 1;
        image_info.arrayLayers   = 1;
        image_info.format        = format_;
        image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage         = usage;
        image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        image_info.samples       = VK_SAMPLE_COUNT_1_BIT;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = memory_usage;

        VkResult result = vmaCreateImage(allocator_, &image_info, &alloc_info,
                                         &image_, &allocation_, nullptr);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("[gfxcoopa] vmaCreateImage failed: " +
                                     util::vk_result_string(result));
        }
    }

    /**
     * @brief Creates a VkImageView for the full image.
     * @param aspect_mask Which aspect(s) to expose in the view.
     */
    void create_view(VkImageAspectFlags aspect_mask) {
        VkImageViewCreateInfo view_info{};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = image_;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format                          = format_;
        view_info.subresourceRange.aspectMask     = aspect_mask;
        view_info.subresourceRange.baseMipLevel   = 0;
        view_info.subresourceRange.levelCount     = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount     = 1;

        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view_));
    }

    core::Device& device_;                        /**< Owning logical device (not owned). */
    VmaAllocator  allocator_ = VK_NULL_HANDLE;    /**< VMA allocator handle (not owned). */
    VkImage       image_     = VK_NULL_HANDLE;    /**< The Vulkan image. */
    VmaAllocation allocation_ = VK_NULL_HANDLE;   /**< VMA allocation for the image memory. */
    VkImageView   view_      = VK_NULL_HANDLE;    /**< Image view covering the full image. */
    uint32_t      width_     = 0;                 /**< Image width in pixels. */
    uint32_t      height_    = 0;                 /**< Image height in pixels. */
    VkFormat      format_    = VK_FORMAT_UNDEFINED;/**< Pixel format. */
};

} // namespace memory
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_MEMORY_IMAGE_H
