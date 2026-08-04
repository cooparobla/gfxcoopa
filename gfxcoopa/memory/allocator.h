/**
 * @file allocator.h
 * @brief VMA (Vulkan Memory Allocator) initialization and lifecycle wrapper.
 *
 * Provides a thin RAII shell around VmaAllocator, created from a Device,
 * and exposes the raw handle for use by Buffer and Image. Must be constructed
 * before any Buffer or Image and destroyed after them.
 */

#ifndef COOPA_GFX_MEMORY_ALLOCATOR_H
#define COOPA_GFX_MEMORY_ALLOCATOR_H

// VMA requires Vulkan function pointers — must be included after volk.
#include <volk/volk.h>

// VMA configuration: use volk's already-loaded function pointers.
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vma/vk_mem_alloc.h>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace memory {

/**
 * @class Allocator
 * @brief RAII owner of a VmaAllocator.
 *
 * Configures VMA to use the function pointers already loaded by volk,
 * so there is no separate dynamic loading path needed. The resulting
 * VmaAllocator handle is passed to Buffer and Image constructors.
 */
class Allocator {
public:
    /**
     * @brief Creates the VmaAllocator.
     * @param instance The Vulkan instance.
     * @param device   The logical device.
     */
    Allocator(const core::Instance& instance, const core::Device& device) {
        // Bind volk's already-loaded global function pointers into VMA.
        VmaVulkanFunctions vma_funcs{};
        vma_funcs.vkGetInstanceProcAddr               = vkGetInstanceProcAddr;
        vma_funcs.vkGetDeviceProcAddr                 = vkGetDeviceProcAddr;
        vma_funcs.vkGetPhysicalDeviceProperties       = vkGetPhysicalDeviceProperties;
        vma_funcs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
        vma_funcs.vkAllocateMemory                    = vkAllocateMemory;
        vma_funcs.vkFreeMemory                        = vkFreeMemory;
        vma_funcs.vkMapMemory                         = vkMapMemory;
        vma_funcs.vkUnmapMemory                       = vkUnmapMemory;
        vma_funcs.vkFlushMappedMemoryRanges           = vkFlushMappedMemoryRanges;
        vma_funcs.vkInvalidateMappedMemoryRanges      = vkInvalidateMappedMemoryRanges;
        vma_funcs.vkBindBufferMemory                  = vkBindBufferMemory;
        vma_funcs.vkBindImageMemory                   = vkBindImageMemory;
        vma_funcs.vkGetBufferMemoryRequirements       = vkGetBufferMemoryRequirements;
        vma_funcs.vkGetImageMemoryRequirements        = vkGetImageMemoryRequirements;
        vma_funcs.vkCreateBuffer                      = vkCreateBuffer;
        vma_funcs.vkDestroyBuffer                     = vkDestroyBuffer;
        vma_funcs.vkCreateImage                       = vkCreateImage;
        vma_funcs.vkDestroyImage                      = vkDestroyImage;
        vma_funcs.vkCmdCopyBuffer                     = vkCmdCopyBuffer;

        VmaAllocatorCreateInfo info{};
        info.vulkanApiVersion = VK_API_VERSION_1_3;
        info.instance         = instance.handle();
        info.physicalDevice   = device.physical();
        info.device           = device.handle();
        info.pVulkanFunctions = &vma_funcs;

        VkResult result = vmaCreateAllocator(&info, &allocator_);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("[gfxcoopa] vmaCreateAllocator failed: " +
                                     util::vk_result_string(result));
        }
    }

    /**
     * @brief Destroys the VmaAllocator.
     */
    ~Allocator() {
        if (allocator_ != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator_);
        }
    }

    /// @brief Non-copyable.
    Allocator(const Allocator&) = delete;
    /// @brief Non-copyable.
    Allocator& operator=(const Allocator&) = delete;

    /**
     * @brief Returns the underlying VmaAllocator handle.
     * @return Raw VmaAllocator.
     */
    VmaAllocator handle() const { return allocator_; }

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE; /**< The VMA allocator handle. */
};

} // namespace memory
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_MEMORY_ALLOCATOR_H
