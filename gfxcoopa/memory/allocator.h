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

// VMA configuration: let VMA dynamically resolve function pointers using
// volk's already-loaded vkGetInstanceProcAddr / vkGetDeviceProcAddr.
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

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
 * Configures VMA to dynamically resolve Vulkan function pointers via the
 * vkGetInstanceProcAddr / vkGetDeviceProcAddr already loaded by volk.
 * The resulting VmaAllocator handle is passed to Buffer and Image constructors.
 */
class Allocator {
public:
    /**
     * @brief Creates the VmaAllocator.
     * @param instance The Vulkan instance.
     * @param device   The logical device.
     */
    Allocator(const core::Instance& instance, const core::Device& device) {
        // Provide volk's proc addr functions so VMA can dynamically resolve
        // all Vulkan 1.0 / 1.1 / 1.3 functions it needs internally.
        VmaVulkanFunctions vma_funcs{};
        vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vma_funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

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
