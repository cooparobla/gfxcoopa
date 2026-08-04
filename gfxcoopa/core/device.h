/**
 * @file device.h
 * @brief Physical and logical Vulkan device selection and queue management.
 *
 * Selects the best available GPU (discrete > integrated > virtual > CPU),
 * verifies swapchain extension support, finds graphics/present queue families,
 * and creates the VkDevice with those queues.
 *
 * The logical Device is the primary object passed to almost every other
 * gfxcoopa class. Analogous to the OpenGL context's underlying driver connection.
 */

#ifndef COOPA_GFX_CORE_DEVICE_H
#define COOPA_GFX_CORE_DEVICE_H

#include <volk/volk.h>
#include <vector>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <limits>
#include <cstring>

#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/core/surface.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace core {

/**
 * @struct QueueFamilyIndices
 * @brief Holds the queue family index for each required queue type.
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics; /**< Queue family supporting VK_QUEUE_GRAPHICS_BIT. */
    std::optional<uint32_t> present;  /**< Queue family supporting surface presentation. */

    /**
     * @brief Returns true when all required queue families have been found.
     * @return True if graphics and present families are both valid.
     */
    bool is_complete() const { return graphics.has_value() && present.has_value(); }
};

/**
 * @class Device
 * @brief Manages the physical GPU selection and the logical Vulkan device.
 *
 * On construction:
 * 1. Enumerates all physical devices and scores them (discrete GPU preferred).
 * 2. Verifies VK_KHR_swapchain extension availability.
 * 3. Finds graphics + present queue family indices.
 * 4. Creates the logical VkDevice and retrieves queue handles.
 * 5. Loads all device-level function pointers via volkLoadDevice.
 */
class Device {
public:
    /**
     * @brief Selects the best physical device and creates the logical device.
     * @param instance The Vulkan instance.
     * @param surface  The window surface (used for present queue family detection).
     * @throws std::runtime_error if no suitable GPU is found or device creation fails.
     */
    Device(const Instance& instance, const Surface& surface) {
        pick_physical_device(instance.handle(), surface.handle());
        create_logical_device();
    }

    /**
     * @brief Destroys the logical VkDevice.
     */
    ~Device() {
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
        }
    }

    /// @brief Non-copyable.
    Device(const Device&) = delete;
    /// @brief Non-copyable.
    Device& operator=(const Device&) = delete;

    // --- Accessors ---

    /**
     * @brief Returns the logical device handle.
     * @return Raw VkDevice.
     */
    VkDevice handle() const { return device_; }

    /**
     * @brief Returns the selected physical device handle.
     * @return Raw VkPhysicalDevice.
     */
    VkPhysicalDevice physical() const { return physical_device_; }

    /**
     * @brief Returns the graphics queue handle.
     * @return Raw VkQueue for graphics submission.
     */
    VkQueue graphics_queue() const { return graphics_queue_; }

    /**
     * @brief Returns the present queue handle.
     * @return Raw VkQueue for presentation submission.
     */
    VkQueue present_queue()  const { return present_queue_; }

    /**
     * @brief Returns the graphics queue family index.
     * @return Graphics queue family index.
     */
    uint32_t graphics_family() const { return queue_families_.graphics.value(); }

    /**
     * @brief Returns the present queue family index.
     * @return Present queue family index.
     */
    uint32_t present_family()  const { return queue_families_.present.value(); }

    /**
     * @brief Stalls until all device queues have finished executing.
     *
     * Equivalent to glFinish() in OpenGL — use before destroying resources
     * that may still be in use by the GPU.
     */
    void wait_idle() const { vkDeviceWaitIdle(device_); }

    /**
     * @brief Returns the cached queue family indices for this device.
     * @return QueueFamilyIndices struct.
     */
    QueueFamilyIndices queue_family_indices() const { return queue_families_; }

    /**
     * @brief Queries and returns the memory properties of the physical device.
     * @return VkPhysicalDeviceMemoryProperties struct.
     */
    VkPhysicalDeviceMemoryProperties memory_properties() const {
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &props);
        return props;
    }

private:
    // --- Physical device selection ---

    /**
     * @brief Enumerates physical devices and picks the highest-scoring one.
     *
     * Score is based on device type: discrete GPU > integrated > virtual > CPU.
     * Devices that lack swapchain support or valid queue families are skipped.
     *
     * @param instance The VkInstance to enumerate from.
     * @param surface  The surface used for present-queue detection.
     */
    void pick_physical_device(VkInstance instance, VkSurfaceKHR surface) {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) {
            throw std::runtime_error("[gfxcoopa] No Vulkan-capable GPU found.");
        }

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        VkPhysicalDevice best = VK_NULL_HANDLE;
        int best_score = -1;

        for (VkPhysicalDevice dev : devices) {
            if (!is_device_suitable(dev, surface)) continue;

            int score = rate_device(dev);
            if (score > best_score) {
                best_score     = score;
                best           = dev;
                queue_families_ = find_queue_families(dev, surface);
            }
        }

        if (best == VK_NULL_HANDLE) {
            throw std::runtime_error("[gfxcoopa] No suitable GPU found. "
                                     "(Missing swapchain support or queue families.)");
        }

        physical_device_ = best;
    }

    /**
     * @brief Assigns a numeric score to a physical device based on its type.
     * @param device The physical device to score.
     * @return Integer score (higher is better).
     */
    static int rate_device(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 1000;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 100;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 10;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1;
            default:                                     return 0;
        }
    }

    /**
     * @brief Returns true if the device supports all required extensions and
     *        has at least one valid format and present mode for the surface.
     * @param device  The physical device to test.
     * @param surface The target surface.
     * @return True if the device is suitable.
     */
    static bool is_device_suitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
        // Check extension support.
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, exts.data());

        bool has_swapchain = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                has_swapchain = true; break;
            }
        }
        if (!has_swapchain) return false;

        // Check queue families.
        QueueFamilyIndices indices = find_queue_families(device, surface);
        if (!indices.is_complete()) return false;

        // Check swapchain adequacy.
        uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
        uint32_t mode_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, nullptr);
        return format_count > 0 && mode_count > 0;
    }

    /**
     * @brief Finds graphics and present queue family indices for a device/surface pair.
     * @param device  The physical device to query.
     * @param surface The target surface for present support.
     * @return QueueFamilyIndices with graphics and present optionals filled where found.
     */
    static QueueFamilyIndices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) {
        QueueFamilyIndices indices;

        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphics = i;
            }

            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
            if (present_support) {
                indices.present = i;
            }

            if (indices.is_complete()) break;
        }

        return indices;
    }

    // --- Logical device creation ---

    /**
     * @brief Creates the logical VkDevice with graphics and present queues.
     *
     * If graphics and present share the same queue family, only one queue is
     * created (common on most desktop GPUs). Enables VK_KHR_swapchain.
     */
    void create_logical_device() {
        std::set<uint32_t> unique_families = {
            queue_families_.graphics.value(),
            queue_families_.present.value()
        };

        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        for (uint32_t family : unique_families) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = family;
            qi.queueCount       = 1;
            qi.pQueuePriorities = &priority;
            queue_infos.push_back(qi);
        }

        VkPhysicalDeviceFeatures features{};
        features.samplerAnisotropy = VK_TRUE; // Enable anisotropic filtering

        const char* extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        VkDeviceCreateInfo create_info{};
        create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount    = static_cast<uint32_t>(queue_infos.size());
        create_info.pQueueCreateInfos       = queue_infos.data();
        create_info.enabledExtensionCount   = 1;
        create_info.ppEnabledExtensionNames = extensions;
        create_info.pEnabledFeatures        = &features;

        GFX_VK_CHECK(vkCreateDevice(physical_device_, &create_info, nullptr, &device_));

        // Load all device-level function pointers via volk.
        volkLoadDevice(device_);

        vkGetDeviceQueue(device_, queue_families_.graphics.value(), 0, &graphics_queue_);
        vkGetDeviceQueue(device_, queue_families_.present.value(),  0, &present_queue_);
    }

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE; /**< Selected GPU. */
    VkDevice         device_          = VK_NULL_HANDLE; /**< Logical device. */
    VkQueue          graphics_queue_  = VK_NULL_HANDLE; /**< Graphics submission queue. */
    VkQueue          present_queue_   = VK_NULL_HANDLE; /**< Presentation queue. */
    QueueFamilyIndices queue_families_;                  /**< Cached queue family indices. */
};

} // namespace core
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_CORE_DEVICE_H
