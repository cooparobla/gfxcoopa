/**
 * @file instance.h
 * @brief Vulkan instance creation and lifetime management.
 *
 * The Instance is the entry point for all Vulkan API calls. It initializes
 * volk, creates VkInstance, and (in debug builds) installs a validation
 * layer debug messenger. Analogous to creating an OpenGL context.
 */

#ifndef COOPA_GFX_CORE_INSTANCE_H
#define COOPA_GFX_CORE_INSTANCE_H

#include <volk/volk.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <memory>

// GLFW for extension enumeration (no OpenGL)
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <gfxcoopa/util/error.h>
#include <gfxcoopa/util/volk_init.h>
#include <gfxcoopa/util/debug_messenger.h>

namespace coopa {
namespace gfx {
namespace core {

/**
 * @class Instance
 * @brief RAII wrapper around VkInstance.
 *
 * Initializes the Vulkan loader via volk, queries and enables all GLFW-
 * required extensions plus VK_EXT_debug_utils when validation is on.
 * Installs a DebugMessenger that routes validation output to stderr.
 *
 * Must be the first gfxcoopa object created and the last destroyed.
 */
class Instance {
public:
    /**
     * @brief Creates a VkInstance and optionally enables validation layers.
     *
     * - Calls volkInitialize() once via the static volk guard.
     * - Queries GLFW for required surface extensions.
     * - Enables VK_LAYER_KHRONOS_validation and VK_EXT_debug_utils when
     *   enable_validation is true and the layer is available.
     *
     * @param app_name Application display name embedded in the instance.
     * @param enable_validation Enable Khronos validation layer + debug messenger.
     * @throws std::runtime_error if instance creation fails or validation is
     *         requested but not available.
     */
    Instance(const std::string& app_name, bool enable_validation = true) {
        // Initialize volk (loads vulkan-1.so / vulkan-1.dll at runtime).
        util::ensure_volk_initialized();

        // --- Application info ---
        VkApplicationInfo app_info{};
        app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName   = app_name.c_str();
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName        = "gfxcoopa";
        app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion         = VK_API_VERSION_1_3;

        // --- Extensions ---
        uint32_t glfw_ext_count = 0;
        const char** glfw_exts  = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
        std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);

        if (enable_validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // --- Validation layers ---
        const char* k_validation_layer = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> layers;

        if (enable_validation) {
            if (!layer_available(k_validation_layer)) {
                throw std::runtime_error(
                    "[gfxcoopa] Requested VK_LAYER_KHRONOS_validation but it is not installed.");
            }
            layers.push_back(k_validation_layer);
        }

        // --- Instance create info ---
        VkDebugUtilsMessengerCreateInfoEXT debug_info =
            util::DebugMessenger::make_create_info();

        VkInstanceCreateInfo create_info{};
        create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo        = &app_info;
        create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
        create_info.enabledLayerCount       = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames     = layers.data();
        // Chain the debug info so validation catches errors during instance creation.
        if (enable_validation) {
            create_info.pNext = &debug_info;
        }

        GFX_VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));

        // Load all instance-level function pointers via volk.
        volkLoadInstance(instance_);

        // Install the permanent debug messenger now that the instance exists.
        if (enable_validation) {
            debug_messenger_ = std::make_unique<util::DebugMessenger>(instance_);
        }
    }

    /**
     * @brief Destroys the debug messenger (if any) and the VkInstance.
     */
    ~Instance() {
        debug_messenger_.reset(); // Must be destroyed before the instance.
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    /// @brief Non-copyable.
    Instance(const Instance&) = delete;
    /// @brief Non-copyable.
    Instance& operator=(const Instance&) = delete;

    /**
     * @brief Returns the underlying VkInstance handle.
     * @return Raw VkInstance.
     */
    VkInstance handle() const { return instance_; }

private:
    /**
     * @brief Checks whether a given Vulkan layer name is available on this system.
     * @param name Layer name string (e.g. "VK_LAYER_KHRONOS_validation").
     * @return True if the layer exists in the enumerated instance layer list.
     */
    static bool layer_available(const char* name) {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> props(count);
        vkEnumerateInstanceLayerProperties(&count, props.data());

        for (const auto& p : props) {
            if (std::strcmp(p.layerName, name) == 0) return true;
        }
        return false;
    }

    VkInstance                              instance_        = VK_NULL_HANDLE; /**< The Vulkan instance handle. */
    std::unique_ptr<util::DebugMessenger>   debug_messenger_;                  /**< Validation debug messenger (null when validation is off). */
};

} // namespace core
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_CORE_INSTANCE_H
