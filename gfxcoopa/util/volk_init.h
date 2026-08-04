/**
 * @file volk_init.h
 * @brief One-time volk initialization guard for gfxcoopa.
 *
 * volk must be initialized exactly once per process before any Vulkan
 * function is called. This header provides a static guard that is called
 * by Instance on construction. All library headers include volk/volk.h
 * rather than vulkan/vulkan.h directly.
 */

#ifndef COOPA_GFX_UTIL_VOLK_INIT_H
#define COOPA_GFX_UTIL_VOLK_INIT_H

#include <volk/volk.h>
#include <stdexcept>

namespace coopa {
namespace gfx {
namespace util {

/**
 * @brief Initializes volk (and therefore the Vulkan loader) exactly once.
 *
 * Subsequent calls are no-ops. Must be called before any vkXxx function
 * is invoked. Automatically called by Instance's constructor.
 *
 * @throws std::runtime_error if volkInitialize() fails (Vulkan not present).
 */
inline void ensure_volk_initialized() {
    static bool initialized = false;
    if (initialized) return;

    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "[gfxcoopa] volkInitialize() failed — is the Vulkan loader installed?");
    }

    initialized = true;
}

} // namespace util
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_UTIL_VOLK_INIT_H
