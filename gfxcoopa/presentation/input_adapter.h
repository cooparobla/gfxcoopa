/**
 * @file input_adapter.h
 * @brief Adapts a real presentation::Window to the input::KeyState predicate
 * that input::InputMap consumes.
 *
 * Kept as a separate header from gfxcoopa/input/input_map.h (rather than a
 * method on Window or InputMap itself) because it must include the
 * GLFW-backed Window, and gfxcoopa/input/ is required to stay Vulkan- and
 * GLFW-free so tools/check_no_vulkan.sh's inverted purity check on that
 * directory holds.
 */

#ifndef COOPA_GFX_PRESENTATION_INPUT_ADAPTER_H
#define COOPA_GFX_PRESENTATION_INPUT_ADAPTER_H

#include <gfxcoopa/input/input_map.h>
#include <gfxcoopa/presentation/window.h>

namespace coopa {
namespace gfx {
namespace presentation {

/**
 * @brief Returns an input::KeyState predicate backed by a live Window.
 *
 * The returned callable captures `window` by reference; it must not outlive
 * the Window it was built from. Typical use:
 *
 * @code
 * auto is_pressed = presentation::key_state_of(window);
 * if (input_map.is_down("jump", is_pressed)) { ... }
 * @endcode
 *
 * @param window The window to poll.
 * @return A KeyState predicate delegating to window.is_key_pressed().
 */
inline input::KeyState key_state_of(const Window& window) {
    return [&window](input::Key key) { return window.is_key_pressed(key); };
}

} // namespace presentation
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_PRESENTATION_INPUT_ADAPTER_H
