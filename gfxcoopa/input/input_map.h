/**
 * @file input_map.h
 * @brief Named action -> key/button bindings, decoupled from any concrete
 * window so it stays unit-testable with a mock key-state predicate.
 *
 * This consolidates what was previously two nearly-identical copies
 * (toyengine's and pixengine's `input/input_map.h`, both keyed on raw GLFW
 * `int` codes) into one gfxcoopa-owned implementation keyed on the sealed
 * Key/MouseButton enums. blendy, which had no InputMap at all and hardcoded
 * GLFW key literals inline in its main loop, gains one for free.
 */

#ifndef COOPA_GFX_INPUT_INPUT_MAP_H
#define COOPA_GFX_INPUT_INPUT_MAP_H

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gfxcoopa/input/keys.h>

namespace coopa {
namespace gfx {
namespace input {

/// @brief A predicate answering "is this key currently held down?". Callers
/// adapt a real presentation::Window with key_state_of() (see
/// gfxcoopa/presentation/input_adapter.h -- kept out of input/ itself since
/// it must include the GLFW-backed Window; input/ stays Vulkan/GLFW-free).
/// Tests supply a lambda over a fake key set instead.
using KeyState = std::function<bool(Key)>;

/**
 * @class InputMap
 * @brief Maps named actions and axes to one or more physical keys.
 *
 * Deliberately takes a KeyState predicate rather than a concrete Window&, so
 * it can be exercised in tests without any windowing or graphics context —
 * the same design the two pre-seal copies already used, kept unchanged here.
 */
class InputMap {
public:
    /// @brief Binds a key to an action. An action may have multiple keys
    /// bound; is_down() returns true if any of them is held.
    void bind(std::string_view action, Key key) {
        bindings_[std::string(action)].push_back(key);
    }

    /// @brief Removes every key binding for `action` (present on both
    /// pre-seal copies this consolidates; leaves `action` unbound
    /// afterward, same as if bind() had never been called for it).
    void unbind(std::string_view action) {
        bindings_.erase(std::string(action));
    }

    /// @brief Returns the keys currently bound to `action` (empty if none).
    const std::vector<Key>& bindings(std::string_view action) const {
        static const std::vector<Key> empty;
        auto it = bindings_.find(std::string(action));
        return it != bindings_.end() ? it->second : empty;
    }

    /// @brief True if any key bound to `action` is currently down, per `is_key_pressed`.
    bool is_down(std::string_view action, const KeyState& is_key_pressed) const {
        for (Key k : bindings(action)) {
            if (is_key_pressed(k)) return true;
        }
        return false;
    }

    /// @brief Binds a two-key axis (e.g. "move_x" -> D/A), read via axis().
    void bind_axis(std::string_view axis, Key positive, Key negative) {
        axes_[std::string(axis)] = AxisBinding{positive, negative};
    }

    /// @brief Returns -1, 0, or +1 depending on which of the axis's two keys
    /// (if any) is held. If both are held, they cancel to 0.
    float axis(std::string_view axis, const KeyState& is_key_pressed) const {
        auto it = axes_.find(std::string(axis));
        if (it == axes_.end()) return 0.0f;
        float value = 0.0f;
        if (is_key_pressed(it->second.positive)) value += 1.0f;
        if (is_key_pressed(it->second.negative)) value -= 1.0f;
        return value;
    }

private:
    struct AxisBinding { Key positive; Key negative; };

    std::unordered_map<std::string, std::vector<Key>> bindings_;
    std::unordered_map<std::string, AxisBinding>       axes_;
};

} // namespace input
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_INPUT_INPUT_MAP_H
