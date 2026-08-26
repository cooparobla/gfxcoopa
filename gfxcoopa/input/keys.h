/**
 * @file keys.h
 * @brief gfxcoopa-owned keyboard/mouse vocabulary — Vulkan- and
 * windowing-library-free.
 *
 * Key/MouseButton values are dense, 0-based, and gfxcoopa-defined —
 * deliberately NOT GLFW key codes. The GLFW mapping table lives in the
 * internal gfxcoopa/detail/glfw_keys.h, which only gfxcoopa's own
 * presentation::Window may include. This is what lets consumer code write
 * `input_map.bind("jump", Key::Space)` instead of naming `GLFW_KEY_SPACE`.
 */

#ifndef COOPA_GFX_INPUT_KEYS_H
#define COOPA_GFX_INPUT_KEYS_H

#include <cstdint>

namespace coopa {
namespace gfx {
namespace input {

/**
 * @enum Key
 * @brief A physical/logical keyboard key.
 *
 * Values are dense and 0-based so `size_t(Key::Count)` sizes lookup arrays
 * directly — unlike GLFW's key codes, which are sparse and not usable that
 * way without an offset table.
 */
enum class Key : int16_t {
    Unknown = -1,

    Space = 0, Apostrophe, Comma, Minus, Period, Slash,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Semicolon, Equal,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    LeftBracket, Backslash, RightBracket, GraveAccent,

    Escape, Enter, Tab, Backspace, Insert, Delete,
    Right, Left, Down, Up,
    PageUp, PageDown, Home, End,
    CapsLock, ScrollLock, NumLock, PrintScreen, Pause,

    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25,

    Kp0, Kp1, Kp2, Kp3, Kp4, Kp5, Kp6, Kp7, Kp8, Kp9,
    KpDecimal, KpDivide, KpMultiply, KpSubtract, KpAdd, KpEnter, KpEqual,

    LeftShift, LeftControl, LeftAlt, LeftSuper,
    RightShift, RightControl, RightAlt, RightSuper,
    Menu,

    Count, ///< Not a real key. Sizes dense lookup arrays: `size_t(Key::Count)`.
};

/// @brief A mouse button. Dense and 0-based, unlike GLFW's button indices
/// (which happen to also be 0-based, but this keeps the two vocabularies
/// independent so gfxcoopa is free to diverge later).
enum class MouseButton : uint8_t {
    Left = 0, Right, Middle,
    Button4, Button5, Button6, Button7, Button8,
    Count, ///< Not a real button. Sizes dense lookup arrays.
};

/// @brief The transition a KeyEvent represents.
enum class KeyAction : uint8_t {
    Release,
    Press,
    Repeat,
};

/// @brief Modifier-key bitmask, active at the moment of a KeyEvent.
enum class Mods : uint8_t {
    None     = 0,
    Shift    = 1u << 0,
    Control  = 1u << 1,
    Alt      = 1u << 2,
    Super    = 1u << 3,
    CapsLock = 1u << 4,
    NumLock  = 1u << 5,
};

/// @brief Combines two modifier masks.
constexpr Mods operator|(Mods a, Mods b) {
    return static_cast<Mods>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

/// @brief True if `set` contains every bit set in `m`.
constexpr bool has(Mods set, Mods m) {
    return (static_cast<uint8_t>(set) & static_cast<uint8_t>(m)) != 0;
}

/// @brief A single discrete keyboard event captured during the last
/// presentation::Window::poll_events() call. Unlike
/// Window::is_key_pressed() (level-triggered polling), this captures edges
/// and OS key-repeat.
struct KeyEvent {
    Key       key = Key::Unknown;
    int       scancode = 0;  ///< Platform-specific scancode, passed through unchanged.
    KeyAction action = KeyAction::Release;
    Mods      mods = Mods::None;
};

/// @brief Standard cursor shapes for hover feedback (e.g. text fields, buttons).
enum class CursorShape {
    Arrow,
    IBeam,
    Hand,
};

} // namespace input
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_INPUT_KEYS_H
