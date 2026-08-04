/**
 * @file window.h
 * @brief GLFW window wrapper configured for Vulkan surface creation.
 *
 * Handles GLFW initialization, window creation, and cleanup. Designed to
 * be constructed first and passed to Instance, Surface, and Renderer.
 */

#ifndef COOPA_GFX_PRESENTATION_WINDOW_H
#define COOPA_GFX_PRESENTATION_WINDOW_H

#define GLFW_INCLUDE_NONE // Prevent GLFW from including its own OpenGL/Vulkan headers
#include <GLFW/glfw3.h>
#include <string>
#include <utility>
#include <stdexcept>

namespace coopa {
namespace gfx {
namespace presentation {

/**
 * @class Window
 * @brief RAII wrapper around a GLFW window configured for Vulkan rendering.
 *
 * Initializes GLFW, creates a non-resizable (by default) Vulkan-compatible
 * window, and handles cleanup on destruction. Exposes the underlying
 * GLFWwindow* handle for use with the Vulkan surface and ImGui.
 */
class Window {
public:
    /**
     * @brief Initializes GLFW and creates a Vulkan-compatible window.
     *
     * GLFW_CLIENT_API is set to GLFW_NO_API so no implicit OpenGL context
     * is created. The window is non-resizable by default; call
     * set_resizable(true) before construction or use the extended constructor.
     *
     * @param title Window title bar string.
     * @param width Window width in pixels.
     * @param height Window height in pixels.
     * @param resizable Allow the user to resize the window.
     * @throws std::runtime_error if GLFW init or window creation fails.
     */
    Window(const std::string& title, uint32_t width, uint32_t height,
           bool resizable = false)
        : width_(width), height_(height)
    {
        if (!glfwInit()) {
            throw std::runtime_error("[gfxcoopa] glfwInit() failed.");
        }

        // No OpenGL context — Vulkan provides its own surface.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,  resizable ? GLFW_TRUE : GLFW_FALSE);

        window_ = glfwCreateWindow(
            static_cast<int>(width_),
            static_cast<int>(height_),
            title.c_str(), nullptr, nullptr);

        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("[gfxcoopa] glfwCreateWindow() failed.");
        }

        // Store a back-pointer so resize callbacks can update our members.
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebuffer_resize_callback);
    }

    /**
     * @brief Destroys the GLFW window and terminates GLFW.
     */
    ~Window() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    /// @brief Non-copyable.
    Window(const Window&) = delete;
    /// @brief Non-copyable.
    Window& operator=(const Window&) = delete;

    // --- Query ---

    /**
     * @brief Returns true if the user has requested the window to close.
     * @return True when GLFW_CLOSE was signalled.
     */
    bool should_close() const { return glfwWindowShouldClose(window_); }

    /**
     * @brief Signals the window to close.
     * @param value Set to true to request the window to close.
     */
    void set_should_close(bool value = true) {
        glfwSetWindowShouldClose(window_, value ? GLFW_TRUE : GLFW_FALSE);
    }

    /**
     * @brief Returns true if the given keyboard key is currently pressed.
     * @param key GLFW key code (e.g., GLFW_KEY_ESCAPE).
     * @return True if key state is GLFW_PRESS.
     */
    bool is_key_pressed(int key) const {
        return glfwGetKey(window_, key) == GLFW_PRESS;
    }

    /**
     * @brief Polls pending OS events (keyboard, mouse, resize, close).
     *
     * Must be called once per frame on the main thread.
     */
    void poll_events() { glfwPollEvents(); }

    /**
     * @brief Returns the underlying GLFWwindow pointer.
     *
     * Required for Vulkan surface creation and ImGui integration.
     *
     * @return Raw GLFWwindow*.
     */
    GLFWwindow* handle() const { return window_; }

    /**
     * @brief Returns the current framebuffer size in pixels.
     *
     * Use this (not the window size) when creating or recreating the swapchain,
     * as they differ on HiDPI/Retina displays.
     *
     * @return Pair of (width, height) in pixels.
     */
    std::pair<uint32_t, uint32_t> framebuffer_size() const {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        return { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
    }

    /**
     * @brief Returns true if the framebuffer was resized since last checked.
     *
     * Call reset_resized() after handling the resize event (e.g. after
     * recreating the swapchain).
     *
     * @return True if a resize event has been recorded.
     */
    bool was_resized() const { return resized_; }

    /**
     * @brief Clears the resized flag after the caller has handled it.
     */
    void reset_resized() { resized_ = false; }

private:
    /**
     * @brief GLFW framebuffer resize callback.
     *
     * Sets the resized_ flag so the render loop can detect and recreate
     * the swapchain. Handles the case where the window is minimized
     * (size 0x0) by waiting until a non-zero size is reported.
     *
     * @param window The GLFW window that was resized.
     * @param width New framebuffer width.
     * @param height New framebuffer height.
     */
    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
        (void)width; (void)height;
        auto* self = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self) {
            self->resized_ = true;
        }
    }

    GLFWwindow* window_  = nullptr; /**< The underlying GLFW window handle. */
    uint32_t    width_;             /**< Initial window width in pixels. */
    uint32_t    height_;            /**< Initial window height in pixels. */
    bool        resized_ = false;   /**< Set to true when a framebuffer resize event arrives. */
};

} // namespace presentation
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_PRESENTATION_WINDOW_H
