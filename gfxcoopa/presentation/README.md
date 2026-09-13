# Presentation Submodule (`coopa::gfx::presentation`)

The `coopa::gfx::presentation` submodule provides GLFW windowing wrappers and multi-buffered frame orchestration. `Renderer::draw_frame()` runs the whole acquire -> record -> submit -> present cycle in one call; `begin_frame()` is a deprecated alias for it.

---

## Presentation Frame Loop Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           FRAME ORCHESTRATION ARCHITECTURE                               │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────────────┐
                                │        Window         │  (GLFW Handles & Input)
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │       Renderer        │
                                ├───────────────────────┤
                                │ - Framebuffers        │
                                │ - Fences & Semaphores │
                                │ - Command Buffers     │
                                └───────────┬───────────┘
                                            │
       ┌────────────────────────────────────┼────────────────────────────────────┐
       ▼                                    ▼                                    ▼
┌───────────────┐                  ┌─────────────────┐                  ┌────────────────┐
│ Acquire Image │                  │ Record Commands │                  │ Submit/Present │
└───────────────┘                  └─────────────────┘                  └────────────────┘
```

---

## File Breakdown

### [renderer.h](renderer.h)
- **Role**: High-level double/triple-buffered frame orchestration (`acquire` → `record` → `submit` → `present`).
- **Key Classes / Structs**: `Renderer`.
- **Details**: Manages `MAX_FRAMES_IN_FLIGHT` (default 2) sets of fences, semaphores, command buffers, and swapchain framebuffers. Detects out-of-date swapchains and invokes user resize callbacks.

### [window.h](window.h)
- **Role**: GLFW window creation, event polling, and input handling wrapper.
- **Key Classes / Structs**: `Window`.
- **Details**: Encapsulates GLFW window lifecycle, framebuffer size callbacks, input polling (`should_close()`, `poll_events()`), and window minimization state checking.

---

## Usage Example

```cpp
#include <gfxcoopa/presentation/window.h>
#include <gfxcoopa/presentation/renderer.h>

coopa::gfx::presentation::Window window("Application", 1280, 720);
coopa::gfx::presentation::Renderer renderer(device, swapchain, render_pass, cmd_pool);

while (!window.should_close()) {
    window.poll_events();

    renderer.draw_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
        cmd.bind_pipeline(pipeline);
        cmd.draw(3);
    });
}
device.wait_idle();
```
