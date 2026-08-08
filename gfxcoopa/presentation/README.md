# Presentation Submodule (`coopa::gfx::presentation`)

The `coopa::gfx::presentation` submodule provides GLFW windowing wrappers and multi-buffered frame renderer orchestration (`begin_frame`/`end_frame`).

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

### [renderer.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/presentation/renderer.h)
- **Role**: High-level double/triple-buffered frame orchestration (`acquire` → `record` → `submit` → `present`).
- **Key Classes / Structs**: `Renderer`.
- **Details**: Manages `MAX_FRAMES_IN_FLIGHT` (default 2) sets of fences, semaphores, command buffers, and swapchain framebuffers. Detects out-of-date swapchains and invokes user resize callbacks.

### [window.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/presentation/window.h)
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

    renderer.begin_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
        cmd.bind_pipeline(pipeline);
        cmd.draw(3);
    });
}
device.wait_idle();
```
