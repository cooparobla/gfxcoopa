# `gfxcoopa::presentation` Submodule

The `gfxcoopa::presentation` submodule handles desktop window management via GLFW and double-buffered frame orchestration through the high-level `Renderer` class.

---

## Frame Render Loop Lifecycle Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             FRAME RENDER LOOP LIFECYCLE                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

  Application            Window                Renderer              Swapchain / GPU
      │                    │                      │                        │
      │── poll_events() ──►│                      │                        │
      │                    │                      │                        │
      │── begin_frame() ─────────────────────────►│                        │
      │                    │                      │── acquire_next_image ─►│
      │                    │                      │◄─ image_index ─────────│
      │                    │                      │
      │                    │                      │── Record Commands
      │                    │                      │   (Draw calls via lambda)
      │                    │                      │
      │                    │                      │── submit(cmd) ────────►│
      │                    │                      │── present() ──────────►│
      │◄─ frame complete ─────────────────────────│
```

---

## Header Files

| File | Primary Class / Struct | Description |
|---|---|---|
| [`window.h`](window.h) | [`Window`](window.h) | RAII wrapper around GLFW windowing. Handles window creation, title management, event polling (`poll_events()`), framebuffer resize detection (`was_resized()`, `framebuffer_size()`), keyboard/mouse input states, and window close checks (`should_close()`). |
| [`renderer.h`](renderer.h) | [`Renderer`](renderer.h) | High-level frame engine manager. Handles per-frame double-buffering synchronization (fences and semaphores), swapchain image acquisition (`acquire`), command buffer recording callback dispatch, queue submission (`submit`), swapchain image presentation (`present`), and automatic swapchain recreation on resize. |

---

## Usage Example

```cpp
#include <gfxcoopa/presentation/window.h>
#include <gfxcoopa/presentation/renderer.h>

coopa::gfx::presentation::Window window("Application", 1280, 720);
// ... setup instance, device, swapchain, render_pass, command_pool ...
coopa::gfx::presentation::Renderer renderer(device, swapchain, render_pass, command_pool);

while (!window.should_close()) {
    window.poll_events();

    renderer.begin_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
        auto extent = swapchain.extent();
        cmd.set_viewport(0, 0, extent.width, extent.height);
        cmd.set_scissor(0, 0, extent.width, extent.height);
        // ... record render commands ...
    });
}

device.wait_idle();
```
