# `gfxcoopa::command` Submodule

The `gfxcoopa::command` submodule provides RAII abstractions for Vulkan command buffers, command pools, and synchronization primitives.

---

## Command Recording & Synchronization Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                       COMMAND RECORDING & SYNCHRONIZATION FLOW                           │
└──────────────────────────────────────────────────────────────────────────────────────────┘

       ┌───────────────────────┐
       │      CommandPool      │
       └───────────┬───────────┘
                   │ allocates
                   ▼
       ┌───────────────────────┐                     ┌───────────────────────┐
       │     CommandBuffer     │                     │     Fence / Sync      │
       ├───────────────────────┤                     ├───────────────────────┤
       │ - begin()             │                     │ - Fence (CPU ↔ GPU)   │
       │ - bind_pipeline()     │                     │ - Semaphore (GPU ↔ GPU)│
       │ - set_viewport()      │                     └───────────┬───────────┘
       │ - draw()              │                                 │
       │ - end()               │◄────────────────────────────────┘
       └───────────┬───────────┘  synchronizes submit
                   │
                   ▼
       ┌───────────────────────┐
       │     VkQueueSubmit     │
       └───────────────────────┘
```

---

## Header Files

| File | Primary Class / Struct | Description |
|---|---|---|
| [`command_buffer.h`](command_buffer.h) | [`CommandBuffer`](command_buffer.h) | RAII wrapper for `VkCommandBuffer`. Offers command recording utilities for pipeline binding, descriptor binding, vertex/index buffer binding, draw calls, viewport/scissor dynamic states, push constants, pipeline barriers, and buffer/image copy commands. |
| [`command_pool.h`](command_pool.h) | [`CommandPool`](command_pool.h) | RAII wrapper around `VkCommandPool`. Manages command buffer allocation, resetting, and provides convenience helpers for transient single-use command execution (`begin_single_use()` / `end_single_use()`). |
| [`sync.h`](sync.h) | [`Fence`](sync.h), [`Semaphore`](sync.h) | RAII wrappers for Vulkan synchronization primitives: `Fence` for CPU↔GPU execution signaling (e.g. frame pacing) and `Semaphore` for GPU↔GPU queue submit synchronization. |

---

## Usage Example

```cpp
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/command/sync.h>

// Create a command pool for the graphics queue family
coopa::gfx::command::CommandPool pool(device, device.graphics_family());

// Allocate a command buffer
auto cmd = pool.allocate();

// Synchronize CPU and GPU
coopa::gfx::command::Fence fence(device, true); // initial signaled state
coopa::gfx::command::Semaphore image_available(device);
coopa::gfx::command::Semaphore render_finished(device);

// Single-use one-time command helper:
pool.single_use([&](coopa::gfx::command::CommandBuffer& transient_cmd) {
    transient_cmd.copy_buffer(src_buffer, dst_buffer, size);
});
```
