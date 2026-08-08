# Command Submodule (`coopa::gfx::command`)

The `coopa::gfx::command` submodule provides RAII abstractions for Vulkan command buffers, command pools, and synchronization primitives.

---

## Command Recording Architecture

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

## File Breakdown

### [command_buffer.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/command/command_buffer.h)
- **Role**: Lightweight RAII wrapper around `VkCommandBuffer` for recording graphics, compute, and transfer commands.
- **Key Classes / Structs**: `CommandBuffer`.
- **Details**: Exposes inline methods for render pass recording (`begin_render_pass`/`end_render_pass`), pipeline binding, descriptor set binding, vertex/index buffer binding, viewport/scissor dynamic states, push constants, pipeline barriers, and buffer/image copy commands.

### [command_pool.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/command/command_pool.h)
- **Role**: RAII manager for `VkCommandPool` allocation and execution of transient single-use command buffers.
- **Key Classes / Structs**: `CommandPool`.
- **Details**: Handles allocation of primary/secondary command buffers, pool resetting, and provides a synchronous lambda helper `single_use()` for one-time GPU transfer and copy operations.

### [sync.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/command/sync.h)
- **Role**: RAII wrappers for Vulkan synchronization primitives.
- **Key Classes / Structs**: `Fence`, `Semaphore`.
- **Details**: `Fence` manages CPU-GPU execution synchronization with `wait()` and `reset()` helpers; `Semaphore` handles GPU-GPU queue submission signaling for swapchain image acquire and presentation.

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
