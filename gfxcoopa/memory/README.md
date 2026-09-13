# Memory Submodule (`coopa::gfx::memory`)

The `coopa::gfx::memory` submodule provides Vulkan Memory Allocator (VMA) RAII integration, GPU buffer management, and image allocation/transitions.

---

## Memory Subsystem Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            MEMORY SYSTEM MANAGEMENT GRAPH                                │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────────────┐
                                │       Allocator       │  (VmaAllocator)
                                └───────────┬───────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
        ┌───────────────────────┐                       ┌───────────────────────┐
        │        Buffer         │                       │         Image         │
        ├───────────────────────┤                       ├───────────────────────┤
        │ - VkBuffer            │                       │ - VkImage             │
        │ - VmaAllocation       │                       │ - VkImageView         │
        │ - map() / upload()    │                       │ - current_usage()     │
        └───────────────────────┘                       └───────────────────────┘
```

---

## File Breakdown

### [allocator.h](allocator.h)
- **Role**: RAII wrapper around Vulkan Memory Allocator (`VmaAllocator`).
- **Key Classes / Structs**: `Allocator`.
- **Details**: Initializes VMA using device and instance handles, providing low-overhead sub-allocated memory management for GPU buffers and textures.

### [buffer.h](buffer.h)
- **Role**: RAII abstraction for GPU memory buffers (`VkBuffer` + `VmaAllocation`).
- **Key Classes / Structs**: `Buffer`.
- **Details**: Manages staging buffers, vertex/index buffers, uniform buffers (UBOs), and storage buffers (SSBOs). Provides host-mapping methods (`map()`, `unmap()`, `upload()`) with explicit VMA memory usage flags.

### [image.h](image.h)
- **Role**: RAII abstraction for GPU images (`VkImage`, `VkImageView`, `VmaAllocation`).
- **Key Classes / Structs**: `Image`.
- **Details**: Owns a 2D `VkImage` + `VmaAllocation` + `VkImageView` and tracks the `TextureUsage` it was last transitioned to (`current_usage()`), which is what lets `command::CommandBuffer::transition()` take only a destination. Staged pixel upload lives in `image_upload.h`.

---

## Usage Example

```cpp
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>

// Create VMA Allocator
coopa::gfx::memory::Allocator allocator(instance, device);

// Allocate a GPU vertex buffer
coopa::gfx::memory::Buffer vertex_buffer(
    allocator,
    sizeof(Vertex) * vertices.size(),
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VMA_MEMORY_USAGE_GPU_ONLY
);

// Upload data using staging buffer
vertex_buffer.upload(cmd, vertices.data(), sizeof(Vertex) * vertices.size());
```
