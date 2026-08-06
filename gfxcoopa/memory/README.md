# `gfxcoopa::memory` Submodule

The `gfxcoopa::memory` submodule provides Vulkan memory management powered by Vulkan Memory Allocator (VMA), offering RAII abstractions for GPU allocators, buffers, and images.

---

## Memory & Resource Allocation Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             MEMORY & RESOURCE ALLOCATION                                 │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                               ┌───────────────────────┐
                               │     VmaAllocator      │
                               ├───────────────────────┤
                               │       Allocator       │
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │ owns memory allocation                      │ owns memory allocation
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │        Buffer         │                     │         Image         │
        ├───────────────────────┤                     ├───────────────────────┤
        │ - vertex()            │                     │ - VkImage             │
        │ - index()             │                     │ - VkImageView         │
        │ - uniform()           │                     │ - transition_layout() │
        │ - staging()           │                     │ - copy_from_buffer()  │
        │ - upload()            │                     └───────────────────────┘
        └───────────────────────┘
```

---

## Header Files

| File | Primary Class / Struct | Description |
|---|---|---|
| [`allocator.h`](allocator.h) | [`Allocator`](allocator.h) | RAII wrapper for `VmaAllocator`. Initializes VMA using Volk dynamic Vulkan function table pointers and handles allocation context lifecycle. |
| [`buffer.h`](buffer.h) | [`Buffer`](buffer.h) | RAII wrapper around `VkBuffer` + `VmaAllocation`. Provides factory functions (`vertex()`, `index()`, `uniform()`, `staging()`, `storage()`), host-mapping, and data upload capabilities. |
| [`image.h`](image.h) | [`Image`](image.h) | RAII wrapper around `VkImage` + `VkImageView` + `VmaAllocation`. Manages 2D/cubemap image creation, image view generation, layout transitions (`transition_layout()`), and staging copy operations. |

---

## Usage Example

```cpp
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>

// Create VMA allocator
coopa::gfx::memory::Allocator allocator(instance, device);

// Create vertex buffer
std::vector<float> vertices = { ... };
auto vbo = coopa::gfx::memory::Buffer::vertex(allocator, vertices.data(), sizeof(float) * vertices.size());

// Create uniform buffer
auto ubo = coopa::gfx::memory::Buffer::uniform(allocator, sizeof(CameraData));
ubo.upload(&camera_data, sizeof(CameraData));

// Create texture image
coopa::gfx::memory::Image texture(
    allocator, device,
    width, height,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
);
texture.transition_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
```
