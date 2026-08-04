# gfxcoopa

A header-only C++20 Vulkan wrapper designed for OpenGL developers. Provides
RAII objects, named methods, and sensible defaults — so you can draw a
triangle in ~20 lines instead of 800.

All public API lives in the `coopa::gfx::` namespace. Every module is a
single `.h` file under `gfxcoopa/`. Build with `cbuild`, run with `cplay`.

---

## Quick start

```cpp
#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/presentation/window.h>
#include <gfxcoopa/core/surface.h>
#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/presentation/renderer.h>

int main() {
    coopa::gfx::presentation::Window   window("My App", 1280, 720);
    coopa::gfx::core::Instance         instance("my_app");
    coopa::gfx::core::Surface          surface(instance, window.handle());
    coopa::gfx::core::Device           device(instance, surface);
    coopa::gfx::core::Swapchain        swapchain(device, surface, 1280, 720);
    coopa::gfx::pipeline::RenderPass   render_pass(device, swapchain.image_format());
    coopa::gfx::command::CommandPool   cmd_pool(device, device.graphics_family());
    coopa::gfx::pipeline::Shader       vert(device, "vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    coopa::gfx::pipeline::Shader       frag(device, "frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    coopa::gfx::pipeline::Pipeline     pipeline(device, render_pass, {&vert, &frag}, {}, {});
    coopa::gfx::presentation::Renderer renderer(device, swapchain, render_pass, cmd_pool);

    while (!window.should_close()) {
        window.poll_events();
        renderer.begin_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
            auto ext = swapchain.extent();
            cmd.bind_pipeline(pipeline);
            cmd.set_viewport(0, 0, ext.width, ext.height);
            cmd.set_scissor(0, 0, ext.width, ext.height);
            cmd.draw(3);
        });
    }
    device.wait_idle();
}
```

---

## Build

```bash
# Configure and build
cbuild

# Run the test suite
cplay
```

Or manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
./gfxcoopa
```

Shaders must be compiled to SPIR-V before running:

```bash
glslc assets/shaders/test.vert -o assets/shaders/test.vert.spv
glslc assets/shaders/test.frag -o assets/shaders/test.frag.spv
```

---

## Module overview

### `gfxcoopa/util/`

| File | Description |
|---|---|
| [error.h](gfxcoopa/util/error.h) | `GFX_VK_CHECK(expr)` macro, `vk_result_string()` |
| [volk_init.h](gfxcoopa/util/volk_init.h) | One-time `volkInitialize()` guard |
| [debug_messenger.h](gfxcoopa/util/debug_messenger.h) | RAII `VkDebugUtilsMessengerEXT` with stderr logging |
| [format.h](gfxcoopa/util/format.h) | `format_from_channels()`, `format_byte_size()`, `format_has_depth()` |

### `gfxcoopa/core/`

| File | Description |
|---|---|
| [instance.h](gfxcoopa/core/instance.h) | `class Instance` — volk init, validation layers, VkInstance |
| [surface.h](gfxcoopa/core/surface.h) | `class Surface` — GLFW → VkSurfaceKHR, `query_support()` |
| [device.h](gfxcoopa/core/device.h) | `class Device` — GPU selection, queues, `wait_idle()` |
| [swapchain.h](gfxcoopa/core/swapchain.h) | `class Swapchain` — images, views, `recreate()` |

### `gfxcoopa/memory/`

| File | Description |
|---|---|
| [allocator.h](gfxcoopa/memory/allocator.h) | `class Allocator` — VMA lifecycle |
| [buffer.h](gfxcoopa/memory/buffer.h) | `class Buffer` — vertex/index/uniform/staging factories, `upload()` |
| [image.h](gfxcoopa/memory/image.h) | `class Image` — 2D image + view, `transition_layout()` |

### `gfxcoopa/pipeline/`

| File | Description |
|---|---|
| [shader.h](gfxcoopa/pipeline/shader.h) | `class Shader` — loads `.spv`, provides `stage_info()` |
| [render_pass.h](gfxcoopa/pipeline/render_pass.h) | `class RenderPass` — color + optional depth subpass |
| [descriptor.h](gfxcoopa/pipeline/descriptor.h) | `DescriptorPool`, `DescriptorSetLayout`, `DescriptorSet` |
| [pipeline.h](gfxcoopa/pipeline/pipeline.h) | `class Pipeline` — full graphics pipeline, `PipelineConfig`, `set_viewport()` |

### `gfxcoopa/command/`

| File | Description |
|---|---|
| [command_pool.h](gfxcoopa/command/command_pool.h) | `class CommandPool` — allocation, `begin_single_use()` / `end_single_use()` |
| [command_buffer.h](gfxcoopa/command/command_buffer.h) | `class CommandBuffer` — `begin/end`, `bind_pipeline`, `draw`, `set_viewport` |
| [sync.h](gfxcoopa/command/sync.h) | `class Fence` (CPU↔GPU), `class Semaphore` (GPU↔GPU) |

### `gfxcoopa/presentation/`

| File | Description |
|---|---|
| [window.h](gfxcoopa/presentation/window.h) | `class Window` — GLFW init, `poll_events()`, `framebuffer_size()`, `was_resized()` |
| [renderer.h](gfxcoopa/presentation/renderer.h) | `class Renderer` — acquire → record → submit → present frame loop |

---

## Vendored dependencies

| Library | Location | Purpose |
|---|---|---|
| [volk](https://github.com/zeux/volk) | `includes/volk/` | Dynamic Vulkan function loading (no `libvulkan.so` link) |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `includes/vma/vk_mem_alloc.h` | GPU memory allocation |

---

## Conventions

| Convention | Detail |
|---|---|
| **Namespace** | `coopa::gfx::core`, `coopa::gfx::pipeline`, `coopa::gfx::memory`, `coopa::gfx::command`, `coopa::gfx::presentation`, `coopa::gfx::util` |
| **Include guards** | `#ifndef COOPA_GFX_<SUBSYSTEM>_<FILE>_H` |
| **Vulkan header** | Always `#include <volk/volk.h>` — never `<vulkan/vulkan.h>` directly |
| **Documentation** | Doxygen `@brief`, `@param`, `@return`, `/**<` inline member docs |
| **RAII** | Every Vulkan handle has constructor creation + destructor destruction |
| **Naming** | snake_case, private members have trailing `_`, OpenGL-named methods where applicable |

---

## Documentation

```bash
coopadocs build
coopadocs show
```

Generated HTML covers all public classes, methods, and parameters.
