# gfxcoopa

A header-only C++20 Vulkan wrapper designed for OpenGL developers. Provides RAII objects, named methods, and sensible defaults — so you can draw a triangle in ~20 lines instead of 800.

All public API lives in the `coopa::gfx::` namespace. Every module is a header file under `gfxcoopa/`. Build with `cbuild`, run with `cplay`.

---

## High-Level Engine Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             GFXCOOPA ENGINE & FRAMEWORK MAP                              │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────────────┐
                                │     Window App        │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │      Renderer         │
                                └───────────┬───────────┘
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│     engine::Mesh      │       │   engine::Offscreen   │       │ engine::ShadowTarget  │
│  (YAML Geometry Load) │       │ (Toon/Upscale Pass)   │       │ (Dir / Point Shadows) │
└──────────┬────────────┘       └───────────┬───────────┘       └───────────┬───────────┘
           │                                │                               │
           └────────────────────────────────┼───────────────────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │  pipeline::Pipeline   │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │ command::CommandBuffer│
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │     memory & core     │
                                └───────────────────────┘
```

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

# Run the test suite / demo
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

For detailed documentation, see the submodule README files under [`gfxcoopa/`](gfxcoopa/README.md).

### [`gfxcoopa/util/`](gfxcoopa/util/README.md)

| File | Description |
|---|---|
| [error.h](gfxcoopa/util/error.h) | `GFX_VK_CHECK(expr)` macro, `vk_result_string()` |
| [volk_init.h](gfxcoopa/util/volk_init.h) | One-time `volkInitialize()` guard |
| [debug_messenger.h](gfxcoopa/util/debug_messenger.h) | RAII `VkDebugUtilsMessengerEXT` with stderr logging |
| [format.h](gfxcoopa/util/format.h) | `format_from_channels()`, `format_byte_size()`, `format_has_depth()`, `format_has_stencil()` |

### [`gfxcoopa/core/`](gfxcoopa/core/README.md)

| File | Description |
|---|---|
| [instance.h](gfxcoopa/core/instance.h) | `class Instance` — volk init, validation layers, VkInstance |
| [surface.h](gfxcoopa/core/surface.h) | `class Surface` — GLFW → VkSurfaceKHR, `query_support()` |
| [device.h](gfxcoopa/core/device.h) | `class Device` — GPU selection, queues, `wait_idle()` |
| [swapchain.h](gfxcoopa/core/swapchain.h) | `class Swapchain` — images, views, `recreate()` |

### [`gfxcoopa/memory/`](gfxcoopa/memory/README.md)

| File | Description |
|---|---|
| [allocator.h](gfxcoopa/memory/allocator.h) | `class Allocator` — VMA lifecycle wrapper |
| [buffer.h](gfxcoopa/memory/buffer.h) | `class Buffer` — vertex/index/uniform/staging/storage factories, `upload()` |
| [image.h](gfxcoopa/memory/image.h) | `class Image` — 2D image + view wrapper, `transition_layout()` |

### [`gfxcoopa/pipeline/`](gfxcoopa/pipeline/README.md)

| File | Description |
|---|---|
| [shader.h](gfxcoopa/pipeline/shader.h) | `class Shader` — loads `.spv`, provides `stage_info()` |
| [render_pass.h](gfxcoopa/pipeline/render_pass.h) | `class RenderPass` — color + optional depth subpass, attachment layouts |
| [descriptor.h](gfxcoopa/pipeline/descriptor.h) | `DescriptorPool`, `DescriptorSetLayout`, `DescriptorSet` wrappers |
| [pipeline.h](gfxcoopa/pipeline/pipeline.h) | `class Pipeline` — full graphics pipeline, `PipelineConfig`, `set_viewport()` |

### [`gfxcoopa/command/`](gfxcoopa/command/README.md)

| File | Description |
|---|---|
| [command_pool.h](gfxcoopa/command/command_pool.h) | `class CommandPool` — allocation, `begin_single_use()` / `end_single_use()` |
| [command_buffer.h](gfxcoopa/command/command_buffer.h) | `class CommandBuffer` — `begin/end`, `bind_pipeline`, `draw`, `set_viewport`, `copy_buffer` |
| [sync.h](gfxcoopa/command/sync.h) | `class Fence` (CPU↔GPU), `class Semaphore` (GPU↔GPU) |

### [`gfxcoopa/presentation/`](gfxcoopa/presentation/README.md)

| File | Description |
|---|---|
| [window.h](gfxcoopa/presentation/window.h) | `class Window` — GLFW init, `poll_events()`, `framebuffer_size()`, `was_resized()` |
| [renderer.h](gfxcoopa/presentation/renderer.h) | `class Renderer` — acquire → record → submit → present frame loop |

### [`gfxcoopa/engine/`](gfxcoopa/engine/README.md)

| File | Description |
|---|---|
| [brdf_lut.h](gfxcoopa/engine/brdf_lut.h) | `class BRDFLUT` — 512x512 R16G16_SFLOAT BRDF LUT generation for specular IBL |
| [camera_ubo.h](gfxcoopa/engine/camera_ubo.h) | `struct CameraData`, `class CameraUBO` — host-visible camera matrices UBO with pixel snapping |
| [cubemap_target.h](gfxcoopa/engine/cubemap_target.h) | `class CubemapTarget` — offscreen HDR cubemap render target for reflection probes & skybox |
| [fullscreen_quad.h](gfxcoopa/engine/fullscreen_quad.h) | `class FullscreenQuad` — 3-vertex screen-space triangle draw helper (no VBO needed) |
| [gi_data.h](gfxcoopa/engine/gi_data.h) | `SHProbe`, `GiUniforms`, `GiSystemData` — Spherical Harmonics light probe grid and reflection probe UBOs |
| [light_data.h](gfxcoopa/engine/light_data.h) | `DirectionalLightGPU`, `PointLightGPU`, `LightUBO` — directional light & multi-point light UBOs |
| [mesh.h](gfxcoopa/engine/mesh.h) | `struct Vertex`, `class Mesh` — GPU mesh data loaded from YAML/raw data with vertex & index buffers |
| [model_ubo.h](gfxcoopa/engine/model_ubo.h) | `struct ModelPushConstants` — 128-byte push constant struct (`model` + `normal_matrix`) |
| [offscreen_target.h](gfxcoopa/engine/offscreen_target.h) | `class OffscreenTarget` — low-res / offscreen target with auto-transition to `SHADER_READ_ONLY_OPTIMAL` |
| [sampler.h](gfxcoopa/engine/sampler.h) | `class Sampler` — RAII `VkSampler` with factory methods (`nearest()`, `linear()`, shadow, cubemap) |
| [shadow_map_target.h](gfxcoopa/engine/shadow_map_target.h) | `class ShadowMapTarget` — directional light and point light cubemap shadow depth targets |
| [shadow_pipeline.h](gfxcoopa/engine/shadow_pipeline.h) | `class ShadowPipeline` — graphics pipelines specialized for shadow map depth pass rendering |

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
| **Namespace** | `coopa::gfx::core`, `coopa::gfx::pipeline`, `coopa::gfx::memory`, `coopa::gfx::command`, `coopa::gfx::presentation`, `coopa::gfx::engine`, `coopa::gfx::util` |
| **Include guards** | `#ifndef COOPA_GFX_<SUBSYSTEM>_<FILE>_H` |
| **Vulkan header** | Always `#include <volk/volk.h>` — never `<vulkan/vulkan.h>` directly |
| **Documentation** | Doxygen `@brief`, `@param`, `@return`, `/**<` inline member docs |
| **RAII** | Every Vulkan handle has constructor creation + destructor destruction |
| **Naming** | snake_case, private members have trailing `_`, OpenGL-named methods where applicable |
| **Layering** | gfxcoopa is the sole owner of Vulkan and windowing (GLFW). Consumers (blendy, uicoopa, pixengine, toyengine) interact only through gfxcoopa's sealed types (`Format`, `TextureView`, `SamplerDesc`, `VertexLayout`, `gfx::app::Context`, ...) and must never name a `Vk*`/`VK_*`/`vk*`/`Vma*`/`vma*`/`GLFW*`/`glfw*` symbol or `coopa::gfx::detail::*` directly. `types/` stays Vulkan/GLFW-free by design (the `coopa::gfx_pure` CMake target enforces this structurally); `detail/` holds the raw<->sealed conversions and is off-limits to consumers. The keyboard/mouse vocabulary and state model (`Key`, `MouseButton`, `Input`, `InputMap`, ...) live in `coopa::input` in libcoopa, not here — `presentation::Window` is only a *backend* that feeds a `coopa::input::Input` from GLFW callbacks (see `coopa/input/README.md`); a consumer reaches it via `ctx.input()`/`window.input()` and never touches GLFW input calls at all. Enforced per-consumer by `tools/check_no_vulkan.sh` (wired as `GFX_LEAK_CHECK` in each consumer's CMakeLists — ON where the repo is fully clean, OFF with a documented residual count where its own `engine/passes/*`/`engine/targets/*` call sites still force raw types through). |

---

## Documentation

```bash
coopadocs build
coopadocs show
```

Generated HTML covers all public classes, methods, and parameters.
