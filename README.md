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
        renderer.draw_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
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
| [image_readback.h](gfxcoopa/util/image_readback.h) | `read_image()`, `save_png()` — GPU image download to host memory or PNG |

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
| [image.h](gfxcoopa/memory/image.h) | `class Image` — 2D image + view wrapper, tracks its own `current_usage()` |
| [image_upload.h](gfxcoopa/memory/image_upload.h) | `upload_image_2d()` — staged pixel upload, leaves the image in `ShaderRead` |

### [`gfxcoopa/pipeline/`](gfxcoopa/pipeline/README.md)

| File | Description |
|---|---|
| [shader.h](gfxcoopa/pipeline/shader.h) | `class Shader` — loads `.spv`, provides `stage_info()` |
| [render_pass.h](gfxcoopa/pipeline/render_pass.h) | `class RenderPass` — color + optional depth subpass, attachment layouts |
| [descriptor.h](gfxcoopa/pipeline/descriptor.h) | `DescriptorPool`, `DescriptorSetLayout`, `DescriptorSet` wrappers |
| [pipeline.h](gfxcoopa/pipeline/pipeline.h) | `class Pipeline` — full graphics pipeline, `PipelineDesc`/`BlendMode`, `set_viewport()` |
| [shader_library.h](gfxcoopa/pipeline/shader_library.h) | `class ShaderLibrary` — ordered search path from a logical shader name to its `.spv` |
| [surface_shader.h](gfxcoopa/pipeline/surface_shader.h) | `SurfaceShaderDesc`, `SurfaceShaderLibrary` — named per-material shader variants |

### [`gfxcoopa/command/`](gfxcoopa/command/README.md)

| File | Description |
|---|---|
| [command_pool.h](gfxcoopa/command/command_pool.h) | `class CommandPool` — allocation, `submit_once()` for one-shot GPU work |
| [command_buffer.h](gfxcoopa/command/command_buffer.h) | `class CommandBuffer` — `begin/end`, `bind_pipeline`, `draw`, `set_viewport`, `transition`, `copy_*`, `blit` |
| [sync.h](gfxcoopa/command/sync.h) | `class Fence` (CPU↔GPU), `class Semaphore` (GPU↔GPU) |

### [`gfxcoopa/presentation/`](gfxcoopa/presentation/README.md)

| File | Description |
|---|---|
| [window.h](gfxcoopa/presentation/window.h) | `class Window` — GLFW init, `poll_events()`, `framebuffer_size()`, `was_resized()` |
| [renderer.h](gfxcoopa/presentation/renderer.h) | `class Renderer` — acquire → record → submit → present frame loop |

### [`gfxcoopa/app/`](gfxcoopa/app/context.h)

| File | Description |
|---|---|
| [context.h](gfxcoopa/app/context.h) | `ContextConfig`, `FrameCallbacks`, `class Context` — owns the whole Window → Instance → Surface → Device → Allocator → Swapchain → CommandPool → RenderPass → Renderer bring-up, frame timing, and the main loop |

### [`gfxcoopa/types/`](gfxcoopa/types.h)

Vulkan- and GLFW-free by construction; the `coopa::gfx_pure` CMake target enforces it.
[types.h](gfxcoopa/types.h) is an umbrella that includes all of them.

| File | Description |
|---|---|
| [enums.h](gfxcoopa/types/enums.h) | `Format`, `TextureUsage`, `ImageUsage`, `ShaderStage`, `CullMode`, `MemoryResidency`, … — the sealed vocabulary replacing `Vk*` enums in the public API |
| [format.h](gfxcoopa/types/format.h) | `Format` helpers — `is_depth()`, `is_stencil()`, `format_byte_size()` |
| [vertex_layout.h](gfxcoopa/types/vertex_layout.h) | `VertexBinding`, `VertexAttribute`, `VertexLayout` — a vertex type's full input description, built fluently |
| [sampler_desc.h](gfxcoopa/types/sampler_desc.h) | `SamplerDesc` + presets (`linear_repeat()`, `nearest_clamp()`, …) |
| [texture_view.h](gfxcoopa/types/texture_view.h) | `TextureView` — opaque, hashable, null-able identity token for a texture view |
| [clear.h](gfxcoopa/types/clear.h) | `ClearColor`, `Extent2D`, `ImageRegion` |

### [`gfxcoopa/detail/`](gfxcoopa/detail/vk_convert.h)

Internal. Consumers must not include these or name `coopa::gfx::detail::*`.

| File | Description |
|---|---|
| [vk_convert.h](gfxcoopa/detail/vk_convert.h) | `to_vk()` / `from_vk()` conversions, barrier masks, `RawRenderPass` |
| [glfw_keys.h](gfxcoopa/detail/glfw_keys.h) | GLFW key/button code → `coopa::input` vocabulary |

### [`gfxcoopa/engine/`](gfxcoopa/engine/README.md)

The rendering engine built on the layers above. See the
[engine README](gfxcoopa/engine/README.md) for the full file-by-file breakdown.

| Directory | Contents |
|---|---|
| [`components/`](gfxcoopa/engine/components/register.h) | Scene components parsed from YAML — `MeshRenderer`/`PBRMaterial`, `CameraComponent`, `DirectionalLight`, `PointLight`, `EnvironmentLight`, `ReflectionProbe`, `GiProbeVolume`, `Volume`, `SdfShape`/`SdfRenderer`, plus `register_render_components()` |
| [`data/`](gfxcoopa/engine/data/mesh.h) | GPU-side data and UBO layouts — `Vertex`/`Mesh`, `Texture`, `CameraUBO`, `LightUBO`, `ModelPushConstants`, `FogData`, `VolumetricsData`, `SdfData`, `PaletteLut` |
| [`targets/`](gfxcoopa/engine/targets/offscreen_target.h) | Render targets — `OffscreenTarget`, `GBufferTarget`, `ShadowMapTarget`, `CubemapTarget`, `TransparentCaptureTarget` |
| [`passes/`](gfxcoopa/engine/passes/fullscreen_stage.h) | The render passes. `FullscreenStage` is the shared scaffold most post-processing passes are built from; `GBufferPipeline`, `ShadowPipeline`, `TransparentPass` and the `sdf_*` passes draw real geometry instead |
| [`gi/`](gfxcoopa/engine/gi/gi_system.h) | Global illumination — `GiSystem`, `GiBaker` (SH probe baking), `SHProbe`/`GiUniforms`, `BRDFLUT` |
| [`loaders/`](gfxcoopa/engine/loaders/mesh_loader.h) | `coopa::asset::AssetManager` loaders for meshes and textures |
| [`util/`](gfxcoopa/engine/util/sampler.h) | `Sampler`, `MaterialTextureCache`, `InstanceBatcher`, `SmaaTextures`, `SsaoKernel`, SH math |
| [render_features.h](gfxcoopa/engine/render_features.h) | `IndirectParams` — indirect-lighting terms shared by the lighting and SSR passes, plus the runtime-vs-startup feature-flag convention |

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
