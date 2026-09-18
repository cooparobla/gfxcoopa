# GfxCoopa Module (`coopa::gfx`)

`gfxcoopa` is a modern, header-only C++20 Vulkan rendering library designed to bring clean RAII ergonomics, low-overhead resource management, and state-of-the-art graphics feature sets to Vulkan applications. Built around dynamic API function loading via Volk and GPU memory management via Vulkan Memory Allocator (VMA), `gfxcoopa` hides low-level Vulkan boilerplate without sacrificing performance or direct hardware control.

The module provides a complete Physically-Based Rendering (PBR) engine incorporating Cook-Torrance BRDF shading (GGX microfacet distribution, Smith masking-shadowing, Schlick Fresnel), 2nd-order (9-coefficient) Spherical Harmonics (SH) Global Illumination probe volumes, Hierarchical Z-Buffer (Hi-Z) Screen-Space Reflections (SSR), directional and omnidirectional point light shadow mapping, multi-mode Anti-Aliasing (Jimenez SMAA 1x Ultra, TAA with temporal accumulation, FXAA 3.11), ACES filmic tonemapping, and GLFW presentation loop integration.

---

## GfxCoopa Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                GFXCOOPA LAYER ARCHITECTURE                               │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────────────┐
                                │   Application Code    │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │     presentation      │  (Window, Renderer)
                                └───────────┬───────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
        ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
        │        engine         │ │     pipeline      │ │        command        │
        │(Targets, Mesh, Lights)│ │(Shaders, Passes)  │ │(CmdPool, CmdBuffer)   │
        └───────────┬───────────┘ └─────────┬─────────┘ └───────────┬───────────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
        ┌───────────────────────┐                       ┌───────────────────────┐
        │        memory         │                       │         core          │
        │(Allocator, VMA, Image)│                       │(Instance, Device, Swap)│
        └───────────┬───────────┘                       └───────────┬───────────┘
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │         util          │  (Volk, Error, Debug)
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │      Vulkan API       │
                                └───────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             ENGINE RENDER PIPELINE FLOW                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │       coopa::gfx::engine::Mesh        │
                         │ ── CameraUBO, LightUBO, ModelPush     │
                         └───────────────────┬───────────────────┘
                                             │
      ┌──────────────────────┬───────────────┴───────────────┬──────────────────────┐
      ▼                      ▼                               ▼                      ▼
┌───────────┐      ┌──────────────────┐           ┌───────────────────┐    ┌──────────────────┐
│ SHADOWS   │      │ G-BUFFER PASS    │           │ GI BAKING SYSTEM  │    │ VULKAN DESCR.    │
├───────────┤      ├──────────────────┤           ├───────────────────┤    ├──────────────────┤
│ Direction │      │ G0: Albedo + AO  │           │ SH Ray Probe Bakes│    │ Set 0: Camera UBO│
│  - 2048²  │      │ G1: Normal + Met │           │ 9 L2 SH Coeffs    │    │ Set 1: Light UBO │
│ Point Cube│      │ G2: Pos + Rough  │           │ BRDF LUT Bakes    │    │ Set 2: Shadows   │
│  - 512²   │      │ Depth Attachment │           │ Reflection Maps   │    │ Set 3: GI SSBO   │
└─────┬─────┘      └─────────┬────────┘           └─────────┬─────────┘    └─────────┬────────┘
      │                      │                              │                        │
      │   ┌──────────────────┴──────────────────────────────┴────────────────────────┘
      ▼   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         DEFERRED LIGHTING & SKYBOX PASS (HDR PASS)                       │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  • Target Format: VK_FORMAT_R16G16B16A16_SFLOAT Offscreen Target                        │
│  • Shading: Cook-Torrance BRDF (GGX Distribution + Smith Masking + Schlick Fresnel)     │
│  • Indirect Lighting: Spherical Harmonics L2 Cosine-Lobe Convolution                     │
└──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│        HI-Z & SSR PASS (OPTIONAL)        │   │        SCENE COLOR MIP CHAIN PASS        │
├──────────────────────────────────────────┤   ├──────────────────────────────────────────┤
│  • Hi-Z Mipmap Pyramid Construction      │   │  • Prefiltered Scene Color Downsampling  │
│  • Hierarchical Ray-Marching SSR         │   │  • Glossy Cone Tapping Mip Chain         │
└────────────────────┬─────────────────────┘   └────────────────────┬─────────────────────┘
                     │                                              │
                     └─────────────────────┬────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        TONEMAPPING & ANTI-ALIASING POST-PROCESS PASS                     │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  • ACES Filmic Tone Curve & Gamma 2.2 Correction                                         │
│  • Anti-Aliasing Modes: SMAA 1x Ultra (Jimenez) / TAA (Halton Jitter) / FXAA 3.11         │
└──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────────┐
                         │           Vulkan Swapchain            │
                         │      (Present Frame to Display)       │
                         └───────────────────────────────────────┘
```

---

## Submodule Architecture Map

| Submodule | Namespace | Description |
|---|---|---|
| [`command`](command/README.md) | `coopa::gfx::command` | Command buffers, command pools, and CPU/GPU sync primitives (Fences, Semaphores). |
| [`core`](core/README.md) | `coopa::gfx::core` | Vulkan instance initialization, GLFW surface integration, physical/logical device selection, and swapchain management. |
| [`engine`](engine/README.md) | `coopa::gfx::engine` | PBR rendering engine, offscreen targets, shadow targets, SH probe GI volume baking, Hi-Z SSR, and AA passes. |
| [`memory`](memory/README.md) | `coopa::gfx::memory` | Vulkan Memory Allocator (VMA) RAII integration, GPU buffer management, and image allocation/transitions. |
| [`pipeline`](pipeline/README.md) | `coopa::gfx::pipeline` | Shaders, render passes, descriptor pools/layouts/sets, and graphics pipeline state configuration. |
| [`presentation`](presentation/README.md) | `coopa::gfx::presentation` | GLFW windowing wrapper and multi-buffered frame orchestration (`Renderer::draw_frame()`). |
| [`util`](util/README.md) | `coopa::gfx::util` | Debug messenger validation layers, format helpers, Volk dynamic loader initialization, and error macros. |

---

## Descriptor Set Layout Architecture

The `engine` rendering system organizes Vulkan descriptor set layouts into four specialized sets:

| Set | Owner / Source | Contents & Bindings | Stage Flags |
|---|---|---|---|
| **Set 0** | `CameraUBO` | Camera view/projection matrices, jittered projection, and eye position (UBO) | Vertex & Fragment |
| **Set 1** | `LightUBO` | Directional light vectors/colors, shadow matrices, and point light array (UBO) | Fragment |
| **Set 2** | `ShadowMapTarget` | 2D directional shadow map + 4 omnidirectional cubemap shadow depth samplers | Fragment |
| **Set 3** | `GiSystem` | SH Light Probe SSBO, GI Uniform UBO, BRDF Integration LUT, Reflection Cubemap | Fragment |

---

## File Breakdown

### Command Submodule (`coopa::gfx::command`)

#### [command_buffer.h](command/command_buffer.h)
- **Role**: Lightweight RAII wrapper around `VkCommandBuffer` for recording graphics, compute, and transfer commands.
- **Key Classes / Structs**: `CommandBuffer`.
- **Details**: Exposes inline methods for render pass recording (`begin_render_pass`/`end_render_pass`), pipeline binding, descriptor set binding, vertex/index buffer binding, viewport/scissor dynamic states, push constants, pipeline barriers, and buffer/image copy commands.

#### [command_pool.h](command/command_pool.h)
- **Role**: RAII manager for `VkCommandPool` allocation and execution of transient single-use command buffers.
- **Key Classes / Structs**: `CommandPool`.
- **Details**: Handles allocation of primary/secondary command buffers, pool resetting, and provides a synchronous lambda helper `single_use()` for one-time GPU transfer and copy operations.

#### [sync.h](command/sync.h)
- **Role**: RAII wrappers for Vulkan synchronization primitives.
- **Key Classes / Structs**: `Fence`, `Semaphore`.
- **Details**: `Fence` manages CPU-GPU execution synchronization with `wait()` and `reset()` helpers; `Semaphore` handles GPU-GPU queue submission signaling for swapchain image acquire and presentation.

---

### Core Submodule (`coopa::gfx::core`)

#### [device.h](core/device.h)
- **Role**: Manages physical GPU selection and logical `VkDevice` instantiation.
- **Key Classes / Structs**: `Device`, `QueueFamilyIndices`.
- **Details**: Automatically rates physical GPUs (prioritizing discrete graphics cards), queries queue families for graphics and present support, creates logical devices with required extensions (e.g. `VK_KHR_swapchain`), retrieves queues, and enforces idle wait on teardown (`wait_idle()`).

#### [instance.h](core/instance.h)
- **Role**: Manages Vulkan instance lifecycle (`VkInstance`) and Volk loader initialization.
- **Key Classes / Structs**: `Instance`.
- **Details**: Triggers `volkInitialize()` upon creation, queries required GLFW window extensions, enables `VK_LAYER_KHRONOS_validation` in debug builds, and sets up instance-level API layers.

#### [surface.h](core/surface.h)
- **Role**: Encapsulates GLFW window surface integration (`VkSurfaceKHR`).
- **Key Classes / Structs**: `Surface`, `SwapChainSupportDetails`.
- **Details**: Uses `glfwCreateWindowSurface` to bind Vulkan to the platform window and queries surface capabilities, supported surface formats, and presentation modes.

#### [swapchain.h](core/swapchain.h)
- **Role**: Encapsulates `VkSwapchainKHR` creation, image view management, and dynamic recreation.
- **Key Classes / Structs**: `Swapchain`.
- **Details**: Chooses optimal surface format (`VK_FORMAT_B8G8R8A8_SRGB`), present mode (`VK_PRESENT_MODE_MAILBOX_KHR` or `FIFO`), extent resolution, builds swapchain image views, and handles full recreation (`recreate()`) on window resize events.

---

### Memory Submodule (`coopa::gfx::memory`)

#### [allocator.h](memory/allocator.h)
- **Role**: RAII wrapper around Vulkan Memory Allocator (`VmaAllocator`).
- **Key Classes / Structs**: `Allocator`.
- **Details**: Initializes VMA using device and instance handles, providing low-overhead sub-allocated memory management for GPU buffers and textures.

#### [buffer.h](memory/buffer.h)
- **Role**: RAII abstraction for GPU memory buffers (`VkBuffer` + `VmaAllocation`).
- **Key Classes / Structs**: `Buffer`.
- **Details**: Manages staging buffers, vertex/index buffers, uniform buffers (UBOs), and storage buffers (SSBOs). Provides host-mapping methods (`map()`, `unmap()`, `upload()`) with explicit VMA memory usage flags.

#### [image.h](memory/image.h)
- **Role**: RAII abstraction for GPU images (`VkImage`, `VkImageView`, `VmaAllocation`).
- **Key Classes / Structs**: `Image`.
- **Details**: Owns a 2D `VkImage` + `VmaAllocation` + `VkImageView` and tracks the `TextureUsage` it was last transitioned to (`current_usage()`). Layout transitions are recorded by `command::CommandBuffer::transition()`; staged pixel upload lives in `memory/image_upload.h`.

---

### Pipeline Submodule (`coopa::gfx::pipeline`)

#### [descriptor.h](pipeline/descriptor.h)
- **Role**: RAII abstractions for Vulkan descriptor sets, layouts, and pools.
- **Key Classes / Structs**: `DescriptorSetLayout`, `DescriptorPool`, `DescriptorWriter`.
- **Details**: `DescriptorSetLayout` configures layout bindings; `DescriptorPool` allocates sets; `DescriptorWriter` provides a builder pattern for updating buffer (`bind_buffer`) and image (`bind_image`) descriptors via `vkUpdateDescriptorSets`.

#### [pipeline.h](pipeline/pipeline.h)
- **Role**: RAII encapsulation of graphics and compute pipelines (`VkPipeline`, `VkPipelineLayout`).
- **Key Classes / Structs**: `Pipeline`, `PipelineConfig`.
- **Details**: Provides a builder pattern (`PipelineConfig`) for vertex input bindings, input assembly, rasterization state, multisampling, depth-stencil testing, color blending, dynamic states, and push constant ranges.

#### [render_pass.h](pipeline/render_pass.h)
- **Role**: RAII wrapper for `VkRenderPass`.
- **Key Classes / Structs**: `RenderPass`, `RenderPassBuilder`.
- **Details**: Simplifies creation of multi-attachment color/depth render passes, subpass descriptions, and subpass dependencies.

#### [shader.h](pipeline/shader.h)
- **Role**: Encapsulates SPIR-V shader module loading (`VkShaderModule`).
- **Key Classes / Structs**: `ShaderModule`.
- **Details**: Loads binary SPIR-V bytecode from file paths and creates shader stage info structures (`VkPipelineShaderStageCreateInfo`).

---

### Presentation Submodule (`coopa::gfx::presentation`)

#### [renderer.h](presentation/renderer.h)
- **Role**: High-level double/triple-buffered frame orchestration (`acquire` → `record` → `submit` → `present`).
- **Key Classes / Structs**: `Renderer`.
- **Details**: Manages `MAX_FRAMES_IN_FLIGHT` (default 2) sets of fences, semaphores, command buffers, and swapchain framebuffers. Detects out-of-date swapchains and invokes user resize callbacks.

#### [window.h](presentation/window.h)
- **Role**: GLFW window creation, event polling, and input handling wrapper.
- **Key Classes / Structs**: `Window`.
- **Details**: Encapsulates GLFW window lifecycle, framebuffer size callbacks, input polling (`should_close()`, `poll_events()`), and window minimization state checking.

---

### Utility Submodule (`coopa::gfx::util`)

#### [debug_messenger.h](util/debug_messenger.h)
- **Role**: Vulkan validation layer debug messenger (`VkDebugUtilsMessengerEXT`).
- **Key Classes / Structs**: `DebugMessenger`.
- **Details**: Captures Vulkan API warning/error logs and routes formatted debug reports to stdout/stderr.

#### [error.h](util/error.h)
- **Role**: Error checking macros and string translation utilities.
- **Key Classes / Structs**: `GFX_VK_CHECK()`, `vk_result_string()`.
- **Details**: Throws `std::runtime_error` with source filename and line number when Vulkan API functions return non-success codes.

#### [format.h](util/format.h)
- **Role**: Vulkan format selection helpers.
- **Key Classes / Structs**: `find_supported_format()`, `find_depth_format()`.
- **Details**: Queries physical device format properties to select depth/stencil attachment formats (`VK_FORMAT_D32_SFLOAT`, `VK_FORMAT_D24_UNORM_S8_UINT`, etc.).

#### [volk_init.h](util/volk_init.h)
- **Role**: Dynamic Vulkan loader initialization wrapper.
- **Key Classes / Structs**: `init_volk()`.
- **Details**: Calls `volkInitialize()` to dynamically load Vulkan entry points without linking against static Vulkan loader libraries.

---

### Engine Submodule (`coopa::gfx::engine`)

#### Data Layer (`engine/data`)

##### [camera_ubo.h](engine/data/camera_ubo.h)
- **Role**: Camera uniform data structure and host-visible UBO buffer manager.
- **Key Classes / Structs**: `CameraData`, `CameraUBO`.
- **Details**: Uploads 16-byte aligned camera matrices (`view`, `proj`, `view_proj`, `inv_proj`, `inv_view`, `eye_pos`) and sub-pixel snapping jitter offsets for retro/TAA rendering.

##### [light_data.h](engine/data/light_data.h)
- **Role**: Uniform data structures for directional and point light sources.
- **Key Classes / Structs**: `DirectionalLightGPU`, `PointLightGPU`, `SpotLightGPU`, `LightDataGPU`, `LightUBO`.
- **Details**: Manages host-visible UBO allocations storing directional light matrix/color/direction and up to 4 omnidirectional point lights with attenuation parameters.

##### [mesh.h](engine/data/mesh.h)
- **Role**: GPU-resident mesh data buffer and geometry loading manager.
- **Key Classes / Structs**: `Vertex`, `Mesh`.
- **Details**: Defines 3D vertex attributes (position, normal, texcoord, tangent), creates GPU VMA vertex/index buffers, and parses Blender-exported YAML scene format files.

##### [model_ubo.h](engine/data/model_ubo.h)
- **Role**: Per-object push constant data layout structure.
- **Key Classes / Structs**: `ModelPushConstants`.
- **Details**: 128-byte push constant layout containing model matrix (`mat4`) and normal matrix (`mat4 normal_matrix = transpose(inverse(model))`).

#### Global Illumination Layer (`engine/gi`)

##### [brdf_lut.h](engine/gi/brdf_lut.h)
- **Role**: Precomputed BRDF integration look-up table generator.
- **Key Classes / Structs**: `BRDFLUT`.
- **Details**: Computes a 512x512 R16G16_SFLOAT texture encoding Cook-Torrance BRDF scale and bias values for environment map specular IBL.

##### [gi_baker.h](engine/gi/gi_baker.h)
- **Role**: CPU/GPU ray-tracing probe baker for Spherical Harmonics GI.
- **Key Classes / Structs**: `GiBaker`.
- **Details**: Casts 128 hemispherically distributed rays per probe position against scene geometry, projecting radiance into 9 L2 Spherical Harmonics basis coefficients.

##### [gi_data.h](engine/gi/gi_data.h)
- **Role**: Data structures and SSBO memory layout for GI probe grids and reflection probes.
- **Key Classes / Structs**: `SHProbe`, `ReflectionProbeGPU`, `GiUniforms`, `GiSystemData`.
- **Details**: Defines GPU memory layouts for 9-coefficient L2 SH probes, probe bounding volumes, uniform parameters, and reflection probe cubemap matrices.

##### [gi_system.h](engine/gi/gi_system.h)
- **Role**: Global Illumination subsystem orchestrator.
- **Key Classes / Structs**: `GiSystem`.
- **Details**: Manages probe grid SSBOs, BRDF LUT generation, reflection probe cubemap array, and descriptor set 3 layout bindings for real-time indirect lighting.

#### Render Targets (`engine/targets`)

##### [cubemap_target.h](engine/targets/cubemap_target.h)
- **Role**: Multi-face HDR cubemap render target manager.
- **Key Classes / Structs**: `CubemapTarget`.
- **Details**: Allocates 6-face cubemap images with individual face views and render passes for environment capture and reflection probe rendering.

##### [gbuffer_target.h](engine/targets/gbuffer_target.h)
- **Role**: Offscreen multi-attachment G-Buffer render target manager.
- **Key Classes / Structs**: `GBufferTarget`.
- **Details**: Manages geometry pass attachments: G0 (`R8G8B8A8_UNORM` Albedo + AO), G1 (`R16G16B16A16_SFLOAT` World Normal + Metallic), G2 (`R16G16B16A16_SFLOAT` World Position + Roughness), and Depth (`D32_SFLOAT`).

##### [offscreen_target.h](engine/targets/offscreen_target.h)
- **Role**: Offscreen HDR color render target with custom resolution support.
- **Key Classes / Structs**: `OffscreenTarget`.
- **Details**: Allocates `VK_FORMAT_R16G16B16A16_SFLOAT` color attachments for HDR rendering, automatically transitioning color images to `SHADER_READ_ONLY_OPTIMAL` for post-processing passes.

##### [shadow_map_target.h](engine/targets/shadow_map_target.h)
- **Role**: Shadow map depth attachment target manager.
- **Key Classes / Structs**: `ShadowMapTarget`.
- **Details**: Allocates 2048x2048 2D depth textures for directional lights and 512x512 6-face cubemap depth textures for omnidirectional point lights.

#### Render Passes & Pipelines (`engine/passes`)

##### [deferred_lighting_pass.h](engine/passes/deferred_lighting_pass.h)
- **Role**: Screen-space deferred shading pass evaluator.
- **Key Classes / Structs**: `DeferredLightingPass`.
- **Details**: Binds G-Buffer attachments, shadow map samplers, and GI probe SSBOs, evaluating Cook-Torrance direct PBR and indirect SH irradiance per pixel.

##### [env_prefilter_pass.h](engine/passes/env_prefilter_pass.h)
- **Role**: Specular environment cubemap pre-filtering pass.
- **Key Classes / Structs**: `EnvPrefilterPass`.
- **Details**: Generates mipmapped GGX pre-filtered environment cubemaps for glossy IBL reflections.

##### [gbuffer_pipeline.h](engine/passes/gbuffer_pipeline.h)
- **Role**: Graphics pipeline manager for G-Buffer geometry rasterization.
- **Key Classes / Structs**: `GBufferPipeline`.
- **Details**: Configures multi-render target (MRT) color blending, depth testing, backface culling, and shader stages for geometry pass rendering.

##### [hiz_pass.h](engine/passes/hiz_pass.h)
- **Role**: Hierarchical Z-Buffer depth pyramid generator.
- **Key Classes / Structs**: `HiZPass`.
- **Details**: Computes a downsampled depth mipmap pyramid using minimum depth reduction for fast hierarchical SSR raymarching.

##### [present_pass.h](engine/passes/present_pass.h)
- **Role**: Final blit pass transferring post-processed images to the swapchain.
- **Key Classes / Structs**: `PresentPass`.
- **Details**: Renders a full-screen quad sampling the final post-processing color buffer directly into the Vulkan swapchain framebuffer.

##### [probe_capture_pass.h](engine/passes/probe_capture_pass.h)
- **Role**: Scene capture pass into reflection probe cubemaps.
- **Key Classes / Structs**: `ProbeCapturePass`.
- **Details**: Renders scene geometry from 6 orthogonal directions into cubemap face attachments for localized reflection probes.

##### [scene_color_mip_pass.h](engine/passes/scene_color_mip_pass.h)
- **Role**: Downsampled scene color mipmap chain generator.
- **Key Classes / Structs**: `SceneColorMipPass`.
- **Details**: Builds a Gaussian/box downsampled color mip chain enabling glossy reflection cone sampling during SSR post-processing.

##### [shadow_pipeline.h](engine/passes/shadow_pipeline.h)
- **Role**: Depth-only graphics pipeline for shadow map rendering.
- **Key Classes / Structs**: `ShadowPipeline`.
- **Details**: Configures depth bias, slope-scaled depth bias, rasterizer state, and vertex shaders for 2D directional and omnidirectional cubemap shadow passes.

##### [skybox_pass.h](engine/passes/skybox_pass.h)
- **Role**: Background skybox rendering pass.
- **Key Classes / Structs**: `SkyboxPass`.
- **Details**: Evaluates analytic atmospheric scattering gradients or samples environment cubemaps for background pixels unreached by scene geometry.

##### [smaa_pass.h](engine/passes/smaa_pass.h)
- **Role**: Subpixel Morphological Anti-Aliasing (SMAA 1x Ultra) post-process pass.
- **Key Classes / Structs**: `SMAAPass`.
- **Details**: Implements Jimenez 2013 3-pass SMAA (edge detection, blend weight calculation, neighborhood blending) using precomputed area/search textures.

##### [ssr_pass.h](engine/passes/ssr_pass.h)
- **Role**: Hierarchical Screen-Space Reflections (SSR) pass.
- **Key Classes / Structs**: `SSRPass`.
- **Details**: Performs Hi-Z raymarching across depth pyramids, computing glossy specular reflection contributions with temporal reprojection.

##### [taa_pass.h](engine/passes/taa_pass.h)
- **Role**: Temporal Anti-Aliasing (TAA) pass.
- **Key Classes / Structs**: `TAAPass`.
- **Details**: Accumulates history frames using sub-pixel Halton jittering, motion vector reprojection, and color bounding-box neighborhood clamping to eliminate aliasing.

##### [tonemapping_pass.h](engine/passes/tonemapping_pass.h)
- **Role**: ACES filmic tonemapping and FXAA post-processing pass.
- **Key Classes / Structs**: `TonemappingPass`.
- **Details**: Applies exposure scaling, ACES filmic s-curve tonemapping, gamma 2.2 correction, and optional FXAA 3.11 edge smoothing.

#### Engine Utilities (`engine/util`)

##### [sampler.h](engine/util/sampler.h)
- **Role**: RAII `VkSampler` wrapper and factory utility.
- **Key Classes / Structs**: `Sampler`.
- **Details**: Factory helpers for `nearest()`, `linear()`, `shadow()` (depth comparison), and `cubemap()` texture samplers with anisotropy settings.

##### [sh_math.h](engine/util/sh_math.h)
- **Role**: 2nd-order Spherical Harmonics mathematical functions.
- **Key Classes / Structs**: `sh_basis()`, `sh_evaluate_irradiance()`.
- **Details**: Evaluates 9 L2 real Spherical Harmonics basis functions ($Y_{lm}(\theta, \phi)$) and Cosine-lobe convolution factors for diffuse indirect lighting.

##### [smaa_textures.h](engine/util/smaa_textures.h)
- **Role**: Precomputed SMAA area and search texture loader.
- **Key Classes / Structs**: `SmaaTextures`.
- **Details**: Loads embedded Jimenez SMAA 1x look-up tables into GPU textures for edge blending calculations.

---

## Usage Example

```cpp
#include <gfxcoopa/presentation/window.h>
#include <gfxcoopa/presentation/renderer.h>
#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/core/surface.h>
#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/pipeline/render_pass.h>

int main() {
    // 1. Create window context
    coopa::gfx::presentation::Window window("GfxCoopa Application", 1280, 720);

    // 2. Initialize Vulkan Core components (Volk loader initialized inside Instance)
    coopa::gfx::core::Instance instance("gfxcoopa_demo");
    coopa::gfx::core::Surface surface(instance, window.handle());
    coopa::gfx::core::Device device(instance, surface);
    coopa::gfx::core::Swapchain swapchain(device, surface, 1280, 720);

    // 3. Initialize Memory Allocator and Command Pool
    coopa::gfx::memory::Allocator allocator(instance, device);
    coopa::gfx::command::CommandPool cmd_pool(device, device.graphics_family());

    // 4. Create simple presentation render pass
    coopa::gfx::pipeline::RenderPassBuilder rp_builder(device);
    auto render_pass = rp_builder
        .add_color_attachment(swapchain.format(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        .build();

    // 5. Instantiate Renderer orchestrator
    coopa::gfx::presentation::Renderer renderer(device, swapchain, *render_pass, cmd_pool);

    // 6. Primary frame loop
    while (!window.should_close()) {
        window.poll_events();

        // Handle window resizing
        if (window.was_resized()) {
            auto [width, height] = window.framebuffer_size();
            if (width > 0 && height > 0) {
                swapchain.recreate(width, height);
                renderer.recreate_framebuffers();
            }
        }

        // Render frame with acquire -> record -> submit -> present pipeline
        renderer.draw_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
            // Record draw commands here
        }, VkClearColorValue{{0.1f, 0.1f, 0.12f, 1.0f}});
    }

    // 7. Clean GPU teardown
    device.wait_idle();
    return 0;
}
```
