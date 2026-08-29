# Engine Submodule (`coopa::gfx::engine`)

The `coopa::gfx::engine` module provides high-level rendering engine abstractions built on top of `gfxcoopa` lower-level Vulkan modules. It is structured into 5 specialized submodules:

- **`coopa::gfx::engine::passes`** ([`passes`](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes)): Render pass execution and graphics pipelines (`DeferredLightingPass`, `GBufferPipeline`, `PbrPipeline`, `SSRPass`, `SMAAPass`, `TAAPass`, `ToneMappingPass`, `PresentPass`, `HiZPass`, `SceneColorMipPass`, `SkyboxPass`, `ShadowPipeline`, `EnvPrefilterPass`, `ProbeCapturePass`, `FogPass`).
- **`coopa::gfx::engine::targets`** ([`targets`](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/targets)): Framebuffer and render target resource managers (`OffscreenTarget`, `GBufferTarget`, `ShadowMapTarget`, `CubemapTarget`).
- **`coopa::gfx::engine::gi`** ([`gi`](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/gi)): Global Illumination system, CPU/GPU probe baking, and SH data structures (`GiSystem`, `GiBaker`, `GiData`, `BRDFLUT`).
- **`coopa::gfx::engine::data`** ([`data`](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/data)): Vertex layout, mesh data, and camera/light uniform structures (`Mesh`, `Vertex`, `CameraUBO`, `LightData`, `ModelPushConstants`).
- **`coopa::gfx::engine::util`** ([`util`](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/util)): Rendering utility wrappers (`Sampler`, `FullscreenQuad`, `SmaaTextures`, `sh_math`).

---

## Render Target & Shading Flow Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                          ENGINE RENDER TARGET & SHADING FLOW                             │
└──────────────────────────────────────────────────────────────────────────────────────────┘

  CameraUBO / LightUBO / ModelPushConstants / GiSystemData
                            │
                            ▼
                ┌───────────────────────┐
                │    Mesh / Geometry    │
                └───────────┬───────────┘
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
┌───────────────────────┐             ┌───────────────────────┐
│   ShadowMapTarget     │             │    OffscreenTarget    │
├───────────────────────┤             ├───────────────────────┤
│ Directional & Point   │             │ Color Attachment      │
│ Depth Shadow Passes   │             │ Depth Attachment      │
└───────────┬───────────┘             └───────────┬───────────┘
            │                                     │ transitions color to
            │ shadow depth map                    │ SHADER_READ_ONLY_OPTIMAL
            └───────────────────┬─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    FullscreenQuad     │  (Sampler::nearest / linear)
                    ├───────────────────────┤
                    │ Post-processing /     │
                    │ Upscale Composite Pass│
                    └───────────────────────┘
```

---

## File Breakdown

### Data Submodule (`engine/data`)

#### [camera_ubo.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/data/camera_ubo.h)
- **Role**: Host-visible uniform buffer storing camera matrices (`view`, `proj`, `view_pos`).
- **Key Classes / Structs**: `CameraData`, `CameraUBO`.
- **Details**: Includes optional sub-pixel position snapping and jitter offsets for low-res retro rendering and temporal anti-aliasing.

#### [light_data.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/data/light_data.h)
- **Role**: Data structures and host-visible UBO management for directional and point light sources.
- **Key Classes / Structs**: `DirectionalLightGPU`, `PointLightGPU`, `LightDataGPU`, `LightUBO`.
- **Details**: Uploads light matrix, color, intensity, attenuation factors, and light counts to GPU shader stages.

#### [mesh.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/data/mesh.h)
- **Role**: GPU-resident mesh data loading and vertex/index buffer management.
- **Key Classes / Structs**: `Vertex`, `Mesh`.
- **Details**: Supports loading Blender-exported YAML scene formats and raw geometry arrays into VMA GPU-only memory buffers.

#### [model_ubo.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/data/model_ubo.h)
- **Role**: Push constant data structure for per-object matrix transformations.
- **Key Classes / Structs**: `ModelPushConstants`.
- **Details**: 128-byte layout containing model matrix (`mat4`) and normal matrix (`mat4 normal_matrix = transpose(inverse(model))`).

#### [fog_data.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/data/fog_data.h)
- **Role**: Host-visible uniform buffer for global Unity-style fog parameters plus up to 8 local fog volumes.
- **Key Classes / Structs**: `FogVolumeGPU`, `FogUBO`, `FogData`.
- **Details**: Self-contained (own `inv_view_proj`/`camera_pos`) so it never touches CameraUBO/LightData's std140 layout; consumed by `FogPass`.

### Global Illumination Submodule (`engine/gi`)

#### [brdf_lut.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/gi/brdf_lut.h)
- **Role**: BRDF Integration Look-Up Table generator.
- **Key Classes / Structs**: `BRDFLUT`.
- **Details**: Computes a 512x512 R16G16_SFLOAT texture encoding Cook-Torrance BRDF scale and bias values for specular IBL split-sum approximation.

#### [gi_baker.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/gi/gi_baker.h)
- **Role**: CPU/GPU probe baking engine for Spherical Harmonics indirect illumination.
- **Key Classes / Structs**: `GiBaker`.
- **Details**: Simulates hemispherical ray bakes per probe position, projecting radiance into 9 2nd-order SH basis coefficients.

#### [gi_data.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/gi/gi_data.h)
- **Role**: Spherical Harmonics probe grid data structures and GPU uniform buffers.
- **Key Classes / Structs**: `SHProbe`, `ReflectionProbeGPU`, `GiUniforms`, `GiSystemData`.
- **Details**: Encapsulates 9 L2 SH coefficients per probe, bounding volumes, and reflection probe cubemap matrices.

#### [gi_system.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/gi/gi_system.h)
- **Role**: Real-time Global Illumination subsystem orchestrator.
- **Key Classes / Structs**: `GiSystem`.
- **Details**: Manages GI probe baking, SSBO updates, BRDF LUT integration, and descriptor set 3 layout bindings.

### Render Targets Submodule (`engine/targets`)

#### [cubemap_target.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/targets/cubemap_target.h)
- **Role**: Offscreen multi-face HDR color cubemap render target utility.
- **Key Classes / Structs**: `CubemapTarget`.
- **Details**: Manages 6-face color attachments for skybox capturing and localized environment reflection probes.

#### [gbuffer_target.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/targets/gbuffer_target.h)
- **Role**: Offscreen multi-attachment G-Buffer render target manager.
- **Key Classes / Structs**: `GBufferTarget`.
- **Details**: Manages deferred geometry pass attachments: Albedo+AO, Normal+Metallic, Position+Roughness, and Depth.

#### [offscreen_target.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/targets/offscreen_target.h)
- **Role**: Custom-resolution offscreen HDR render target.
- **Key Classes / Structs**: `OffscreenTarget`.
- **Details**: Owns color (`R16G16B16A16_SFLOAT`) and depth attachments, automatically transitioning color images to `SHADER_READ_ONLY_OPTIMAL` for post-processing.

#### [shadow_map_target.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/targets/shadow_map_target.h)
- **Role**: Depth attachment targets for directional and point light shadows.
- **Key Classes / Structs**: `ShadowMapTarget`.
- **Details**: Allocates 2048x2048 2D depth maps for directional lights and 512x512 cubemap depth textures for omnidirectional point lights.

### Render Passes & Pipelines Submodule (`engine/passes`)

#### [deferred_lighting_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/deferred_lighting_pass.h)
- **Role**: Deferred lighting evaluation pass.
- **Key Classes / Structs**: `DeferredLightingPass`.
- **Details**: Combines G-Buffer attachments, shadow maps, and SH GI inputs to compute Cook-Torrance direct and indirect shading.

#### [env_prefilter_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/env_prefilter_pass.h)
- **Role**: GGX-prefiltered environment map generator.
- **Key Classes / Structs**: `EnvPrefilterPass`.
- **Details**: Generates roughness mip chains for environment cubemap specular reflection probes.

#### [gbuffer_pipeline.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/gbuffer_pipeline.h)
- **Role**: Graphics pipeline for geometry rasterization into G-Buffer attachments.
- **Key Classes / Structs**: `GBufferPipeline`.
- **Details**: Configures MRT color blending, depth testing, and backface culling for geometry passes.

#### [hiz_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/hiz_pass.h)
- **Role**: Hierarchical Z-Buffer depth pyramid generator.
- **Key Classes / Structs**: `HiZPass`.
- **Details**: Downsamples scene depth into a mip pyramid for accelerated SSR raymarching.

#### [pbr_pipeline.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/pbr_pipeline.h)
- **Role**: Physically-based rendering pipeline wrapper.
- **Key Classes / Structs**: `PbrPipeline`.
- **Details**: Manages graphics pipeline layouts and descriptor set bindings for forward and deferred shading.

#### [present_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/present_pass.h)
- **Role**: Final blit pass to swapchain framebuffers.
- **Key Classes / Structs**: `PresentPass`.
- **Details**: Renders a full-screen quad transferring final color target output to the Vulkan swapchain.

#### [probe_capture_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/probe_capture_pass.h)
- **Role**: Scene capture pass into reflection probe cubemaps.
- **Key Classes / Structs**: `ProbeCapturePass`.
- **Details**: Renders scene geometry into 6 cubemap directions for localized reflection baking.

#### [scene_color_mip_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/scene_color_mip_pass.h)
- **Role**: Downsampled color mip chain generator.
- **Key Classes / Structs**: `SceneColorMipPass`.
- **Details**: Builds a downsampled color pyramid enabling glossy reflection cone sampling in SSR.

#### [shadow_pipeline.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/shadow_pipeline.h)
- **Role**: Depth-only shadow map pipeline.
- **Key Classes / Structs**: `ShadowPipeline`.
- **Details**: Configures depth bias, slope-scaled depth bias, and vertex shaders for 2D and cubemap shadow passes.

#### [skybox_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/skybox_pass.h)
- **Role**: Analytic skybox and atmosphere rendering pass.
- **Key Classes / Structs**: `SkyboxPass`.
- **Details**: Renders gradient sky backgrounds behind scene geometry.

#### [smaa_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/smaa_pass.h)
- **Role**: Subpixel Morphological Anti-Aliasing (SMAA 1x Ultra) pass.
- **Key Classes / Structs**: `SMAAPass`.
- **Details**: Executes 3-pass SMAA post-processing (edge detection, blend weights, neighborhood blending).

#### [ssr_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/ssr_pass.h)
- **Role**: Screen-Space Reflections (SSR) pass.
- **Key Classes / Structs**: `SSRPass`.
- **Details**: Implements Hi-Z raymarching and glossy reflection compositing.

#### [taa_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/taa_pass.h)
- **Role**: Temporal Anti-Aliasing (TAA) pass.
- **Key Classes / Structs**: `TAAPass`.
- **Details**: Performs sub-pixel jitter accumulation, motion vector reprojection, and color clamping.

#### [tonemapping_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/tonemapping_pass.h)
- **Role**: Fullscreen tonemapping and FXAA post-processing pass.
- **Key Classes / Structs**: `TonemappingPass`.
- **Details**: Evaluates ACES filmic s-curve, exposure adjustments, gamma 2.2 correction, and FXAA 3.11 edge smoothing.

#### [fog_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/passes/fog_pass.h)
- **Role**: Fullscreen Unity-style fog composite (Linear/Exponential/Exp2 global fog + local box/sphere fog volumes).
- **Key Classes / Structs**: `FogPass`.
- **Details**: Reads scene colour plus the G-buffer's normal/world-position, writes a fogged copy into its own HDR target (see `assets/shaders/fog.frag`, `assets/shaders/gfx/fog.glsl`).

### Engine Utilities Submodule (`engine/util`)

#### [fullscreen_quad.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/util/fullscreen_quad.h)
- **Role**: Stateless 3-vertex full-screen triangle renderer helper.
- **Key Classes / Structs**: `FullscreenQuad`.
- **Details**: Draws screen-filling triangles using shader-generated vertex coordinates without vertex buffer bindings.

#### [sampler.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/util/sampler.h)
- **Role**: RAII `VkSampler` wrapper and factory methods.
- **Key Classes / Structs**: `Sampler`.
- **Details**: Provides static factories for `nearest()`, `linear()`, `shadow()`, and `cubemap()` samplers.

#### [sh_math.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/util/sh_math.h)
- **Role**: 2nd-order Spherical Harmonics math functions.
- **Key Classes / Structs**: `sh_basis()`, `sh_evaluate_irradiance()`.
- **Details**: Computes 9 L2 SH basis functions and cosine-lobe irradiance convolution.

#### [smaa_textures.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/engine/util/smaa_textures.h)
- **Role**: Embedded SMAA area/search texture loader.
- **Key Classes / Structs**: `SmaaTextures`.
- **Details**: Creates GPU textures initialized with precomputed SMAA 1x blend weights.

---

## Usage Example

```cpp
#include <gfxcoopa/engine/offscreen_target.h>
#include <gfxcoopa/engine/shadow_map_target.h>
#include <gfxcoopa/engine/sampler.h>
#include <gfxcoopa/engine/fullscreen_quad.h>

// Create an offscreen render target for retro low-resolution rendering
coopa::gfx::engine::OffscreenTarget offscreen(device, allocator, 320, 240);

// Create shadow map targets for light depth passes
coopa::gfx::engine::ShadowMapTarget shadow_target(device, allocator, 2048, 512);

// Create a nearest-neighbor sampler for pixel-art upscaling
coopa::gfx::engine::Sampler nearest_sampler = coopa::gfx::engine::Sampler::nearest(device);

// Record offscreen render pass
offscreen.begin(cmd);
// ... record geometry draw calls ...
offscreen.end(cmd); // Color attachment automatically transitions to SHADER_READ_ONLY_OPTIMAL
```
