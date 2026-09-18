# Engine Submodule (`coopa::gfx::engine`)

The `coopa::gfx::engine` module provides high-level rendering engine abstractions built on top of `gfxcoopa` lower-level Vulkan modules. It is structured into 5 specialized submodules:

- **`coopa::gfx::engine::passes`** ([`passes`](passes)): Render pass execution and graphics pipelines (`FullscreenStage` — the shared scaffold most post-processing passes are built from — plus `DeferredLightingPass`, `GBufferPipeline`, `SsrPass`, `SmaaPass`, `TaaPass`, `ToneMappingPass`, `PresentPass`, `HiZPass`, `SceneColorMipPass`, `SkyboxPass`, `ShadowPipeline`, `EnvPrefilterPass`, `ProbeCapturePass`, `FogPass`, `BloomPass`, `DofPass`, `TiltShiftPass`, `VolumetricsPass`, the `Sdf*` passes).
- **`coopa::gfx::engine::targets`** ([`targets`](targets)): Framebuffer and render target resource managers (`OffscreenTarget`, `GBufferTarget`, `ShadowMapTarget`, `CubemapTarget`).
- **`coopa::gfx::engine::gi`** ([`gi`](gi)): Global Illumination system, CPU/GPU probe baking, and SH data structures (`GiSystem`, `GiBaker`, `GiData`, `BRDFLUT`).
- **`coopa::gfx::engine::data`** ([`data`](data)): Vertex layout, mesh data, and camera/light uniform structures (`Mesh`, `Vertex`, `CameraUBO`, `LightData`, `ModelPushConstants`).
- **`coopa::gfx::engine::util`** ([`util`](util)): Rendering utility wrappers (`Sampler`, `MaterialTextureCache`, `InstanceBatcher`, `SmaaTextures`, `SsaoKernel`, `sh_math`).

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
                    │   FullscreenStage     │  (Sampler::nearest / linear)
                    ├───────────────────────┤
                    │ Post-processing /     │
                    │ Upscale Composite Pass│
                    └───────────────────────┘
```

---

## File Breakdown

### Data Submodule (`engine/data`)

#### [camera_ubo.h](data/camera_ubo.h)
- **Role**: Host-visible uniform buffer storing camera matrices (`view`, `proj`, `view_pos`).
- **Key Classes / Structs**: `CameraData`, `CameraUBO`.
- **Details**: Includes optional sub-pixel position snapping and jitter offsets for low-res retro rendering and temporal anti-aliasing.

#### [light_data.h](data/light_data.h)
- **Role**: Data structures and host-visible UBO management for directional and point light sources.
- **Key Classes / Structs**: `DirectionalLightGPU`, `PointLightGPU`, `SpotLightGPU`, `LightDataGPU`, `LightUBO`.
- **Details**: Uploads light matrix, color, intensity, attenuation factors, and light counts to GPU shader stages.

#### [mesh.h](data/mesh.h)
- **Role**: GPU-resident mesh data loading and vertex/index buffer management.
- **Key Classes / Structs**: `Vertex`, `Mesh`.
- **Details**: Supports loading Blender-exported YAML scene formats and raw geometry arrays into VMA GPU-only memory buffers.

#### [model_ubo.h](data/model_ubo.h)
- **Role**: Push constant data structure for per-object matrix transformations.
- **Key Classes / Structs**: `ModelPushConstants`.
- **Details**: 128-byte layout containing model matrix (`mat4`) and normal matrix (`mat4 normal_matrix = transpose(inverse(model))`).

#### [fog_data.h](data/fog_data.h)
- **Role**: Host-visible uniform buffer for GLOBAL Unity-style fog parameters. Fog is global-only; bounded volumes live in `volumetrics_data.h`.
- **Key Classes / Structs**: `FogUBO`, `FogData`.
- **Details**: Self-contained (own `inv_view_proj`/`camera_pos`) so it never touches CameraUBO/LightData's std140 layout; consumed by `FogPass`.

### Global Illumination Submodule (`engine/gi`)

#### [brdf_lut.h](gi/brdf_lut.h)
- **Role**: BRDF Integration Look-Up Table generator.
- **Key Classes / Structs**: `BRDFLUT`.
- **Details**: Computes a 512x512 R16G16_SFLOAT texture encoding Cook-Torrance BRDF scale and bias values for specular IBL split-sum approximation.

#### [gi_baker.h](gi/gi_baker.h)
- **Role**: CPU/GPU probe baking engine for Spherical Harmonics indirect illumination.
- **Key Classes / Structs**: `GiBaker`.
- **Details**: Simulates hemispherical ray bakes per probe position, projecting radiance into 9 2nd-order SH basis coefficients.

#### [gi_data.h](gi/gi_data.h)
- **Role**: Spherical Harmonics probe grid data structures and GPU uniform buffers.
- **Key Classes / Structs**: `SHProbe`, `ReflectionProbeGPU`, `GiUniforms`, `GiSystemData`.
- **Details**: Encapsulates 9 L2 SH coefficients per probe, bounding volumes, and reflection probe cubemap matrices.

#### [gi_system.h](gi/gi_system.h)
- **Role**: Real-time Global Illumination subsystem orchestrator.
- **Key Classes / Structs**: `GiSystem`.
- **Details**: Manages GI probe baking, SSBO updates, BRDF LUT integration, and descriptor set 3 layout bindings.

### Render Targets Submodule (`engine/targets`)

#### [cubemap_target.h](targets/cubemap_target.h)
- **Role**: Offscreen multi-face HDR color cubemap render target utility.
- **Key Classes / Structs**: `CubemapTarget`.
- **Details**: Manages 6-face color attachments for skybox capturing and localized environment reflection probes.

#### [gbuffer_target.h](targets/gbuffer_target.h)
- **Role**: Offscreen multi-attachment G-Buffer render target manager.
- **Key Classes / Structs**: `GBufferTarget`.
- **Details**: Manages deferred geometry pass attachments: Albedo+AO, Normal+Metallic, Position+Roughness, Emissive, and Depth.

#### [offscreen_target.h](targets/offscreen_target.h)
- **Role**: Custom-resolution offscreen HDR render target.
- **Key Classes / Structs**: `OffscreenTarget`.
- **Details**: Owns color (`R16G16B16A16_SFLOAT`) and depth attachments, automatically transitioning color images to `SHADER_READ_ONLY_OPTIMAL` for post-processing.

#### [shadow_map_target.h](targets/shadow_map_target.h)
- **Role**: Depth attachment targets for directional and point light shadows.
- **Key Classes / Structs**: `ShadowMapTarget`.
- **Details**: Allocates 2048x2048 2D depth maps for directional lights and 512x512 cubemap depth textures for omnidirectional point lights.

### Render Passes & Pipelines Submodule (`engine/passes`)

#### [deferred_lighting_pass.h](passes/deferred_lighting_pass.h)
- **Role**: Deferred lighting evaluation pass.
- **Key Classes / Structs**: `DeferredLightingPass`.
- **Details**: Combines G-Buffer attachments, shadow maps, and SH GI inputs to compute Cook-Torrance direct and indirect shading.

#### [env_prefilter_pass.h](passes/env_prefilter_pass.h)
- **Role**: GGX-prefiltered environment map generator.
- **Key Classes / Structs**: `EnvPrefilterPass`.
- **Details**: Generates roughness mip chains for environment cubemap specular reflection probes.

#### [gbuffer_pipeline.h](passes/gbuffer_pipeline.h)
- **Role**: Graphics pipeline for geometry rasterization into G-Buffer attachments.
- **Key Classes / Structs**: `GBufferPipeline`.
- **Details**: Configures MRT color blending, depth testing, and backface culling for geometry passes.

#### [hiz_pass.h](passes/hiz_pass.h)
- **Role**: Hierarchical Z-Buffer depth pyramid generator.
- **Key Classes / Structs**: `HiZPass`.
- **Details**: Downsamples scene depth into a mip pyramid for accelerated SSR raymarching.

#### [present_pass.h](passes/present_pass.h)
- **Role**: Final blit pass to swapchain framebuffers.
- **Key Classes / Structs**: `PresentPass`.
- **Details**: Renders a full-screen quad transferring final color target output to the Vulkan swapchain.

#### [probe_capture_pass.h](passes/probe_capture_pass.h)
- **Role**: Scene capture pass into reflection probe cubemaps.
- **Key Classes / Structs**: `ProbeCapturePass`.
- **Details**: Renders scene geometry into 6 cubemap directions for localized reflection baking.

#### [scene_color_mip_pass.h](passes/scene_color_mip_pass.h)
- **Role**: Downsampled color mip chain generator.
- **Key Classes / Structs**: `SceneColorMipPass`.
- **Details**: Builds a downsampled color pyramid enabling glossy reflection cone sampling in SSR.

#### [shadow_pipeline.h](passes/shadow_pipeline.h)
- **Role**: Depth-only shadow map pipeline.
- **Key Classes / Structs**: `ShadowPipeline`.
- **Details**: Configures depth bias, slope-scaled depth bias, and vertex shaders for 2D and cubemap shadow passes.

#### [skybox_pass.h](passes/skybox_pass.h)
- **Role**: Analytic skybox and atmosphere rendering pass.
- **Key Classes / Structs**: `SkyboxPass`.
- **Details**: Renders gradient sky backgrounds behind scene geometry.

#### [smaa_pass.h](passes/smaa_pass.h)
- **Role**: Subpixel Morphological Anti-Aliasing (SMAA 1x Ultra) pass.
- **Key Classes / Structs**: `SMAAPass`.
- **Details**: Executes 3-pass SMAA post-processing (edge detection, blend weights, neighborhood blending).

#### [ssr_pass.h](passes/ssr_pass.h)
- **Role**: Screen-Space Reflections (SSR) pass.
- **Key Classes / Structs**: `SSRPass`.
- **Details**: Implements Hi-Z raymarching and glossy reflection compositing.

#### [taa_pass.h](passes/taa_pass.h)
- **Role**: Temporal Anti-Aliasing (TAA) pass.
- **Key Classes / Structs**: `TAAPass`.
- **Details**: Performs sub-pixel jitter accumulation, motion vector reprojection, and color clamping.

#### [tonemapping_pass.h](passes/tonemapping_pass.h)
- **Role**: Fullscreen tonemapping and FXAA post-processing pass.
- **Key Classes / Structs**: `TonemappingPass`.
- **Details**: Evaluates ACES filmic s-curve, exposure adjustments, gamma 2.2 correction, and FXAA 3.11 edge smoothing.

#### [fog_pass.h](passes/fog_pass.h)
- **Role**: Fullscreen Unity-style fog composite (Linear/Exponential/Exp2 global fog + local box/sphere fog volumes).
- **Key Classes / Structs**: `FogPass`.
- **Details**: Reads scene colour plus the G-buffer's normal/world-position, writes a fogged copy into its own HDR target (see `assets/shaders/fog.frag`, `assets/shaders/gfx/fog.glsl`).

#### [fullscreen_stage.h](passes/fullscreen_stage.h)
- **Role**: The shared scaffold for a fullscreen-triangle render stage — its two shaders, its own descriptor set(s), and its pipeline.
- **Key Classes / Structs**: `FullscreenStageDesc`, `FullscreenStage`.
- **Details**: A pass *owns* one or more stages rather than deriving from anything, so its public constructor and `draw()` stay unchanged. `owned_sets` declares the sets the stage allocates, `leading_layouts`/`extra_layouts` the ones owned elsewhere, and `instances` the number of independent copies (one per pyramid level or blur axis). `rebuild_sets()` reallocates the sets without rebuilding the pipeline. Render targets, barriers and stage ordering deliberately stay in the pass.

#### [extra_sets.h](passes/extra_sets.h)
- **Role**: Optional app-supplied descriptor sets appended after a pass's own.
- **Key Classes / Structs**: `ExtraSets`.
- **Details**: Carries the layouts and the binder together so the two cannot drift apart; `validate()` turns a mismatch into a startup exception. Empty extras mean the set is not declared in the pipeline layout at all.

#### Post-processing passes
- [bloom_pass.h](passes/bloom_pass.h) (`BloomPass`) — bright-pass threshold, multi-tap downsample, tent-filter upsample+combine.
- [dof_pass.h](passes/dof_pass.h) (`DofPass`) — circle-of-confusion depth of field.
- [tilt_shift_pass.h](passes/tilt_shift_pass.h) (`TiltShiftPass`) — separable horizontal/vertical band blur, folded with the upscale.
- [fxaa_pass.h](passes/fxaa_pass.h) (`FxaaPass`) — standalone LDR FXAA 3.11, for a source already tonemapped.
- [ssao_pass.h](passes/ssao_pass.h) (`SsaoPass`) — hemisphere-kernel SSAO with a bilateral blur and temporal resolve.
- [volumetrics_pass.h](passes/volumetrics_pass.h) (`VolumetricsPass`) — wind-driven volumetric scattering.
- [pixel_stylize_pass.h](passes/pixel_stylize_pass.h) (`PixelStylizePass`) — outline, palette quantization, dither and bloom composite.

#### Geometry and transparency passes
- [transparent_pass.h](passes/transparent_pass.h) (`TransparentPass`) — forward BLEND pass, depth-tested against the G-buffer.
- [transparent_capture_pass.h](passes/transparent_capture_pass.h) (`TransparentCapturePass`) — unblended three-target capture of transparent surfaces.
- [textured_quad_2d_pass.h](passes/textured_quad_2d_pass.h) (`TexturedQuad2dPass`) — shared 2D textured-quad pass with a per-texture descriptor cache and streaming geometry.
- [sdf_gbuffer_pass.h](passes/sdf_gbuffer_pass.h), [sdf_forward_pass.h](passes/sdf_forward_pass.h), [sdf_capture_pass.h](passes/sdf_capture_pass.h), [sdf_shadow_pass.h](passes/sdf_shadow_pass.h) — signed-distance-field shapes rendered into the G-buffer, the forward pass, probe capture, and both shadow paths.


### Engine Utilities Submodule (`engine/util`)

#### [sampler.h](util/sampler.h)
- **Role**: RAII `VkSampler` wrapper and factory methods.
- **Key Classes / Structs**: `Sampler`.
- **Details**: Provides static factories for `nearest()`, `linear()`, `shadow()`, and `cubemap()` samplers.

#### [sh_math.h](util/sh_math.h)
- **Role**: 2nd-order Spherical Harmonics math functions.
- **Key Classes / Structs**: `sh_basis()`, `sh_evaluate_irradiance()`.
- **Details**: Computes 9 L2 SH basis functions and cosine-lobe irradiance convolution.

#### [smaa_textures.h](util/smaa_textures.h)
- **Role**: Embedded SMAA area/search texture loader.
- **Key Classes / Structs**: `SmaaTextures`.
- **Details**: Creates GPU textures initialized with precomputed SMAA 1x blend weights.

---

## Usage Example

```cpp
#include <gfxcoopa/engine/targets/offscreen_target.h>
#include <gfxcoopa/engine/targets/shadow_map_target.h>
#include <gfxcoopa/engine/util/sampler.h>

// Create an offscreen render target for retro low-resolution rendering
coopa::gfx::engine::targets::OffscreenTarget offscreen(device, allocator, 320, 240);

// Create shadow map targets for light depth passes
coopa::gfx::engine::targets::ShadowMapTarget shadow_target(device, allocator, 2048, 512);

// Create a nearest-neighbor sampler for pixel-art upscaling
coopa::gfx::engine::util::Sampler nearest_sampler = coopa::gfx::engine::util::Sampler::nearest(device);

// Record offscreen render pass
offscreen.begin(cmd);
// ... record geometry draw calls ...
offscreen.end(cmd); // Color attachment automatically transitions to SHADER_READ_ONLY_OPTIMAL
```
