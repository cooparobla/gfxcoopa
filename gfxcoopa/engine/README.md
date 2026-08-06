# `gfxcoopa::engine` Submodule

The `gfxcoopa::engine` submodule provides high-level rendering engine abstractions built on top of `gfxcoopa` lower-level Vulkan modules. It includes mesh handling, UBO management, render targets, shadow map pipelines, samplers, cubemaps, BRDF lookup tables, and Global Illumination (GI) probe data structures.

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

## Header Files

| File | Primary Class / Struct | Description |
|---|---|---|
| [`brdf_lut.h`](brdf_lut.h) | [`BRDFLUT`](brdf_lut.h) | Generates and manages a 512x512 R16G16_SFLOAT BRDF Integration Look-Up Table for PBR specular IBL split-sum approximation. |
| [`camera_ubo.h`](camera_ubo.h) | [`CameraData`](camera_ubo.h), [`CameraUBO`](camera_ubo.h) | Host-visible uniform buffer storing camera matrices (`view`, `proj`, `view_pos`). Includes optional sub-pixel position snapping for low-res retro rendering. |
| [`cubemap_target.h`](cubemap_target.h) | [`CubemapTarget`](cubemap_target.h) | Offscreen multi-face HDR color cubemap render target utility for skyboxes, environment maps, and reflection probes. |
| [`fullscreen_quad.h`](fullscreen_quad.h) | [`FullscreenQuad`](fullscreen_quad.h) | Stateless helper for recording 3-vertex screen-space triangle draw calls without vertex buffer bindings (using `gl_VertexIndex`). |
| [`gi_data.h`](gi_data.h) | [`SHProbe`](gi_data.h), [`GiUniforms`](gi_data.h), [`GiSystemData`](gi_data.h) | Spherical Harmonics (SH) probe grid data structures and uniform buffers for real-time global illumination probe volumes and reflection probes. |
| [`light_data.h`](light_data.h) | [`DirectionalLightGPU`](light_data.h), [`PointLightGPU`](light_data.h), [`LightUBO`](light_data.h) | Data structures and host-visible UBO management for directional lights and multi-point lights. |
| [`mesh.h`](mesh.h) | [`Vertex`](mesh.h), [`Mesh`](mesh.h) | GPU-resident mesh data loading (supporting Blender-exported YAML scene formats and raw geometry), managing vertex/index buffers and draw commands. |
| [`model_ubo.h`](model_ubo.h) | [`ModelPushConstants`](model_ubo.h) | 128-byte push constant struct containing `model` object-to-world transform matrix and `normal_matrix` (`transpose(inverse(model))`). |
| [`offscreen_target.h`](offscreen_target.h) | [`OffscreenTarget`](offscreen_target.h) | Low-resolution / custom-resolution offscreen render target owning color and depth attachments. Automatically transitions color to `SHADER_READ_ONLY_OPTIMAL` for post-processing and upscaling. |
| [`sampler.h`](sampler.h) | [`Sampler`](sampler.h) | RAII `VkSampler` wrapper with factory methods for nearest-neighbor (`nearest()`), linear (`linear()`), shadow map comparison, and cubemap samplers. |
| [`shadow_map_target.h`](shadow_map_target.h) | [`ShadowMapTarget`](shadow_map_target.h) | Depth render targets for Directional Light (2D depth map) and Point Light Cubemap depth maps. |
| [`shadow_pipeline.h`](shadow_pipeline.h) | [`ShadowPipeline`](shadow_pipeline.h) | Specialized graphics pipelines and push constant layouts for directional and point-light cubemap shadow depth rendering passes. |

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
