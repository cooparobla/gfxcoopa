# `gfxcoopa::pipeline` Submodule

The `gfxcoopa::pipeline` submodule provides Vulkan pipeline infrastructure, including shader bytecode loading, render pass setup, descriptor pools/layouts/sets management, and graphics pipeline state configuration.

---

## Pipeline & Descriptor Setup Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE & DESCRIPTOR SETUP MODEL                              │
└──────────────────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────┐                     ┌───────────────────────┐
    │        Shader         │                     │      RenderPass       │
    ├───────────────────────┤                     ├───────────────────────┤
    │ SPIR-V .spv Bytecode  │                     │ Color & Depth Attach  │
    └───────────┬───────────┘                     └───────────┬───────────┘
                │                                             │
                └───────────────────┬─────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
    │ DescriptorSetLayout   │─►│       Pipeline        │◄─│ Push Constant Ranges  │
    └───────────────────────┘  └───────────────────────┘  └───────────────────────┘
                                           ▲
                                           │ binds layout & resources
    ┌───────────────────────┐              │
    │    DescriptorPool     │              │
    └───────────┬───────────┘              │
                │ allocates                │
                ▼                          │
    ┌───────────────────────┐              │
    │     DescriptorSet     │──────────────┘
    ├───────────────────────┤
    │ - write_buffer()      │
    │ - write_image()       │
    └───────────────────────┘
```

---

## Header Files

| File | Primary Class / Struct | Description |
|---|---|---|
| [`shader.h`](shader.h) | [`Shader`](shader.h) | RAII wrapper around `VkShaderModule`. Loads compiled SPIR-V `.spv` files from disk and returns `VkPipelineShaderStageCreateInfo`. |
| [`render_pass.h`](render_pass.h) | [`RenderPassConfig`](render_pass.h), [`RenderPass`](render_pass.h) | RAII wrapper around `VkRenderPass`. Configures color and optional depth attachments, subpass definitions, clear values, load/store ops, and layout transitions. |
| [`descriptor.h`](descriptor.h) | [`DescriptorPool`](descriptor.h), [`DescriptorSetLayout`](descriptor.h), [`DescriptorSet`](descriptor.h) | RAII wrappers for Vulkan descriptor resource management: `DescriptorPool` allocates sets, `DescriptorSetLayout` specifies binding contracts, and `DescriptorSet` binds buffers (`write_buffer()`) and samplers/images (`write_image()`). |
| [`pipeline.h`](pipeline.h) | [`PipelineConfig`](pipeline.h), [`Pipeline`](pipeline.h) | RAII wrapper for `VkPipeline` and `VkPipelineLayout`. Configures input assembly, rasterization, depth testing, blending, dynamic states (viewport/scissor), push constant ranges, and layout binding. |

---

## Usage Example

```cpp
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/pipeline.h>

// Create RenderPass
coopa::gfx::pipeline::RenderPass render_pass(device, swapchain.image_format(), VK_FORMAT_D32_SFLOAT);

// Load Shaders
coopa::gfx::pipeline::Shader vert(device, "vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
coopa::gfx::pipeline::Shader frag(device, "frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

// Create Descriptor Layout
coopa::gfx::pipeline::DescriptorSetLayout layout(device, {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT}
});

// Build Graphics Pipeline
coopa::gfx::pipeline::Pipeline pipeline(
    device, render_pass,
    {&vert, &frag},
    {layout.handle()},
    {/* push constant ranges */}
);
```
