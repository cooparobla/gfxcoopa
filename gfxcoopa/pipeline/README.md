# Pipeline Submodule (`coopa::gfx::pipeline`)

The `coopa::gfx::pipeline` submodule provides RAII wrappers and builders for Vulkan shaders, render passes, descriptor pools/layouts/sets, and graphics pipeline state configuration.

---

## Pipeline Architecture Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE CREATION & LAYOUT FLOW                               │
└──────────────────────────────────────────────────────────────────────────────────────────┘

       ┌───────────────────────┐                        ┌───────────────────────┐
       │     ShaderModule      │                        │  DescriptorSetLayout  │
       └───────────┬───────────┘                        └───────────┬───────────┘
                   │                                                │
                   ▼                                                ▼
       ┌───────────────────────┐                        ┌───────────────────────┐
       │    PipelineConfig     ├───────────────────────►│    PipelineLayout     │
       └───────────┬───────────┘                        └───────────┬───────────┘
                   │                                                │
                   ▼                                                ▼
       ┌───────────────────────┐                        ┌───────────────────────┐
       │      RenderPass       ├───────────────────────►│       Pipeline        │
       └───────────────────────┘                        └───────────────────────┘
```

---

## File Breakdown

### [descriptor.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/pipeline/descriptor.h)
- **Role**: RAII abstractions for Vulkan descriptor sets, layouts, and pools.
- **Key Classes / Structs**: `DescriptorSetLayout`, `DescriptorPool`, `DescriptorWriter`.
- **Details**: `DescriptorSetLayout` configures layout bindings; `DescriptorPool` allocates sets; `DescriptorWriter` provides a builder pattern for updating buffer (`bind_buffer`) and image (`bind_image`) descriptors via `vkUpdateDescriptorSets`.

### [pipeline.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/pipeline/pipeline.h)
- **Role**: RAII encapsulation of graphics and compute pipelines (`VkPipeline`, `VkPipelineLayout`).
- **Key Classes / Structs**: `Pipeline`, `PipelineConfig`.
- **Details**: Provides a builder pattern (`PipelineConfig`) for vertex input bindings, input assembly, rasterization state, multisampling, depth-stencil testing, color blending, dynamic states, and push constant ranges.

### [render_pass.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/pipeline/render_pass.h)
- **Role**: RAII wrapper for `VkRenderPass`.
- **Key Classes / Structs**: `RenderPass`, `RenderPassBuilder`.
- **Details**: Simplifies creation of multi-attachment color/depth render passes, subpass descriptions, and subpass dependencies.

### [shader.h](file:///home/coopa/git/gfxcoopa/gfxcoopa/pipeline/shader.h)
- **Role**: Encapsulates SPIR-V shader module loading (`VkShaderModule`).
- **Key Classes / Structs**: `ShaderModule`.
- **Details**: Loads binary SPIR-V bytecode from file paths and creates shader stage info structures (`VkPipelineShaderStageCreateInfo`).

---

## Usage Example

```cpp
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/pipeline.h>

// 1. Create render pass
coopa::gfx::pipeline::RenderPassBuilder rp_builder(device);
auto render_pass = rp_builder
    .add_color_attachment(swapchain.format(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    .build();

// 2. Load SPIR-V shaders
coopa::gfx::pipeline::ShaderModule vert_shader(device, "shaders/vert.spv");
coopa::gfx::pipeline::ShaderModule frag_shader(device, "shaders/frag.spv");

// 3. Build descriptor set layout
coopa::gfx::pipeline::DescriptorSetLayout layout = coopa::gfx::pipeline::DescriptorSetLayout::Builder(device)
    .add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
    .build();

// 4. Configure and create graphics pipeline
coopa::gfx::pipeline::PipelineConfig config{};
config.shader_stages = {
    vert_shader.stage_info(VK_SHADER_STAGE_VERTEX_BIT),
    frag_shader.stage_info(VK_SHADER_STAGE_FRAGMENT_BIT)
};
config.descriptor_set_layouts = { layout.handle() };
config.render_pass = render_pass->handle();

coopa::gfx::pipeline::Pipeline pipeline(device, config);
```
