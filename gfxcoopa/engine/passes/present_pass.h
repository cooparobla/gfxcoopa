#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file present_pass.h
 * @brief Blits final post-processed (Tonemapped + FXAA) offscreen color target to the Vulkan swapchain.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_PRESENT_PASS_H
#define GFXCOOPA_ENGINE_PASSES_PRESENT_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/util/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class PresentPass {
public:
    PresentPass(coopa::gfx::core::Device& device,
                coopa::gfx::pipeline::RenderPass& swapchain_pass,
                const util::Sampler& linear_sampler,
                const std::string& vert_spv,
                const std::string& frag_spv)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .build(device)
        );

        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder()
                .add_sets(*desc_layout_, 1)
                .build(device)
        );

        desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
            device, *desc_pool_, *desc_layout_
        );

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.raster.cull = coopa::gfx::CullMode::None;
        desc.depth.test  = false;
        desc.depth.write = false;
        desc.descriptor_layouts = {desc_layout_.get()};

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, swapchain_pass, desc);

        (void)linear_sampler;
    }

    void set_source_image(coopa::gfx::TextureView image_view, const util::Sampler& linear_sampler) {
        desc_set_->bind_image(0, image_view, linear_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t viewport_w, uint32_t viewport_h) const {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);

        cmd.bind_descriptor_set(*desc_set_, 0);
        cmd.draw(3); // Fullscreen triangle
    }

private:
    coopa::gfx::core::Device& device_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       desc_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_PRESENT_PASS_H
