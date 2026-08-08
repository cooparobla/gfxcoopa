#include <gfxcoopa/engine/targets/gbuffer_target.h>
#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file deferred_lighting_pass.h
 * @brief Deferred lighting pass header for gfxcoopa.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_DEFERRED_LIGHTING_PASS_H
#define GFXCOOPA_ENGINE_PASSES_DEFERRED_LIGHTING_PASS_H

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
#include <gfxcoopa/engine/gi/gi_system.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class DeferredLightingPass {
public:
    DeferredLightingPass(coopa::gfx::core::Device& device,
                         coopa::gfx::pipeline::RenderPass& offscreen_pass,
                         VkDescriptorSetLayout camera_layout,
                         VkDescriptorSetLayout light_layout,
                         VkDescriptorSetLayout shadow_layout,
                         VkDescriptorSetLayout gi_layout,
                         const util::Sampler& linear_sampler,
                         const std::string& vert_spv,
                         const std::string& frag_spv)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layout (Set 4): 3 G-Buffer samplers + 1 SSAO sampler
        std::vector<VkDescriptorSetLayoutBinding> gbuffer_bindings;
        for (uint32_t i = 0; i < 4; ++i) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = i;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            gbuffer_bindings.push_back(b);
        }

        gbuffer_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            device, gbuffer_bindings
        );

        gbuffer_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1,
            std::vector<VkDescriptorPoolSize>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}
            }
        );

        gbuffer_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
            device, *gbuffer_desc_pool_, *gbuffer_desc_layout_
        );

        // Pipeline configuration
        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        std::vector<VkDescriptorSetLayout> layouts = {
            camera_layout,
            light_layout,
            shadow_layout
        };
        if (gi_layout != VK_NULL_HANDLE) {
            layouts.push_back(gi_layout);
        }
        layouts.push_back(gbuffer_desc_layout_->handle());

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, offscreen_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            layouts,
            cfg,
            std::vector<VkPushConstantRange>{}
        );

        (void)linear_sampler;
    }

    void set_gbuffer_images(VkImageView g0_view,
                            VkImageView g1_view,
                            VkImageView g2_view,
                            const util::Sampler& linear_sampler) {
        gbuffer_desc_set_->bind_image(0, g0_view, linear_sampler.handle());
        gbuffer_desc_set_->bind_image(1, g1_view, linear_sampler.handle());
        gbuffer_desc_set_->bind_image(2, g2_view, linear_sampler.handle());
    }

    /// Binding 3 must be rebound every frame -- callers pass the SSAO pass's blurred output when
    /// enabled, or its permanent neutral (fully-unoccluded) texture when disabled/absent, so this
    /// binding is never left pointing at an image still in VK_IMAGE_LAYOUT_UNDEFINED.
    void set_ssao_image(VkImageView ssao_view, VkSampler ssao_sampler) {
        gbuffer_desc_set_->bind_image(3, ssao_view, ssao_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd,
              const coopa::gfx::pipeline::DescriptorSet& camera_set,
              const coopa::gfx::pipeline::DescriptorSet& light_set,
              const coopa::gfx::pipeline::DescriptorSet& shadow_set,
              const gi::GiSystem* gi_system,
              uint32_t viewport_w, uint32_t viewport_h) const
    {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);

        cmd.bind_descriptor_set(pipeline_->layout(), camera_set, 0);
        cmd.bind_descriptor_set(pipeline_->layout(), light_set, 1);
        cmd.bind_descriptor_set(pipeline_->layout(), shadow_set, 2);
        if (gi_system) {
            gi_system->bind(cmd, pipeline_->layout());
        }
        cmd.bind_descriptor_set(pipeline_->layout(), *gbuffer_desc_set_, 4);

        cmd.draw(3); // Fullscreen triangle
    }

    VkPipelineLayout layout() const {
        return pipeline_->layout();
    }

private:
    coopa::gfx::core::Device& device_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> gbuffer_desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      gbuffer_desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       gbuffer_desc_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_DEFERRED_LIGHTING_PASS_H
