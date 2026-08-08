#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file skybox_pass.h
 * @brief Analytic gradient skybox pass, fills background pixels left blank by deferred lighting.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SKYBOX_PASS_H
#define GFXCOOPA_ENGINE_PASSES_SKYBOX_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

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



class SkyboxPass {
public:
    struct SkyboxPushConstants {
        glm::mat4 inv_view_proj;
    };

    SkyboxPass(coopa::gfx::core::Device& device,
               coopa::gfx::pipeline::RenderPass& offscreen_pass,
               VkDescriptorSetLayout camera_layout,
               const std::string& vert_spv,
               const std::string& frag_spv)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layout (Set 1): G-Buffer normal/metallic sampler
        std::vector<VkDescriptorSetLayoutBinding> normal_bindings(1);
        normal_bindings[0].binding            = 0;
        normal_bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normal_bindings[0].descriptorCount    = 1;
        normal_bindings[0].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
        normal_bindings[0].pImmutableSamplers = nullptr;

        normal_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            device, normal_bindings
        );

        normal_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1,
            std::vector<VkDescriptorPoolSize>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
            }
        );

        normal_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
            device, *normal_desc_pool_, *normal_desc_layout_
        );

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(SkyboxPushConstants);

        std::vector<VkDescriptorSetLayout> layouts = {
            camera_layout,
            normal_desc_layout_->handle()
        };

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, offscreen_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            layouts,
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
    }

    void set_gbuffer_normal_image(VkImageView g1_view, const util::Sampler& linear_sampler) {
        normal_desc_set_->bind_image(0, g1_view, linear_sampler.handle());
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd,
              const coopa::gfx::pipeline::DescriptorSet& camera_set,
              const glm::mat4& view,
              const glm::mat4& proj,
              uint32_t viewport_w, uint32_t viewport_h) const
    {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);

        SkyboxPushConstants pc{};
        pc.inv_view_proj = glm::inverse(proj * view);
        cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyboxPushConstants), &pc);

        cmd.bind_descriptor_set(pipeline_->layout(), camera_set, 0);
        cmd.bind_descriptor_set(pipeline_->layout(), *normal_desc_set_, 1);

        cmd.draw(3); // Fullscreen triangle
    }

private:
    coopa::gfx::core::Device& device_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> normal_desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      normal_desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       normal_desc_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>            pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SKYBOX_PASS_H
