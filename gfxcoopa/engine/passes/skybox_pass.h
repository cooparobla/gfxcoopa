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
#include <gfxcoopa/engine/render_features.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class SkyboxPass {
public:
    struct SkyboxPushConstants {
        glm::mat4 inv_view_proj;
        // Trailing sky colours, sourced from the same IndirectParams the lighting
        // pass and SSR composite read -- see IndirectParams' doc (render_features.h)
        // for why these three must never be plumbed from a different instance.
        glm::vec4 sky_zenith  = glm::vec4(0.05f, 0.18f, 0.55f, 0.0f);
        glm::vec4 sky_horizon = glm::vec4(0.25f, 0.35f, 0.45f, 0.0f);
        glm::vec4 sky_ground  = glm::vec4(0.05f, 0.045f, 0.04f, 0.0f);
    }; // 112 bytes

    SkyboxPass(coopa::gfx::core::Device& device,
               coopa::gfx::pipeline::RenderPass& offscreen_pass,
               const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
               const std::string& vert_spv,
               const std::string& frag_spv)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layout (Set 1): G-Buffer normal/metallic sampler
        normal_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .build(device)
        );

        normal_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder()
                .add_sets(*normal_desc_layout_, 1)
                .build(device)
        );

        normal_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
            device, *normal_desc_pool_, *normal_desc_layout_
        );

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.raster.cull = coopa::gfx::CullMode::None;
        desc.depth.test  = false;
        desc.depth.write = false;
        desc.descriptor_layouts = {&camera_layout, normal_desc_layout_.get()};
        desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(SkyboxPushConstants)}};

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, offscreen_pass, desc);
    }

    void set_gbuffer_normal_image(coopa::gfx::TextureView g1_view, const util::Sampler& linear_sampler) {
        normal_desc_set_->bind_image(0, g1_view, linear_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd,
              const coopa::gfx::pipeline::DescriptorSet& camera_set,
              const glm::mat4& view,
              const glm::mat4& proj,
              uint32_t viewport_w, uint32_t viewport_h,
              const IndirectParams& indirect = IndirectParams{}) const
    {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);

        SkyboxPushConstants pc{};
        pc.inv_view_proj = glm::inverse(proj * view);
        pc.sky_zenith    = glm::vec4(indirect.sky_zenith, 0.0f);
        pc.sky_horizon   = glm::vec4(indirect.sky_horizon, 0.0f);
        pc.sky_ground    = glm::vec4(indirect.sky_ground, 0.0f);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);

        cmd.bind_descriptor_set(camera_set, 0);
        cmd.bind_descriptor_set(*normal_desc_set_, 1);

        cmd.draw(3); // Fullscreen triangle
    }

private:
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
