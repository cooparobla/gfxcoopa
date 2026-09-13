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
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/passes/fullscreen_stage.h>
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
        : stage_(device, offscreen_pass, describe(camera_layout, vert_spv, frag_spv))
    {}

    SkyboxPass(const SkyboxPass&) = delete;
    SkyboxPass& operator=(const SkyboxPass&) = delete;

    void set_gbuffer_normal_image(coopa::gfx::TextureView g1_view, const util::Sampler& linear_sampler) {
        stage_.set().bind_image(0, g1_view, linear_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd,
              const coopa::gfx::pipeline::DescriptorSet& camera_set,
              const glm::mat4& view,
              const glm::mat4& proj,
              uint32_t viewport_w, uint32_t viewport_h,
              const IndirectParams& indirect = IndirectParams{}) const
    {
        // Binds the pipeline, viewport/scissor, and this pass's own set at set 1;
        // the caller's camera set goes at set 0 below.
        stage_.bind(cmd, viewport_w, viewport_h);

        SkyboxPushConstants pc{};
        pc.inv_view_proj = glm::inverse(proj * view);
        pc.sky_zenith    = glm::vec4(indirect.sky_zenith, 0.0f);
        pc.sky_horizon   = glm::vec4(indirect.sky_horizon, 0.0f);
        pc.sky_ground    = glm::vec4(indirect.sky_ground, 0.0f);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);

        cmd.bind_descriptor_set(camera_set, 0);
        stage_.draw(cmd);
    }

private:
    /// @brief Set 0 is the caller's camera layout; set 1 is this pass's own
    /// G-buffer normal/metallic sampler.
    static FullscreenStageDesc describe(const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
                                        const std::string& vert_spv, const std::string& frag_spv) {
        FullscreenStageDesc d;
        d.vert_spv = vert_spv;
        d.frag_spv = frag_spv;
        d.leading_layouts = {&camera_layout};
        d.owned_sets = {{{0, coopa::gfx::DescriptorType::CombinedImageSampler,
                          coopa::gfx::ShaderStage::Fragment, 1}}};
        d.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(SkyboxPushConstants)}};
        return d;
    }

    FullscreenStage stage_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SKYBOX_PASS_H
