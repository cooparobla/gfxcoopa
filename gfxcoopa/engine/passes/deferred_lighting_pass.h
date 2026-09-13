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
#include <gfxcoopa/engine/passes/extra_sets.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class DeferredLightingPass {
public:
    DeferredLightingPass(coopa::gfx::core::Device& device,
                         coopa::gfx::pipeline::RenderPass& offscreen_pass,
                         const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
                         const coopa::gfx::pipeline::DescriptorSetLayout& light_layout,
                         const coopa::gfx::pipeline::DescriptorSetLayout& shadow_layout,
                         const util::Sampler& linear_sampler,
                         const std::string& vert_spv,
                         const std::string& frag_spv,
                         ExtraSets extra = {},
                         std::vector<coopa::gfx::pipeline::PushConstantRange> pc_ranges = {})
        : extra_(std::move(extra))
    {
        extra_.validate("DeferredLightingPass");

        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layout (the pass's last set -- see gbuffer_set_index_): 4
        // G-Buffer samplers (albedo/ao, normal/metallic, position/roughness, emissive)
        // + 1 SSAO sampler.
        coopa::gfx::pipeline::DescriptorLayoutBuilder gbuffer_layout_builder;
        for (uint32_t i = 0; i < 5; ++i) {
            gbuffer_layout_builder.combined_sampler(i, coopa::gfx::ShaderStage::Fragment);
        }
        gbuffer_desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            gbuffer_layout_builder.build(device)
        );

        gbuffer_desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder()
                .add_sets(*gbuffer_desc_layout_, 1)
                .build(device)
        );

        gbuffer_desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
            device, *gbuffer_desc_pool_, *gbuffer_desc_layout_
        );

        // Pipeline configuration
        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.raster.cull = coopa::gfx::CullMode::None;
        desc.depth.test  = false;
        desc.depth.write = false;

        std::vector<const coopa::gfx::pipeline::DescriptorSetLayout*> layouts = {
            &camera_layout,
            &light_layout,
            &shadow_layout
        };
        // Both indices are derived, never hardcoded: they shift with the number of extra
        // sets the caller supplies, and with any set this pass might later own ahead of them.
        // A stale constant here means vkCmdBindDescriptorSets with a firstSet past the end of
        // the layout -- a validation error and undefined behaviour.
        extra_first_set_ = static_cast<uint32_t>(layouts.size());
        layouts.insert(layouts.end(), extra_.layouts.begin(), extra_.layouts.end());
        gbuffer_set_index_ = static_cast<uint32_t>(layouts.size());
        layouts.push_back(gbuffer_desc_layout_.get());

        desc.descriptor_layouts = layouts;
        desc.push_constants     = pc_ranges;

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, offscreen_pass, desc);

        (void)linear_sampler;
    }

    void set_gbuffer_images(coopa::gfx::TextureView g0_view,
                            coopa::gfx::TextureView g1_view,
                            coopa::gfx::TextureView g2_view,
                            coopa::gfx::TextureView g3_view,
                            const util::Sampler& linear_sampler) {
        gbuffer_desc_set_->bind_image(0, g0_view, linear_sampler);
        gbuffer_desc_set_->bind_image(1, g1_view, linear_sampler);
        gbuffer_desc_set_->bind_image(2, g2_view, linear_sampler);
        // Binding 4, not 3 -- SSAO keeps binding 3 (see set_ssao_image() below), so adding
        // emissive is a pure append in every consumer's fragment shader rather than a
        // renumber, and this set stays aligned with SsrPass's composite G-buffer set,
        // which also treats binding 3 as SSAO.
        gbuffer_desc_set_->bind_image(4, g3_view, linear_sampler);
    }

    /// Binding 3 must be rebound every frame -- callers pass the SSAO pass's blurred output when
    /// enabled, or its permanent neutral (fully-unoccluded) texture when disabled/absent, so this
    /// binding is never left pointing at an image still in VK_IMAGE_LAYOUT_UNDEFINED.
    ///
    /// Still raw (VkImageView/VkSampler), unlike set_gbuffer_images() above: SsaoPass -- the only
    /// source of this argument -- is one of this refactor's "hard tier" files (its output_view()/
    /// neutral_view() accessors have no TextureView-returning sibling), so a sealed parameter here
    /// would have no caller who could actually satisfy it yet.
    void set_ssao_image(VkImageView ssao_view, VkSampler ssao_sampler) {
        gbuffer_desc_set_->bind_image(3, ssao_view, ssao_sampler);
    }

    /// @brief Draws the fullscreen deferred-lighting pass, with no push-constant block
    /// (only valid if the ctor's pc_ranges was left empty).
    void draw(coopa::gfx::command::CommandBuffer& cmd,
              const coopa::gfx::pipeline::DescriptorSet& camera_set,
              const coopa::gfx::pipeline::DescriptorSet& light_set,
              const coopa::gfx::pipeline::DescriptorSet& shadow_set,
              uint32_t viewport_w, uint32_t viewport_h) const
    {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);

        bind_sets_and_draw_(cmd, camera_set, light_set, shadow_set);
    }

    /// @brief Draws the fullscreen deferred-lighting pass, pushing `push_constants` to the
    /// fragment stage first -- the sealed replacement for a caller fetching layout() to push
    /// its own app-defined struct before calling the no-push-constant draw() above. `PushConstants`
    /// must match the byte layout the ctor's pc_ranges declared (this pass doesn't know that
    /// type -- it's app-defined, e.g. blendy's LightingPushConstants).
    template<typename PushConstants>
    void draw(coopa::gfx::command::CommandBuffer& cmd,
              const coopa::gfx::pipeline::DescriptorSet& camera_set,
              const coopa::gfx::pipeline::DescriptorSet& light_set,
              const coopa::gfx::pipeline::DescriptorSet& shadow_set,
              const PushConstants& push_constants,
              uint32_t viewport_w, uint32_t viewport_h) const
    {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, push_constants);

        bind_sets_and_draw_(cmd, camera_set, light_set, shadow_set);
    }

    VkPipelineLayout layout() const {
        return pipeline_->layout();
    }

private:
    /// @brief Shared tail of both draw() overloads -- binds the pass's own 3 sets, any
    /// caller-supplied ExtraSets, the G-buffer/SSAO set, and issues the fullscreen draw.
    /// Relies on the caller having already called cmd.bind_pipeline(*pipeline_).
    void bind_sets_and_draw_(coopa::gfx::command::CommandBuffer& cmd,
                             const coopa::gfx::pipeline::DescriptorSet& camera_set,
                             const coopa::gfx::pipeline::DescriptorSet& light_set,
                             const coopa::gfx::pipeline::DescriptorSet& shadow_set) const
    {
        cmd.bind_descriptor_set(camera_set, 0);
        cmd.bind_descriptor_set(light_set, 1);
        cmd.bind_descriptor_set(shadow_set, 2);
        if (extra_.bind) {
            extra_.bind(cmd, extra_first_set_);
        }
        cmd.bind_descriptor_set(*gbuffer_desc_set_, gbuffer_set_index_);

        cmd.draw(3); // Fullscreen triangle
    }

    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> gbuffer_desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      gbuffer_desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       gbuffer_desc_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           pipeline_;

    ExtraSets extra_;
    uint32_t  extra_first_set_   = 0;
    uint32_t  gbuffer_set_index_ = 0;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_DEFERRED_LIGHTING_PASS_H
