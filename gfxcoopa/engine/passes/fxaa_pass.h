/**
 * @file fxaa_pass.h
 * @brief Standalone LDR FXAA 3.11 (Quality Preset 39) post-processing pass.
 *
 * Follows FogPass's shape line for line -- a fullscreen triangle, one combined-image-sampler
 * descriptor, writing into a separate target the caller owns begin()/end() for (see
 * pipeline::RenderPass's hardcoded LOAD_OP_CLEAR, documented on FogPass itself, for why that
 * target can't be the same image this pass reads from).
 *
 * Unlike blendy's ToneMappingPass (tonemapping_pass.h), which fuses FXAA into the HDR->LDR
 * tonemap step and re-tonemaps every one of FXAA's ~30 taps, this pass expects an
 * already-tonemapped LDR source and does no exposure/ACES work itself -- see fxaa.frag's own
 * file doc for why that's a deliberate behavior change, not just a refactor.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_FXAA_PASS_H
#define GFXCOOPA_ENGINE_PASSES_FXAA_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/types/texture_view.h>
#include <gfxcoopa/engine/util/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class FxaaPass {
public:
    /**
     * @brief Push-constant layout for fxaa.frag -- fragment stage only.
     */
    struct PushConstants {
        float screen_width;
        float screen_height;
        float subpixel_quality      = 0.75f;   /**< Blend weight of the subpixel-aliasing term. */
        float edge_threshold        = 0.166f;  /**< Local contrast (fraction of lumaMax) below which no AA is applied. */
        float edge_threshold_min    = 0.0312f; /**< Absolute contrast floor -- avoids AA-ing near-black noise. */
    };

    /**
     * @brief Builds the FXAA pipeline and its descriptor set.
     * @param device      Logical device.
     * @param target_pass Render pass of the (separate) LDR target this pass writes into.
     * @param vert_spv    Fullscreen-triangle vertex shader.
     * @param frag_spv    fxaa.frag.
     */
    FxaaPass(coopa::gfx::core::Device& device,
             coopa::gfx::pipeline::RenderPass& target_pass,
             const std::string& vert_spv,
             const std::string& frag_spv)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder().add_sets(*layout_, 1).build(device));
        set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *layout_);

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.raster.cull = coopa::gfx::CullMode::None;
        desc.depth.test  = false;
        desc.depth.write = false;
        desc.descriptor_layouts = {layout_.get()};
        desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(PushConstants)}};

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, target_pass, desc);
    }

    FxaaPass(const FxaaPass&) = delete;
    FxaaPass& operator=(const FxaaPass&) = delete;

    /** @brief Rebinds the LDR source image (linear-filtered, matching FXAA's own edge-search taps). */
    void set_source_image(coopa::gfx::TextureView color_view, const coopa::gfx::engine::util::Sampler& linear_sampler) {
        set_->bind_image(0, color_view, linear_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& pc,
              uint32_t viewport_w, uint32_t viewport_h) const {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(*set_, 0);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);
        cmd.draw(3);
    }

private:
    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>            pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_FXAA_PASS_H
