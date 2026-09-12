/**
 * @file wind_pass.h
 * @brief Fullscreen volumetric wind composite (see assets/shaders/volumetrics.frag).
 *
 * Structural copy of FogPass -- same two descriptor sets, same fullscreen
 * triangle, same G-buffer taps -- because the two passes differ only in their
 * fragment shader's middle (fog integrates analytically, wind raymarches). Like
 * FogPass, it writes into its own target: pipeline::RenderPass's hardcoded
 * LOAD_OP_CLEAR (gfxcoopa/pipeline/render_pass.h) means the target it reads
 * scene colour from can't be reopened and composited onto in place.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_VOLUMETRICS_PASS_H
#define GFXCOOPA_ENGINE_PASSES_VOLUMETRICS_PASS_H

#include <volk/volk.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/buffer.h>
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

class VolumetricsPass {
public:
    /**
     * @brief Builds the wind pipeline and its two descriptor sets.
     * @param device      Logical device.
     * @param target_pass Render pass of the (separate) HDR target this pass writes into.
     * @param wind_ubo    The VolumetricsData's uniform buffer (see engine/data/wind_data.h). Bound
     *                    once here, at set 1 binding 0 -- the buffer's VkBuffer handle never
     *                    changes after creation, so only its per-frame upload() repeats.
     * @param vert_spv    Fullscreen-triangle vertex shader (the shared fullscreen.vert).
     * @param frag_spv    volumetrics.frag.
     */
    VolumetricsPass(coopa::gfx::core::Device& device,
             coopa::gfx::pipeline::RenderPass& target_pass,
             const coopa::gfx::memory::Buffer& wind_ubo,
             const std::string& vert_spv,
             const std::string& frag_spv)
        : nearest_sampler_(coopa::gfx::engine::util::Sampler::nearest(device))
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Set 0: scene colour + G-buffer normal/position, all fragment-stage combined samplers.
        coopa::gfx::pipeline::DescriptorLayoutBuilder image_layout_builder;
        for (uint32_t i = 0; i < 3; ++i) {
            image_layout_builder.combined_sampler(i, coopa::gfx::ShaderStage::Fragment);
        }
        image_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(image_layout_builder.build(device));

        // Set 1: the VolumetricsUBO.
        ubo_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .uniform_buffer(0, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder()
                .add_sets(*image_layout_, 1)
                .add_sets(*ubo_layout_, 1)
                .build(device));

        image_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *image_layout_);
        ubo_set_   = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *ubo_layout_);
        ubo_set_->bind_buffer(0, wind_ubo);

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.raster.cull = coopa::gfx::CullMode::None;
        desc.depth.test  = false;
        desc.depth.write = false;
        desc.descriptor_layouts = {image_layout_.get(), ubo_layout_.get()};

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, target_pass, desc);
    }

    VolumetricsPass(const VolumetricsPass&) = delete;
    VolumetricsPass& operator=(const VolumetricsPass&) = delete;

    /**
     * @brief Binds the three source images. Call ONCE, at construction time.
     *
     * DescriptorSet::bind_image issues vkUpdateDescriptorSets immediately, so
     * rebinding mid-frame races this pipeline's in-flight command buffers.
     *
     * g_normal/g_position use the nearest sampler this pass owns, not linear:
     * linear filtering would blend world positions across silhouette edges,
     * producing a wrong ray-termination distance at every object outline -- the
     * same reasoning FogPass and PixelStylizePass both document.
     */
    void set_source_images(coopa::gfx::TextureView scene_color, coopa::gfx::TextureView g_normal,
                           coopa::gfx::TextureView g_position,
                           const coopa::gfx::engine::util::Sampler& linear_sampler) {
        image_set_->bind_image(0, scene_color, linear_sampler);
        image_set_->bind_image(1, g_normal, nearest_sampler_);
        image_set_->bind_image(2, g_position, nearest_sampler_);
    }

    /**
     * @brief Records the fullscreen wind composite into the caller's open render pass.
     *
     * Sets a POSITIVE-height viewport, overriding the negative-height one
     * OffscreenTarget::begin() installs for geometry passes -- a fullscreen
     * triangle must undo it (the same trap TiltShiftPass::begin_stage_ documents).
     */
    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t viewport_w, uint32_t viewport_h) const {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(*image_set_, 0);
        cmd.bind_descriptor_set(*ubo_set_, 1);
        cmd.draw(3);
    }

private:
    coopa::gfx::engine::util::Sampler                          nearest_sampler_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> image_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> ubo_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       image_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       ubo_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>            pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_VOLUMETRICS_PASS_H
