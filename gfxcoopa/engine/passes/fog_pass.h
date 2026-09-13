/**
 * @file fog_pass.h
 * @brief Fullscreen Unity-style fog composite (see assets/shaders/fog.frag).
 *
 * Follows PixelStylizePass's shape almost line for line -- the closest existing
 * pass (fullscreen triangle, reads scene colour + G-buffer, owns its own
 * descriptor layout/pool/set). Writes into its own target: pipeline::RenderPass's
 * hardcoded LOAD_OP_CLEAR (gfxcoopa/pipeline/render_pass.h) means the target it
 * reads scene colour from can't be reopened and composited onto in place.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_FOG_PASS_H
#define GFXCOOPA_ENGINE_PASSES_FOG_PASS_H

#include <volk/volk.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/passes/fullscreen_stage.h>
#include <gfxcoopa/types/texture_view.h>
#include <gfxcoopa/engine/util/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class FogPass {
public:
    /**
     * @brief Builds the fog pipeline and its two descriptor sets.
     * @param device     Logical device.
     * @param target_pass Render pass of the (separate) HDR target this pass writes into.
     * @param fog_ubo    The FogData's uniform buffer (see engine/data/fog_data.h). Bound
     *                   once here, at set 1 binding 0 -- like camera_set_/light_set_ in
     *                   PbrRenderPipeline, the buffer's VkBuffer handle never changes
     *                   after creation, so only its per-frame upload() needs repeating.
     * @param vert_spv   Fullscreen-triangle vertex shader (the shared tonemapping.vert).
     * @param frag_spv   fog.frag.
     */
    FogPass(coopa::gfx::core::Device& device,
            coopa::gfx::pipeline::RenderPass& target_pass,
            const coopa::gfx::memory::Buffer& fog_ubo,
            const std::string& vert_spv,
            const std::string& frag_spv)
        : nearest_sampler_(coopa::gfx::engine::util::Sampler::nearest(device)),
          stage_(device, target_pass, describe(vert_spv, frag_spv))
    {
        stage_.set(1).bind_buffer(0, fog_ubo);
    }

    FogPass(const FogPass&) = delete;
    FogPass& operator=(const FogPass&) = delete;

    /**
     * @brief Rebinds the three source images. Called every frame the pass runs, since
     * scene_color may be a different view depending on whether SSR ran this frame
     * (see PbrRenderPipeline's hdr_source_view).
     *
     * g_normal/g_position use the nearest sampler this pass owns, not linear: linear
     * filtering would blend world positions across silhouette edges, producing a wrong
     * fog distance at every object outline -- the same reasoning PixelStylizePass
     * documents for its own depth/normal taps.
     */
    void set_source_images(coopa::gfx::TextureView scene_color, coopa::gfx::TextureView g_normal,
                           coopa::gfx::TextureView g_position, const coopa::gfx::engine::util::Sampler& linear_sampler) {
        stage_.set(0).bind_image(0, scene_color, linear_sampler);
        stage_.set(0).bind_image(1, g_normal, nearest_sampler_);
        stage_.set(0).bind_image(2, g_position, nearest_sampler_);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t viewport_w, uint32_t viewport_h) const {
        stage_.bind(cmd, viewport_w, viewport_h);
        stage_.draw(cmd);
    }

private:
    /// @brief Set 0: scene colour + G-buffer normal/position. Set 1: the pass's UBO.
    static FullscreenStageDesc describe(const std::string& vert_spv, const std::string& frag_spv) {
        using coopa::gfx::DescriptorType;
        using coopa::gfx::ShaderStage;
        FullscreenStageDesc d;
        d.vert_spv = vert_spv;
        d.frag_spv = frag_spv;
        d.owned_sets = {
            {{0, DescriptorType::CombinedImageSampler, ShaderStage::Fragment, 1},
             {1, DescriptorType::CombinedImageSampler, ShaderStage::Fragment, 1},
             {2, DescriptorType::CombinedImageSampler, ShaderStage::Fragment, 1}},
            {{0, DescriptorType::UniformBuffer, ShaderStage::Fragment, 1}},
        };
        return d;
    }

    coopa::gfx::engine::util::Sampler nearest_sampler_;
    FullscreenStage                   stage_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_FOG_PASS_H
