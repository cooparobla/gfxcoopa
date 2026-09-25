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
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/passes/fullscreen_stage.h>
#include <gfxcoopa/types/texture_view.h>
#include <gfxcoopa/engine/util/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

/**
 * @class VolumetricsPass
 * @brief Composites wind-driven volumetric scattering over a scene colour image.
 *
 * Same shape as FogPass: scene colour plus G-buffer normal and position at set
 * 0, the caller's VolumetricsData UBO at set 1, writing into its own target.
 */
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
    /**
     * @param fog_ubo The FogData uniform buffer (engine/data/fog_data.h), for the
     *                MERGED path where this pass also applies the global fog term
     *                instead of reading an image FogPass already wrote it into --
     *                see volumetrics.frag's `merged_fog`. Bound unconditionally (the
     *                buffer exists whether or not fog is enabled), so the pipeline
     *                layout never depends on the runtime flag; the caller selects the
     *                path through VolumetricsUBO::counts.w.
     */
    VolumetricsPass(coopa::gfx::core::Device& device,
             coopa::gfx::pipeline::RenderPass& target_pass,
             const coopa::gfx::memory::Buffer& wind_ubo,
             const coopa::gfx::memory::Buffer& fog_ubo,
             const std::string& vert_spv,
             const std::string& frag_spv)
        : nearest_sampler_(coopa::gfx::engine::util::Sampler::nearest(device)),
          stage_(device, target_pass, describe(vert_spv, frag_spv))
    {
        stage_.set(1).bind_buffer(0, wind_ubo);
        stage_.set(3).bind_buffer(0, fog_ubo);
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
        stage_.set(0).bind_image(0, scene_color, linear_sampler);
        stage_.set(0).bind_image(1, g_normal, nearest_sampler_);
        stage_.set(0).bind_image(2, g_position, nearest_sampler_);
    }

    /**
     * @brief Binds the directional and spot shadow maps the march's in-scatter
     *        terms sample. Call ONCE, at construction time (same rule as
     *        set_source_images above).
     *
     * @param dir_shadow    The directional shadow depth map.
     * @param spot_shadow   The spot shadow depth map.
     * @param shadow_sampler A compare-enabled sampler (util::Sampler::shadow()) --
     *                      volumetrics.frag declares both bindings as
     *                      sampler2DShadow, so each march tap is one hardware
     *                      depth-compare read.
     *
     * The bindings are non-optional in the pipeline layout; a consumer that never
     * shadows its volumes still binds real maps and gates the taps off through
     * VolumetricsUBO::shadow_params instead (a runtime flag, unlike this binding).
     */
    void set_shadow_images(coopa::gfx::TextureView dir_shadow, coopa::gfx::TextureView spot_shadow,
                           const coopa::gfx::engine::util::Sampler& shadow_sampler) {
        stage_.set(2).bind_image(0, dir_shadow, shadow_sampler);
        stage_.set(2).bind_image(1, spot_shadow, shadow_sampler);
    }

    /**
     * @brief Records the fullscreen wind composite into the caller's open render pass.
     *
     * Sets a POSITIVE-height viewport, overriding the negative-height one
     * OffscreenTarget::begin() installs for geometry passes -- a fullscreen
     * triangle must undo it (the same trap TiltShiftPass::begin_stage_ documents).
     */
    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t viewport_w, uint32_t viewport_h) const {
        stage_.bind(cmd, viewport_w, viewport_h);
        stage_.draw(cmd);
    }

private:
    /// @brief Set 0: scene colour + G-buffer normal/position. Set 1: the pass's UBO.
    ///        Set 2: directional + spot shadow maps (see set_shadow_images).
    ///        Set 3: the fog UBO, for the merged global-fog path.
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
            {{0, DescriptorType::CombinedImageSampler, ShaderStage::Fragment, 1},
             {1, DescriptorType::CombinedImageSampler, ShaderStage::Fragment, 1}},
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

#endif // GFXCOOPA_ENGINE_PASSES_VOLUMETRICS_PASS_H
