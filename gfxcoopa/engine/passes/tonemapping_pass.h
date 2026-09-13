/**
 * @file tonemapping_pass.h
 * @brief Fullscreen tonemapping and FXAA render pass header for gfxcoopa.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_TONEMAPPING_PASS_H
#define GFXCOOPA_ENGINE_PASSES_TONEMAPPING_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <cstdint>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/types/texture_view.h>
#include <gfxcoopa/engine/passes/fullscreen_stage.h>
#include <gfxcoopa/engine/util/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class ToneMappingPass {
public:
    struct PushConstants {
        float   exposure           = 1.0f;
        float   screen_width       = 1920.0f;
        float   screen_height      = 1080.0f;
        int32_t fxaa_enabled       = 1;
        float   subpixel_quality   = 0.75f;
        float   edge_threshold     = 0.166f;
        float   edge_threshold_min = 0.0312f;
    };

    /**
     * @brief Builds the tonemapping pipeline and its descriptor set.
     * @param device         Logical device.
     * @param swapchain_pass Render pass this writes into.
     * @param linear_sampler Accepted for signature compatibility and unused;
     *   the sampler that matters is the one passed to set_source_image().
     * @param vert_spv       Fullscreen-triangle vertex shader.
     * @param frag_spv       Tonemap (+ optional FXAA) fragment shader.
     */
    ToneMappingPass(coopa::gfx::core::Device& device,
                    coopa::gfx::pipeline::RenderPass& swapchain_pass,
                    const util::Sampler& linear_sampler,
                    const std::string& vert_spv,
                    const std::string& frag_spv)
        : stage_(device, swapchain_pass, describe(vert_spv, frag_spv))
    {
        (void)linear_sampler;
    }

    ToneMappingPass(const ToneMappingPass&) = delete;
    ToneMappingPass& operator=(const ToneMappingPass&) = delete;

    /**
     * @brief Points the pass at the HDR image to tonemap.
     * @param hdr_view       The HDR color image.
     * @param linear_sampler Sampler used to read it.
     */
    void set_source_image(coopa::gfx::TextureView hdr_view, const util::Sampler& linear_sampler) {
        stage_.set().bind_image(0, hdr_view, linear_sampler);
    }

    void set_exposure(float exposure) {
        exposure_ = exposure;
    }

    float exposure() const {
        return exposure_;
    }

    void set_fxaa_config(bool enabled, float subpixel = 0.75f, float threshold = 0.166f, float threshold_min = 0.0312f) {
        fxaa_enabled_       = enabled ? 1 : 0;
        subpixel_quality_   = subpixel;
        edge_threshold_     = threshold;
        edge_threshold_min_ = threshold_min;
    }

    /**
     * @brief Records the fullscreen tonemap draw.
     * @param cmd        Command buffer, inside the target render pass.
     * @param viewport_w Target width in pixels.
     * @param viewport_h Target height in pixels.
     */
    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t viewport_w, uint32_t viewport_h) const {
        PushConstants pc{
            exposure_,
            static_cast<float>(viewport_w),
            static_cast<float>(viewport_h),
            fxaa_enabled_,
            subpixel_quality_,
            edge_threshold_,
            edge_threshold_min_
        };
        stage_.draw(cmd, viewport_w, viewport_h, coopa::gfx::ShaderStage::Fragment, pc);
    }

private:
    /// @brief One sampled HDR image at binding 0, plus the fragment push constants.
    static FullscreenStageDesc describe(const std::string& vert_spv, const std::string& frag_spv) {
        FullscreenStageDesc d;
        d.vert_spv = vert_spv;
        d.frag_spv = frag_spv;
        d.owned_sets = {{{0, coopa::gfx::DescriptorType::CombinedImageSampler,
                          coopa::gfx::ShaderStage::Fragment, 1}}};
        d.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(PushConstants)}};
        return d;
    }

    float   exposure_;
    int32_t fxaa_enabled_;
    float   subpixel_quality_;
    float   edge_threshold_;
    float   edge_threshold_min_;

    FullscreenStage stage_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_TONEMAPPING_PASS_H
