/**
 * @file pixel_stylize_pass.h
 * @brief Optional bloom + tonemap + outline + ordered dither + palette
 *        quantization overlay (see assets/shaders/pixel_stylize.frag).
 *
 * The tonemap step is optional (PushConstants::exposure <= 0 disables it):
 * a full PBR renderer with its own tonemap/AA chain (e.g. blendy) feeds
 * already-tonemapped LDR input and composites the retro-look effects as a
 * final overlay, with no double-tonemapping. A consumer with no tonemap
 * step of its own (e.g. toyengine) sets exposure > 0 and feeds raw HDR
 * scene color instead.
 *
 * Reads scene color plus the G-buffer's depth and normal (for the outline
 * edge detector), a palette LUT, and (optionally) a pre-blurred bloom image --
 * see set_source_images()'s bloom_result param. Bloom itself is NOT computed
 * here: it's a separate BloomPass pyramid (bright-pass threshold -> multi-tap
 * downsample -> tent-filter upsample+combine), and this pass just adds its
 * finished result before the tonemap. Writes into its own target --
 * pipeline::RenderPass's hardcoded LOAD_OP_CLEAR (gfxcoopa/pipeline/render_pass.h)
 * means the target it reads from can't be reopened and composited onto in place.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_PIXEL_STYLIZE_PASS_H
#define GFXCOOPA_ENGINE_PASSES_PIXEL_STYLIZE_PASS_H

#include <volk/volk.h>
#include <glm/glm.hpp>

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

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class PixelStylizePass {
public:
    /**
     * @brief Matches pixel_stylize.frag's StylizePushConstants block.
     *
     * outline_color (vec4) is listed first so its GLSL std430 16-byte
     * alignment lands at offset 0 for free -- everywhere else in gfxcoopa
     * that mixes a vec4 with scalars in a push-constant block (e.g.
     * SsrPass::ResolvePushConstants, gbuffer.frag's material block) follows
     * the same ordering, since a vec4 placed mid-struct forces the GLSL
     * side to insert padding the plain C++ struct below doesn't replicate.
     */
    struct PushConstants {
        glm::vec4 outline_color      = glm::vec4(0.05f, 0.04f, 0.08f, 1.0f);
        glm::vec2 inv_render_size;
        float     outline_thickness  = 0.0f;  ///< In texels; <= 0 disables.
        float     depth_threshold    = 0.02f;
        float     normal_threshold   = 0.75f;
        float     dither_strength    = 0.0f;  ///< <= 0 disables.
        float     palette_count      = 0.0f;  ///< <= 0 disables palette quantization.
        float     camera_near           = 0.1f;
        float     camera_far            = 1000.0f;
        float     camera_is_perspective = 1.0f;  ///< >= 0.5 => perspective, else orthographic.
        float     exposure              = 0.0f;  ///< <= 0 disables the tonemap step (input is already LDR).
        /// Final multiplier on the pre-blurred bloom image bound at binding 4 (see
        /// set_source_images()'s bloom_result param and BloomPass). <= 0 disables it.
        /// No threshold/LOD fields here anymore: BloomPass's own bright-pass shader
        /// thresholds once, per source texel, before any blurring -- re-thresholding
        /// the finished blurred result here would eat the halo falloff the pyramid
        /// exists to produce, and there is no mip chain left to pick an LOD from.
        float     bloom_intensity        = 0.0f;
    };
    static_assert(sizeof(PushConstants) == 64,
                 "PushConstants must match pixel_stylize.frag's StylizePushConstants byte-for-byte");

    PixelStylizePass(coopa::gfx::core::Device& device,
                     coopa::gfx::pipeline::RenderPass& target_pass,
                     const std::string& vert_spv,
                     const std::string& frag_spv)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        coopa::gfx::pipeline::DescriptorLayoutBuilder layout_builder;
        for (uint32_t i = 0; i < 5; ++i) {
            layout_builder.combined_sampler(i, coopa::gfx::ShaderStage::Fragment);
        }
        desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(layout_builder.build(device));
        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder().add_sets(*desc_layout_, 1).build(device));
        desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *desc_layout_);

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.raster.cull = coopa::gfx::CullMode::None;
        desc.depth.test  = false;
        desc.depth.write = false;
        desc.descriptor_layouts = {desc_layout_.get()};
        desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(PushConstants)}};

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, target_pass, desc);
    }

    PixelStylizePass(const PixelStylizePass&) = delete;
    PixelStylizePass& operator=(const PixelStylizePass&) = delete;

    void set_source_images(coopa::gfx::TextureView scene_color, coopa::gfx::TextureView scene_depth,
                           coopa::gfx::TextureView scene_normal, coopa::gfx::TextureView palette_lut,
                           const coopa::gfx::engine::util::Sampler& linear_sampler,
                           const coopa::gfx::engine::util::Sampler& nearest_sampler,
                           coopa::gfx::TextureView bloom_result = {},
                           const coopa::gfx::engine::util::Sampler* bloom_sampler = nullptr) {
        // scene_depth/scene_normal use the nearest sampler, not linear: the outline edge
        // detector's taps need exact texel values (linear filtering would blend across the
        // very discontinuities it's looking for), and linear filtering of a D32_SFLOAT depth
        // image is an optional Vulkan format feature that isn't queried anywhere in this engine.
        desc_set_->bind_image(0, scene_color, linear_sampler);
        desc_set_->bind_image(1, scene_depth, nearest_sampler);
        desc_set_->bind_image(2, scene_normal, nearest_sampler);
        desc_set_->bind_image(3, palette_lut, nearest_sampler);
        // bloom_result: the finished, pre-blurred output of a dedicated BloomPass pyramid
        // (bright-pass threshold -> multi-tap downsample -> tent-filter upsample+combine),
        // sampled with a plain texture() through a LINEAR sampler: one already-composited
        // image at half this pass's resolution, bilinearly upsampled for free, with no LOD
        // to select. Optional -- omitting it re-binds binding 0's own view/sampler, which is
        // harmless because bloom_intensity <= 0 makes the shader skip the read. Only pass a
        // real view once the BloomPass has executed at least once, or the descriptor points
        // at a target that has never been rendered into.
        desc_set_->bind_image(4, bloom_sampler ? bloom_result : scene_color,
                              bloom_sampler ? *bloom_sampler : linear_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& params,
              uint32_t viewport_w, uint32_t viewport_h) const {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(*desc_set_, 0);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, params);
        cmd.draw(3);
    }

private:
    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       desc_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>            pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_PIXEL_STYLIZE_PASS_H
