/**
 * @file pixel_stylize_pass.h
 * @brief Optional tonemap + outline + ordered dither + palette quantization
 *        overlay (see assets/shaders/pixel_stylize.frag).
 *
 * The tonemap step is optional (PushConstants::exposure <= 0 disables it):
 * a full PBR renderer with its own tonemap/AA chain (e.g. blendy) feeds
 * already-tonemapped LDR input and composites the retro-look effects as a
 * final overlay, with no double-tonemapping. A consumer with no tonemap
 * step of its own (e.g. toyengine) sets exposure > 0 and feeds raw HDR
 * scene color instead.
 *
 * Reads scene color plus the G-buffer's depth and normal (for the outline
 * edge detector), and a palette LUT. Writes into its own target --
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
    };
    static_assert(sizeof(PushConstants) == 60,
                 "PushConstants must match pixel_stylize.frag's StylizePushConstants byte-for-byte");

    PixelStylizePass(coopa::gfx::core::Device& device,
                     coopa::gfx::pipeline::RenderPass& target_pass,
                     const std::string& vert_spv,
                     const std::string& frag_spv)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 4; ++i) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = i;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings.push_back(b);
        }
        desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, bindings);
        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}});
        desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *desc_layout_);

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, target_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{desc_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc_range});
    }

    PixelStylizePass(const PixelStylizePass&) = delete;
    PixelStylizePass& operator=(const PixelStylizePass&) = delete;

    void set_source_images(VkImageView scene_color, VkImageView scene_depth, VkImageView scene_normal,
                           VkImageView palette_lut, const coopa::gfx::engine::util::Sampler& linear_sampler,
                           const coopa::gfx::engine::util::Sampler& nearest_sampler) {
        // scene_depth/scene_normal use the nearest sampler, not linear: the outline edge
        // detector's taps need exact texel values (linear filtering would blend across the
        // very discontinuities it's looking for), and linear filtering of a D32_SFLOAT depth
        // image is an optional Vulkan format feature that isn't queried anywhere in this engine.
        desc_set_->bind_image(0, scene_color, linear_sampler.handle());
        desc_set_->bind_image(1, scene_depth, nearest_sampler.handle());
        desc_set_->bind_image(2, scene_normal, nearest_sampler.handle());
        desc_set_->bind_image(3, palette_lut, nearest_sampler.handle());
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& params,
              uint32_t viewport_w, uint32_t viewport_h) const {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(pipeline_->layout(), *desc_set_, 0);
        cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &params);
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
