/**
 * @file post_process_pipeline.h
 * @brief Screen-space post-processing pipeline for edge detection and color quantization.
 *
 * Two stages rendered via ping-pong OffscreenTargets:
 *   1. Edge detection (edge_detect.frag): Sobel on depth buffer, composites
 *      dark lines over the toon-rendered color image.
 *   2. Color quantization (color_quantize.frag): optional posterization
 *      for a retro limited-palette feel.
 *
 * Each stage can be toggled independently. When all stages are disabled,
 * the source image is passed through unchanged.
 */

#ifndef COOPA_GFX_ENGINE_POST_PROCESS_PIPELINE_H
#define COOPA_GFX_ENGINE_POST_PROCESS_PIPELINE_H

#include <volk/volk.h>
#include <memory>
#include <string>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/offscreen_target.h>
#include <gfxcoopa/engine/fullscreen_quad.h>
#include <gfxcoopa/engine/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @struct EdgeDetectPushConstants
 * @brief Push constants for the edge detection stage.
 */
struct EdgeDetectPushConstants {
    float edge_threshold = 0.005f; /**< Depth difference threshold for edge detection. */
    float edge_strength  = 0.85f;  /**< How dark the detected edges are (0=off, 1=full black). */
    float _pad0          = 0.0f;
    float _pad1          = 0.0f;
};

/**
 * @struct QuantizePushConstants
 * @brief Push constants for the color quantization stage.
 */
struct QuantizePushConstants {
    int   color_levels = 0;    /**< Number of discrete levels per channel (0 = disabled). */
    float brightness   = 1.0f; /**< Brightness multiplier. */
    float saturation   = 1.0f; /**< Saturation boost. */
    float _pad0        = 0.0f;
};

/**
 * @class PostProcessPipeline
 * @brief Applies edge detection and optional color quantization post-processing.
 *
 * Usage:
 * @code
 * PostProcessPipeline pp(device, allocator, 320, 240, ...);
 * // Between the offscreen toon pass and the upscale pass:
 * VkImageView result = pp.process(cmd, color_view, depth_view);
 * // result is ready in SHADER_READ_ONLY_OPTIMAL for the upscale pass.
 * @endcode
 */
class PostProcessPipeline {
public:
    /**
     * @brief Creates the post-processing pipeline stages.
     *
     * @param device            Logical device.
     * @param allocator         VMA allocator.
     * @param render_width      Render resolution width.
     * @param render_height     Render resolution height.
     * @param linear_sampler    Linear sampler for intermediate sampling.
     * @param edge_vert_spv     Path to upscale.vert.spv (reused fullscreen vert).
     * @param edge_frag_spv     Path to edge_detect.frag.spv.
     * @param quantize_vert_spv Path to upscale.vert.spv (shared).
     * @param quantize_frag_spv Path to color_quantize.frag.spv.
     */
    PostProcessPipeline(core::Device&        device,
                        memory::Allocator&   allocator,
                        uint32_t             render_width,
                        uint32_t             render_height,
                        const Sampler&       linear_sampler,
                        const std::string&   edge_vert_spv,
                        const std::string&   edge_frag_spv,
                        const std::string&   quantize_vert_spv,
                        const std::string&   quantize_frag_spv)
        : device_(device), allocator_(allocator),
          linear_sampler_(linear_sampler)
    {
        // Intermediate target for edge detection output.
        edge_target_ = std::make_unique<OffscreenTarget>(
            device, allocator, render_width, render_height
        );

        // Intermediate target for quantization output.
        quant_target_ = std::make_unique<OffscreenTarget>(
            device, allocator, render_width, render_height
        );

        // Build the edge detection pipeline.
        build_edge_pipeline_(edge_vert_spv, edge_frag_spv);

        // Build the quantization pipeline.
        build_quant_pipeline_(quantize_vert_spv, quantize_frag_spv);
    }

    /** @brief Toggles the edge detection stage. */
    void set_edge_detection_enabled(bool enabled) { edge_enabled_ = enabled; }
    /** @brief Sets edge detection parameters. */
    void set_edge_params(float threshold, float strength) {
        edge_pc_.edge_threshold = threshold;
        edge_pc_.edge_strength  = strength;
    }

    /** @brief Toggles the color quantization stage. */
    void set_quantization_enabled(bool enabled) { quantize_enabled_ = enabled; }
    /** @brief Sets quantization parameters. */
    void set_quantize_params(int levels, float brightness, float saturation) {
        quant_pc_.color_levels = levels;
        quant_pc_.brightness   = brightness;
        quant_pc_.saturation   = saturation;
    }

    /**
     * @brief Runs all enabled post-processing stages.
     *
     * Must be called outside any render pass scope (stages begin/end their own passes).
     *
     * @param cmd        Command buffer (outside any render pass).
     * @param color_view Source color image view (from offscreen toon+outline pass).
     * @param depth_view Source depth image view (for edge detection).
     * @return VkImageView of the final processed image (ready as SHADER_READ_ONLY_OPTIMAL).
     */
    VkImageView process(command::CommandBuffer& cmd,
                        VkImageView             color_view,
                        VkImageView             depth_view)
    {
        VkImageView current_color = color_view;

        // Stage 1: Edge detection.
        if (edge_enabled_) {
            edge_target_->begin(cmd);

            cmd.bind_pipeline(*edge_pipeline_);
            // Update descriptor set with current color + depth views.
            edge_desc_set_->bind_image(0, current_color,     linear_sampler_.handle());
            edge_desc_set_->bind_image(1, depth_view,        linear_sampler_.handle());
            cmd.bind_descriptor_set(edge_pipeline_->layout(), *edge_desc_set_, 0);
            cmd.push_constants(edge_pipeline_->layout(),
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(EdgeDetectPushConstants), &edge_pc_);
            fsq_.draw(cmd);

            edge_target_->end(cmd);
            current_color = edge_target_->color_view();
        }

        // Stage 2: Color quantization.
        if (quantize_enabled_) {
            quant_target_->begin(cmd);

            cmd.bind_pipeline(*quant_pipeline_);
            quant_desc_set_->bind_image(0, current_color, linear_sampler_.handle());
            cmd.bind_descriptor_set(quant_pipeline_->layout(), *quant_desc_set_, 0);
            cmd.push_constants(quant_pipeline_->layout(),
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(QuantizePushConstants), &quant_pc_);
            fsq_.draw(cmd);

            quant_target_->end(cmd);
            current_color = quant_target_->color_view();
        }

        return current_color;
    }

private:
    /**
     * @brief Builds the edge detection pipeline.
     */
    void build_edge_pipeline_(const std::string& vert_spv, const std::string& frag_spv) {
        edge_vert_ = std::make_unique<pipeline::Shader>(device_, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        edge_frag_ = std::make_unique<pipeline::Shader>(device_, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor layout: binding 0 = color image, binding 1 = depth image.
        std::vector<VkDescriptorSetLayoutBinding> bindings(2);
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        edge_layout_ = std::make_unique<pipeline::DescriptorSetLayout>(device_, bindings);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(EdgeDetectPushConstants);

        pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        edge_pipeline_ = std::make_unique<pipeline::Pipeline>(
            device_, edge_target_->render_pass_object(),
            std::vector<pipeline::Shader*>{edge_vert_.get(), edge_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{edge_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc}
        );

        // Pool and set.
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 2;
        edge_pool_ = std::make_unique<pipeline::DescriptorPool>(
            device_, 1, std::vector<VkDescriptorPoolSize>{ps}
        );
        edge_desc_set_ = std::make_unique<pipeline::DescriptorSet>(device_, *edge_pool_, *edge_layout_);
    }

    /**
     * @brief Builds the quantization pipeline.
     */
    void build_quant_pipeline_(const std::string& vert_spv, const std::string& frag_spv) {
        quant_vert_ = std::make_unique<pipeline::Shader>(device_, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        quant_frag_ = std::make_unique<pipeline::Shader>(device_, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        quant_layout_ = std::make_unique<pipeline::DescriptorSetLayout>(
            device_, std::vector<VkDescriptorSetLayoutBinding>{binding}
        );

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(QuantizePushConstants);

        pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        quant_pipeline_ = std::make_unique<pipeline::Pipeline>(
            device_, quant_target_->render_pass_object(),
            std::vector<pipeline::Shader*>{quant_vert_.get(), quant_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{quant_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc}
        );

        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 1;
        quant_pool_ = std::make_unique<pipeline::DescriptorPool>(
            device_, 1, std::vector<VkDescriptorPoolSize>{ps}
        );
        quant_desc_set_ = std::make_unique<pipeline::DescriptorSet>(device_, *quant_pool_, *quant_layout_);
    }

    core::Device&       device_;
    memory::Allocator&  allocator_;
    const Sampler&      linear_sampler_;

    // Edge detection stage.
    bool                                          edge_enabled_  = true;
    EdgeDetectPushConstants                       edge_pc_;
    std::unique_ptr<OffscreenTarget>              edge_target_;
    std::unique_ptr<pipeline::Shader>             edge_vert_;
    std::unique_ptr<pipeline::Shader>             edge_frag_;
    std::unique_ptr<pipeline::DescriptorSetLayout>edge_layout_;
    std::unique_ptr<pipeline::Pipeline>           edge_pipeline_;
    std::unique_ptr<pipeline::DescriptorPool>     edge_pool_;
    std::unique_ptr<pipeline::DescriptorSet>      edge_desc_set_;

    // Color quantization stage.
    bool                                           quantize_enabled_ = false;
    QuantizePushConstants                          quant_pc_;
    std::unique_ptr<OffscreenTarget>               quant_target_;
    std::unique_ptr<pipeline::Shader>              quant_vert_;
    std::unique_ptr<pipeline::Shader>              quant_frag_;
    std::unique_ptr<pipeline::DescriptorSetLayout> quant_layout_;
    std::unique_ptr<pipeline::Pipeline>            quant_pipeline_;
    std::unique_ptr<pipeline::DescriptorPool>      quant_pool_;
    std::unique_ptr<pipeline::DescriptorSet>       quant_desc_set_;

    FullscreenQuad                                 fsq_;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_POST_PROCESS_PIPELINE_H
