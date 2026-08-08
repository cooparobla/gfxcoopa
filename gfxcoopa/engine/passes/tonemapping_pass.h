#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file tonemapping_pass.h
 * @brief Fullscreen tonemapping and FXAA render pass header for gfxcoopa.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_TONEMAPPING_PASS_H
#define GFXCOOPA_ENGINE_PASSES_TONEMAPPING_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

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

    ToneMappingPass(coopa::gfx::core::Device& device,
                    coopa::gfx::pipeline::RenderPass& swapchain_pass,
                    const util::Sampler& linear_sampler,
                    const std::string& vert_spv,
                    const std::string& frag_spv)
        : device_(device),
          exposure_(1.0f),
          fxaa_enabled_(1),
          subpixel_quality_(0.75f),
          edge_threshold_(0.166f),
          edge_threshold_min_(0.0312f)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layout: Binding 0 = sampler2D (HDR image)
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        desc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            device,
            std::vector<VkDescriptorSetLayoutBinding>{binding}
        );

        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device,
            1,
            std::vector<VkDescriptorPoolSize>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
            }
        );

        desc_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(
            device, *desc_pool_, *desc_layout_
        );

        // Pipeline config: backface cull off, no depth test, no blending
        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, swapchain_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{desc_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );

        (void)linear_sampler;
    }

    void set_source_image(VkImageView hdr_view, const util::Sampler& linear_sampler) {
        desc_set_->bind_image(0, hdr_view, linear_sampler.handle());
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

    void draw(coopa::gfx::command::CommandBuffer& cmd, uint32_t viewport_w, uint32_t viewport_h) const {
        cmd.bind_pipeline(*pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);

        cmd.bind_descriptor_set(pipeline_->layout(), *desc_set_, 0);

        PushConstants pc{
            exposure_,
            static_cast<float>(viewport_w),
            static_cast<float>(viewport_h),
            fxaa_enabled_,
            subpixel_quality_,
            edge_threshold_,
            edge_threshold_min_
        };
        cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);

        cmd.draw(3); // Fullscreen triangle
    }

private:
    coopa::gfx::core::Device& device_;
    float   exposure_;
    int32_t fxaa_enabled_;
    float   subpixel_quality_;
    float   edge_threshold_;
    float   edge_threshold_min_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>              frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> desc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       desc_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_TONEMAPPING_PASS_H
