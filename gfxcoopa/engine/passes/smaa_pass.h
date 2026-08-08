#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file smaa_pass.h
 * @brief Subpixel Morphological Anti-Aliasing (SMAA 1x) post-processing pass for gfxcoopa.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SMAA_PASS_H
#define GFXCOOPA_ENGINE_PASSES_SMAA_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/targets/offscreen_target.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/engine/util/smaa_textures.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class SmaaPass {
public:
    struct EdgePush {
        glm::vec4 rt_metrics;
        float     threshold = 0.1f;
    };

    struct BlendPush {
        glm::vec4 rt_metrics;
        int32_t   max_search_steps = 16;
    };

    struct NeighborhoodPush {
        glm::vec4 rt_metrics;
        float     exposure = 1.0f;
    };

    SmaaPass(coopa::gfx::core::Device& device,
             coopa::gfx::memory::Allocator& allocator,
             coopa::gfx::pipeline::RenderPass& output_render_pass,
             coopa::gfx::command::CommandPool& cmd_pool,
             uint32_t width, uint32_t height,
             const util::Sampler& linear_sampler,
             const std::string& shader_dir)
        : device_(device), allocator_(allocator), width_(width), height_(height)
    {
        smaa_textures_ = std::make_unique<util::SmaaTextures>(device, allocator, cmd_pool);

        edges_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, width, height, VK_FORMAT_R8G8_UNORM, VK_SAMPLE_COUNT_1_BIT
        );

        blend_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT
        );

        // --- Stage 1: Edge Detection ---
        edge_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shader_dir + "/smaa_edge.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        edge_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shader_dir + "/smaa_edge.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        VkDescriptorSetLayoutBinding b0{};
        b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        edge_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, std::vector<VkDescriptorSetLayoutBinding>{b0});

        edge_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}});
        edge_set_  = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *edge_pool_, *edge_layout_);

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode = VK_CULL_MODE_NONE; cfg.depth_test = false; cfg.depth_write = false;

        VkPushConstantRange pc_edge{}; pc_edge.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; pc_edge.offset = 0; pc_edge.size = sizeof(EdgePush);
        edge_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, edges_target_->render_pass_object(),
            std::vector<coopa::gfx::pipeline::Shader*>{edge_vert_.get(), edge_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{}, std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{edge_layout_->handle()}, cfg, std::vector<VkPushConstantRange>{pc_edge}
        );

        // --- Stage 2: Blending Weight Calculation ---
        blend_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shader_dir + "/smaa_blend.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        blend_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shader_dir + "/smaa_blend.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayoutBinding> b_blend(3);
        b_blend[0].binding = 0; b_blend[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b_blend[0].descriptorCount = 1; b_blend[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b_blend[1].binding = 1; b_blend[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b_blend[1].descriptorCount = 1; b_blend[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b_blend[2].binding = 2; b_blend[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b_blend[2].descriptorCount = 1; b_blend[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        blend_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, b_blend);
        blend_pool_   = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}});
        blend_set_    = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *blend_pool_, *blend_layout_);

        VkPushConstantRange pc_blend{}; pc_blend.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; pc_blend.offset = 0; pc_blend.size = sizeof(BlendPush);
        blend_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, blend_target_->render_pass_object(),
            std::vector<coopa::gfx::pipeline::Shader*>{blend_vert_.get(), blend_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{}, std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{blend_layout_->handle()}, cfg, std::vector<VkPushConstantRange>{pc_blend}
        );

        blend_set_->bind_image(0, edges_target_->color_view(), linear_sampler.handle());
        blend_set_->bind_image(1, smaa_textures_->area_view(), smaa_textures_->sampler().handle());
        blend_set_->bind_image(2, smaa_textures_->search_view(), smaa_textures_->sampler().handle());

        // --- Stage 3: Neighborhood Blending ---
        neigh_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shader_dir + "/smaa_neighborhood.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        neigh_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shader_dir + "/smaa_neighborhood.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayoutBinding> b_neigh(2);
        b_neigh[0].binding = 0; b_neigh[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b_neigh[0].descriptorCount = 1; b_neigh[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b_neigh[1].binding = 1; b_neigh[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b_neigh[1].descriptorCount = 1; b_neigh[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        neigh_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, b_neigh);
        neigh_pool_   = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(device, 1, std::vector<VkDescriptorPoolSize>{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}});
        neigh_set_    = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *neigh_pool_, *neigh_layout_);

        VkPushConstantRange pc_neigh{}; pc_neigh.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; pc_neigh.offset = 0; pc_neigh.size = sizeof(NeighborhoodPush);
        neigh_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, output_render_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{neigh_vert_.get(), neigh_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{}, std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{neigh_layout_->handle()}, cfg, std::vector<VkPushConstantRange>{pc_neigh}
        );

        neigh_set_->bind_image(1, blend_target_->color_view(), linear_sampler.handle());
    }

    void set_source_image(VkImageView color_view, const util::Sampler& linear_sampler) {
        edge_set_->bind_image(0, color_view, linear_sampler.handle());
        neigh_set_->bind_image(0, color_view, linear_sampler.handle());
    }

    void recreate(uint32_t width, uint32_t height, const util::Sampler& linear_sampler) {
        width_  = width;
        height_ = height;
        edges_target_->recreate(width, height);
        blend_target_->recreate(width, height);

        blend_set_->bind_image(0, edges_target_->color_view(), linear_sampler.handle());
        neigh_set_->bind_image(1, blend_target_->color_view(), linear_sampler.handle());
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd,
              targets::OffscreenTarget& output_target,
              float exposure, float threshold, int max_search_steps,
              uint32_t viewport_w, uint32_t viewport_h)
    {
        glm::vec4 rt_metrics(1.0f / viewport_w, 1.0f / viewport_h, static_cast<float>(viewport_w), static_cast<float>(viewport_h));

        // Pass 1: Edge Detection
        edges_target_->begin(cmd);
        cmd.bind_pipeline(*edge_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(edge_pipeline_->layout(), *edge_set_, 0);

        EdgePush pc_edge{rt_metrics, threshold};
        cmd.push_constants(edge_pipeline_->layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(EdgePush), &pc_edge);
        cmd.draw(3);
        edges_target_->end(cmd);

        // Pass 2: Blending Weight Calculation
        blend_target_->begin(cmd);
        cmd.bind_pipeline(*blend_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(blend_pipeline_->layout(), *blend_set_, 0);

        BlendPush pc_blend{rt_metrics, max_search_steps};
        cmd.push_constants(blend_pipeline_->layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BlendPush), &pc_blend);
        cmd.draw(3);
        blend_target_->end(cmd);

        // Pass 3: Neighborhood Blending into output render pass
        output_target.begin(cmd);
        cmd.bind_pipeline(*neigh_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(neigh_pipeline_->layout(), *neigh_set_, 0);

        NeighborhoodPush pc_neigh{rt_metrics, exposure};
        cmd.push_constants(neigh_pipeline_->layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(NeighborhoodPush), &pc_neigh);
        cmd.draw(3);
        output_target.end(cmd);
    }

private:
    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;
    uint32_t width_;
    uint32_t height_;

    std::unique_ptr<util::SmaaTextures>                        smaa_textures_;
    std::unique_ptr<targets::OffscreenTarget> edges_target_;
    std::unique_ptr<targets::OffscreenTarget> blend_target_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              edge_vert_, edge_frag_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> edge_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      edge_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       edge_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           edge_pipeline_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              blend_vert_, blend_frag_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> blend_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      blend_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       blend_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           blend_pipeline_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>              neigh_vert_, neigh_frag_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> neigh_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      neigh_pool_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet>       neigh_set_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline>           neigh_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SMAA_PASS_H
