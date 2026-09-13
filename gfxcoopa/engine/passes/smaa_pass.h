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
#include <gfxcoopa/pipeline/shader_library.h>
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
             const coopa::gfx::pipeline::ShaderLibrary& shaders)
        : allocator_(allocator), width_(width), height_(height)
    {
        smaa_textures_ = std::make_unique<util::SmaaTextures>(device, allocator, cmd_pool);

        edges_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, width, height, coopa::gfx::Format::RG8_Unorm, coopa::gfx::SampleCount::X1
        );

        blend_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, width, height, coopa::gfx::Format::RGBA8_Unorm, coopa::gfx::SampleCount::X1
        );

        // --- Stage 1: Edge Detection ---
        edge_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shaders("smaa_edge.vert"), VK_SHADER_STAGE_VERTEX_BIT);
        edge_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shaders("smaa_edge.frag"), VK_SHADER_STAGE_FRAGMENT_BIT);

        edge_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .build(device));
        edge_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder().add_sets(*edge_layout_, 1).build(device));
        edge_set_  = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *edge_pool_, *edge_layout_);

        coopa::gfx::pipeline::PipelineDesc common_desc;
        common_desc.vertex = coopa::gfx::VertexLayout::none();
        common_desc.raster.cull = coopa::gfx::CullMode::None;
        common_desc.depth.test  = false;
        common_desc.depth.write = false;
        const auto both_stages = coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment;

        coopa::gfx::pipeline::PipelineDesc edge_desc = common_desc;
        edge_desc.shaders = {edge_vert_.get(), edge_frag_.get()};
        edge_desc.descriptor_layouts = {edge_layout_.get()};
        edge_desc.push_constants = {{both_stages, 0, sizeof(EdgePush)}};
        edge_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, edges_target_->render_pass_object(), edge_desc);

        // --- Stage 2: Blending Weight Calculation ---
        blend_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shaders("smaa_blend.vert"), VK_SHADER_STAGE_VERTEX_BIT);
        blend_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shaders("smaa_blend.frag"), VK_SHADER_STAGE_FRAGMENT_BIT);

        blend_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                .build(device));
        blend_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder().add_sets(*blend_layout_, 1).build(device));
        blend_set_  = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *blend_pool_, *blend_layout_);

        coopa::gfx::pipeline::PipelineDesc blend_desc = common_desc;
        blend_desc.shaders = {blend_vert_.get(), blend_frag_.get()};
        blend_desc.descriptor_layouts = {blend_layout_.get()};
        blend_desc.push_constants = {{both_stages, 0, sizeof(BlendPush)}};
        blend_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, blend_target_->render_pass_object(), blend_desc);

        blend_set_->bind_image(0, edges_target_->color_view_typed(), linear_sampler);
        blend_set_->bind_image(1, smaa_textures_->area_view_typed(), smaa_textures_->sampler());
        blend_set_->bind_image(2, smaa_textures_->search_view_typed(), smaa_textures_->sampler());

        // --- Stage 3: Neighborhood Blending ---
        neigh_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shaders("smaa_neighborhood.vert"), VK_SHADER_STAGE_VERTEX_BIT);
        neigh_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, shaders("smaa_neighborhood.frag"), VK_SHADER_STAGE_FRAGMENT_BIT);

        neigh_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .build(device));
        neigh_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder().add_sets(*neigh_layout_, 1).build(device));
        neigh_set_  = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *neigh_pool_, *neigh_layout_);

        coopa::gfx::pipeline::PipelineDesc neigh_desc = common_desc;
        neigh_desc.shaders = {neigh_vert_.get(), neigh_frag_.get()};
        neigh_desc.descriptor_layouts = {neigh_layout_.get()};
        neigh_desc.push_constants = {{both_stages, 0, sizeof(NeighborhoodPush)}};
        neigh_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, output_render_pass, neigh_desc);

        neigh_set_->bind_image(1, blend_target_->color_view_typed(), linear_sampler);
    }

    void set_source_image(coopa::gfx::TextureView color_view, const util::Sampler& linear_sampler) {
        edge_set_->bind_image(0, color_view, linear_sampler);
        neigh_set_->bind_image(0, color_view, linear_sampler);
    }

    void recreate(uint32_t width, uint32_t height, const util::Sampler& linear_sampler) {
        width_  = width;
        height_ = height;
        edges_target_->recreate(width, height);
        blend_target_->recreate(width, height);

        blend_set_->bind_image(0, edges_target_->color_view_typed(), linear_sampler);
        neigh_set_->bind_image(1, blend_target_->color_view_typed(), linear_sampler);
    }

    void draw(coopa::gfx::command::CommandBuffer& cmd,
              targets::OffscreenTarget& output_target,
              float exposure, float threshold, int max_search_steps,
              uint32_t viewport_w, uint32_t viewport_h)
    {
        glm::vec4 rt_metrics(1.0f / viewport_w, 1.0f / viewport_h, static_cast<float>(viewport_w), static_cast<float>(viewport_h));

        const auto both_stages = coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment;

        // Pass 1: Edge Detection
        edges_target_->begin(cmd);
        cmd.bind_pipeline(*edge_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(*edge_set_, 0);

        EdgePush pc_edge{rt_metrics, threshold};
        cmd.push_constants(both_stages, pc_edge);
        cmd.draw(3);
        edges_target_->end(cmd);

        // Pass 2: Blending Weight Calculation
        blend_target_->begin(cmd);
        cmd.bind_pipeline(*blend_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(*blend_set_, 0);

        BlendPush pc_blend{rt_metrics, max_search_steps};
        cmd.push_constants(both_stages, pc_blend);
        cmd.draw(3);
        blend_target_->end(cmd);

        // Pass 3: Neighborhood Blending into output render pass
        output_target.begin(cmd);
        cmd.bind_pipeline(*neigh_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(viewport_w), static_cast<float>(viewport_h));
        cmd.set_scissor(0, 0, viewport_w, viewport_h);
        cmd.bind_descriptor_set(*neigh_set_, 0);

        NeighborhoodPush pc_neigh{rt_metrics, exposure};
        cmd.push_constants(both_stages, pc_neigh);
        cmd.draw(3);
        output_target.end(cmd);
    }

private:
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
