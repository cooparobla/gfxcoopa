/**
 * @file pbr_pipeline.h
 * @brief Graphics pipeline for physically-based rendering pass.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_PBR_PIPELINE_H
#define GFXCOOPA_ENGINE_PASSES_PBR_PIPELINE_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/data/mesh.h>
#include <gfxcoopa/engine/data/model_ubo.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class PbrPipeline {
public:
    struct PushConstants {
        coopa::gfx::engine::data::ModelPushConstants model; // 128 bytes
        glm::vec4 albedo       = {0.8f, 0.8f, 0.8f, 1.0f}; // 16 bytes; .w = alpha
        float     metallic     = 0.0f;
        float     roughness    = 0.5f;
        float     ao           = 1.0f;
        float     alpha_cutoff = 0.0f;                     // 16 bytes; 0.0 = no alpha test
    };

    PbrPipeline(coopa::gfx::core::Device& device,
                coopa::gfx::pipeline::RenderPass& render_pass,
                VkDescriptorSetLayout camera_layout,
                VkDescriptorSetLayout light_layout,
                VkDescriptorSetLayout shadow_layout,
                VkDescriptorSetLayout material_layout,
                const std::string& vert_spv,
                const std::string& frag_spv,
                VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_4_BIT)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        auto binding = coopa::gfx::engine::data::Vertex::binding_description();
        auto attrs   = coopa::gfx::engine::data::Vertex::attribute_descriptions();
        std::vector<VkVertexInputAttributeDescription> attr_vec(attrs.begin(), attrs.end());

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_BACK_BIT;
        cfg.front_face  = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        cfg.depth_test  = true;
        cfg.depth_write = true;
        cfg.samples     = samples;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        std::vector<VkDescriptorSetLayout> layouts = {
            camera_layout,
            light_layout,
            shadow_layout
        };
        if (material_layout != VK_NULL_HANDLE) {
            layouts.push_back(material_layout);
        }

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, render_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{binding},
            attr_vec,
            layouts,
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
    }

    void bind(coopa::gfx::command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*pipeline_);
    }

    void push(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& pc) const {
        cmd.push_constants(pipeline_->layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
    }

    VkPipelineLayout layout() const {
        return pipeline_->layout();
    }

private:
    coopa::gfx::core::Device& device_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_PBR_PIPELINE_H
