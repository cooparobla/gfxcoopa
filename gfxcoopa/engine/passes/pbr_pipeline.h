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
                const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
                const coopa::gfx::pipeline::DescriptorSetLayout& light_layout,
                const coopa::gfx::pipeline::DescriptorSetLayout& shadow_layout,
                const coopa::gfx::pipeline::DescriptorSetLayout* material_layout,
                const std::string& vert_spv,
                const std::string& frag_spv)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::engine::data::Vertex::layout();
        desc.raster.cull  = coopa::gfx::CullMode::Back;
        desc.raster.front = coopa::gfx::FrontFace::CounterClockwise;
        desc.depth.test  = true;
        desc.depth.write = true;
        // samples: no longer a caller-set param -- the sealed Pipeline ctor reads it from
        // `render_pass` itself (RenderPass::samples()), which can never disagree.

        desc.descriptor_layouts = {&camera_layout, &light_layout, &shadow_layout};
        if (material_layout) {
            desc.descriptor_layouts.push_back(material_layout);
        }
        desc.push_constants = {{coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment,
                                0, sizeof(PushConstants)}};

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, render_pass, desc);
    }

    void bind(coopa::gfx::command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*pipeline_);
    }

    void push(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& pc) const {
        cmd.push_constants(coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment, pc);
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
