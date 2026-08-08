/**
 * @file shadow_pipeline.h
 * @brief Vulkan pipelines for directional and cubemap shadow map depth passes.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SHADOW_PIPELINE_H
#define GFXCOOPA_ENGINE_PASSES_SHADOW_PIPELINE_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/data/mesh.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {




/**
 * @struct DirectionalShadowPushConstants
 * @brief Push constant block for directional shadow depth pass (128 bytes).
 */
struct DirectionalShadowPushConstants {
    glm::mat4 light_space_matrix;
    glm::mat4 model;
};

/**
 * @struct CubeShadowPushConstants
 * @brief Push constant block for point light cubemap shadow pass (144 bytes).
 */
struct CubeShadowPushConstants {
    glm::mat4 light_space_matrix;
    glm::mat4 model;
    glm::vec4 light_pos_range; // xyz = light pos, w = range
};

/**
 * @class ShadowPipeline
 * @brief Manages graphics pipelines for directional & point light shadow map rendering.
 */
class ShadowPipeline {
public:
    ShadowPipeline(core::Device&         device,
                   pipeline::RenderPass& dir_pass,
                   pipeline::RenderPass& cube_pass,
                   const std::string&    dir_vert_spv,
                   const std::string&    dir_frag_spv,
                   const std::string&    cube_vert_spv,
                   const std::string&    cube_frag_spv)
        : device_(device)
    {
        // 1. Directional Shadow Pipeline
        dir_vert_ = std::make_unique<pipeline::Shader>(device, dir_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        dir_frag_ = std::make_unique<pipeline::Shader>(device, dir_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        auto binding = data::Vertex::binding_description();

        // Shadow depth shaders only consume position (location 0).
        // Providing only that attribute eliminates validation warnings about
        // unconsumed locations 1/2/3 for normal, uv, and tangent.
        std::vector<VkVertexInputAttributeDescription> attr_vec = {{
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = offsetof(data::Vertex, position)
        }};

        pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = true;
        cfg.depth_write = true;

        VkPushConstantRange dir_pc{};
        dir_pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        dir_pc.offset     = 0;
        dir_pc.size       = sizeof(DirectionalShadowPushConstants);

        dir_pipeline_ = std::make_unique<pipeline::Pipeline>(
            device, dir_pass,
            std::vector<pipeline::Shader*>{dir_vert_.get(), dir_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{binding},
            attr_vec,
            std::vector<VkDescriptorSetLayout>{},
            cfg,
            std::vector<VkPushConstantRange>{dir_pc}
        );

        // 2. Cube Shadow Pipeline
        cube_vert_ = std::make_unique<pipeline::Shader>(device, cube_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        cube_frag_ = std::make_unique<pipeline::Shader>(device, cube_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        VkPushConstantRange cube_pc{};
        cube_pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        cube_pc.offset     = 0;
        cube_pc.size       = sizeof(CubeShadowPushConstants);

        cube_pipeline_ = std::make_unique<pipeline::Pipeline>(
            device, cube_pass,
            std::vector<pipeline::Shader*>{cube_vert_.get(), cube_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{binding},
            attr_vec,
            std::vector<VkDescriptorSetLayout>{},
            cfg,
            std::vector<VkPushConstantRange>{cube_pc}
        );
    }

    void bind_directional(command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*dir_pipeline_);
    }

    void push_directional(command::CommandBuffer& cmd, const DirectionalShadowPushConstants& pc) const {
        cmd.push_constants(dir_pipeline_->layout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DirectionalShadowPushConstants), &pc);
    }

    void bind_cube(command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*cube_pipeline_);
    }

    void push_cube(command::CommandBuffer& cmd, const CubeShadowPushConstants& pc) const {
        cmd.push_constants(cube_pipeline_->layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CubeShadowPushConstants), &pc);
    }

private:
    core::Device&                       device_;
    std::unique_ptr<pipeline::Shader>   dir_vert_;
    std::unique_ptr<pipeline::Shader>   dir_frag_;
    std::unique_ptr<pipeline::Pipeline> dir_pipeline_;

    std::unique_ptr<pipeline::Shader>   cube_vert_;
    std::unique_ptr<pipeline::Shader>   cube_frag_;
    std::unique_ptr<pipeline::Pipeline> cube_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_SHADOW_PIPELINE_H
