/**
 * @file toon_pipeline.h
 * @brief Cel/toon shading graphics pipeline for the retro render pass.
 *
 * Wraps a gfxcoopa Pipeline configured for:
 *   - Vertex/fragment shaders: toon.vert / toon.frag
 *   - Standard back-face culling
 *   - Depth test + depth write enabled
 *   - Two descriptor sets: camera (set 0), light (set 1)
 *   - Push constants: ModelPushConstants (128 bytes = model + normal matrix)
 */

#ifndef COOPA_GFX_ENGINE_TOON_PIPELINE_H
#define COOPA_GFX_ENGINE_TOON_PIPELINE_H

#include <volk/volk.h>
#include <memory>
#include <string>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/mesh.h>
#include <gfxcoopa/engine/model_ubo.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @class ToonPipeline
 * @brief Cel-shading pipeline for rendering scene meshes.
 *
 * Usage:
 * @code
 * ToonPipeline toon(device, offscreen_target.render_pass_object(),
 *                   camera_layout, light_layout,
 *                   "assets/shaders/toon.vert.spv",
 *                   "assets/shaders/toon.frag.spv");
 * // Per frame:
 * toon.bind(cmd);
 * toon.push_model(cmd, model_push);
 * mesh->draw(cmd);
 * @endcode
 */
class ToonPipeline {
public:
    /**
     * @brief Creates the toon shading pipeline.
     *
     * @param device          Logical device.
     * @param render_pass     The offscreen render pass this pipeline renders into.
     * @param camera_layout   Descriptor set layout for set 0 (camera UBO).
     * @param light_layout    Descriptor set layout for set 1 (light UBO).
     * @param vert_spv_path   Path to compiled toon.vert.spv.
     * @param frag_spv_path   Path to compiled toon.frag.spv.
     */
    ToonPipeline(core::Device&                    device,
                 pipeline::RenderPass&            render_pass,
                 pipeline::DescriptorSetLayout&   camera_layout,
                 pipeline::DescriptorSetLayout&   light_layout,
                 pipeline::DescriptorSetLayout&   shadow_layout,
                 const std::string&               vert_spv_path,
                 const std::string&               frag_spv_path)
        : device_(device)
    {
        // Load shaders.
        vert_shader_ = std::make_unique<pipeline::Shader>(device, vert_spv_path, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<pipeline::Shader>(device, frag_spv_path, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex input.
        auto binding = Vertex::binding_description();
        auto attrs   = Vertex::attribute_descriptions();
        std::vector<VkVertexInputAttributeDescription> attr_vec(attrs.begin(), attrs.end());

        // Pipeline config: back-face cull (standard).
        pipeline::PipelineConfig cfg{};
        cfg.cull_mode = VK_CULL_MODE_BACK_BIT;

        // Push constant range: model + normal matrix (128 bytes, both stages).
        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(ModelPushConstants); // 128 bytes

        std::vector<VkDescriptorSetLayout> layouts = {
            camera_layout.handle(),
            light_layout.handle(),
            shadow_layout.handle()
        };

        pipeline_ = std::make_unique<pipeline::Pipeline>(
            device, render_pass,
            std::vector<pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{binding},
            attr_vec,
            layouts,
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
    }

    /**
     * @brief Binds this pipeline to the command buffer.
     * @param cmd The command buffer to bind into.
     */
    void bind(command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*pipeline_);
    }

    /**
     * @brief Uploads per-object model data as push constants.
     * @param cmd   Command buffer to record into.
     * @param model The model push constant data.
     */
    void push_model(command::CommandBuffer& cmd, const ModelPushConstants& model) const {
        cmd.push_constants(
            pipeline_->layout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(ModelPushConstants),
            &model
        );
    }

    /** @brief Returns the underlying pipeline layout (for descriptor binding). */
    VkPipelineLayout layout() const { return pipeline_->layout(); }

private:
    core::Device&                        device_;       /**< Logical device (not owned). */
    std::unique_ptr<pipeline::Shader>    vert_shader_;  /**< Toon vertex shader. */
    std::unique_ptr<pipeline::Shader>    frag_shader_;  /**< Toon fragment shader. */
    std::unique_ptr<pipeline::Pipeline>  pipeline_;     /**< The Vulkan graphics pipeline. */
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_TOON_PIPELINE_H
