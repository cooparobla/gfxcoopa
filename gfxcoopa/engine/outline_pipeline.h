/**
 * @file outline_pipeline.h
 * @brief Inverted-hull outline pipeline using front-face culling.
 *
 * Renders the mesh a second pass with front-face culling (back-faces only).
 * Vertices are extruded along normals by outline_width (via push constants).
 * The result is a solid-color "shell" visible around the object's silhouette.
 *
 * Push constant layout (total 144 bytes):
 *   [0..127]:   ModelPushConstants (model + normal_matrix)
 *   [128]:      float outline_width
 *   [132..143]: vec3 outline_color (rgb)
 */

#ifndef COOPA_GFX_ENGINE_OUTLINE_PIPELINE_H
#define COOPA_GFX_ENGINE_OUTLINE_PIPELINE_H

#include <volk/volk.h>
#include <glm/glm.hpp>
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
 * @struct OutlinePushConstants
 * @brief Full push constant block for the outline pipeline.
 *
 * Layout: model (128 bytes) + outline_params vec4 (16 bytes) = 144 bytes.
 * outline_params: xyz = outline RGB color, w = extrusion width in world units.
 */
struct OutlinePushConstants {
    ModelPushConstants model_data;    /**< Model + normal matrix (128 bytes). */
    glm::vec4 outline_params;         /**< xyz=outline_color, w=outline_width. */

    OutlinePushConstants()
        : outline_params(0.0f, 0.0f, 0.0f, 0.04f) // Default: black outline, 0.04 units wide
    {}

    /** @brief Convenience setter for color and width. */
    void set_outline(const glm::vec3& color, float width) {
        outline_params = glm::vec4(color, width);
    }
};

/**
 * @class OutlinePipeline
 * @brief Inverted-hull outline pipeline (front-face cull, extrude-along-normal).
 *
 * Usage:
 * @code
 * OutlinePipeline outline(device, render_pass, camera_layout, vert_spv, frag_spv);
 * // Per frame (after toon pass):
 * outline.bind(cmd);
 * outline.push(cmd, push_data);
 * mesh->draw(cmd);
 * @endcode
 */
class OutlinePipeline {
public:
    /**
     * @brief Creates the outline pipeline with front-face culling.
     *
     * @param device        Logical device.
     * @param render_pass   Offscreen render pass.
     * @param camera_layout Camera descriptor set layout (set 0).
     * @param vert_spv_path Path to outline.vert.spv.
     * @param frag_spv_path Path to outline.frag.spv.
     */
    OutlinePipeline(core::Device&                  device,
                    pipeline::RenderPass&          render_pass,
                    pipeline::DescriptorSetLayout& camera_layout,
                    const std::string&             vert_spv_path,
                    const std::string&             frag_spv_path)
        : device_(device)
    {
        vert_shader_ = std::make_unique<pipeline::Shader>(device, vert_spv_path, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<pipeline::Shader>(device, frag_spv_path, VK_SHADER_STAGE_FRAGMENT_BIT);

        auto binding = Vertex::binding_description();
        auto attrs   = Vertex::attribute_descriptions();
        std::vector<VkVertexInputAttributeDescription> attr_vec(attrs.begin(), attrs.end());

        // KEY: front-face culling → only back faces rendered → inverted hull outline.
        pipeline::PipelineConfig cfg{};
        cfg.cull_mode = VK_CULL_MODE_FRONT_BIT;

        // Push constant range: model (128 bytes) + outline params (16 bytes) = 144 bytes.
        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(OutlinePushConstants); // 144 bytes

        pipeline_ = std::make_unique<pipeline::Pipeline>(
            device, render_pass,
            std::vector<pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{binding},
            attr_vec,
            std::vector<VkDescriptorSetLayout>{camera_layout.handle()},
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
    }

    /**
     * @brief Binds the outline pipeline.
     * @param cmd Command buffer.
     */
    void bind(command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*pipeline_);
    }

    /**
     * @brief Uploads the full outline push constants (model + outline params).
     * @param cmd  Command buffer.
     * @param push Outline push constant data.
     */
    void push(command::CommandBuffer& cmd, const OutlinePushConstants& push) const {
        cmd.push_constants(
            pipeline_->layout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(OutlinePushConstants),
            &push
        );
    }

    /** @brief Returns the pipeline layout. */
    VkPipelineLayout layout() const { return pipeline_->layout(); }

private:
    core::Device&                       device_;
    std::unique_ptr<pipeline::Shader>   vert_shader_;
    std::unique_ptr<pipeline::Shader>   frag_shader_;
    std::unique_ptr<pipeline::Pipeline> pipeline_;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_OUTLINE_PIPELINE_H
