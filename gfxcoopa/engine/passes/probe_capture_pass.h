#include <gfxcoopa/engine/targets/cubemap_target.h>
#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file probe_capture_pass.h
 * @brief Renders real scene geometry into one reflection-probe cubemap face.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_PROBE_CAPTURE_PASS_H
#define GFXCOOPA_ENGINE_PASSES_PROBE_CAPTURE_PASS_H

#include <volk/volk.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/data/mesh.h>
#include <gfxcoopa/engine/data/model_ubo.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {




/// Two pipelines, both built against a depth-inclusive render pass (in
/// practice targets::CubemapTarget::render_pass()), used together within a single
/// begin_face_pass()/end_face_pass() instance:
///   1. A sky-background fullscreen triangle (depth off), drawn FIRST to fill
///      every texel with the analytic sky.
///   2. Real scene geometry (depth on), drawn on top -- RenderPass hardcodes
///      LOAD_OP_CLEAR, so the sky fill can't be a separate pass; it has to be
///      the first draw inside the same instance.
///
/// Deliberately NOT built on PbrPipeline/pbr.frag: that shader samples the
/// reflection cubemap recursively, which would be a hazard against the very
/// image this pass is writing, and PbrPipeline's ctor mandates a shadow
/// descriptor set this pass doesn't use (see probe_capture.frag -- no
/// shadows in v1).
class ProbeCapturePass {
public:
    /// 32 bytes -- byte-identical to GBufferPipeline::PushConstants /
    /// TransparentPass::PushConstants. model/normal_matrix moved to the
    /// per-instance vertex stream (data::InstanceData); this block is now
    /// shared once per instanced batch, not pushed per object.
    struct PushConstants {
        glm::vec4 albedo       = {0.8f, 0.8f, 0.8f, 1.0f};
        float     metallic     = 0.0f;
        float     roughness    = 0.5f;
        float     ao           = 1.0f;
        float     alpha_cutoff = 0.0f;
    };

    struct SkyPushConstants {
        int32_t face;
    };

    ProbeCapturePass(coopa::gfx::core::Device& device,
                     coopa::gfx::pipeline::RenderPass& face_pass,
                     VkDescriptorSetLayout camera_layout,
                     VkDescriptorSetLayout light_layout,
                     VkDescriptorSetLayout brdf_layout,
                     const std::string& shader_dir)
        : device_(device)
    {
        // --- Sky background pipeline ---
        sky_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(
            device, shader_dir + "/env_prefilter.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        sky_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(
            device, shader_dir + "/probe_sky_background.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        coopa::gfx::pipeline::PipelineConfig sky_cfg{};
        sky_cfg.cull_mode   = VK_CULL_MODE_NONE;
        sky_cfg.depth_test  = false;
        sky_cfg.depth_write = false;

        VkPushConstantRange sky_pc_range{};
        sky_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        sky_pc_range.offset     = 0;
        sky_pc_range.size       = sizeof(SkyPushConstants);

        sky_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, face_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{sky_vert_.get(), sky_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{},
            sky_cfg,
            std::vector<VkPushConstantRange>{sky_pc_range}
        );

        // --- Geometry pipeline ---
        geom_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(
            device, shader_dir + "/pbr.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        geom_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(
            device, shader_dir + "/probe_capture.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        auto binding          = coopa::gfx::engine::data::Vertex::binding_description();
        auto instance_binding = coopa::gfx::engine::data::InstanceData::binding_description();
        std::vector<VkVertexInputBindingDescription> binding_vec = {binding, instance_binding};

        auto attrs = coopa::gfx::engine::data::Vertex::attribute_descriptions();
        std::vector<VkVertexInputAttributeDescription> attr_vec(attrs.begin(), attrs.end());
        auto instance_attrs = coopa::gfx::engine::data::InstanceData::attribute_descriptions();
        attr_vec.insert(attr_vec.end(), instance_attrs.begin(), instance_attrs.end());

        coopa::gfx::pipeline::PipelineConfig geom_cfg{};
        // CULL_MODE_NONE, not the main pass's back-face/CCW convention: this
        // render pass's projection deliberately omits the Vulkan Y-flip (see
        // targets::CubemapTarget::get_face_projection()), which mirrors framebuffer-
        // space winding relative to the main pass.
        geom_cfg.cull_mode   = VK_CULL_MODE_NONE;
        geom_cfg.depth_test  = true;
        geom_cfg.depth_write = true;

        VkPushConstantRange geom_pc_range{};
        geom_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        geom_pc_range.offset     = 0;
        geom_pc_range.size       = sizeof(PushConstants);

        std::vector<VkDescriptorSetLayout> geom_layouts = { camera_layout, light_layout, brdf_layout };

        geom_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, face_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{geom_vert_.get(), geom_frag_.get()},
            binding_vec,
            attr_vec,
            geom_layouts,
            geom_cfg,
            std::vector<VkPushConstantRange>{geom_pc_range}
        );
    }

    /// Binds the sky pipeline and draws a fullscreen triangle for one face.
    /// Must be called first within a begin_face_pass()/end_face_pass()
    /// instance (depth off, so it doesn't block the geometry draw after it).
    void draw_sky_background(coopa::gfx::command::CommandBuffer& cmd, uint32_t face) const {
        cmd.bind_pipeline(*sky_pipeline_);
        SkyPushConstants pc{ static_cast<int32_t>(face) };
        cmd.push_constants(sky_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(SkyPushConstants), &pc);
        cmd.draw(3);
    }

    /// Binds the geometry pipeline. Caller then binds the camera/light/brdf
    /// descriptor sets (sets 0/1/2) and issues push()/mesh draws per object.
    void bind_geometry(coopa::gfx::command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*geom_pipeline_);
    }

    void push(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& pc) const {
        cmd.push_constants(geom_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
    }

    VkPipelineLayout geometry_layout() const { return geom_pipeline_->layout(); }

private:
    coopa::gfx::core::Device& device_;

    std::unique_ptr<coopa::gfx::pipeline::Shader> sky_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> sky_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> sky_pipeline_;

    std::unique_ptr<coopa::gfx::pipeline::Shader> geom_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> geom_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> geom_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_PROBE_CAPTURE_PASS_H
