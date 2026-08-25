#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file transparent_capture_pass.h
 * @brief Forward-shaded, depth-tested capture of transparent geometry -- a second reflection
 *        SOURCE for opaque reflectors' SSR raymarch, not a visible draw of its own.
 *
 * Unlike TransparentPass (transparent_pass.h), which alpha-blends into the live HDR frame and
 * borrows the opaque G-Buffer's depth read-only, this pass owns and WRITES its own depth
 * (standard opaque-style test: front-most transparent surface wins per pixel, among
 * transparent objects only -- see transparent_capture_target.h's doc for why no cross-test
 * against the opaque G-Buffer is needed here) and has no blending on ANY of its three color
 * outputs. That -- no blending anywhere -- is what makes a hand-rolled pipeline like
 * GBufferPipeline's the right template rather than pipeline::Pipeline's PipelineConfig, which
 * replicates ONE blend state across every color attachment (a real blocker for bolting MRT
 * onto TransparentPass itself, irrelevant here since every attachment wants the same
 * blendEnable=false GBufferPipeline already writes).
 *
 * Points its own render pass at an externally-owned VkRenderPass (TransparentCaptureTarget's),
 * the same convention GBufferPipeline uses with GBufferTarget.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_TRANSPARENT_CAPTURE_PASS_H
#define GFXCOOPA_ENGINE_PASSES_TRANSPARENT_CAPTURE_PASS_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/data/mesh.h>
#include <gfxcoopa/engine/data/model_ubo.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class TransparentCapturePass {
public:
    /// 32 bytes -- byte-identical to TransparentPass::PushConstants (alpha_cutoff/alpha are
    /// carried along unused: this pass never blends or alpha-tests, front-most surface always
    /// wins by depth, but keeping the layout identical lets a caller build one PushConstants
    /// value and push it into both this pass and TransparentPass with no translation).
    struct PushConstants {
        glm::vec4 albedo       = {0.8f, 0.8f, 0.8f, 1.0f}; // 16 bytes; .w = alpha, unused here
        float     metallic     = 0.0f;
        float     roughness    = 0.5f;
        float     ao           = 1.0f;
        float     alpha_cutoff = 0.0f;                     // unused
    };

    /**
     * @brief Creates the pass's pipeline against an externally-owned render pass.
     *
     * @param device        Logical device.
     * @param render_pass   Owned by TransparentCaptureTarget -- 3 color attachments (no
     *                      blending) + a depth attachment this pass tests AND writes.
     * @param camera_layout Set 0.
     * @param light_layout  Set 1.
     * @param shadow_layout Set 2.
     * @param vert_spv      Vertex shader (reuses pbr.vert.spv, same convention as
     *                      TransparentPass).
     * @param frag_spv      Fragment shader.
     * @param extra_pc_bytes Additional bytes past PushConstants' own [0, sizeof(PushConstants))
     *                      for frame-level lighting-mode data (e.g. toyengine's
     *                      soft_lighting/light_bands/spec_threshold/rim_strength/
     *                      ambient_intensity/sky_intensity), folded into ONE fragment-stage
     *                      VkPushConstantRange -- same reasoning as TransparentPass's own
     *                      extra_pc_bytes ctor param.
     */
    TransparentCapturePass(coopa::gfx::core::Device& device,
                           VkRenderPass render_pass,
                           VkDescriptorSetLayout camera_layout,
                           VkDescriptorSetLayout light_layout,
                           VkDescriptorSetLayout shadow_layout,
                           const std::string& vert_spv,
                           const std::string& frag_spv,
                           uint32_t extra_pc_bytes = 0)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDescriptorSetLayout> layouts = { camera_layout, light_layout, shadow_layout };

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants) + extra_pc_bytes;

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount         = static_cast<uint32_t>(layouts.size());
        layout_info.pSetLayouts            = layouts.data();
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges    = &pc_range;

        GFX_VK_CHECK(vkCreatePipelineLayout(device_.handle(), &layout_info, nullptr, &pipeline_layout_));

        // Vertex input -- binding 0 (per-vertex) + binding 1 (per-instance model matrix), same
        // as TransparentPass/GBufferPipeline (this pass also reuses pbr.vert).
        auto binding          = coopa::gfx::engine::data::Vertex::binding_description();
        auto instance_binding = coopa::gfx::engine::data::InstanceData::binding_description();
        std::vector<VkVertexInputBindingDescription> binding_vec = {binding, instance_binding};

        auto attrs = coopa::gfx::engine::data::Vertex::attribute_descriptions();
        std::vector<VkVertexInputAttributeDescription> attr_vec(attrs.begin(), attrs.end());
        auto instance_attrs = coopa::gfx::engine::data::InstanceData::attribute_descriptions();
        attr_vec.insert(attr_vec.end(), instance_attrs.begin(), instance_attrs.end());

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount   = static_cast<uint32_t>(binding_vec.size());
        vertex_input.pVertexBindingDescriptions      = binding_vec.data();
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attr_vec.size());
        vertex_input.pVertexAttributeDescriptions    = attr_vec.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = 2;
        dynamic_state.pDynamicStates    = dynamic_states;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;
        rasterizer.lineWidth               = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.sampleShadingEnable  = VK_FALSE;

        // Depth: standard opaque-style test AND write -- front-most transparent surface wins
        // per pixel, among transparent objects only (see file doc).
        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType             = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable   = VK_TRUE;
        depth_stencil.depthWriteEnable  = VK_TRUE;
        depth_stencil.depthCompareOp    = VK_COMPARE_OP_LESS;
        depth_stencil.stencilTestEnable = VK_FALSE;

        // Color blending: 3 attachments, NONE blended -- see file doc for why this pass (unlike
        // TransparentPass) never needs pipeline::PipelineConfig's single replicated blend state.
        VkPipelineColorBlendAttachmentState blend_attachments[3]{};
        for (int i = 0; i < 3; ++i) {
            blend_attachments[i].blendEnable = VK_FALSE;
            blend_attachments[i].colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        }

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable   = VK_FALSE;
        color_blending.attachmentCount = 3;
        color_blending.pAttachments    = blend_attachments;

        VkPipelineShaderStageCreateInfo stages[] = {
            vert_shader_->stage_info(),
            frag_shader_->stage_info()
        };

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount          = 2;
        pipeline_info.pStages             = stages;
        pipeline_info.pVertexInputState   = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState      = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState   = &multisampling;
        pipeline_info.pDepthStencilState  = &depth_stencil;
        pipeline_info.pColorBlendState    = &color_blending;
        pipeline_info.pDynamicState       = &dynamic_state;
        pipeline_info.layout              = pipeline_layout_;
        pipeline_info.renderPass          = render_pass;
        pipeline_info.subpass             = 0;

        GFX_VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_));
    }

    ~TransparentCapturePass() {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
        }
        if (pipeline_layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_.handle(), pipeline_layout_, nullptr);
        }
    }

    TransparentCapturePass(const TransparentCapturePass&) = delete;
    TransparentCapturePass& operator=(const TransparentCapturePass&) = delete;

    void bind(coopa::gfx::command::CommandBuffer& cmd) const {
        vkCmdBindPipeline(cmd.handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    }

    void push(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& pc) const {
        cmd.push_constants(pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
    }

    VkPipelineLayout layout() const {
        return pipeline_layout_;
    }

private:
    coopa::gfx::core::Device& device_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> frag_shader_;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_        = VK_NULL_HANDLE;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_TRANSPARENT_CAPTURE_PASS_H
