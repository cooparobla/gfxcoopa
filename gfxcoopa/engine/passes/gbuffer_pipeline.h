#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file gbuffer_pipeline.h
 * @brief Pipeline for rendering geometry into G-Buffer attachments.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_GBUFFER_PIPELINE_H
#define GFXCOOPA_ENGINE_PASSES_GBUFFER_PIPELINE_H

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



class GBufferPipeline {
public:
    // model/normal_matrix used to live here too (160 bytes total) but are
    // now streamed per-instance instead (see data::InstanceData) — this
    // block is shared once per instanced draw batch rather than pushed per
    // object, so only genuinely per-batch material state remains.
    //
    // 48 bytes total. The first 32 bytes (through alpha_cutoff) are byte-identical to
    // TransparentPass::PushConstants / TransparentCapturePass::PushConstants /
    // ProbeCapturePass::PushConstants, which stay at 32 bytes -- emissive is deferred
    // (opaque G-buffer) only, so those forward-path structs deliberately don't grow.
    struct PushConstants {
        glm::vec4 albedo       = {0.8f, 0.8f, 0.8f, 1.0f}; // 16 bytes; .w = alpha
        float     metallic     = 0.0f;
        float     roughness    = 0.5f;
        float     ao           = 1.0f;
        float     alpha_cutoff = 0.0f;                     // 16 bytes; 0.0 = no alpha test
        glm::vec4 emissive     = {0.0f, 0.0f, 0.0f, 0.0f}; // 16 bytes; xyz = pre-multiplied emissive radiance, w reserved
    };

    GBufferPipeline(coopa::gfx::core::Device& device,
                    VkRenderPass render_pass,
                    VkDescriptorSetLayout camera_layout,
                    VkDescriptorSetLayout material_layout,
                    const std::string& vert_spv,
                    const std::string& frag_spv)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layouts
        std::vector<VkDescriptorSetLayout> layouts = { camera_layout };
        if (material_layout != VK_NULL_HANDLE) {
            layouts.push_back(material_layout);
        }

        // Push constant range -- fragment-only now that model/normal_matrix
        // (the only fields the vertex stage used to read) moved to the
        // per-instance vertex stream.
        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount         = static_cast<uint32_t>(layouts.size());
        layout_info.pSetLayouts            = layouts.data();
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges    = &pc_range;

        GFX_VK_CHECK(vkCreatePipelineLayout(device_.handle(), &layout_info, nullptr, &pipeline_layout_));

        // Vertex Input -- binding 0 (per-vertex) + binding 1 (per-instance model matrix).
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

        // Input assembly
        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        // Dynamic State
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

        // Rasterization
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;
        rasterizer.lineWidth               = 1.0f;

        // Multisampling (single-sampled G-Buffer)
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.sampleShadingEnable  = VK_FALSE;

        // Depth / Stencil
        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable  = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp   = VK_COMPARE_OP_LESS;
        depth_stencil.stencilTestEnable= VK_FALSE;

        // Color blending (4 G-Buffer attachments)
        VkPipelineColorBlendAttachmentState blend_attachments[4]{};
        for (int i = 0; i < 4; ++i) {
            blend_attachments[i].blendEnable = VK_FALSE;
            blend_attachments[i].colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        }

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable   = VK_FALSE;
        color_blending.attachmentCount = 4;
        color_blending.pAttachments    = blend_attachments;

        // Shaders
        VkPipelineShaderStageCreateInfo stages[] = {
            vert_shader_->stage_info(),
            frag_shader_->stage_info()
        };

        // Create Graphics Pipeline
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

    ~GBufferPipeline() {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
        }
        if (pipeline_layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_.handle(), pipeline_layout_, nullptr);
        }
    }

    GBufferPipeline(const GBufferPipeline&) = delete;
    GBufferPipeline& operator=(const GBufferPipeline&) = delete;

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

#endif // GFXCOOPA_ENGINE_PASSES_GBUFFER_PIPELINE_H
