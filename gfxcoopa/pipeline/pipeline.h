/**
 * @file pipeline.h
 * @brief Graphics pipeline creation with OpenGL-like defaults and dynamic state.
 *
 * PipelineConfig provides a builder-style struct for controlling all rasterization
 * state. Pipeline wraps VkPipeline and VkPipelineLayout. set_viewport() and
 * set_scissor() static helpers make dynamic state changes as terse as their
 * OpenGL equivalents.
 */

#ifndef COOPA_GFX_PIPELINE_PIPELINE_H
#define COOPA_GFX_PIPELINE_PIPELINE_H

#include <volk/volk.h>
#include <algorithm>
#include <vector>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace pipeline {

/**
 * @struct PipelineConfig
 * @brief Configurable rasterization state for Pipeline construction.
 *
 * Defaults match OpenGL's out-of-the-box behavior: triangle list, filled
 * polygons, back-face culling, CCW winding, depth test on, no blending.
 */
struct PipelineConfig {
    VkPrimitiveTopology    topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; /**< Input primitive topology. */
    VkPolygonMode          polygon_mode           = VK_POLYGON_MODE_FILL;                /**< Fill, line, or point rendering. */
    VkCullModeFlags        cull_mode              = VK_CULL_MODE_BACK_BIT;               /**< Face culling mode. */
    VkFrontFace            front_face             = VK_FRONT_FACE_COUNTER_CLOCKWISE;     /**< CCW winding is front-facing. */
    bool                   depth_test             = true;                                 /**< Enable depth testing. */
    bool                   depth_write            = true;                                 /**< Enable depth writing. */
    bool                   blending               = false;                                /**< Enable alpha blending. */
    float                  line_width             = 1.0f;                                 /**< Rasterized line width. */
    VkSampleCountFlagBits  samples                = VK_SAMPLE_COUNT_1_BIT;               /**< MSAA sample count. */
    VkCompareOp             depth_compare_op       = VK_COMPARE_OP_LESS;                  /**< Depth comparison function. */
    uint32_t                color_attachment_count = 1;                                   /**< Color attachments sharing this blend state. */
};

/**
 * @class Pipeline
 * @brief RAII wrapper around a Vulkan graphics pipeline and its layout.
 *
 * Builds the full VkGraphicsPipeline from shaders, vertex input layouts,
 * render pass, and a PipelineConfig. Dynamic viewport and scissor state
 * are always enabled so that window resizes do not require pipeline recreation.
 */
class Pipeline {
public:
    /**
     * @brief Creates a graphics pipeline.
     *
     * @param device             The logical device.
     * @param render_pass        The compatible render pass.
     * @param shaders            Pointers to Shader objects (vertex + fragment required).
     * @param vertex_bindings    VkVertexInputBindingDescription array (per-buffer stride).
     * @param vertex_attributes  VkVertexInputAttributeDescription array (per-attribute layout).
     * @param descriptor_layouts Descriptor set layouts used by the shaders.
     * @param config             Rasterization state overrides (defaults are GL-like).
     * @param push_constants     Push constant ranges (e.g. for per-object model matrices).
     */
    Pipeline(core::Device&                                           device,
             RenderPass&                                             render_pass,
             const std::vector<Shader*>&                            shaders,
             const std::vector<VkVertexInputBindingDescription>&    vertex_bindings,
             const std::vector<VkVertexInputAttributeDescription>&  vertex_attributes,
             const std::vector<VkDescriptorSetLayout>&              descriptor_layouts = {},
             const PipelineConfig&                                  config = {},
             const std::vector<VkPushConstantRange>&                push_constants = {})
        : device_(device)
    {
        create_layout(descriptor_layouts, push_constants);
        create_pipeline(render_pass.handle(), shaders, vertex_bindings, vertex_attributes, config);
    }

    /**
     * @brief Creates a graphics pipeline against a raw render pass handle.
     *
     * For callers (e.g. a pass with a hand-built VkRenderPass) that don't own a RenderPass
     * wrapper.
     *
     * @param device             The logical device.
     * @param render_pass        Handle of a compatible render pass.
     * @param shaders            Pointers to Shader objects (vertex + fragment required).
     * @param vertex_bindings    VkVertexInputBindingDescription array (per-buffer stride).
     * @param vertex_attributes  VkVertexInputAttributeDescription array (per-attribute layout).
     * @param descriptor_layouts Descriptor set layouts used by the shaders.
     * @param config             Rasterization state overrides (defaults are GL-like).
     * @param push_constants     Push constant ranges (e.g. for per-object model matrices).
     */
    Pipeline(core::Device&                                           device,
             VkRenderPass                                            render_pass,
             const std::vector<Shader*>&                            shaders,
             const std::vector<VkVertexInputBindingDescription>&    vertex_bindings,
             const std::vector<VkVertexInputAttributeDescription>&  vertex_attributes,
             const std::vector<VkDescriptorSetLayout>&              descriptor_layouts = {},
             const PipelineConfig&                                  config = {},
             const std::vector<VkPushConstantRange>&                push_constants = {})
        : device_(device)
    {
        create_layout(descriptor_layouts, push_constants);
        create_pipeline(render_pass, shaders, vertex_bindings, vertex_attributes, config);
    }

    /**
     * @brief Destroys the pipeline and its layout.
     */
    ~Pipeline() {
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
        if (layout_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
    }

    /// @brief Non-copyable.
    Pipeline(const Pipeline&) = delete;
    /// @brief Non-copyable.
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * @brief Returns the underlying VkPipeline handle.
     * @return Raw VkPipeline.
     */
    VkPipeline handle() const { return pipeline_; }

    /**
     * @brief Returns the VkPipelineLayout (needed for vkCmdBindDescriptorSets).
     * @return Raw VkPipelineLayout.
     */
    VkPipelineLayout layout() const { return layout_; }

    // --- Dynamic state helpers (analogous to glViewport / glScissor) ---

    /**
     * @brief Records a dynamic viewport command into the command buffer.
     *
     * @param cmd Command buffer to record into.
     * @param x   Viewport left edge.
     * @param y   Viewport top edge.
     * @param w   Viewport width.
     * @param h   Viewport height.
     */
    static void set_viewport(VkCommandBuffer cmd, float x, float y, float w, float h) {
        VkViewport vp{};
        vp.x        = x;
        vp.y        = y;
        vp.width    = w;
        vp.height   = h;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
    }

    /**
     * @brief Records a dynamic scissor rectangle command into the command buffer.
     *
     * @param cmd Command buffer to record into.
     * @param x   Scissor left edge.
     * @param y   Scissor top edge.
     * @param w   Scissor width.
     * @param h   Scissor height.
     */
    static void set_scissor(VkCommandBuffer cmd,
                            int32_t  x, int32_t  y,
                            uint32_t w, uint32_t h)
    {
        VkRect2D scissor{};
        scissor.offset = {x, y};
        scissor.extent = {w, h};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

private:
    /**
     * @brief Creates the VkPipelineLayout from the given descriptor set layouts and push constants.
     * @param descriptor_layouts Layouts to include in the pipeline layout.
     * @param push_constants     Push constant ranges.
     */
    void create_layout(const std::vector<VkDescriptorSetLayout>& descriptor_layouts,
                       const std::vector<VkPushConstantRange>&   push_constants = {}) {
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount         = static_cast<uint32_t>(descriptor_layouts.size());
        layout_info.pSetLayouts            = descriptor_layouts.empty() ? nullptr
                                                                        : descriptor_layouts.data();
        layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constants.size());
        layout_info.pPushConstantRanges    = push_constants.empty() ? nullptr : push_constants.data();

        GFX_VK_CHECK(vkCreatePipelineLayout(device_.handle(), &layout_info, nullptr, &layout_));
    }

    /**
     * @brief Creates the VkPipeline from all pipeline state.
     * @param render_pass       Handle of a compatible render pass.
     * @param shaders           Shader stages.
     * @param vertex_bindings   Vertex buffer binding descriptions.
     * @param vertex_attributes Vertex attribute descriptions.
     * @param config            Rasterization overrides.
     */
    void create_pipeline(VkRenderPass                                            render_pass,
                         const std::vector<Shader*>&                            shaders,
                         const std::vector<VkVertexInputBindingDescription>&    vertex_bindings,
                         const std::vector<VkVertexInputAttributeDescription>&  vertex_attributes,
                         const PipelineConfig&                                  config)
    {
        // Shader stages.
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        stages.reserve(shaders.size());
        for (auto* s : shaders) {
            stages.push_back(s->stage_info());
        }

        // Vertex input.
        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount   = static_cast<uint32_t>(vertex_bindings.size());
        vertex_input.pVertexBindingDescriptions      = vertex_bindings.empty() ? nullptr : vertex_bindings.data();
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attributes.size());
        vertex_input.pVertexAttributeDescriptions    = vertex_attributes.empty() ? nullptr : vertex_attributes.data();

        // Input assembly.
        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology               = config.topology;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        // Dynamic viewport/scissor — always enabled so resize is handled without rebuild.
        VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = 2;
        dynamic_state.pDynamicStates    = dynamic_states;

        // Viewport state (actual values set dynamically).
        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount  = 1;

        // Rasterizer.
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = config.polygon_mode;
        rasterizer.cullMode                = config.cull_mode;
        rasterizer.frontFace               = config.front_face;
        rasterizer.depthBiasEnable         = VK_FALSE;
        rasterizer.lineWidth               = config.line_width;

        // Multisampling.
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = config.samples;
        multisampling.sampleShadingEnable  = VK_FALSE;

        // Depth/stencil.
        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable  = config.depth_test  ? VK_TRUE : VK_FALSE;
        depth_stencil.depthWriteEnable = config.depth_write ? VK_TRUE : VK_FALSE;
        depth_stencil.depthCompareOp   = config.depth_compare_op;
        depth_stencil.stencilTestEnable= VK_FALSE;

        // Color blending. The same attachment state is replicated across
        // config.color_attachment_count attachments (defaults to 1).
        VkPipelineColorBlendAttachmentState blend_attachment{};
        if (config.blending) {
            blend_attachment.blendEnable         = VK_TRUE;
            blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
            blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;
        } else {
            blend_attachment.blendEnable = VK_FALSE;
        }
        blend_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(
            std::max(1u, config.color_attachment_count), blend_attachment);

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable   = VK_FALSE;
        color_blending.attachmentCount = static_cast<uint32_t>(blend_attachments.size());
        color_blending.pAttachments    = blend_attachments.data();

        // Final pipeline create info.
        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount          = static_cast<uint32_t>(stages.size());
        pipeline_info.pStages             = stages.data();
        pipeline_info.pVertexInputState   = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState      = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState   = &multisampling;
        pipeline_info.pDepthStencilState  = &depth_stencil;
        pipeline_info.pColorBlendState    = &color_blending;
        pipeline_info.pDynamicState       = &dynamic_state;
        pipeline_info.layout              = layout_;
        pipeline_info.renderPass          = render_pass;
        pipeline_info.subpass             = 0;

        GFX_VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), VK_NULL_HANDLE, 1,
                                               &pipeline_info, nullptr, &pipeline_));
    }

    core::Device&    device_;                      /**< Owning logical device (not owned). */
    VkPipeline       pipeline_ = VK_NULL_HANDLE;   /**< The Vulkan graphics pipeline. */
    VkPipelineLayout layout_   = VK_NULL_HANDLE;   /**< The pipeline layout (descriptor bindings + push constants). */
};

} // namespace pipeline
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_PIPELINE_PIPELINE_H
