/**
 * @file command_buffer.h
 * @brief High-level command buffer recording wrapper.
 *
 * Wraps a raw VkCommandBuffer with OpenGL-like named methods:
 * begin/end for recording lifetime, begin_render_pass/end_render_pass for
 * render scope, bind_pipeline/bind_vertex_buffer/draw for draw commands.
 *
 * CommandBuffer does NOT own the underlying VkCommandBuffer — it is a
 * non-owning view that wraps a buffer allocated from a CommandPool.
 */

#ifndef COOPA_GFX_COMMAND_COMMAND_BUFFER_H
#define COOPA_GFX_COMMAND_COMMAND_BUFFER_H

#include <volk/volk.h>
#include <vector>
#include <array>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace command {

/**
 * @class CommandBuffer
 * @brief Non-owning wrapper around VkCommandBuffer providing named recording methods.
 *
 * Intended to be used as a short-lived object during a frame:
 * @code
 * CommandBuffer cmd(raw_vk_cmd);
 * cmd.begin();
 * cmd.begin_render_pass(render_pass, framebuffer, extent, clear_color);
 * cmd.bind_pipeline(pipeline);
 * cmd.bind_vertex_buffer(vertex_buffer);
 * cmd.set_viewport(0, 0, w, h);
 * cmd.draw(3); // triangle
 * cmd.end_render_pass();
 * cmd.end();
 * @endcode
 */
class CommandBuffer {
public:
    /**
     * @brief Constructs a CommandBuffer view around an existing VkCommandBuffer.
     *
     * The raw command buffer must have been allocated from a CommandPool and
     * must remain valid for the lifetime of this wrapper.
     *
     * @param cmd The raw VkCommandBuffer to wrap.
     */
    explicit CommandBuffer(VkCommandBuffer cmd) : cmd_(cmd) {}

    // --- Recording lifecycle ---

    /**
     * @brief Begins command buffer recording (analogous to starting a GL display list).
     *
     * @param one_time_submit If true, hints that the buffer will be submitted once
     *                        (VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).
     * @throws std::runtime_error if vkBeginCommandBuffer fails.
     */
    void begin(bool one_time_submit = false) {
        VkCommandBufferBeginInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags = one_time_submit ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
        GFX_VK_CHECK(vkBeginCommandBuffer(cmd_, &info));
    }

    /**
     * @brief Ends command buffer recording.
     * @throws std::runtime_error if vkEndCommandBuffer fails.
     */
    void end() {
        GFX_VK_CHECK(vkEndCommandBuffer(cmd_));
    }

    // --- Render pass ---

    /**
     * @brief Begins a render pass and sets the clear values.
     *
     * @param render_pass   The render pass to begin.
     * @param framebuffer   The framebuffer to render into.
     * @param extent        Render area (typically the swapchain extent).
     * @param clear_color   RGBA clear color for the color attachment.
     * @param clear_depth   Depth clear value (default 1.0f = far plane).
     * @param clear_stencil Stencil clear value.
     */
    void begin_render_pass(VkRenderPass        render_pass,
                           VkFramebuffer       framebuffer,
                           VkExtent2D          extent,
                           VkClearColorValue   clear_color   = {{0.0f, 0.0f, 0.0f, 1.0f}},
                           float               clear_depth   = 1.0f,
                           uint32_t            clear_stencil = 0)
    {
        std::array<VkClearValue, 2> clear_values{};
        clear_values[0].color        = clear_color;
        clear_values[1].depthStencil = { clear_depth, clear_stencil };

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = render_pass;
        rp_info.framebuffer       = framebuffer;
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = extent;
        rp_info.clearValueCount   = static_cast<uint32_t>(clear_values.size());
        rp_info.pClearValues      = clear_values.data();

        vkCmdBeginRenderPass(cmd_, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
    }

    /**
     * @brief Ends the current render pass.
     */
    void end_render_pass() {
        vkCmdEndRenderPass(cmd_);
    }

    // --- Pipeline binding ---

    /**
     * @brief Binds a graphics pipeline (analogous to glUseProgram).
     * @param pipeline The pipeline to bind.
     */
    void bind_pipeline(const pipeline::Pipeline& p) {
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p.handle());
    }

    /**
     * @brief Binds a vertex buffer at binding slot 0 (analogous to glBindVertexArray + glBindBuffer).
     * @param buffer     The vertex buffer to bind.
     * @param offset     Byte offset into the buffer.
     * @param binding    Vertex input binding slot index.
     */
    void bind_vertex_buffer(const memory::Buffer& buffer,
                            VkDeviceSize offset  = 0,
                            uint32_t     binding = 0)
    {
        VkBuffer     raw  = buffer.handle();
        VkDeviceSize off  = offset;
        vkCmdBindVertexBuffers(cmd_, binding, 1, &raw, &off);
    }

    /**
     * @brief Binds an index buffer (analogous to glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ...)).
     * @param buffer    The index buffer to bind.
     * @param index_type VK_INDEX_TYPE_UINT16 or VK_INDEX_TYPE_UINT32.
     * @param offset    Byte offset into the buffer.
     */
    void bind_index_buffer(const memory::Buffer& buffer,
                           VkIndexType index_type = VK_INDEX_TYPE_UINT32,
                           VkDeviceSize offset    = 0)
    {
        vkCmdBindIndexBuffer(cmd_, buffer.handle(), offset, index_type);
    }

    /**
     * @brief Binds a descriptor set to the graphics pipeline (analogous to glBindBufferBase/glBindTexture).
     * @param pipeline_layout The pipeline layout that owns the descriptor set layout.
     * @param set             The descriptor set to bind.
     * @param set_index       The descriptor set index (binding point).
     */
    void bind_descriptor_set(VkPipelineLayout              pipeline_layout,
                             const pipeline::DescriptorSet& set,
                             uint32_t                       set_index = 0)
    {
        VkDescriptorSet raw = set.handle();
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout, set_index, 1, &raw, 0, nullptr);
    }

    /**
     * @brief Records a push constant update (analogous to setting a uniform directly).
     *
     * @param pipeline_layout The pipeline layout that declared the push constant range.
     * @param stages          Shader stages that read this push constant.
     * @param offset          Byte offset into the push constant block.
     * @param size            Number of bytes to update.
     * @param data            Pointer to the data to upload.
     */
    void push_constants(VkPipelineLayout   pipeline_layout,
                        VkShaderStageFlags stages,
                        uint32_t           offset,
                        uint32_t           size,
                        const void*        data)
    {
        vkCmdPushConstants(cmd_, pipeline_layout, stages, offset, size, data);
    }

    // --- Draw calls ---

    /**
     * @brief Records a non-indexed draw call (analogous to glDrawArrays).
     * @param vertex_count Number of vertices to draw.
     * @param first_vertex Index of the first vertex.
     * @param instance_count Number of instances (default 1).
     */
    void draw(uint32_t vertex_count,
              uint32_t first_vertex   = 0,
              uint32_t instance_count = 1)
    {
        vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, 0);
    }

    /**
     * @brief Records an indexed draw call (analogous to glDrawElements).
     * @param index_count    Number of indices to draw.
     * @param first_index    Offset into the index buffer.
     * @param vertex_offset  Offset added to each index value.
     * @param instance_count Number of instances (default 1).
     * @param first_instance First instance ID; offsets into any bound
     *   VK_VERTEX_INPUT_RATE_INSTANCE stream (gl_InstanceIndex already
     *   includes this — never use gl_InstanceIndex to index anything else).
     *   Affects instance-rate vertex attribute fetching in core Vulkan with
     *   no feature flag required (the drawIndirectFirstInstance device
     *   feature applies only to *indirect* draws, a common confusion).
     */
    void draw_indexed(uint32_t index_count,
                      uint32_t first_index    = 0,
                      int32_t  vertex_offset  = 0,
                      uint32_t instance_count = 1,
                      uint32_t first_instance = 0)
    {
        vkCmdDrawIndexed(cmd_, index_count, instance_count, first_index, vertex_offset, first_instance);
    }

    // --- Dynamic state ---

    /**
     * @brief Sets the viewport dynamically (analogous to glViewport).
     * @param x Left edge.
     * @param y Top edge.
     * @param w Width.
     * @param h Height.
     */
    void set_viewport(float x, float y, float w, float h) {
        pipeline::Pipeline::set_viewport(cmd_, x, y, w, h);
    }

    /**
     * @brief Sets the scissor rectangle dynamically (analogous to glScissor).
     * @param x Left edge.
     * @param y Top edge.
     * @param w Width.
     * @param h Height.
     */
    void set_scissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
        pipeline::Pipeline::set_scissor(cmd_, x, y, w, h);
    }

    /**
     * @brief Returns the raw VkCommandBuffer handle.
     * @return Raw VkCommandBuffer.
     */
    VkCommandBuffer handle() const { return cmd_; }

private:
    VkCommandBuffer cmd_ = VK_NULL_HANDLE; /**< The raw command buffer (not owned). */
};

} // namespace command
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_COMMAND_COMMAND_BUFFER_H
