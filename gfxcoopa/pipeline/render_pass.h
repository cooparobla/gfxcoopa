/**
 * @file render_pass.h
 * @brief Vulkan render pass creation with sensible OpenGL-like defaults.
 *
 * A render pass describes the framebuffer attachments and how they are
 * loaded/stored around a draw call. Analogous to binding an FBO in OpenGL.
 * Defaults to a single color attachment + optional depth/stencil.
 */

#ifndef COOPA_GFX_PIPELINE_RENDER_PASS_H
#define COOPA_GFX_PIPELINE_RENDER_PASS_H

#include <volk/volk.h>
#include <vector>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/util/error.h>
#include <gfxcoopa/util/format.h>

namespace coopa {
namespace gfx {
namespace pipeline {

/**
 * @class RenderPass
 * @brief RAII wrapper around VkRenderPass with one color + optional depth attachment.
 *
 * The default configuration mirrors the most common OpenGL FBO setup:
 * - Color attachment: LOAD_OP_CLEAR, STORE_OP_STORE, final layout PRESENT_SRC_KHR.
 * - Depth attachment (optional): LOAD_OP_CLEAR, STORE_OP_DONT_CARE, final layout
 *   DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
 * - Single graphics subpass with implicit external dependencies.
 */
class RenderPass {
public:
    /**
     * @brief Creates a render pass with one color attachment.
     * @param device        The logical device.
     * @param color_format  The VkFormat of the swapchain/color attachment.
     * @param depth_format  The VkFormat of the depth attachment.
     *                      Pass VK_FORMAT_UNDEFINED to disable depth.
     * @throws std::runtime_error if render pass creation fails.
     */
    RenderPass(core::Device& device,
               VkFormat color_format,
               VkFormat depth_format = VK_FORMAT_D32_SFLOAT)
        : device_(device)
    {
        bool has_depth = depth_format != VK_FORMAT_UNDEFINED;

        // --- Attachment descriptions ---
        std::vector<VkAttachmentDescription> attachments;

        // Color attachment.
        VkAttachmentDescription color_attachment{};
        color_attachment.format         = color_format;
        color_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        attachments.push_back(color_attachment);

        // Depth attachment (optional).
        if (has_depth) {
            VkAttachmentDescription depth_attachment{};
            depth_attachment.format         = depth_format;
            depth_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
            depth_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            depth_attachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            attachments.push_back(depth_attachment);
        }

        // --- Subpass references ---
        VkAttachmentReference color_ref{};
        color_ref.attachment = 0;
        color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 1;
        depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &color_ref;
        subpass.pDepthStencilAttachment = has_depth ? &depth_ref : nullptr;

        // --- Subpass dependencies ---
        // Ensures the render pass waits for the swapchain image to be available
        // before writing to the color attachment.
        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   (has_depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0);

        VkRenderPassCreateInfo create_info{};
        create_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        create_info.pAttachments    = attachments.data();
        create_info.subpassCount    = 1;
        create_info.pSubpasses      = &subpass;
        create_info.dependencyCount = 1;
        create_info.pDependencies   = &dependency;

        GFX_VK_CHECK(vkCreateRenderPass(device_.handle(), &create_info, nullptr, &render_pass_));
    }

    /**
     * @brief Destroys the VkRenderPass.
     */
    ~RenderPass() {
        if (render_pass_ != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device_.handle(), render_pass_, nullptr);
        }
    }

    /// @brief Non-copyable.
    RenderPass(const RenderPass&) = delete;
    /// @brief Non-copyable.
    RenderPass& operator=(const RenderPass&) = delete;

    /**
     * @brief Returns the underlying VkRenderPass handle.
     * @return Raw VkRenderPass.
     */
    VkRenderPass handle() const { return render_pass_; }

private:
    core::Device& device_;                           /**< Owning logical device (not owned). */
    VkRenderPass  render_pass_ = VK_NULL_HANDLE;     /**< The Vulkan render pass. */
};

} // namespace pipeline
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_PIPELINE_RENDER_PASS_H
