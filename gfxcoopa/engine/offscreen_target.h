/**
 * @file offscreen_target.h
 * @brief Low-resolution render target for the retro rendering pipeline.
 *
 * Creates a color attachment (VK_FORMAT_R8G8B8A8_UNORM) and depth attachment
 * at a configurable resolution, with a matching RenderPass that transitions the
 * color image to SHADER_READ_ONLY_OPTIMAL after rendering so it can be sampled
 * by subsequent passes (post-processing, upscale).
 *
 * The depth image is also exposed as an image view so edge detection shaders can
 * read it for depth-discontinuity calculations.
 */

#ifndef COOPA_GFX_ENGINE_OFFSCREEN_TARGET_H
#define COOPA_GFX_ENGINE_OFFSCREEN_TARGET_H

#include <volk/volk.h>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @class OffscreenTarget
 * @brief Owns a low-resolution color + depth render target.
 *
 * The color image is transitioned to SHADER_READ_ONLY_OPTIMAL after the render pass,
 * ready to be bound as a combined image sampler in the upscale pass.
 *
 * Usage:
 * @code
 * OffscreenTarget target(device, allocator, 320, 240);
 *
 * // Per frame:
 * target.begin(cmd);
 *   // Record toon + outline draw calls here...
 * target.end(cmd);
 *
 * // Now target.color_view() can be bound to a sampler for the upscale pass.
 * @endcode
 */
class OffscreenTarget {
public:
    /**
     * @brief Creates the offscreen target at the given resolution.
     *
     * @param device       Vulkan logical device.
     * @param allocator    VMA allocator.
     * @param width        Render width in pixels.
     * @param height       Render height in pixels.
     * @param color_format Color attachment pixel format.
     * @param samples      MSAA sample count.
     */
    OffscreenTarget(core::Device&         device,
                    memory::Allocator&    allocator,
                    uint32_t              width,
                    uint32_t              height,
                    VkFormat              color_format = VK_FORMAT_R8G8B8A8_UNORM,
                    VkSampleCountFlagBits samples      = VK_SAMPLE_COUNT_1_BIT)
        : device_(device), allocator_(allocator),
          width_(width), height_(height),
          color_format_(color_format), samples_(samples)
    {
        create_resources_();
    }

    /**
     * @brief Destroys the framebuffer and resources.
     */
    ~OffscreenTarget() {
        destroy_framebuffer_();
    }

    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    // --- Frame recording ---

    /**
     * @brief Begins the offscreen render pass.
     *
     * Records begin_render_pass with a black clear color and far-depth clear.
     * Sets dynamic viewport and scissor to the full render resolution.
     *
     * @param cmd Command buffer to record into.
     */
    void begin(command::CommandBuffer& cmd) const {
        cmd.begin_render_pass(
            render_pass_->handle(),
            framebuffer_,
            {width_, height_},
            {{0.0f, 0.0f, 0.0f, 1.0f}},
            1.0f
        );
        cmd.set_viewport(0.0f, 0.0f,
                         static_cast<float>(width_),
                         static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);
    }

    /**
     * @brief Ends the offscreen render pass.
     * @param cmd Command buffer to record into.
     */
    void end(command::CommandBuffer& cmd) const {
        cmd.end_render_pass();
    }

    // --- Resolution management ---

    /**
     * @brief Recreates the target at a new resolution.
     *
     * Safe to call between frames (after vkDeviceWaitIdle).
     * Pipelines referencing this target must also be recreated.
     *
     * @param width  New render width.
     * @param height New render height.
     */
    void recreate(uint32_t width, uint32_t height) {
        width_  = width;
        height_ = height;
        destroy_framebuffer_();
        color_image_.reset();
        resolve_image_.reset();
        depth_image_.reset();
        render_pass_.reset();
        create_resources_();
    }

    // --- Accessors ---

    /** @brief Returns the render width in pixels. */
    uint32_t width()  const { return width_; }
    /** @brief Returns the render height in pixels. */
    uint32_t height() const { return height_; }

    /** @brief Returns the color image view for sampling in subsequent passes. */
    VkImageView color_view() const {
        return (samples_ > VK_SAMPLE_COUNT_1_BIT && resolve_image_)
            ? resolve_image_->view()
            : color_image_->view();
    }
    /** @brief Returns the depth image view (for edge detection shaders). */
    VkImageView depth_view() const { return depth_image_->view(); }

    /** @brief Returns the render pass handle. */
    VkRenderPass render_pass() const { return render_pass_->handle(); }

    /** @brief Returns the pipeline::RenderPass object for Pipeline construction. */
    pipeline::RenderPass& render_pass_object() const { return *render_pass_; }

private:
    /**
     * @brief Creates all resources: color image, resolve image (if MSAA), depth image, render pass, framebuffer.
     */
    void create_resources_() {
        bool is_msaa = samples_ > VK_SAMPLE_COUNT_1_BIT;

        // Color attachment: used as color attachment (+ sampled if not MSAA).
        VkImageUsageFlags color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (!is_msaa) {
            color_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }

        color_image_ = std::make_unique<memory::Image>(
            device_, allocator_,
            width_, height_,
            color_format_,
            color_usage,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VMA_MEMORY_USAGE_AUTO,
            samples_
        );

        if (is_msaa) {
            // Resolve attachment: single-sampled image that receives MSAA resolve output.
            resolve_image_ = std::make_unique<memory::Image>(
                device_, allocator_,
                width_, height_,
                color_format_,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VK_SAMPLE_COUNT_1_BIT
            );
        }

        // Depth attachment: D32_SFLOAT.
        depth_image_ = std::make_unique<memory::Image>(
            device_, allocator_,
            width_, height_,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VMA_MEMORY_USAGE_AUTO,
            samples_
        );

        render_pass_ = std::make_unique<pipeline::RenderPass>(
            device_,
            color_format_,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   // color final
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   // depth final
            samples_
        );

        create_framebuffer_();
    }

    /**
     * @brief Creates the VkFramebuffer from the current images and render pass.
     */
    void create_framebuffer_() {
        std::vector<VkImageView> attachments;
        attachments.push_back(color_image_->view());
        attachments.push_back(depth_image_->view());
        if (samples_ > VK_SAMPLE_COUNT_1_BIT && resolve_image_) {
            attachments.push_back(resolve_image_->view());
        }

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass_->handle();
        fb_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        fb_info.pAttachments    = attachments.data();
        fb_info.width           = width_;
        fb_info.height          = height_;
        fb_info.layers          = 1;

        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &framebuffer_));
    }

    /**
     * @brief Destroys the VkFramebuffer (called before recreating or on destruction).
     */
    void destroy_framebuffer_() {
        if (framebuffer_ != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), framebuffer_, nullptr);
            framebuffer_ = VK_NULL_HANDLE;
        }
    }

    core::Device&      device_;    /**< Logical device (not owned). */
    memory::Allocator& allocator_; /**< VMA allocator (not owned). */
    uint32_t           width_;     /**< Render width in pixels. */
    uint32_t           height_;    /**< Render height in pixels. */
    VkFormat           color_format_ = VK_FORMAT_R8G8B8A8_UNORM;
    VkSampleCountFlagBits samples_   = VK_SAMPLE_COUNT_1_BIT;

    std::unique_ptr<memory::Image>          color_image_;   /**< Color attachment. */
    std::unique_ptr<memory::Image>          resolve_image_; /**< MSAA resolve attachment. */
    std::unique_ptr<memory::Image>          depth_image_;   /**< Depth attachment. */
    std::unique_ptr<pipeline::RenderPass>   render_pass_;   /**< Offscreen render pass. */
    VkFramebuffer                           framebuffer_ = VK_NULL_HANDLE; /**< Framebuffer. */
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_OFFSCREEN_TARGET_H
