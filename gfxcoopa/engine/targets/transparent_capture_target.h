/**
 * @file transparent_capture_target.h
 * @brief Forward capture target for transparent geometry, used ONLY as a second reflection
 *        source for opaque reflectors' SSR raymarch -- not the transparent objects' own final
 *        (alpha-blended) draw, which is TransparentPass/TransparentCaptureTarget's sibling
 *        engine/passes/transparent_pass.h.
 *
 * Modeled directly on gbuffer_target.h: same clear-every-frame, opaque-style depth-tested
 * render pass structure, just 3 color attachments instead of 4 (no albedo/AO -- this target's
 * "color" attachment already holds the FINAL shaded result, not raw material to be lit later)
 * and every color attachment HDR (R16G16B16A16_SFLOAT) rather than G0's R8G8B8A8_UNORM, since
 * shaded_color needs HDR range same as any other lit output.
 *
 * Depth-tested/written normally (front-most transparent surface wins per pixel) but ONLY
 * among transparent objects themselves -- this target is NOT depth-tested against the opaque
 * G-buffer's depth. That is intentional, not an oversight: see gfx_ssr_trace_secondary()'s doc
 * (gfx/ssr_trace_secondary_body.glsl) for why the merge between "opaque occludes transparent"
 * happens at raymarch time (nearer-of-two-independent-traces), not at capture time.
 */

#ifndef GFXCOOPA_ENGINE_TARGETS_TRANSPARENT_CAPTURE_TARGET_H
#define GFXCOOPA_ENGINE_TARGETS_TRANSPARENT_CAPTURE_TARGET_H

#include <volk/volk.h>
#include <memory>
#include <vector>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace targets {

/**
 * @class TransparentCaptureTarget
 * @brief Three colour attachments plus depth recording transparent surfaces'
 *        shaded appearance, used only as a reflection source for SSR.
 *
 * Same clear-every-frame, depth-tested shape as GBufferTarget; it is a separate
 * target because SSR must reflect transparent surfaces without them having been
 * composited into the frame.
 */
class TransparentCaptureTarget {
public:
    TransparentCaptureTarget(coopa::gfx::core::Device& device,
                             coopa::gfx::memory::Allocator& allocator,
                             uint32_t width,
                             uint32_t height)
        : device_(device), allocator_(allocator), width_(width), height_(height)
    {
        create_resources_();
    }

    ~TransparentCaptureTarget() {
        destroy_resources_();
    }

    TransparentCaptureTarget(const TransparentCaptureTarget&) = delete;
    TransparentCaptureTarget& operator=(const TransparentCaptureTarget&) = delete;

    void begin(coopa::gfx::command::CommandBuffer& cmd) const {
        VkClearValue clear_values[4]{};
        clear_values[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};  // shaded_color
        // normal_metallic clears to (0,0,0,0) deliberately -- this is the SAME "empty/no
        // surface here" sentinel every dot(N,N) < 0.001 early-out in this codebase already
        // relies on (ssr.frag's own background check, skybox.frag's discard), reused as-is by
        // gfx_ssr_trace_secondary()'s caller to treat "no transparent object at this pixel" as
        // a clean miss rather than needing a separate coverage mask.
        clear_values[1].color = {{0.0f, 0.0f, 0.0f, 0.0f}};  // normal_metallic
        clear_values[2].color = {{0.0f, 0.0f, 0.0f, 0.0f}};  // position_roughness
        clear_values[3].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = render_pass_;
        rp_info.framebuffer       = framebuffer_;
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {width_, height_};
        rp_info.clearValueCount   = 4;
        rp_info.pClearValues      = clear_values;

        vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        // Negative-height viewport for the Vulkan Y-flip, matching every other geometry target
        // in this engine (gbuffer_target.h, offscreen_target.h).
        cmd.set_viewport(0.0f, static_cast<float>(height_),
                         static_cast<float>(width_),
                         -static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);
    }

    void end(coopa::gfx::command::CommandBuffer& cmd) const {
        cmd.end_render_pass();
    }

    void recreate(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        destroy_resources_();
        create_resources_();
    }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    VkImageView shaded_color_view() const { return shaded_color_image_->view(); }
    VkImageView normal_metallic_view() const { return normal_metallic_image_->view(); }
    VkImageView position_roughness_view() const { return position_roughness_image_->view(); }
    VkImageView depth_view() const { return depth_image_->view(); }
    VkImage depth_image_handle() const { return depth_image_->handle(); }

    VkRenderPass render_pass() const { return render_pass_; }

private:
    void create_resources_() {
        VkImageUsageFlags color_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        shaded_color_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width_, height_,
            VK_FORMAT_R16G16B16A16_SFLOAT, color_flags,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );

        normal_metallic_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width_, height_,
            VK_FORMAT_R16G16B16A16_SFLOAT, color_flags,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );

        position_roughness_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width_, height_,
            VK_FORMAT_R16G16B16A16_SFLOAT, color_flags,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );

        depth_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, width_, height_,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT, VMA_MEMORY_USAGE_AUTO
        );

        // Render pass definition with 3 color attachments + 1 depth attachment
        VkAttachmentDescription attachments[4]{};

        // shaded_color
        attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // normal_metallic
        attachments[1].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // position_roughness
        attachments[2].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[2].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Depth -- final layout DEPTH_STENCIL_ATTACHMENT_OPTIMAL, matching GBufferTarget's
        // own depth exactly, so HiZPass::execute()'s hardcoded oldLayout expectation
        // (DEPTH_STENCIL_ATTACHMENT_OPTIMAL, see hiz_pass.h) is satisfied unmodified by a
        // second HiZPass instance pointed at this target's depth.
        attachments[3].format         = VK_FORMAT_D32_SFLOAT;
        attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[3].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_refs[3]{
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
        };
        VkAttachmentReference depth_ref{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 3;
        subpass.pColorAttachments       = color_refs;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 4;
        rp_info.pAttachments    = attachments;
        rp_info.subpassCount    = 1;
        rp_info.pSubpasses      = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies   = &dependency;

        GFX_VK_CHECK(vkCreateRenderPass(device_.handle(), &rp_info, nullptr, &render_pass_));

        // Framebuffer
        std::vector<VkImageView> views = {
            shaded_color_image_->view(),
            normal_metallic_image_->view(),
            position_roughness_image_->view(),
            depth_image_->view()
        };

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass_;
        fb_info.attachmentCount = static_cast<uint32_t>(views.size());
        fb_info.pAttachments    = views.data();
        fb_info.width           = width_;
        fb_info.height          = height_;
        fb_info.layers          = 1;

        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &framebuffer_));
    }

    void destroy_resources_() {
        if (framebuffer_ != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), framebuffer_, nullptr);
            framebuffer_ = VK_NULL_HANDLE;
        }
        if (render_pass_ != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device_.handle(), render_pass_, nullptr);
            render_pass_ = VK_NULL_HANDLE;
        }
        shaded_color_image_.reset();
        normal_metallic_image_.reset();
        position_roughness_image_.reset();
        depth_image_.reset();
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;
    uint32_t                       width_;
    uint32_t                       height_;

    std::unique_ptr<coopa::gfx::memory::Image> shaded_color_image_;
    std::unique_ptr<coopa::gfx::memory::Image> normal_metallic_image_;
    std::unique_ptr<coopa::gfx::memory::Image> position_roughness_image_;
    std::unique_ptr<coopa::gfx::memory::Image> depth_image_;

    VkRenderPass  render_pass_ = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
};

} // namespace targets
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_TARGETS_TRANSPARENT_CAPTURE_TARGET_H
