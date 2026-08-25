#include <gfxcoopa/engine/util/fullscreen_quad.h>
/**
 * @file transparent_pass.h
 * @brief Forward BLEND transparent geometry pass.
 *
 * Renders alpha-blended geometry after opaque lighting (and, when SSR is enabled, after
 * the SSR composite), into the live HDR colour image, depth-tested read-only against the
 * G-Buffer's opaque depth. Owns its own raw VkRenderPass rather than using
 * pipeline::RenderPass, because it needs LOAD_OP_LOAD on both attachments (the colour
 * image already holds the finished opaque result; the depth image is the G-Buffer's, not
 * a depth buffer this pass ever writes) -- pipeline::RenderPass hardcodes
 * VK_ATTACHMENT_LOAD_OP_CLEAR and does not support attaching an externally-owned depth
 * image.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_TRANSPARENT_PASS_H
#define GFXCOOPA_ENGINE_PASSES_TRANSPARENT_PASS_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/data/mesh.h>
#include <gfxcoopa/engine/data/model_ubo.h>
#include <gfxcoopa/engine/passes/extra_sets.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class TransparentPass {
public:
    /// 32 bytes -- byte-identical to GBufferPipeline::PushConstants. model/
    /// normal_matrix moved to the per-instance vertex stream (data::InstanceData);
    /// this block is now shared once per instanced batch, not pushed per object.
    struct PushConstants {
        glm::vec4 albedo       = {0.8f, 0.8f, 0.8f, 1.0f}; // 16 bytes; .w = alpha
        float     metallic     = 0.0f;
        float     roughness    = 0.5f;
        float     ao           = 1.0f;
        float     alpha_cutoff = 0.0f;                     // unused; BLEND never alpha-tests
    };

    /**
     * @brief Creates the pass's render pass and pipeline.
     *
     * @param device        Logical device.
     * @param color_format  Format of the HDR colour image this pass will be pointed at
     *                      (offscreen_target_ / the SSR composite target -- both share it).
     * @param camera_layout Set 0.
     * @param light_layout  Set 1.
     * @param shadow_layout Set 2.
     * @param vert_spv      Vertex shader (reuses pbr.vert.spv -- see plan/transparency docs).
     * @param frag_spv      Fragment shader (transparent.frag.spv).
     * @param extra         Optional trailing sets (e.g. GI), appended after set 2.
     * @param extra_pc_bytes Additional bytes to reserve past PushConstants' own
     *                      [0, sizeof(PushConstants)) -- for a consumer whose transparent.frag
     *                      needs frame-level data (e.g. toyengine's
     *                      soft_lighting/light_bands/spec_threshold toggle) beyond the
     *                      per-object material block this pass owns. Folded into ONE fragment-
     *                      stage VkPushConstantRange, not a second one: the Vulkan spec forbids
     *                      two ranges in the same pipeline layout sharing a stage, even with
     *                      disjoint byte ranges. GLSL likewise permits only one push_constant
     *                      block per stage, so such a consumer's shader declares ONE block
     *                      spanning [0, sizeof(PushConstants) + extra_pc_bytes) and pushes into
     *                      each region independently (this pass's own push() call handles
     *                      [0, sizeof(PushConstants)); the caller issues its own
     *                      cmd.push_constants() at [sizeof(PushConstants), ...) for the rest).
     */
    TransparentPass(coopa::gfx::core::Device& device,
                    VkFormat color_format,
                    VkDescriptorSetLayout camera_layout,
                    VkDescriptorSetLayout light_layout,
                    VkDescriptorSetLayout shadow_layout,
                    const std::string& vert_spv,
                    const std::string& frag_spv,
                    ExtraSets extra = {},
                    uint32_t extra_pc_bytes = 0)
        : device_(device), color_format_(color_format), extra_(std::move(extra))
    {
        extra_.validate("TransparentPass");
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        create_render_pass_();

        std::vector<VkDescriptorSetLayout> layouts = { camera_layout, light_layout, shadow_layout };
        extra_first_set_ = static_cast<uint32_t>(layouts.size());
        layouts.insert(layouts.end(), extra_.layouts.begin(), extra_.layouts.end());

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants) + extra_pc_bytes;

        std::vector<VkPushConstantRange> pc_ranges{pc_range};

        auto binding          = coopa::gfx::engine::data::Vertex::binding_description();
        auto instance_binding = coopa::gfx::engine::data::InstanceData::binding_description();
        std::vector<VkVertexInputBindingDescription> binding_vec = {binding, instance_binding};

        auto attrs = coopa::gfx::engine::data::Vertex::attribute_descriptions();
        std::vector<VkVertexInputAttributeDescription> attr_vec(attrs.begin(), attrs.end());
        auto instance_attrs = coopa::gfx::engine::data::InstanceData::attribute_descriptions();
        attr_vec.insert(attr_vec.end(), instance_attrs.begin(), instance_attrs.end());

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode        = VK_CULL_MODE_BACK_BIT;
        cfg.front_face       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        cfg.depth_test       = true;               // test against the opaque G-Buffer depth
        cfg.depth_write      = false;              // never occlude other transparents
        cfg.depth_compare_op = VK_COMPARE_OP_LESS; // a coplanar decal fails cleanly, no flicker
        cfg.blending         = true;               // src-alpha-over (pipeline.h)

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, render_pass_,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            binding_vec,
            attr_vec,
            layouts,
            cfg,
            pc_ranges
        );
    }

    ~TransparentPass() {
        destroy_framebuffer_();
        if (render_pass_ != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device_.handle(), render_pass_, nullptr);
        }
    }

    TransparentPass(const TransparentPass&) = delete;
    TransparentPass& operator=(const TransparentPass&) = delete;

    /**
     * @brief Points the pass at the current frame's targets, rebuilding the framebuffer
     * only when the view/extent triple actually changed.
     *
     * @param color_view        The live HDR colour image view to blend into (the raw
     *                          offscreen target, or the SSR composite output).
     * @param gbuffer_depth_view The G-Buffer's depth view, bound read-only.
     * @param width  Render width.
     * @param height Render height.
     */
    void set_targets(VkImageView color_view, VkImageView gbuffer_depth_view,
                     uint32_t width, uint32_t height) {
        if (framebuffer_ != VK_NULL_HANDLE &&
            color_view == color_view_ && gbuffer_depth_view == depth_view_ &&
            width == width_ && height == height_) {
            return;
        }
        destroy_framebuffer_();
        color_view_ = color_view;
        depth_view_ = gbuffer_depth_view;
        width_      = width;
        height_     = height;

        VkImageView views[2] = { color_view_, depth_view_ };

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass_;
        fb_info.attachmentCount = 2;
        fb_info.pAttachments    = views;
        fb_info.width           = width_;
        fb_info.height          = height_;
        fb_info.layers          = 1;

        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &framebuffer_));
    }

    /**
     * @brief Transitions the G-Buffer depth image to read-only and begins the render pass.
     *
     * The depth image's layout on entry depends on whether HiZPass ran this frame (it
     * transitions DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL and only
     * runs when SSR is enabled), so the caller must pass the layout it actually left the
     * image in.
     *
     * @param cmd                 Command buffer to record into.
     * @param gbuffer_depth       The G-Buffer depth VkImage (for the barrier).
     * @param depth_current_layout The image's layout on entry.
     */
    void begin(coopa::gfx::command::CommandBuffer& cmd, VkImage gbuffer_depth,
              VkImageLayout depth_current_layout) {
        const bool from_shader_read = (depth_current_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = depth_current_layout;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = gbuffer_depth;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask = from_shader_read
            ? VK_ACCESS_SHADER_READ_BIT
            : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        vkCmdPipelineBarrier(cmd.handle(),
                             from_shader_read ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                              : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        cmd.begin_render_pass(render_pass_, framebuffer_, {width_, height_});

        // Negative-height viewport for the Vulkan Y-flip, matching every other geometry
        // target in this engine (gbuffer_target.h, offscreen_target.h).
        cmd.set_viewport(0.0f, static_cast<float>(height_),
                         static_cast<float>(width_),
                         -static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);
    }

    /** @brief Binds the pipeline. Call after begin(), before push()/draws. */
    void bind(coopa::gfx::command::CommandBuffer& cmd) {
        cmd.bind_pipeline(*pipeline_);
    }

    /// Binds the caller's extra sets (e.g. GI), if any were provided at construction. Call
    /// after bind(), alongside binding sets 0-2, before draws. A no-op when `extra` was empty.
    void bind_extra(coopa::gfx::command::CommandBuffer& cmd) {
        if (extra_.bind) {
            extra_.bind(cmd, pipeline_->layout(), extra_first_set_);
        }
    }

    /** @brief Uploads per-batch material push constants. */
    void push(coopa::gfx::command::CommandBuffer& cmd, const PushConstants& pc) {
        cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
    }

    /** @brief Ends the render pass. */
    void end(coopa::gfx::command::CommandBuffer& cmd) {
        cmd.end_render_pass();
    }

    VkPipelineLayout layout() const { return pipeline_->layout(); }

private:
    void create_render_pass_() {
        VkAttachmentDescription attachments[2]{};

        // 0: the live HDR colour image. Both OffscreenTarget instances this pass can be
        // pointed at (the raw offscreen target and the SSR composite target) leave their
        // colour image in SHADER_READ_ONLY_OPTIMAL when their own render pass ends.
        attachments[0].format         = color_format_;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // 1: the G-Buffer's own depth image, read-only -- this pass tests against it but
        // never writes it. storeOp is DONT_CARE, not STORE: nothing here ever modifies depth
        // (depth_write=false above, and the READ_ONLY layout disallows it regardless), so there
        // is nothing to preserve -- and DONT_CARE matters beyond the pass itself: sync
        // validation (VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT) models any
        // depth/stencil attachment with storeOp=STORE as a WRITE access at vkCmdEndRenderPass
        // regardless of the actual layout or pipeline state, which produced a false
        // WRITE_AFTER_WRITE hazard for any barrier a caller issues afterward to read this same
        // depth image again later in the frame (e.g. toyengine's PixelStylizePass usage, which
        // samples it for outline detection right after this pass runs).
        attachments[1].format         = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;

        // Two dependencies: the tonemapper reads the colour image after we're done with
        // it (out), and we read it (having been written by deferred lighting / skybox /
        // SSR composite) before we're allowed to write it (in).
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 2;
        rp_info.pAttachments    = attachments;
        rp_info.subpassCount    = 1;
        rp_info.pSubpasses      = &subpass;
        rp_info.dependencyCount = 2;
        rp_info.pDependencies   = deps;

        GFX_VK_CHECK(vkCreateRenderPass(device_.handle(), &rp_info, nullptr, &render_pass_));
    }

    void destroy_framebuffer_() {
        if (framebuffer_ != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), framebuffer_, nullptr);
            framebuffer_ = VK_NULL_HANDLE;
        }
    }

    coopa::gfx::core::Device& device_;
    VkFormat                  color_format_;

    VkRenderPass  render_pass_  = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_  = VK_NULL_HANDLE;
    VkImageView   color_view_   = VK_NULL_HANDLE;
    VkImageView   depth_view_   = VK_NULL_HANDLE;
    uint32_t      width_        = 0;
    uint32_t      height_       = 0;

    std::unique_ptr<coopa::gfx::pipeline::Shader> vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> pipeline_;

    ExtraSets extra_;
    uint32_t  extra_first_set_ = 0;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_TRANSPARENT_PASS_H
