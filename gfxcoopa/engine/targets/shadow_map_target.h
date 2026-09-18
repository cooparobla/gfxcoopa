/**
 * @file shadow_map_target.h
 * @brief Depth render targets for Directional Light, Point Light Cubemap, and
 *        Spot Light shadow maps.
 */

#ifndef GFXCOOPA_ENGINE_TARGETS_SHADOW_MAP_TARGET_H
#define GFXCOOPA_ENGINE_TARGETS_SHADOW_MAP_TARGET_H

#include <volk/volk.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <stdexcept>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace targets {

/**
 * @class ShadowMapTarget
 * @brief Manages directional shadow depth map, point light cubemap depth target,
 *        and spot light depth map.
 */
class ShadowMapTarget {
public:
    ShadowMapTarget(core::Device& device, memory::Allocator& allocator,
                    uint32_t dir_res = 2048, uint32_t cube_res = 512, uint32_t spot_res = 1024)
        : device_(device), allocator_(allocator),
          dir_res_(dir_res), cube_res_(cube_res), spot_res_(spot_res)
    {
        create_resources_();
    }

    ~ShadowMapTarget() {
        destroy_framebuffers_();
    }

    ShadowMapTarget(const ShadowMapTarget&) = delete;
    ShadowMapTarget& operator=(const ShadowMapTarget&) = delete;

    // --- Directional Shadow Pass ---

    void begin_directional_pass(command::CommandBuffer& cmd) const {
        VkClearValue clear_value{};
        clear_value.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = dir_render_pass_->handle();
        rp_info.framebuffer       = dir_framebuffer_;
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {dir_res_, dir_res_};
        rp_info.clearValueCount   = 1;
        rp_info.pClearValues      = &clear_value;

        vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(dir_res_), static_cast<float>(dir_res_));
        cmd.set_scissor(0, 0, dir_res_, dir_res_);
    }

    void end_directional_pass(command::CommandBuffer& cmd) const {
        cmd.end_render_pass();
    }

    // --- Spot Light Pass ---
    //
    // Deliberately reuses dir_render_pass_ (not a new VkRenderPass) -- its attachment
    // description (D32_SFLOAT, UNDEFINED -> SHADER_READ_ONLY_OPTIMAL) is identical to
    // what a spot map needs, and render-pass *compatibility* (matching attachment
    // descriptions, not matching framebuffer size) is what lets the existing
    // directional shadow pipeline record into spot_framebuffer_ with no new pipeline
    // variant. Only the framebuffer and resolution differ.

    void begin_spot_pass(command::CommandBuffer& cmd) const {
        VkClearValue clear_value{};
        clear_value.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = dir_render_pass_->handle();
        rp_info.framebuffer       = spot_framebuffer_;
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {spot_res_, spot_res_};
        rp_info.clearValueCount   = 1;
        rp_info.pClearValues      = &clear_value;

        vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(spot_res_), static_cast<float>(spot_res_));
        cmd.set_scissor(0, 0, spot_res_, spot_res_);
    }

    void end_spot_pass(command::CommandBuffer& cmd) const {
        cmd.end_render_pass();
    }

    // --- Point Light Cubemap Face Pass ---

    void begin_cube_face_pass(command::CommandBuffer& cmd, uint32_t face_index) const {
        VkClearValue clear_value{};
        clear_value.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = cube_render_pass_->handle();
        rp_info.framebuffer       = cube_face_framebuffers_[face_index];
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {cube_res_, cube_res_};
        rp_info.clearValueCount   = 1;
        rp_info.pClearValues      = &clear_value;

        vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(cube_res_), static_cast<float>(cube_res_));
        cmd.set_scissor(0, 0, cube_res_, cube_res_);
    }

    void end_cube_face_pass(command::CommandBuffer& cmd) const {
        cmd.end_render_pass();
    }

    // --- Layout Transitions ---

    void transition_cube_to_shader_read(command::CommandBuffer& cmd) const {
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = cube_image_;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 6;
        barrier.srcAccessMask                   = 0;
        barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd.handle(),
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    void transition_spot_to_shader_read(command::CommandBuffer& cmd) const {
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = spot_depth_image_->handle();
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd.handle(),
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    void transition_dir_to_shader_read(command::CommandBuffer& cmd) const {
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = dir_depth_image_->handle();
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd.handle(),
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    // --- Cubemap View Matrices ---

    static glm::mat4 get_cube_face_matrix(uint32_t face_index, const glm::vec3& light_pos, float range) {
        glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, range);

        // 6 cubemap directions: +X, -X, +Y, -Y, +Z, -Z
        glm::vec3 targets[6] = {
            light_pos + glm::vec3( 1.0f,  0.0f,  0.0f), // +X
            light_pos + glm::vec3(-1.0f,  0.0f,  0.0f), // -X
            light_pos + glm::vec3( 0.0f,  1.0f,  0.0f), // +Y
            light_pos + glm::vec3( 0.0f, -1.0f,  0.0f), // -Y
            light_pos + glm::vec3( 0.0f,  0.0f,  1.0f), // +Z
            light_pos + glm::vec3( 0.0f,  0.0f, -1.0f)  // -Z
        };
        glm::vec3 ups[6] = {
            glm::vec3(0.0f, -1.0f,  0.0f), // +X
            glm::vec3(0.0f, -1.0f,  0.0f), // -X
            glm::vec3(0.0f,  0.0f,  1.0f), // +Y
            glm::vec3(0.0f,  0.0f, -1.0f), // -Y
            glm::vec3(0.0f, -1.0f,  0.0f), // +Z
            glm::vec3(0.0f, -1.0f,  0.0f)  // -Z
        };

        glm::mat4 view = glm::lookAt(light_pos, targets[face_index], ups[face_index]);
        return proj * view;
    }

    /**
     * @brief Builds a spot light's light-space (projection * view) matrix.
     *
     * A single perspective frustum aimed along `dir`, sized to exactly cover the
     * cone at `outer_degrees` -- unlike the cube map's fixed 90-degree faces, a
     * spot's FOV is the cone itself, so no shadow-map texels are wasted outside it.
     * Uses the same RH_ZO convention (no Y-flip) as compute_dir_shadow_fit()'s
     * orthoRH_ZO, so both maps share the shader's `proj_coords.xy * 0.5 + 0.5`
     * mapping in calc_dir_shadow()/calc_spot_shadow().
     *
     * @param pos            World-space light position.
     * @param dir            Normalized world-space aim direction.
     * @param outer_degrees  Cone outer half-angle, in degrees (clamped to [1, 89]
     *                       by the caller -- SpotLightComponent::clamped_outer_angle()).
     * @param range          Light range; used as the far clip plane.
     * @return Combined projection * view matrix.
     */
    static glm::mat4 get_spot_matrix(const glm::vec3& pos, const glm::vec3& dir,
                                     float outer_degrees, float range) {
        // Same degenerate-up guard as compute_dir_shadow_fit() (pixel_math.h) --
        // this engine is Z-up, so a near-vertical aim needs a different up axis to
        // keep lookAt() from degenerating.
        const glm::vec3 up = (std::abs(dir.z) < 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                                        : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(2.0f * outer_degrees), 1.0f,
                                               0.1f, std::max(range, 0.2f));
        glm::mat4 view = glm::lookAt(pos, pos + dir, up);
        return proj * view;
    }

    // --- Accessors ---

    VkImageView dir_shadow_view() const { return dir_depth_image_->view(); }
    VkImageView cube_shadow_view() const { return cube_array_view_; }
    VkImageView spot_shadow_view() const { return spot_depth_image_->view(); }

    pipeline::RenderPass& dir_render_pass() const { return *dir_render_pass_; }
    pipeline::RenderPass& cube_render_pass() const { return *cube_render_pass_; }

private:
    void create_resources_() {
        // 1. Directional Depth Image (2D)
        dir_depth_image_ = std::make_unique<memory::Image>(
            device_, allocator_, dir_res_, dir_res_,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );

        dir_render_pass_ = std::make_unique<pipeline::RenderPass>(
            device_,
            VK_FORMAT_UNDEFINED, // no color
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL // depth final layout
        );

        VkImageView dir_view = dir_depth_image_->view();
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = dir_render_pass_->handle();
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = &dir_view;
        fb_info.width           = dir_res_;
        fb_info.height          = dir_res_;
        fb_info.layers          = 1;
        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &dir_framebuffer_));

        // 2. Spot Light Depth Image (2D) -- reuses dir_render_pass_ (see begin_spot_pass()'s
        //    doc for why that's compatible), its own image/view/framebuffer at spot_res_.
        spot_depth_image_ = std::make_unique<memory::Image>(
            device_, allocator_, spot_res_, spot_res_,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );

        VkImageView spot_view = spot_depth_image_->view();
        VkFramebufferCreateInfo spot_fb_info{};
        spot_fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        spot_fb_info.renderPass      = dir_render_pass_->handle();
        spot_fb_info.attachmentCount = 1;
        spot_fb_info.pAttachments    = &spot_view;
        spot_fb_info.width           = spot_res_;
        spot_fb_info.height          = spot_res_;
        spot_fb_info.layers          = 1;
        GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &spot_fb_info, nullptr, &spot_framebuffer_));

        // 3. Point Light Cubemap Depth Image (Cubemap, 6 layers)
        VkImageCreateInfo image_info{};
        image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType     = VK_IMAGE_TYPE_2D;
        image_info.extent.width  = cube_res_;
        image_info.extent.height = cube_res_;
        image_info.extent.depth  = 1;
        image_info.mipLevels     = 1;
        image_info.arrayLayers   = 6;
        image_info.format        = VK_FORMAT_D32_SFLOAT;
        image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
        image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        image_info.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        GFX_VK_CHECK(vmaCreateImage(allocator_.handle(), &image_info, &alloc_info,
                                    &cube_image_, &cube_allocation_, nullptr));

        // Create Cubemap ImageView (all 6 layers)
        VkImageViewCreateInfo view_info{};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = cube_image_;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_CUBE;
        view_info.format                          = VK_FORMAT_D32_SFLOAT;
        view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        view_info.subresourceRange.baseMipLevel   = 0;
        view_info.subresourceRange.levelCount     = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount     = 6;
        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &cube_array_view_));

        // Create per-face 2D ImageViews & Framebuffers
        cube_render_pass_ = std::make_unique<pipeline::RenderPass>(
            device_,
            VK_FORMAT_UNDEFINED,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        cube_face_views_.resize(6);
        cube_face_framebuffers_.resize(6);
        for (uint32_t i = 0; i < 6; ++i) {
            VkImageViewCreateInfo face_view_info{};
            face_view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            face_view_info.image                           = cube_image_;
            face_view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            face_view_info.format                          = VK_FORMAT_D32_SFLOAT;
            face_view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
            face_view_info.subresourceRange.baseMipLevel   = 0;
            face_view_info.subresourceRange.levelCount     = 1;
            face_view_info.subresourceRange.baseArrayLayer = i;
            face_view_info.subresourceRange.layerCount     = 1;
            GFX_VK_CHECK(vkCreateImageView(device_.handle(), &face_view_info, nullptr, &cube_face_views_[i]));

            VkFramebufferCreateInfo face_fb_info{};
            face_fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            face_fb_info.renderPass      = cube_render_pass_->handle();
            face_fb_info.attachmentCount = 1;
            face_fb_info.pAttachments    = &cube_face_views_[i];
            face_fb_info.width           = cube_res_;
            face_fb_info.height          = cube_res_;
            face_fb_info.layers          = 1;
            GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &face_fb_info, nullptr, &cube_face_framebuffers_[i]));
        }
    }

    void destroy_framebuffers_() {
        if (dir_framebuffer_ != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), dir_framebuffer_, nullptr);
            dir_framebuffer_ = VK_NULL_HANDLE;
        }
        if (spot_framebuffer_ != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_.handle(), spot_framebuffer_, nullptr);
            spot_framebuffer_ = VK_NULL_HANDLE;
        }
        for (auto fb : cube_face_framebuffers_) {
            if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device_.handle(), fb, nullptr);
        }
        cube_face_framebuffers_.clear();
        for (auto v : cube_face_views_) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device_.handle(), v, nullptr);
        }
        cube_face_views_.clear();
        if (cube_array_view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_.handle(), cube_array_view_, nullptr);
            cube_array_view_ = VK_NULL_HANDLE;
        }
        if (cube_image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_.handle(), cube_image_, cube_allocation_);
            cube_image_ = VK_NULL_HANDLE;
            cube_allocation_ = VK_NULL_HANDLE;
        }
    }

    core::Device&      device_;
    memory::Allocator& allocator_;
    uint32_t           dir_res_;
    uint32_t           cube_res_;
    uint32_t           spot_res_;

    std::unique_ptr<memory::Image>        dir_depth_image_;
    std::unique_ptr<pipeline::RenderPass> dir_render_pass_;
    VkFramebuffer                         dir_framebuffer_ = VK_NULL_HANDLE;

    std::unique_ptr<memory::Image>        spot_depth_image_;
    VkFramebuffer                         spot_framebuffer_ = VK_NULL_HANDLE; // uses dir_render_pass_

    VkImage                               cube_image_      = VK_NULL_HANDLE;
    VmaAllocation                         cube_allocation_ = VK_NULL_HANDLE;
    VkImageView                           cube_array_view_ = VK_NULL_HANDLE;
    std::vector<VkImageView>              cube_face_views_;
    std::vector<VkFramebuffer>            cube_face_framebuffers_;
    std::unique_ptr<pipeline::RenderPass> cube_render_pass_;
};

} // namespace targets
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_SHADOW_MAP_TARGET_H
