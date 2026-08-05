/**
 * @file cubemap_target.h
 * @brief HDR color cubemap render target for reflection probe capture.
 */

#ifndef COOPA_GFX_ENGINE_CUBEMAP_TARGET_H
#define COOPA_GFX_ENGINE_CUBEMAP_TARGET_H

#include <volk/volk.h>
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

/**
 * @class CubemapTarget
 * @brief Offscreen multi-face render target utility for reflection probes and skybox capturing.
 */
class CubemapTarget {
public:
    CubemapTarget(core::Device& device, memory::Allocator& allocator,
                  uint32_t face_resolution = 256, uint32_t mip_levels = 1)
        : device_(device), allocator_(allocator),
          res_(face_resolution), mip_levels_(mip_levels)
    {
        create_resources_();
    }

    ~CubemapTarget() {
        destroy_framebuffers_();
    }

    CubemapTarget(const CubemapTarget&) = delete;
    CubemapTarget& operator=(const CubemapTarget&) = delete;

    pipeline::RenderPass& render_pass() { return *render_pass_; }
    const pipeline::RenderPass& render_pass() const { return *render_pass_; }

    void begin_face_pass(command::CommandBuffer& cmd, uint32_t face, glm::vec4 clear_color = {0.0f, 0.0f, 0.0f, 1.0f}) {
        VkClearValue clear_values[2]{};
        clear_values[0].color = {{clear_color.r, clear_color.g, clear_color.b, clear_color.a}};
        clear_values[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass        = render_pass_->handle();
        rp_info.framebuffer       = face_fbs_[face];
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {res_, res_};
        rp_info.clearValueCount   = 2;
        rp_info.pClearValues      = clear_values;

        vkCmdBeginRenderPass(cmd.handle(), &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(res_), static_cast<float>(res_));
        cmd.set_scissor(0, 0, res_, res_);
    }

    void end_face_pass(command::CommandBuffer& cmd) {
        cmd.end_render_pass();
    }

    void transition_to_shader_read(command::CommandBuffer& cmd, VkImageLayout old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) const {
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = old_layout;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = cubemap_image_;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = mip_levels_;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 6;
        barrier.srcAccessMask                   = (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0;
        barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

        VkPipelineStageFlags src_stage = (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        vkCmdPipelineBarrier(
            cmd.handle(),
            src_stage,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier
        );
    }

    VkImageView cubemap_view() const { return cubemap_view_; }
    uint32_t resolution() const { return res_; }

    static glm::mat4 get_face_matrix(uint32_t face, const glm::vec3& position,
                                     float near_plane = 0.1f, float far_plane = 100.0f) {
        glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, near_plane, far_plane);
        proj[1][1] *= -1.0f; // Vulkan Y-flip

        // Standard 6 Vulkan cubemap face directions (+X, -X, +Y, -Y, +Z, -Z)
        glm::vec3 targets[6] = {
            position + glm::vec3( 1.0f,  0.0f,  0.0f), // +X
            position + glm::vec3(-1.0f,  0.0f,  0.0f), // -X
            position + glm::vec3( 0.0f,  1.0f,  0.0f), // +Y
            position + glm::vec3( 0.0f, -1.0f,  0.0f), // -Y
            position + glm::vec3( 0.0f,  0.0f,  1.0f), // +Z
            position + glm::vec3( 0.0f,  0.0f, -1.0f)  // -Z
        };
        glm::vec3 ups[6] = {
            glm::vec3(0.0f, -1.0f,  0.0f), // +X
            glm::vec3(0.0f, -1.0f,  0.0f), // -X
            glm::vec3(0.0f,  0.0f,  1.0f), // +Y
            glm::vec3(0.0f,  0.0f, -1.0f), // -Y
            glm::vec3(0.0f, -1.0f,  0.0f), // +Z
            glm::vec3(0.0f, -1.0f,  0.0f)  // -Z
        };

        glm::mat4 view = glm::lookAt(position, targets[face], ups[face]);
        return proj * view;
    }

private:
    void create_resources_() {
        // 1. Create Cubemap Color Image (R16G16B16A16_SFLOAT, 6 array layers)
        VkImageCreateInfo image_info{};
        image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType     = VK_IMAGE_TYPE_2D;
        image_info.extent.width  = res_;
        image_info.extent.height = res_;
        image_info.extent.depth  = 1;
        image_info.mipLevels     = mip_levels_;
        image_info.arrayLayers   = 6;
        image_info.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
        image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        image_info.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        GFX_VK_CHECK(vmaCreateImage(allocator_.handle(), &image_info, &alloc_info,
                                    &cubemap_image_, &cubemap_allocation_, nullptr));

        // 2. Full Cubemap View for Sampling
        VkImageViewCreateInfo view_info{};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = cubemap_image_;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_CUBE;
        view_info.format                          = VK_FORMAT_R16G16B16A16_SFLOAT;
        view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel   = 0;
        view_info.subresourceRange.levelCount     = mip_levels_;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount     = 6;
        GFX_VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &cubemap_view_));

        // 3. Per-face Image Views for Render Targets
        for (uint32_t f = 0; f < 6; ++f) {
            VkImageViewCreateInfo face_view_info{};
            face_view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            face_view_info.image                           = cubemap_image_;
            face_view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            face_view_info.format                          = VK_FORMAT_R16G16B16A16_SFLOAT;
            face_view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            face_view_info.subresourceRange.baseMipLevel   = 0;
            face_view_info.subresourceRange.levelCount     = 1;
            face_view_info.subresourceRange.baseArrayLayer = f;
            face_view_info.subresourceRange.layerCount     = 1;
            GFX_VK_CHECK(vkCreateImageView(device_.handle(), &face_view_info, nullptr, &face_views_[f]));
        }

        // 4. Depth Attachment Image for face pass
        depth_image_ = std::make_unique<memory::Image>(
            device_, allocator_, res_, res_,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );

        // 5. Render Pass (Color + Depth)
        render_pass_ = std::make_unique<pipeline::RenderPass>(
            device_,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        );

        // 6. Framebuffers per face
        for (uint32_t f = 0; f < 6; ++f) {
            VkImageView attachments[] = { face_views_[f], depth_image_->view() };

            VkFramebufferCreateInfo fb_info{};
            fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb_info.renderPass      = render_pass_->handle();
            fb_info.attachmentCount = 2;
            fb_info.pAttachments    = attachments;
            fb_info.width           = res_;
            fb_info.height          = res_;
            fb_info.layers          = 1;
            GFX_VK_CHECK(vkCreateFramebuffer(device_.handle(), &fb_info, nullptr, &face_fbs_[f]));
        }
    }

    void destroy_framebuffers_() {
        for (uint32_t f = 0; f < 6; ++f) {
            if (face_fbs_[f] != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(device_.handle(), face_fbs_[f], nullptr);
                face_fbs_[f] = VK_NULL_HANDLE;
            }
            if (face_views_[f] != VK_NULL_HANDLE) {
                vkDestroyImageView(device_.handle(), face_views_[f], nullptr);
                face_views_[f] = VK_NULL_HANDLE;
            }
        }
        if (cubemap_view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_.handle(), cubemap_view_, nullptr);
            cubemap_view_ = VK_NULL_HANDLE;
        }
        if (cubemap_image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_.handle(), cubemap_image_, cubemap_allocation_);
            cubemap_image_ = VK_NULL_HANDLE;
            cubemap_allocation_ = VK_NULL_HANDLE;
        }
    }

    core::Device& device_;
    memory::Allocator& allocator_;
    uint32_t res_;
    uint32_t mip_levels_;

    VkImage       cubemap_image_      = VK_NULL_HANDLE;
    VmaAllocation cubemap_allocation_ = VK_NULL_HANDLE;
    VkImageView   cubemap_view_       = VK_NULL_HANDLE;
    VkImageView   face_views_[6]      = {VK_NULL_HANDLE};
    VkFramebuffer face_fbs_[6]        = {VK_NULL_HANDLE};

    std::unique_ptr<memory::Image> depth_image_;
    std::unique_ptr<pipeline::RenderPass> render_pass_;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_CUBEMAP_TARGET_H
