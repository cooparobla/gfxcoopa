/**
 * @file brdf_lut.h
 * @brief BRDF Integration LUT generation & management for PBR indirect specular split-sum approximation.
 */

#ifndef COOPA_GFX_ENGINE_BRDF_LUT_H
#define COOPA_GFX_ENGINE_BRDF_LUT_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/sampler.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @class BRDFLUT
 * @brief Generates and stores a 512x512 R16G16_SFLOAT BRDF Integration Look-Up Table.
 */
class BRDFLUT {
public:
    BRDFLUT(core::Device& device,
            memory::Allocator& allocator,
            command::CommandPool& cmd_pool,
            const std::string& shader_dir)
        : device_(device)
    {
        // 1. Create 512x512 R16G16_SFLOAT Image
        lut_image_ = std::make_unique<memory::Image>(
            device, allocator, 512, 512,
            VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        // 2. Create Linear Clamp-to-Edge Sampler
        lut_sampler_ = std::make_unique<Sampler>(
            device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
        );

        // 3. Generate LUT texture contents via rasterization pass
        generate_(device, allocator, cmd_pool, shader_dir);
    }

    ~BRDFLUT() = default;

    BRDFLUT(const BRDFLUT&) = delete;
    BRDFLUT& operator=(const BRDFLUT&) = delete;

    VkImageView view() const { return lut_image_->view(); }
    VkSampler sampler() const { return lut_sampler_->handle(); }
    const memory::Image& image() const { return *lut_image_; }

private:
    void generate_(core::Device& device,
                   memory::Allocator& allocator,
                   command::CommandPool& cmd_pool,
                   const std::string& shader_dir)
    {
        pipeline::RenderPass render_pass(
            device, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        VkImageView attachments[] = { lut_image_->view() };
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass.handle();
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = attachments;
        fb_info.width           = 512;
        fb_info.height          = 512;
        fb_info.layers          = 1;

        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        GFX_VK_CHECK(vkCreateFramebuffer(device.handle(), &fb_info, nullptr, &framebuffer));

        std::string vert_path = shader_dir + "/brdf_lut.vert.spv";
        std::string frag_path = shader_dir + "/brdf_lut.frag.spv";

        pipeline::Shader vert_shader(device, vert_path, VK_SHADER_STAGE_VERTEX_BIT);
        pipeline::Shader frag_shader(device, frag_path, VK_SHADER_STAGE_FRAGMENT_BIT);

        pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;
        cfg.samples     = VK_SAMPLE_COUNT_1_BIT;

        pipeline::Pipeline lut_pipeline(
            device, render_pass,
            std::vector<pipeline::Shader*>{ &vert_shader, &frag_shader },
            {}, {}, {}, cfg, {}
        );

        VkCommandBuffer raw_cmd = cmd_pool.begin_single_use();
        command::CommandBuffer cmd(raw_cmd);

        VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 0.0f}}};
        VkRenderPassBeginInfo pass_begin{};
        pass_begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        pass_begin.renderPass        = render_pass.handle();
        pass_begin.framebuffer       = framebuffer;
        pass_begin.renderArea.offset = {0, 0};
        pass_begin.renderArea.extent = {512, 512};
        pass_begin.clearValueCount   = 1;
        pass_begin.pClearValues      = &clear_color;

        vkCmdBeginRenderPass(cmd.handle(), &pass_begin, VK_SUBPASS_CONTENTS_INLINE);
        cmd.bind_pipeline(lut_pipeline);
        cmd.set_viewport(0.0f, 0.0f, 512.0f, 512.0f);
        cmd.set_scissor(0, 0, 512, 512);
        cmd.draw(3);
        cmd.end_render_pass();

        cmd_pool.end_single_use(raw_cmd, device.graphics_queue());

        vkDestroyFramebuffer(device.handle(), framebuffer, nullptr);
    }

    core::Device& device_;
    std::unique_ptr<memory::Image> lut_image_;
    std::unique_ptr<Sampler>       lut_sampler_;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_BRDF_LUT_H
