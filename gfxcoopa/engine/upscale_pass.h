/**
 * @file upscale_pass.h
 * @brief Nearest-neighbour upscale pass from low-res offscreen to swapchain.
 *
 * Renders a fullscreen triangle sampling the final low-res post-processed image
 * with a VK_FILTER_NEAREST sampler, producing the characteristic pixel-art look.
 *
 * Pixel-perfect mode: if the window size is an integer multiple of the render
 * resolution, the viewport is centered with black bars. Otherwise nearest-neighbour
 * sampling still produces a visually correct retro result.
 */

#ifndef COOPA_GFX_ENGINE_UPSCALE_PASS_H
#define COOPA_GFX_ENGINE_UPSCALE_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <cmath>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/fullscreen_quad.h>
#include <gfxcoopa/engine/sampler.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @class UpscalePass
 * @brief Nearest-neighbour upscale from the low-res offscreen target to the swapchain.
 *
 * Usage:
 * @code
 * UpscalePass upscale(device, swapchain_render_pass, nn_sampler,
 *                     "assets/shaders/upscale.vert.spv",
 *                     "assets/shaders/upscale.frag.spv");
 * // After post-processing, inside the swapchain render pass:
 * upscale.draw(cmd, final_color_view, window_w, window_h, render_w, render_h);
 * @endcode
 */
class UpscalePass {
public:
    /**
     * @brief Creates the upscale pass.
     *
     * @param device         Logical device.
     * @param swapchain_pass The swapchain render pass (PRESENT_SRC_KHR final layout).
     * @param nn_sampler     Nearest-neighbour sampler.
     * @param vert_spv_path  Path to upscale.vert.spv.
     * @param frag_spv_path  Path to upscale.frag.spv.
     */
    UpscalePass(core::Device&          device,
                pipeline::RenderPass&  swapchain_pass,
                const Sampler&         nn_sampler,
                const std::string&     vert_spv_path,
                const std::string&     frag_spv_path)
        : device_(device), nn_sampler_(nn_sampler)
    {
        vert_shader_ = std::make_unique<pipeline::Shader>(device, vert_spv_path, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<pipeline::Shader>(device, frag_spv_path, VK_SHADER_STAGE_FRAGMENT_BIT);

        // Descriptor set layout: set 0 binding 0 = combined image sampler.
        VkDescriptorSetLayoutBinding sampler_binding{};
        sampler_binding.binding         = 0;
        sampler_binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler_binding.descriptorCount = 1;
        sampler_binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        image_layout_ = std::make_unique<pipeline::DescriptorSetLayout>(
            device, std::vector<VkDescriptorSetLayoutBinding>{sampler_binding}
        );

        // Pipeline: no vertex input, no culling, no depth test.
        pipeline::PipelineConfig cfg{};
        cfg.cull_mode  = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        pipeline_ = std::make_unique<pipeline::Pipeline>(
            device, swapchain_pass,
            std::vector<pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{image_layout_->handle()},
            cfg
        );

        // Descriptor pool + set for the color image.
        VkDescriptorPoolSize pool_size{};
        pool_size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_size.descriptorCount = 1;

        desc_pool_ = std::make_unique<pipeline::DescriptorPool>(
            device, 1, std::vector<VkDescriptorPoolSize>{pool_size}
        );
        desc_set_ = std::make_unique<pipeline::DescriptorSet>(device, *desc_pool_, *image_layout_);
    }

    /**
     * @brief Updates the bound source image (call when the offscreen color changes).
     * @param color_view The VkImageView of the final post-processed low-res image.
     */
    void set_source_image(VkImageView color_view) {
        desc_set_->bind_image(0, color_view, nn_sampler_.handle());
    }

    /**
     * @brief Records the upscale draw call into the swapchain render pass.
     *
     * Computes the pixel-perfect viewport (integer scaling with letterboxing/pillarboxing).
     *
     * @param cmd          Command buffer (must be inside the swapchain render pass).
     * @param window_w     Window framebuffer width.
     * @param window_h     Window framebuffer height.
     * @param render_w     Low-res render width.
     * @param render_h     Low-res render height.
     */
    void draw(command::CommandBuffer& cmd,
              uint32_t window_w, uint32_t window_h,
              uint32_t render_w, uint32_t render_h) const
    {
        // Compute the largest integer scale factor that fits in the window.
        float scale_x = static_cast<float>(window_w) / static_cast<float>(render_w);
        float scale_y = static_cast<float>(window_h) / static_cast<float>(render_h);
        float scale   = std::floor(std::min(scale_x, scale_y));
        if (scale < 1.0f) scale = 1.0f; // Fallback: at least 1:1

        float vp_w = static_cast<float>(render_w) * scale;
        float vp_h = static_cast<float>(render_h) * scale;
        float vp_x = std::floor((static_cast<float>(window_w) - vp_w) * 0.5f);
        float vp_y = std::floor((static_cast<float>(window_h) - vp_h) * 0.5f);

        cmd.set_viewport(vp_x, vp_y, vp_w, vp_h);
        cmd.set_scissor(
            static_cast<int32_t>(vp_x),
            static_cast<int32_t>(vp_y),
            static_cast<uint32_t>(vp_w),
            static_cast<uint32_t>(vp_h)
        );

        cmd.bind_pipeline(*pipeline_);
        cmd.bind_descriptor_set(pipeline_->layout(), *desc_set_, 0);
        fsq_.draw(cmd);
    }

private:
    core::Device&                              device_;
    const Sampler&                             nn_sampler_;  /**< Nearest-neighbour sampler (not owned). */
    std::unique_ptr<pipeline::Shader>          vert_shader_;
    std::unique_ptr<pipeline::Shader>          frag_shader_;
    std::unique_ptr<pipeline::DescriptorSetLayout> image_layout_;
    std::unique_ptr<pipeline::Pipeline>        pipeline_;
    std::unique_ptr<pipeline::DescriptorPool>  desc_pool_;
    std::unique_ptr<pipeline::DescriptorSet>   desc_set_;
    FullscreenQuad                             fsq_;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_UPSCALE_PASS_H
