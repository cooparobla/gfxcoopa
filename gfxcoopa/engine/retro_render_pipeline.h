/**
 * @file retro_render_pipeline.h
 * @brief Orchestrates the full retro frame render sequence.
 *
 * Frame sequence:
 *   1. Pre-frame: Begin a separate command buffer for offscreen work.
 *   2. Begin offscreen render pass (low-res, toon+outline).
 *   3. Bind ToonPipeline, draw all MeshRenderer objects.
 *   4. Bind OutlinePipeline, draw same objects (back-face extruded).
 *   5. End offscreen render pass.
 *   6. Run PostProcessPipeline (edge detection, optional color quantize).
 *   7. Submit pre-frame work via the command pool.
 *   8. Inside Renderer::begin_frame() callback:
 *      - Bind UpscalePass, draw fullscreen triangle sampling the post-processed image.
 *
 * The render pipeline exposes a single entry point: render(renderer, scene).
 *
 * Render resolution is configurable at construction and via set_render_resolution().
 */

#ifndef COOPA_GFX_ENGINE_RETRO_RENDER_PIPELINE_H
#define COOPA_GFX_ENGINE_RETRO_RENDER_PIPELINE_H

#include <volk/volk.h>
#include <memory>
#include <string>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/command/sync.h>
#include <gfxcoopa/presentation/renderer.h>
#include <gfxcoopa/engine/offscreen_target.h>
#include <gfxcoopa/engine/camera_ubo.h>
#include <gfxcoopa/engine/light_data.h>
#include <gfxcoopa/engine/model_ubo.h>
#include <gfxcoopa/engine/sampler.h>
#include <gfxcoopa/engine/toon_pipeline.h>
#include <gfxcoopa/engine/outline_pipeline.h>
#include <gfxcoopa/engine/post_process_pipeline.h>
#include <gfxcoopa/engine/upscale_pass.h>
#include <gfxcoopa/engine/shadow_map_target.h>
#include <gfxcoopa/engine/shadow_pipeline.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @struct RetroRenderConfig
 * @brief Configuration for the retro render pipeline.
 */
struct RetroRenderConfig {
    uint32_t render_width  = 320;  /**< Internal render resolution width. */
    uint32_t render_height = 240;  /**< Internal render resolution height. */

    std::string shader_dir = "assets/shaders"; /**< Base path for compiled .spv shaders. */

    float outline_width = 0.03f;   /**< Inverted-hull outline extrusion amount (world units). */
    glm::vec3 outline_color = glm::vec3(0.0f); /**< Outline colour (default black). */

    bool edge_detection_enabled = true;  /**< Enable depth-based Sobel edge detection. */
    bool color_quantize_enabled = false; /**< Enable palette quantization. */
    int  color_quantize_levels  = 0;     /**< Palette quantization levels (0 = off). */
};

/**
 * @class RetroRenderPipeline
 * @brief Owns the entire retro rendering stack and orchestrates per-frame rendering.
 *
 * Usage:
 * @code
 * RetroRenderPipeline rp(device, allocator, swapchain, swapchain_render_pass,
 *                        cmd_pool, {320, 240, "assets/shaders"});
 * // Per frame:
 * rp.render(renderer, scene);
 * @endcode
 */
class RetroRenderPipeline {
public:
    /**
     * @brief Constructs the full render pipeline.
     *
     * @param device              Logical device.
     * @param allocator           VMA allocator.
     * @param swapchain           Swapchain (for extent queries on upscale).
     * @param swapchain_pass      Swapchain-compatible render pass for the upscale pass.
     * @param cmd_pool            Command pool for the pre-frame offscreen command buffer.
     * @param config              Render pipeline configuration.
     */
    RetroRenderPipeline(core::Device&           device,
                        memory::Allocator&       allocator,
                        core::Swapchain&         swapchain,
                        pipeline::RenderPass&    swapchain_pass,
                        command::CommandPool&    cmd_pool,
                        RetroRenderConfig        config = {})
        : device_(device), allocator_(allocator),
          swapchain_(swapchain), config_(std::move(config))
    {
        auto spv = [&](const std::string& name) {
            return config_.shader_dir + "/" + name;
        };

        // --- Samplers ---
        nn_sampler_  = std::make_unique<Sampler>(Sampler::nearest(device));
        lin_sampler_ = std::make_unique<Sampler>(Sampler::linear(device));

        // --- Offscreen target (the main low-res render target) ---
        offscreen_ = std::make_unique<OffscreenTarget>(
            device, allocator, config_.render_width, config_.render_height
        );

        // --- Camera + Light UBOs ---
        camera_ubo_ = std::make_unique<CameraUBO>(device, allocator);
        light_data_ = std::make_unique<LightData>(device, allocator);

        // --- Shadow Map Target & Shadow Pipeline ---
        shadow_target_ = std::make_unique<ShadowMapTarget>(device, allocator, 2048, 512);
        shadow_pipeline_ = std::make_unique<ShadowPipeline>(
            device,
            shadow_target_->dir_render_pass(),
            shadow_target_->cube_render_pass(),
            spv("shadow_depth.vert.spv"),
            spv("shadow_depth.frag.spv"),
            spv("shadow_cube.vert.spv"),
            spv("shadow_cube.frag.spv")
        );

        // --- Descriptor layouts for camera (set 0), light (set 1), shadow maps (set 2) ---
        {
            VkDescriptorSetLayoutBinding cam_binding{};
            cam_binding.binding         = 0;
            cam_binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            cam_binding.descriptorCount = 1;
            cam_binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            camera_layout_ = std::make_unique<pipeline::DescriptorSetLayout>(
                device, std::vector<VkDescriptorSetLayoutBinding>{cam_binding}
            );
        }
        {
            VkDescriptorSetLayoutBinding light_binding{};
            light_binding.binding         = 0;
            light_binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            light_binding.descriptorCount = 1;
            light_binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            light_layout_ = std::make_unique<pipeline::DescriptorSetLayout>(
                device, std::vector<VkDescriptorSetLayoutBinding>{light_binding}
            );
        }
        {
            // Set 2: binding 0 = sampler2D (directional shadow), binding 1 = samplerCube (point light shadow)
            std::vector<VkDescriptorSetLayoutBinding> shadow_bindings(2);
            shadow_bindings[0].binding         = 0;
            shadow_bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            shadow_bindings[0].descriptorCount = 1;
            shadow_bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

            shadow_bindings[1].binding         = 1;
            shadow_bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            shadow_bindings[1].descriptorCount = 1;
            shadow_bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

            shadow_layout_ = std::make_unique<pipeline::DescriptorSetLayout>(device, shadow_bindings);
        }

        // --- Descriptor pool for camera, light, and shadow sets ---
        {
            std::vector<VkDescriptorPoolSize> pool_sizes{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}
            };
            desc_pool_ = std::make_unique<pipeline::DescriptorPool>(device, 3, pool_sizes);
        }

        // --- Descriptor sets ---
        camera_set_ = std::make_unique<pipeline::DescriptorSet>(device, *desc_pool_, *camera_layout_);
        camera_set_->bind_buffer(0, camera_ubo_->buffer());

        light_set_ = std::make_unique<pipeline::DescriptorSet>(device, *desc_pool_, *light_layout_);
        light_set_->bind_buffer(0, light_data_->buffer());

        shadow_set_ = std::make_unique<pipeline::DescriptorSet>(device, *desc_pool_, *shadow_layout_);
        shadow_set_->bind_image(0, shadow_target_->dir_shadow_view(), nn_sampler_->handle());
        shadow_set_->bind_image(1, shadow_target_->cube_shadow_view(), nn_sampler_->handle());

        // --- Toon pipeline ---
        toon_pipeline_ = std::make_unique<ToonPipeline>(
            device,
            offscreen_->render_pass_object(),
            *camera_layout_,
            *light_layout_,
            *shadow_layout_,
            spv("toon.vert.spv"),
            spv("toon.frag.spv")
        );

        // --- Outline pipeline ---
        outline_pipeline_ = std::make_unique<OutlinePipeline>(
            device,
            offscreen_->render_pass_object(),
            *camera_layout_,
            spv("outline.vert.spv"),
            spv("outline.frag.spv")
        );

        // --- Post-process pipeline ---
        post_process_ = std::make_unique<PostProcessPipeline>(
            device, allocator,
            config_.render_width, config_.render_height,
            *lin_sampler_,
            spv("upscale.vert.spv"),   // Shared fullscreen vert shader
            spv("edge_detect.frag.spv"),
            spv("upscale.vert.spv"),
            spv("color_quantize.frag.spv")
        );
        post_process_->set_edge_detection_enabled(config_.edge_detection_enabled);
        post_process_->set_quantization_enabled(config_.color_quantize_enabled);
        if (config_.color_quantize_levels > 0) {
            post_process_->set_quantize_params(config_.color_quantize_levels, 1.0f, 1.0f);
        }

        // --- Upscale pass ---
        upscale_pass_ = std::make_unique<UpscalePass>(
            device, swapchain_pass, *nn_sampler_,
            spv("upscale.vert.spv"),
            spv("upscale.frag.spv")
        );

        // --- Pre-frame command buffer (for offscreen work) ---
        allocate_pre_frame_cmd_(cmd_pool);
    }

    /**
     * @brief Renders one complete frame: offscreen toon+outline, post-process, upscale.
     *
     * Call this inside the main loop. Uses Renderer::begin_frame() for swapchain management.
     *
     * @param renderer  The Renderer managing the swapchain frame loop.
     * @param scene     The scene to render (must have a valid active camera + renderable objects).
     * @return The result of Renderer::begin_frame() (false if window minimized/resizing).
     */
    template<typename Scene>
    bool render(presentation::Renderer& renderer, Scene& scene) {
        // Update camera UBO from the active scene camera.
        update_camera_(scene);

        // Pre-frame: record offscreen toon+outline + post-processing.
        VkImageView final_color_view = record_offscreen_(scene);

        // Swapchain pass: upscale the final low-res image to the window.
        VkExtent2D swapchain_extent = swapchain_.extent();
        return renderer.begin_frame(
            [&](command::CommandBuffer& cmd) {
                upscale_pass_->set_source_image(final_color_view);
                upscale_pass_->draw(cmd,
                                    swapchain_extent.width, swapchain_extent.height,
                                    config_.render_width,    config_.render_height);
            }
        );
    }

    /**
     * @brief Changes the internal render resolution.
     *
     * Call after vkDeviceWaitIdle(). Recreates all resolution-dependent resources.
     *
     * @param width  New render width.
     * @param height New render height.
     */
    void set_render_resolution(uint32_t width, uint32_t height) {
        config_.render_width  = width;
        config_.render_height = height;
        offscreen_->recreate(width, height);
        // Post-process targets recreate automatically inside PostProcessPipeline on next process() call.
    }

    /** @brief Returns mutable light data (modify then it's uploaded per-frame). */
    LightData& light_data() { return *light_data_; }

private:
    /**
     * @brief Allocates the pre-frame command buffer.
     */
    void allocate_pre_frame_cmd_(command::CommandPool& cmd_pool) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool        = cmd_pool.handle();
        alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_.handle(), &alloc_info, &pre_frame_cmd_);
        pre_frame_cmd_pool_ = cmd_pool.handle();
    }

    /**
     * @brief Updates camera UBO from the active scene camera.
     */
    template<typename Scene>
    void update_camera_(Scene& scene) {
        if (!scene.active_camera()) return;

        auto* cam = scene.active_camera();
        float aspect = static_cast<float>(config_.render_width)
                     / static_cast<float>(config_.render_height);

        camera_ubo_->update(cam->get_view_matrix(),
                            cam->get_projection_matrix(aspect),
                            cam->get_world_position());

        auto& ubo = light_data_->data();

        // 1. Directional Light
        if (auto* dl = scene.active_light()) {
            ubo.dir_direction = glm::vec4(dl->direction, dl->intensity);
            ubo.dir_color     = glm::vec4(dl->color, 1.0f);
            ubo.dir_ambient   = glm::vec4(dl->ambient, 1.0f);

            // Compute Light Space Matrix for directional shadow mapping
            glm::vec3 light_dir = glm::normalize(dl->direction);
            glm::vec3 light_pos = -light_dir * 30.0f; // Position light source along inverse vector
            glm::mat4 light_view = glm::lookAt(light_pos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 light_proj = glm::ortho(-15.0f, 15.0f, -15.0f, 15.0f, 0.1f, 60.0f);
            light_proj[1][1] *= -1.0f; // Vulkan Y-flip
            ubo.dir_light_space_matrix = light_proj * light_view;
            ubo.dir_shadow_params = glm::vec4(0.005f, 0.0f, 1.0f, 0.0f); // bias=0.005, shadow_enabled=1
            ubo.light_counts.x = 1;
        } else {
            ubo.light_counts.x = 0;
        }

        // 2. Point Lights
        auto point_lights = scene.get_point_lights();
        uint32_t num_points = std::min(static_cast<uint32_t>(point_lights.size()), MAX_POINT_LIGHTS);
        ubo.light_counts.y = num_points;

        for (uint32_t i = 0; i < num_points; ++i) {
            auto* pl = point_lights[i];
            ubo.point_lights[i].position_range  = glm::vec4(pl->get_world_position(), pl->range);
            ubo.point_lights[i].color_intensity = glm::vec4(pl->color, pl->intensity);
            ubo.point_lights[i].attenuation     = glm::vec4(pl->attenuation_constant, pl->attenuation_linear, pl->attenuation_quadratic, pl->cast_shadows ? 1.0f : 0.0f);
        }

        light_data_->upload();
    }

    /**
     * @brief Records and submits the offscreen (shadows + toon + outline + post-process) command buffer.
     * @return The final post-processed color image view, ready for sampling.
     */
    template<typename Scene>
    VkImageView record_offscreen_(Scene& scene) {
        // Reset and begin the pre-frame command buffer.
        vkResetCommandBuffer(pre_frame_cmd_, 0);

        command::CommandBuffer cmd(pre_frame_cmd_);
        cmd.begin(true); // One-time submit.

        auto renderables = scene.get_renderable_objects();
        auto& ubo = light_data_->data();

        // --- Pass 0a: Directional Light Shadow Pass ---
        if (ubo.light_counts.x > 0) {
            shadow_target_->begin_directional_pass(cmd);
            shadow_pipeline_->bind_directional(cmd);

            for (auto* obj : renderables) {
                auto* mr = obj->get_mesh_renderer();
                auto* tc = obj->get_transform();
                if (!mr || !mr->is_ready() || !tc) continue;

                DirectionalShadowPushConstants pc;
                pc.light_space_matrix = ubo.dir_light_space_matrix;
                pc.model = tc->get_world_matrix();
                shadow_pipeline_->push_directional(cmd, pc);
                mr->get_mesh()->bind(cmd);
                mr->get_mesh()->draw(cmd);
            }
            shadow_target_->end_directional_pass(cmd);
        }

        // --- Pass 0b: Point Light Shadow Pass ---
        auto point_lights = scene.get_point_lights();
        if (!point_lights.empty() && point_lights[0]->cast_shadows) {
            auto* pl = point_lights[0];
            glm::vec3 pl_pos = pl->get_world_position();
            float pl_range   = pl->range;

            for (uint32_t face = 0; face < 6; ++face) {
                shadow_target_->begin_cube_face_pass(cmd, face);
                shadow_pipeline_->bind_cube(cmd);

                CubeShadowPushConstants pc;
                pc.light_space_matrix = ShadowMapTarget::get_cube_face_matrix(face, pl_pos, pl_range);
                pc.light_pos_range    = glm::vec4(pl_pos, pl_range);

                for (auto* obj : renderables) {
                    auto* mr = obj->get_mesh_renderer();
                    auto* tc = obj->get_transform();
                    if (!mr || !mr->is_ready() || !tc) continue;

                    pc.model = tc->get_world_matrix();
                    shadow_pipeline_->push_cube(cmd, pc);
                    mr->get_mesh()->bind(cmd);
                    mr->get_mesh()->draw(cmd);
                }
                shadow_target_->end_cube_face_pass(cmd);
            }
        }

        // --- Offscreen toon + outline pass ---
        offscreen_->begin(cmd);

        // Bind toon pipeline.
        toon_pipeline_->bind(cmd);
        cmd.bind_descriptor_set(toon_pipeline_->layout(), *camera_set_, 0);
        cmd.bind_descriptor_set(toon_pipeline_->layout(), *light_set_,  1);
        cmd.bind_descriptor_set(toon_pipeline_->layout(), *shadow_set_, 2);

        // Draw all renderable objects.
        for (auto* obj : renderables) {
            // Duck-typed: works with any SceneObject that has MeshRenderer and TransformComponent.
            auto* mr = obj->get_mesh_renderer();
            auto* tc = obj->get_transform();
            if (!mr || !mr->is_ready() || !tc) continue;

            ModelPushConstants model_pc;
            model_pc.set_model(tc->get_world_matrix());
            toon_pipeline_->push_model(cmd, model_pc);
            mr->get_mesh()->bind(cmd);
            mr->get_mesh()->draw(cmd);
        }

        // --- Outline pass (same objects, front-face cull) ---
        outline_pipeline_->bind(cmd);
        cmd.bind_descriptor_set(outline_pipeline_->layout(), *camera_set_, 0);

        for (auto* obj : renderables) {
            auto* mr = obj->get_mesh_renderer();
            auto* tc = obj->get_transform();
            if (!mr || !mr->is_ready() || !tc) continue;

            OutlinePushConstants out_pc;
            out_pc.model_data.set_model(tc->get_world_matrix());
            out_pc.set_outline(config_.outline_color, config_.outline_width);
            outline_pipeline_->push(cmd, out_pc);
            mr->get_mesh()->bind(cmd);
            mr->get_mesh()->draw(cmd);
        }

        offscreen_->end(cmd);

        // --- Post-processing ---
        VkImageView final_view = post_process_->process(
            cmd,
            offscreen_->color_view(),
            offscreen_->depth_view()
        );

        cmd.end();

        // Submit pre-frame work.
        VkCommandBuffer raw = cmd.handle();
        VkSubmitInfo submit{};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &raw;
        vkQueueSubmit(device_.graphics_queue(), 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(device_.graphics_queue());

        return final_view;
    }

    core::Device&       device_;
    memory::Allocator&  allocator_;
    core::Swapchain&    swapchain_;
    RetroRenderConfig   config_;

    // Samplers.
    std::unique_ptr<Sampler> nn_sampler_;
    std::unique_ptr<Sampler> lin_sampler_;

    // Render targets.
    std::unique_ptr<OffscreenTarget> offscreen_;
    std::unique_ptr<ShadowMapTarget> shadow_target_;

    // UBOs.
    std::unique_ptr<CameraUBO>  camera_ubo_;
    std::unique_ptr<LightData>  light_data_;

    // Descriptor infrastructure.
    std::unique_ptr<pipeline::DescriptorSetLayout> camera_layout_;
    std::unique_ptr<pipeline::DescriptorSetLayout> light_layout_;
    std::unique_ptr<pipeline::DescriptorSetLayout> shadow_layout_;
    std::unique_ptr<pipeline::DescriptorPool>      desc_pool_;
    std::unique_ptr<pipeline::DescriptorSet>       camera_set_;
    std::unique_ptr<pipeline::DescriptorSet>       light_set_;
    std::unique_ptr<pipeline::DescriptorSet>       shadow_set_;

    // Rendering stages.
    std::unique_ptr<ToonPipeline>        toon_pipeline_;
    std::unique_ptr<OutlinePipeline>     outline_pipeline_;
    std::unique_ptr<ShadowPipeline>      shadow_pipeline_;
    std::unique_ptr<PostProcessPipeline> post_process_;
    std::unique_ptr<UpscalePass>         upscale_pass_;

    // Pre-frame command buffer (for offscreen work).
    VkCommandBuffer  pre_frame_cmd_      = VK_NULL_HANDLE;
    VkCommandPool    pre_frame_cmd_pool_ = VK_NULL_HANDLE;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_RETRO_RENDER_PIPELINE_H
