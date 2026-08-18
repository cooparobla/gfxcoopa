/**
 * @file ssr_pass.h
 * @brief Screen-Space Reflections (SSR) pass with Hi-Z raymarching and BRDF compositing.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SSR_PASS_H
#define GFXCOOPA_ENGINE_PASSES_SSR_PASS_H

#include <volk/volk.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/engine/targets/offscreen_target.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/engine/passes/hiz_pass.h>
#include <gfxcoopa/engine/gi/gi_system.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



class SsrPass {
public:
    struct SsrPushConstants {
        glm::mat4 inv_proj;
        float max_distance;      // world-space ray length (ssr_max_distance)
        float bias_texels;       // normal bias, in full-res screen texels (ssr_bias_texels)
        float thickness_min;     // world-space floor for the thickness test (ssr_thickness)
        float thickness_scale;   // thickness as a fraction of |view z| (ssr_thickness_scale)
        float roughness_cutoff;
        int   max_iterations;
        int   max_hiz_mip;
        int   start_mip;         // Hi-Z mip the march starts at (ssr_start_mip)
        int   min_mip0_steps;    // self-reflection gate (ssr_min_mip0_steps)
        int   max_color_mip = 0; // top mip of the prefiltered scene-colour chain
    };

    struct ResolvePushConstants {
        glm::mat4 prev_view_proj;   // offset 0  (mat4 must be 16-byte aligned; put it first)
        glm::vec2 resolution;       // offset 64
        float     blend_factor;     // offset 72
        int       history_valid;    // offset 76
    };                              // 80 bytes

    struct CompositePushConstants {
        glm::vec2 ssr_resolution;      // resolution of u_ssr_map (trace res under half-res)
        glm::vec2 screen_resolution;   // always full screen res
        int       half_res;
    };                                 // 20 bytes

    /// Per-frame parameters for execute(). Collapsed into a struct because the argument list
    /// kept growing phase over phase (scale-relative tuning, then reprojection, then the
    /// glossy/half-res knobs) -- past ~6 positional args of the same type, a call site is not
    /// self-documenting and is easy to mis-order.
    struct Params {
        glm::mat4 proj;
        int   max_iterations   = 64;
        float thickness_min    = 0.05f;
        float thickness_scale  = 0.01f;
        float max_distance     = 15.0f;
        float bias_texels      = 3.5f;
        float roughness_cutoff = 0.6f;
        int   max_hiz_mip      = 0;      // hiz_pass_->max_mip_level(), filled by the caller
        int   start_mip        = 0;
        int   min_mip0_steps   = 1;
        int   max_color_mip    = 0;      // scene_color_mip_pass_->max_mip_level(), filled by caller
        bool  temporal_enabled = true;
        float temporal_blend   = 0.85f;
        // Reprojection: previous frame's JITTERED proj * view, and whether it (and the history
        // buffer) actually exist yet. False for the first two frames and right after a resize.
        glm::mat4 prev_view_proj       = glm::mat4(1.0f);
        bool      prev_view_proj_valid = false;
    };

    SsrPass(coopa::gfx::core::Device& device,
            coopa::gfx::memory::Allocator& allocator,
            VkDescriptorSetLayout camera_layout,
            VkDescriptorSetLayout gi_layout,
            uint32_t width,
            uint32_t height,
            const std::string& ssr_vert_spv,
            const std::string& ssr_frag_spv,
            const std::string& comp_vert_spv,
            const std::string& comp_frag_spv,
            const std::string& resolve_vert_spv,
            const std::string& resolve_frag_spv,
            bool half_res = false)
        : device_(device), allocator_(allocator), width_(width), height_(height), half_res_(half_res)
    {
        // Trace resolution. The march, the temporal resolve, and the history buffer run here;
        // the composite always runs at full screen resolution and bilaterally upsamples. Floor
        // division (not +1 rounding) keeps the composite's "full = 2*half + 1" tap mapping exact
        // at even screen dimensions.
        trace_width_  = half_res_ ? std::max(1u, width_  / 2) : width_;
        trace_height_ = half_res_ ? std::max(1u, height_ / 2) : height_;
        // 0. Nearest-filtered sampler for the in-march G-buffer point-lookups in ssr.frag
        // (self-hit rejection, backface test). Those sample at an arbitrary marched UV, not
        // a texel center, so a LINEAR sampler bilinearly blends world positions/normals across
        // silhouette edges into values that exist on no real surface -- a direct source of
        // edge speckle. The G-buffer is discrete per-pixel data; it should never be smoothed.
        nearest_sampler_ = std::make_unique<util::Sampler>(
            device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
        );

        // 1. Create Offscreen Targets. target_/resolved_target_/history_image_ run at trace
        // resolution (full res, or half under ssr_half_res); composite_target_ always runs at
        // full screen resolution and bilaterally upsamples the resolved SSR buffer into it.
        target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_1_BIT
        );
        composite_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_1_BIT
        );
        // Temporal resolve output: the raymarch result (target_) blended with history, read
        // by the composite pass in place of the raw raymarch output.
        resolved_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_1_BIT
        );

        // History image: a copy of last frame's resolved_target_, TRANSFER_DST so update_history_()
        // can vkCmdCopyImage into it (same pattern as TaaPass::history_image_).
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device, allocator, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        history_initialized_ = false;

        // 2. Shaders
        ssr_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, ssr_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        ssr_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, ssr_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        comp_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, comp_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        comp_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, comp_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        resolve_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, resolve_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        resolve_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, resolve_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // 3. Descriptor Set Layouts
        // G-Buffer layout (3 images: G0, G1, G2) for the SSR raymarch set -- it no longer reads
        // G_depth (Hi-Z supplies depth).
        std::vector<VkDescriptorSetLayoutBinding> gbuf3_bindings;
        for (uint32_t i = 0; i < 3; ++i) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = i;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            gbuf3_bindings.push_back(b);
        }
        gbuf3_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, gbuf3_bindings);

        // Composite G-buffer layout: same G0-G2 plus an SSAO sampler (binding 3), so the
        // composite can attenuate the SSR/env delta by the same ao * ssao term
        // deferred_lighting.frag applies to indirect_specular (see ssr_composite.frag).
        std::vector<VkDescriptorSetLayoutBinding> comp_gbuf_bindings = gbuf3_bindings;
        {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = 3;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            comp_gbuf_bindings.push_back(b);
        }
        comp_gbuf_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, comp_gbuf_bindings);

        // Single image sampler layouts
        std::vector<VkDescriptorSetLayoutBinding> single_img = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        hiz_layout_         = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, single_img);
        scene_color_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, single_img);
        raw_ssr_layout_     = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, single_img);

        // Resolve pass layout: 0 = current raymarch output, 1 = history, 2 = G2 world position
        // (the reprojection source -- G2 already stores world position, so no motion-vector
        // attachment is needed anywhere in the engine).
        std::vector<VkDescriptorSetLayoutBinding> resolve_bindings = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        resolve_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(device, resolve_bindings);

        // 4. Descriptor Pool (6 sets / 12 combined-image-sampler descriptors used; sized with headroom)
        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            device, 12,
            std::vector<VkDescriptorPoolSize>{
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 25}
            }
        );

        // Allocate Descriptor Sets
        ssr_gbuf_set_    = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *gbuf3_layout_);
        hiz_set_         = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *hiz_layout_);
        scene_color_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *scene_color_layout_);
        resolve_set_     = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *resolve_layout_);

        comp_gbuf3_set_       = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *comp_gbuf_layout_);
        comp_raw_ssr_set_     = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *raw_ssr_layout_);
        comp_scene_color_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *scene_color_layout_);

        // 5. SSR Pipeline Creation
        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(SsrPushConstants);

        std::vector<VkDescriptorSetLayout> ssr_layouts = {
            camera_layout,
            gbuf3_layout_->handle(),
            hiz_layout_->handle(),
            scene_color_layout_->handle()
        };

        ssr_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, target_->render_pass_object(),
            std::vector<coopa::gfx::pipeline::Shader*>{ssr_vert_.get(), ssr_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            ssr_layouts,
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );

        // 6. SSR Composite Pipeline Creation
        std::vector<VkDescriptorSetLayout> comp_layouts = {
            camera_layout,
            comp_gbuf_layout_->handle(),
            raw_ssr_layout_->handle(),
            scene_color_layout_->handle(),
            gi_layout
        };

        VkPushConstantRange comp_pc_range{};
        comp_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        comp_pc_range.offset     = 0;
        comp_pc_range.size       = sizeof(CompositePushConstants);

        comp_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, composite_target_->render_pass_object(),
            std::vector<coopa::gfx::pipeline::Shader*>{comp_vert_.get(), comp_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            comp_layouts,
            cfg,
            std::vector<VkPushConstantRange>{comp_pc_range}
        );

        // 7. Temporal Resolve Pipeline Creation
        VkPushConstantRange resolve_pc_range{};
        resolve_pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        resolve_pc_range.offset     = 0;
        resolve_pc_range.size       = sizeof(ResolvePushConstants);

        resolve_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, resolved_target_->render_pass_object(),
            std::vector<coopa::gfx::pipeline::Shader*>{resolve_vert_.get(), resolve_frag_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{resolve_layout_->handle()},
            cfg,
            std::vector<VkPushConstantRange>{resolve_pc_range}
        );
    }

    void recreate(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        trace_width_  = half_res_ ? std::max(1u, width_  / 2) : width_;
        trace_height_ = half_res_ ? std::max(1u, height_ / 2) : height_;

        target_->recreate(trace_width_, trace_height_);
        composite_target_->recreate(width, height);
        resolved_target_->recreate(trace_width_, trace_height_);

        // History no longer matches the new resolution -- drop it and start fresh.
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        history_initialized_ = false;
    }

    void update_descriptors(const targets::GBufferTarget& gbuffer,
                            VkImageView hiz_view,
                            VkSampler hiz_sampler,
                            VkImageView scene_color_mip_view,
                            VkSampler scene_color_mip_sampler,
                            VkImageView scene_color_view,
                            const util::Sampler& linear_sampler)
    {
        // SSR Raymarching descriptors. Nearest sampler: see constructor comment -- these are
        // read both at the texel-centered in_uv (where nearest == linear, no change) and at
        // arbitrary marched UVs during the hit tests (where nearest is required for correctness).
        ssr_gbuf_set_->bind_image(0, gbuffer.g0_view(), nearest_sampler_->handle());
        ssr_gbuf_set_->bind_image(1, gbuffer.g1_view(), nearest_sampler_->handle());
        ssr_gbuf_set_->bind_image(2, gbuffer.g2_view(), nearest_sampler_->handle());

        hiz_set_->bind_image(0, hiz_view, hiz_sampler);
        // The march samples the PREFILTERED chain (cone footprint -> textureLod), while the
        // composite below samples the raw full-res scene colour it is compositing INTO. Same
        // descriptor layout, deliberately different images.
        scene_color_set_->bind_image(0, scene_color_mip_view, scene_color_mip_sampler);

        // Temporal resolve descriptors: current frame's raw raymarch output + last frame's
        // resolved history + G2 (the reprojection source). The current buffer is still sampled
        // at texel-centered in_uv, so nearest_sampler_ avoids implying this HDR data buffer
        // should ever be blurred.
        resolve_set_->bind_image(0, target_->color_view(), nearest_sampler_->handle());
        // LINEAR, not nearest: reprojected UVs are no longer texel-centred, and point-sampling
        // them makes the accumulated reflection stair-step and crawl under camera motion.
        resolve_set_->bind_image(1, history_image_->view(), linear_sampler.handle());
        resolve_set_->bind_image(2, gbuffer.g2_view(), nearest_sampler_->handle());

        // SSR Composite descriptors
        comp_gbuf3_set_->bind_image(0, gbuffer.g0_view(), linear_sampler.handle());
        comp_gbuf3_set_->bind_image(1, gbuffer.g1_view(), linear_sampler.handle());
        comp_gbuf3_set_->bind_image(2, gbuffer.g2_view(), linear_sampler.handle());

        // Composite now reads the temporally-resolved buffer, not the raw single-sample trace.
        comp_raw_ssr_set_->bind_image(0, resolved_target_->color_view(), linear_sampler.handle());
        comp_scene_color_set_->bind_image(0, scene_color_view, linear_sampler.handle());
    }

    /// Binding 3 of the composite G-buffer set must be rebound every frame -- callers pass the
    /// SSAO pass's blurred output when enabled, or its permanent neutral (fully-unoccluded)
    /// texture when disabled/absent, mirroring DeferredLightingPass::set_ssao_image() so the two
    /// passes always attenuate indirect specular by the identical ao * ssao term.
    void set_ssao_image(VkImageView ssao_view, VkSampler ssao_sampler) {
        comp_gbuf3_set_->bind_image(3, ssao_view, ssao_sampler);
    }

    void execute(coopa::gfx::command::CommandBuffer& cmd,
                 const coopa::gfx::pipeline::DescriptorSet& camera_set,
                 const gi::GiSystem* gi_system,
                 const Params& params)
    {
        // On the very first frame (or right after a resize), history_image_ is UNDEFINED --
        // transition it once before it's bound as a sampled image in the resolve pass below.
        // The resolve shader doesn't read it in that case (history_valid = 0 in the push
        // constant), but the descriptor binding still needs a valid layout at draw time
        // regardless of the runtime branch.
        if (!history_initialized_) {
            VkImageMemoryBarrier barrier{};
            barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.image                           = history_image_->handle();
            barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel   = 0;
            barrier.subresourceRange.levelCount     = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount     = 1;
            barrier.srcAccessMask                   = 0;
            barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        // 1. Raymarching Pass -- runs at trace resolution (full res, or half under ssr_half_res).
        target_->begin(cmd);
        cmd.bind_pipeline(*ssr_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(trace_width_), static_cast<float>(trace_height_));
        cmd.set_scissor(0, 0, trace_width_, trace_height_);

        SsrPushConstants pc{};
        pc.inv_proj         = glm::inverse(params.proj);
        pc.max_distance     = params.max_distance;
        pc.bias_texels      = params.bias_texels;
        pc.thickness_min    = params.thickness_min;
        pc.thickness_scale  = params.thickness_scale;
        pc.roughness_cutoff = params.roughness_cutoff;
        pc.max_iterations   = params.max_iterations;
        pc.max_hiz_mip      = params.max_hiz_mip;
        pc.start_mip        = params.start_mip;
        pc.min_mip0_steps   = params.min_mip0_steps;
        pc.max_color_mip    = params.max_color_mip;

        cmd.push_constants(ssr_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SsrPushConstants), &pc);

        cmd.bind_descriptor_set(ssr_pipeline_->layout(), camera_set, 0);
        cmd.bind_descriptor_set(ssr_pipeline_->layout(), *ssr_gbuf_set_, 1);
        cmd.bind_descriptor_set(ssr_pipeline_->layout(), *hiz_set_, 2);
        cmd.bind_descriptor_set(ssr_pipeline_->layout(), *scene_color_set_, 3);

        cmd.draw(3);
        target_->end(cmd);

        // 2. Temporal Resolve Pass: blend target_ (this frame's raw trace) with history_image_
        // (last frame's resolved output) into resolved_target_, which the composite pass below
        // reads instead of target_ directly. When temporal_enabled is false, blend_factor = 0
        // degenerates this into a pure passthrough of the current frame -- one code path, no
        // branching pipeline structure.
        resolved_target_->begin(cmd);
        cmd.bind_pipeline(*resolve_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(trace_width_), static_cast<float>(trace_height_));
        cmd.set_scissor(0, 0, trace_width_, trace_height_);

        ResolvePushConstants rpc{};
        rpc.prev_view_proj = params.prev_view_proj;
        rpc.resolution     = glm::vec2(static_cast<float>(trace_width_), static_cast<float>(trace_height_));
        rpc.blend_factor   = params.temporal_enabled ? params.temporal_blend : 0.0f;
        // Both a history IMAGE (history_initialized_) and a previous view-projection MATRIX
        // (prev_view_proj_valid) must exist -- the matrix lags a frame behind the image on a
        // fresh start (frames 0 and 1), so ANDing them is what keeps the very first reprojected
        // sample from reading a stale/identity prev_view_proj.
        rpc.history_valid = (history_initialized_ && params.prev_view_proj_valid) ? 1 : 0;

        cmd.push_constants(resolve_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ResolvePushConstants), &rpc);
        cmd.bind_descriptor_set(resolve_pipeline_->layout(), *resolve_set_, 0);

        cmd.draw(3);
        resolved_target_->end(cmd);

        // 3. Composite Pass -- always full screen resolution; bilaterally upsamples the
        // (possibly trace-resolution) resolved SSR buffer.
        composite_target_->begin(cmd);
        cmd.bind_pipeline(*comp_pipeline_);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_));
        cmd.set_scissor(0, 0, width_, height_);

        CompositePushConstants cpc{};
        cpc.ssr_resolution    = glm::vec2(static_cast<float>(trace_width_), static_cast<float>(trace_height_));
        cpc.screen_resolution = glm::vec2(static_cast<float>(width_), static_cast<float>(height_));
        cpc.half_res          = half_res_ ? 1 : 0;
        cmd.push_constants(comp_pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CompositePushConstants), &cpc);

        cmd.bind_descriptor_set(comp_pipeline_->layout(), camera_set, 0);
        cmd.bind_descriptor_set(comp_pipeline_->layout(), *comp_gbuf3_set_, 1);
        cmd.bind_descriptor_set(comp_pipeline_->layout(), *comp_raw_ssr_set_, 2);
        cmd.bind_descriptor_set(comp_pipeline_->layout(), *comp_scene_color_set_, 3);
        if (gi_system) {
            gi_system->bind_at_set(cmd, comp_pipeline_->layout(), 4);
        }

        cmd.draw(3);
        composite_target_->end(cmd);

        // 4. Copy resolved_target_ into history_image_ for next frame's resolve pass.
        update_ssr_history_(cmd);
    }

    VkImageView output_view() const { return composite_target_->color_view(); }
    targets::OffscreenTarget& composite_target() { return *composite_target_; }

private:
    // Copies resolved_target_'s color image into history_image_ (same barrier/copy/barrier
    // pattern as TaaPass::update_history), so the next frame's resolve pass has something to
    // blend against.
    void update_ssr_history_(coopa::gfx::command::CommandBuffer& cmd) {
        VkImage src_image = resolved_target_->color_image_object()->handle();
        VkImage dst_image = history_image_->handle();

        VkImageMemoryBarrier barriers[2]{};

        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src_image;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;
        barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // Already in SHADER_READ_ONLY_OPTIMAL: either from the one-time UNDEFINED transition
        // at the top of execute() (frame 0) or from the end of this same function last frame.
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image = dst_image;
        barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[1].subresourceRange.baseMipLevel = 0;
        barriers[1].subresourceRange.levelCount = 1;
        barriers[1].subresourceRange.baseArrayLayer = 0;
        barriers[1].subresourceRange.layerCount = 1;
        barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, barriers);

        VkImageCopy copy_region{};
        copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.srcSubresource.layerCount = 1;
        copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.dstSubresource.layerCount = 1;
        copy_region.extent = { trace_width_, trace_height_, 1 };

        vkCmdCopyImage(cmd.handle(), src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, barriers);

        history_initialized_ = true;
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    uint32_t width_;
    uint32_t height_;

    // Trace resolution -- ctor param, not runtime-toggleable (changing it reallocates targets).
    bool     half_res_ = false;
    uint32_t trace_width_  = 0;
    uint32_t trace_height_ = 0;

    std::unique_ptr<util::Sampler>          nearest_sampler_;

    std::unique_ptr<targets::OffscreenTarget> target_;
    std::unique_ptr<targets::OffscreenTarget> composite_target_;
    std::unique_ptr<targets::OffscreenTarget> resolved_target_;

    std::unique_ptr<coopa::gfx::memory::Image> history_image_;
    bool history_initialized_ = false;

    std::unique_ptr<coopa::gfx::pipeline::Shader> ssr_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> ssr_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> comp_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> comp_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> resolve_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> resolve_frag_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> gbuf3_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> comp_gbuf_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> hiz_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> scene_color_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> raw_ssr_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> resolve_layout_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool> desc_pool_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> ssr_gbuf_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> hiz_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> scene_color_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> resolve_set_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> comp_gbuf3_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> comp_raw_ssr_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> comp_scene_color_set_;

    std::unique_ptr<coopa::gfx::pipeline::Pipeline> ssr_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> comp_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> resolve_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SSR_PASS_H
