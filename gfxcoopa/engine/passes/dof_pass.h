/**
 * @file dof_pass.h
 * @brief Physically-based depth of field: thin-lens circle of confusion ->
 *        half-resolution golden-angle spiral bokeh gather -> full-resolution
 *        composite (see assets/shaders/dof_coc.frag, dof_bokeh.frag,
 *        dof_composite.frag).
 *
 * Modelled on BloomPass's shape (owns its own OffscreenTargets, one Pipeline per
 * distinct shader, every descriptor bound once at construction -- so unlike
 * HiZPass/SceneColorMipPass this needs no per-frame device_.wait_idle()), but
 * with a fixed three-stage chain instead of a pyramid, and three DIFFERENT
 * shaders rather than one shader reused across stages (contrast TiltShiftPass,
 * whose two stages share a single pipeline).
 *
 * The CoC here is a function of scene DEPTH, not screen position -- the opposite
 * of TiltShiftPass, whose own file doc explains why a depth-driven CoC would
 * break a separable horizontal-then-vertical Gaussian: hard discontinuities at
 * foreground/background silhouettes leak across the two passes (sharp foreground
 * bleeding into blurred background, or vice versa). This pass sidesteps that by
 * never separating the blur: dof_bokeh.frag's spiral gather is a single pass
 * whose occlusion-aware tap weight (see its own doc) is exactly the asymmetric
 * behaviour a separable blur cannot express.
 *
 * Runs at RENDER resolution (unlike TiltShiftPass, which deliberately runs at
 * DISPLAY resolution after the upscale -- see its file doc): DOF is a property
 * of the lens forming the image, not a filter over the already-formed pixel-art
 * frame, and depth is only available at render resolution anyway (GBufferTarget
 * never allocates it at display resolution). It is inserted after fog and before
 * BloomPass in PixelRenderPipeline's chain, so defocused HDR highlights bloom
 * into real bokeh instead of DOF blurring an already-tonemapped, already-bloomed
 * image.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_DOF_PASS_H
#define GFXCOOPA_ENGINE_PASSES_DOF_PASS_H

#include <volk/volk.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <memory>
#include <string>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/engine/targets/offscreen_target.h>
#include <gfxcoopa/types/texture_view.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

/**
 * @class DofPass
 * @brief Thin-lens depth of field: circle-of-confusion, half-resolution bokeh
 *        gather, then full-resolution composite.
 *
 * Owns its three OffscreenTargets and one pipeline per stage. Every descriptor
 * is bound at construction, so execute() needs no per-frame descriptor update
 * and no device wait.
 */
class DofPass {
public:
    /// Per-frame tunables. All are push constants -- unlike this pass's descriptor
    /// bindings (fixed at construction, see the file doc) these are free to change
    /// every frame with no rebuild.
    struct Params {
        float focus_distance = 8.0f;    ///< Metres from the camera to the sharp plane.
        float focal_length_mm = 50.0f;  ///< Resolved lens focal length (camera's `lens` or config override).
        float sensor_width_mm = 36.0f;  ///< Resolved sensor width; sets the mm -> pixel scale.
        float aperture = 2.8f;          ///< f-stop N; aperture diameter = focal_length / N.
        float max_radius = 12.0f;       ///< |CoC| ceiling, in FULL-res pixels.
        int   sample_count = 32;        ///< Spiral taps; clamped to [8, MAX_DOF_TAPS] in-shader.
        int   blade_count = 0;          ///< < 3 = perfect disc; else an N-sided polygonal iris.
        float blade_rotation_deg = 0.0f; ///< Iris rotation, degrees.
        float camera_near = 0.1f;
        float camera_far = 1000.0f;
        bool  camera_is_perspective = true;
        bool  debug_view = false;       ///< Composite emits the signed CoC field instead of the image.
    };

    struct PushConstants {         // vec4s first so std430's 16-byte alignment lands at offset 0
        glm::vec4 lens;             // offset  0: focus_distance_m, focal_length_m, aperture_diameter_m, coc_px_per_m
        glm::vec4 camera;           // offset 16: near, far, is_perspective, max_radius (full-res px)
        glm::vec4 bokeh;            // offset 32: sample_count, blade_count, blade_rotation_rad, debug_view
        glm::vec2 inv_size;         // offset 48: 1/full_w, 1/full_h
    };
    static_assert(sizeof(PushConstants) == 56,
                 "PushConstants must match gfx/dof_common.glsl's DofPush byte-for-byte");

    /**
     * @brief Builds the three-stage CoC/gather/composite chain and its pipelines.
     * @param device        Logical device.
     * @param allocator     VMA allocator, for the three owned OffscreenTargets.
     * @param full_width    Render-resolution width in pixels (NOT display resolution).
     * @param full_height   Render-resolution height in pixels.
     * @param source_hdr    Linear HDR scene colour to defocus (e.g. fog_target_'s color view).
     * @param source_depth  Gbuffer depth at the SAME resolution as source_hdr.
     * @param linear_sampler  Sampler for the half-res intermediates and the bokeh upsample.
     * @param nearest_sampler Sampler for depth and the full-res sharp source -- linear
     *                        filtering of D32_Sfloat is an optional Vulkan format feature
     *                        this engine never queries, and would blend across the very
     *                        discontinuities the CoC formula is measuring (see
     *                        pixel_stylize_pass.h's identical reasoning for its own outline).
     * @param vert_spv        Fullscreen-triangle vertex shader.
     * @param coc_frag_spv    dof_coc.frag.
     * @param bokeh_frag_spv  dof_bokeh.frag.
     * @param composite_frag_spv dof_composite.frag.
     */
    DofPass(coopa::gfx::core::Device& device,
            coopa::gfx::memory::Allocator& allocator,
            uint32_t full_width, uint32_t full_height,
            coopa::gfx::TextureView source_hdr,
            coopa::gfx::TextureView source_depth,
            const coopa::gfx::engine::util::Sampler& linear_sampler,
            const coopa::gfx::engine::util::Sampler& nearest_sampler,
            const std::string& vert_spv,
            const std::string& coc_frag_spv,
            const std::string& bokeh_frag_spv,
            const std::string& composite_frag_spv)
        : full_width_(full_width), full_height_(full_height),
          half_width_(std::max(1u, full_width / 2u)), half_height_(std::max(1u, full_height / 2u)),
          coc_target_(device, allocator, half_width_, half_height_, coopa::gfx::Format::RGBA16_Sfloat),
          bokeh_target_(device, allocator, half_width_, half_height_, coopa::gfx::Format::RGBA16_Sfloat),
          result_target_(device, allocator, full_width, full_height, coopa::gfx::Format::RGBA16_Sfloat)
    {
        vert_shader_      = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        coc_frag_         = std::make_unique<coopa::gfx::pipeline::Shader>(device, coc_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        bokeh_frag_       = std::make_unique<coopa::gfx::pipeline::Shader>(device, bokeh_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        composite_frag_   = std::make_unique<coopa::gfx::pipeline::Shader>(device, composite_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        // One layout shape per stage -- coc_layout_ (colour + depth), bokeh_layout_
        // (its own half-res output only), composite_layout_ (sharp colour + bokeh +
        // depth, recomputing its own full-res CoC -- see dof_composite.frag's doc).
        coc_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .build(device));
        bokeh_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .build(device));
        composite_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(
            coopa::gfx::pipeline::DescriptorPoolBuilder()
                .add_sets(*coc_layout_, 1)
                .add_sets(*bokeh_layout_, 1)
                .add_sets(*composite_layout_, 1)
                .build(device));

        coc_set_       = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *coc_layout_);
        bokeh_set_     = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *bokeh_layout_);
        composite_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *composite_layout_);

        coc_set_->bind_image(0, source_hdr, linear_sampler);
        coc_set_->bind_image(1, source_depth, nearest_sampler);
        bokeh_set_->bind_image(0, coc_target_.color_view_typed(), linear_sampler);
        composite_set_->bind_image(0, source_hdr, nearest_sampler);
        composite_set_->bind_image(1, bokeh_target_.color_view_typed(), linear_sampler);
        composite_set_->bind_image(2, source_depth, nearest_sampler);

        coopa::gfx::pipeline::PipelineDesc common;
        common.vertex      = coopa::gfx::VertexLayout::none();
        common.raster.cull = coopa::gfx::CullMode::None;
        common.depth.test  = false;
        common.depth.write = false;
        common.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(PushConstants)}};

        coopa::gfx::pipeline::PipelineDesc coc_desc = common;
        coc_desc.shaders = {vert_shader_.get(), coc_frag_.get()};
        coc_desc.descriptor_layouts = {coc_layout_.get()};
        coc_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, coc_target_.render_pass_object(), coc_desc);

        coopa::gfx::pipeline::PipelineDesc bokeh_desc = common;
        bokeh_desc.shaders = {vert_shader_.get(), bokeh_frag_.get()};
        bokeh_desc.descriptor_layouts = {bokeh_layout_.get()};
        bokeh_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, bokeh_target_.render_pass_object(), bokeh_desc);

        coopa::gfx::pipeline::PipelineDesc composite_desc = common;
        composite_desc.shaders = {vert_shader_.get(), composite_frag_.get()};
        composite_desc.descriptor_layouts = {composite_layout_.get()};
        composite_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, result_target_.render_pass_object(), composite_desc);
    }

    DofPass(const DofPass&) = delete;
    DofPass& operator=(const DofPass&) = delete;

    /// @brief Runs CoC+downsample -> bokeh gather -> composite. Leaves the finished
    /// result in result_view_typed()/result_image().
    ///
    /// No manual VkImageMemoryBarrier between stages: every target here is an
    /// OffscreenTarget whose render pass ends in SHADER_READ_ONLY_OPTIMAL and emits
    /// an exit subpass dependency (see render_pass.h's needs_exit_dependency block),
    /// exactly the write-then-read hazard between consecutive stages here --
    /// BloomPass and TiltShiftPass chain OffscreenTargets the same way.
    void execute(coopa::gfx::command::CommandBuffer& cmd, const Params& params) const {
        PushConstants pc{};

        float focal_length_m = params.focal_length_mm / 1000.0f;
        // aperture_diameter = focal_length / N (the thin-lens f-number definition);
        // guarded against N <= 0 (a bad config value) the same way dof_common.glsl's
        // dof_signed_coc() guards its own denominator, rather than propagating Inf.
        float aperture_diameter_m = params.aperture > 1e-4f ? focal_length_m / params.aperture : 0.0f;
        float sensor_width_m = std::max(params.sensor_width_mm, 1e-3f) / 1000.0f;
        float coc_px_per_m = static_cast<float>(full_width_) / sensor_width_m;

        pc.lens   = glm::vec4(params.focus_distance, focal_length_m, aperture_diameter_m, coc_px_per_m);
        pc.camera = glm::vec4(params.camera_near, params.camera_far,
                               params.camera_is_perspective ? 1.0f : 0.0f, params.max_radius);
        pc.bokeh  = glm::vec4(static_cast<float>(params.sample_count),
                               static_cast<float>(params.blade_count),
                               glm::radians(params.blade_rotation_deg),
                               params.debug_view ? 1.0f : 0.0f);
        pc.inv_size = glm::vec2(1.0f / static_cast<float>(full_width_), 1.0f / static_cast<float>(full_height_));

        begin_stage_(cmd, coc_target_, *coc_pipeline_, *coc_set_);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);
        cmd.draw(3);
        coc_target_.end(cmd);

        begin_stage_(cmd, bokeh_target_, *bokeh_pipeline_, *bokeh_set_);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);
        cmd.draw(3);
        bokeh_target_.end(cmd);

        begin_stage_(cmd, result_target_, *composite_pipeline_, *composite_set_);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);
        cmd.draw(3);
        result_target_.end(cmd);
    }

    /// @brief The finished, defocused HDR image, at (full_width, full_height) --
    /// bit-identical to the source wherever the CoC is under the 1px early-out.
    coopa::gfx::TextureView result_view_typed() const { return result_target_.color_view_typed(); }
    /// @brief Sealed sibling of result_view_typed(), for PNG readback.
    coopa::gfx::memory::Image& result_image() const { return *result_target_.color_image_object(); }

    uint32_t width()  const { return full_width_; }
    uint32_t height() const { return full_height_; }

private:
    /// begin() + bind + POSITIVE-height viewport. The positive height is not
    /// redundant: OffscreenTarget::begin() sets a NEGATIVE-height viewport (its
    /// Vulkan-NDC Y-flip for geometry passes), which would flip a fullscreen-
    /// triangle pass's output vertically -- see TiltShiftPass::begin_stage_() /
    /// BloomPass::begin_stage_() for the same override.
    void begin_stage_(coopa::gfx::command::CommandBuffer& cmd, const targets::OffscreenTarget& target,
                      const coopa::gfx::pipeline::Pipeline& pipe,
                      const coopa::gfx::pipeline::DescriptorSet& set) const {
        target.begin(cmd);
        cmd.bind_pipeline(pipe);
        cmd.set_viewport(0.0f, 0.0f, static_cast<float>(target.width()), static_cast<float>(target.height()));
        cmd.set_scissor(0, 0, target.width(), target.height());
        cmd.bind_descriptor_set(set, 0);
    }

    uint32_t full_width_;
    uint32_t full_height_;
    uint32_t half_width_;
    uint32_t half_height_;

    targets::OffscreenTarget coc_target_;
    targets::OffscreenTarget bokeh_target_;
    targets::OffscreenTarget result_target_;

    std::unique_ptr<coopa::gfx::pipeline::Shader> vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> coc_frag_, bokeh_frag_, composite_frag_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> coc_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> bokeh_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> composite_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool>      desc_pool_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> coc_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> bokeh_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> composite_set_;

    std::unique_ptr<coopa::gfx::pipeline::Pipeline> coc_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> bokeh_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> composite_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_DOF_PASS_H
