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
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/engine/targets/offscreen_target.h>
#include <gfxcoopa/engine/targets/gbuffer_target.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/memory/image_upload.h>
#include <gfxcoopa/engine/passes/hiz_pass.h>
#include <gfxcoopa/engine/passes/extra_sets.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

/**
 * @class SsrPass
 * @brief Screen-space reflections: Hi-Z raymarch, temporal resolve, optional
 *        bilateral blur, then a BRDF composite back into the HDR frame.
 *
 * Owns the trace/resolve/blur targets, the reflection history image, and one
 * pipeline per stage. The march can run at half resolution (see the
 * constructor's half_res flag); the composite always runs at full resolution.
 */
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
        float jitter_strength = 0.0f; // 0 = old deterministic mirror-ray trace (ssr_jitter)
        int   frame_index     = 0;    // decorrelates the jitter's noise frame to frame

        // Secondary source (see set_secondary_source()): != 0 -> ssr.frag also traces
        // gfx_ssr_trace_secondary() and keeps whichever hit is nearer. max_hiz_mip_b/
        // max_color_mip_b are the secondary pyramid's OWN mip counts -- an independent
        // HiZPass/SceneColorMipPass instance over a differently-sized image chain, so these
        // can legitimately differ from max_hiz_mip/max_color_mip above even though every
        // other tuning knob (distance/bias/thickness/roughness_cutoff/iterations/start_mip/
        // min_mip0_steps) is shared between the two traces.
        float has_secondary   = 0.0f;
        int   max_hiz_mip_b   = 0;
        int   max_color_mip_b = 0;
    };
    static_assert(sizeof(SsrPushConstants) == 124, "ssr.frag's SsrPushConstants block must match this layout byte-for-byte");

    /// Matches ssr_blur.frag's push-constant block exactly.
    struct BlurPushConstants {
        float radius;
    };

    struct ResolvePushConstants {
        // Current clip space -> previous frame's clip space. mat4 must be 16-byte aligned, so it
        // goes first; vec2 is flattened to two floats below, matching SsaoPass's own house rule.
        glm::mat4 reproject;        // offset 0
        float     resolution_x;     // offset 64
        float     resolution_y;     // offset 68
        // Accumulation cap for the chain being resolved -- the specular and diffuse chains share
        // one count buffer and clamp it to different depths. 0 makes the resolve a passthrough.
        float     max_accum;        // offset 72
        // Fallback fixed-rate history weight, used only where the shared count buffer is
        // unavailable (the 1x1 neutral texture).
        float     blend_factor;     // offset 76
        int       history_valid;    // offset 80
        float     gamma = 1.0f;     // offset 84 -- variance-clipping width, in std deviations
                                     // (ssr_temporal_gamma); see ssr_resolve.frag's own doc
        int       frozen = 0;       // offset 88 -- hold accepted history verbatim (still camera)
    };                              // 92 bytes
    static_assert(sizeof(ResolvePushConstants) == 92, "ssr_resolve.frag's PushConstants block must match this layout byte-for-byte");

    /// Union layout shared with the composite's GLSL push-constant block, so
    /// every consumer's ssr_composite.frag -- whatever else it does -- reads
    /// max_color_mip/sky_intensity/ssgi_intensity/ssgi_distance at the same
    /// offsets. A consumer that doesn't use a field (blendy has no SSGI
    /// bounce term today) simply never reads it; the bytes are harmless.
    struct CompositePushConstants {
        glm::vec2 ssr_resolution;      // offset 0  -- resolution of u_ssr_map (trace res under half-res)
        glm::vec2 screen_resolution;   // offset 8  -- always full screen res
        int       half_res;            // offset 16
        int       max_color_mip  = 0;  // offset 20 -- top mip of the prefiltered scene-colour chain
        float     sky_intensity  = 1.0f; // offset 24 -- MUST match the lighting pass's sky_intensity,
                                          // or the env-specular subtraction below leaves a residue
        float     ssgi_intensity = 0.0f; // offset 28 -- 0 disables the diffuse-bounce term
        float     ssgi_distance  = 0.5f; // offset 32 -- world-space offset along N for the FALLBACK bounce tap
        float     ssgi_traced    = 0.0f; // offset 36 -- > 0.5: the composite reads the traced+resolved
                                          // SSGI buffer (u_ssgi_map) instead of the normal-offset mip tap
        // offset 40..47: explicit padding up to the next 16-byte boundary, so the vec4s below
        // land at the same offset a hand-written GLSL push_constant block would place them at
        // (std430 requires vec4 to start on a 16-byte boundary).
        float     _pad1 = 0.0f;
        float     _pad2 = 0.0f;
        // Sky colours -- MUST match the lighting pass's, same reasoning as sky_intensity above.
        glm::vec4 sky_zenith  = glm::vec4(0.05f, 0.18f, 0.55f, 0.0f);  // offset 48
        glm::vec4 sky_horizon = glm::vec4(0.25f, 0.35f, 0.45f, 0.0f);  // offset 64
        glm::vec4 sky_ground  = glm::vec4(0.05f, 0.045f, 0.04f, 0.0f); // offset 80
    };                                 // 96 bytes
    static_assert(sizeof(CompositePushConstants) == 96,
                 "ssr_composite.frag's PushConstants block must match this layout byte-for-byte");

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
        // Stochastic ray jitter -- see GfxSsrParams' own doc (gfx/ssr_trace_body.glsl) for why
        // this exists. 0 (the default) reproduces the pre-jitter single-ray trace exactly, so
        // every existing consumer of this pass is unaffected unless it opts in. frame_index is
        // meaningless at jitter_strength == 0.
        float jitter_strength  = 0.0f;
        int   frame_index      = 0;
        bool  temporal_enabled = true;
        // Accumulation depth of the specular chain's temporal resolve: each pixel averages this
        // many frames of the jittered trace (frame N blended at 1/N against the shared count
        // TemporalHistoryPass publishes) before the running mean becomes a fixed-rate blend.
        int   temporal_frames  = 32;
        // Same, for the traced-SSGI chain. Deeper by default: one cosine-hemisphere ray has far
        // higher variance than a near-mirror reflection ray.
        int   ssgi_temporal_frames = 48;
        // Fallback fixed-rate history weight, used only where the shared count buffer is
        // unavailable (see set_temporal_count_image() -- a consumer that never calls it).
        float temporal_blend   = 0.85f;
        // Variance-clipping gamma for the temporal resolve's history rejection (see
        // ResolvePushConstants' own doc) -- widens or tightens the accepted history band as a
        // multiple of the 3x3 neighbourhood's standard deviation.
        float temporal_gamma   = 1.0f;
        // True once the camera has been still long enough for the accumulated average to top up;
        // the resolve then holds accepted history verbatim, which is what makes a resting image
        // byte-static. Same signal SsaoPass::Params::frozen carries.
        bool  temporal_frozen  = false;
        // Reprojection: current clip space -> previous frame's clip space, composed in DOUBLE
        // precision by the caller (glm::dmat4(prev_view_proj) * glm::inverse(glm::dmat4(proj) *
        // glm::dmat4(view))) before truncating to float. World-scale magnitudes cancel inside the
        // double product; a float composition -- or the RGBA16F G-buffer position this replaces --
        // drifts by whole pixels at scene scales of a few hundred units, which made the
        // accumulated reflection slide and boil against the geometry in motion. Same scheme
        // SsaoPass::Params::reproject and TaaPass use. reproject_valid says whether it (and the
        // history buffer) exist yet -- false for the first two frames and right after a resize.
        glm::mat4 reproject       = glm::mat4(1.0f);
        bool      reproject_valid = false;
        // Indirect-specular/SSGI terms fed straight into CompositePushConstants -- see that
        // struct's doc. sky_intensity must match whatever the lighting pass used for the same
        // frame; ssgi_intensity 0 (the default) makes the diffuse-bounce term a no-op for
        // consumers whose composite shader doesn't implement it at all (e.g. blendy's).
        float sky_intensity    = 1.0f;
        float ssgi_intensity   = 0.0f;
        float ssgi_distance    = 0.5f;
        // Ray length for the traced-SSGI stage (see the ctor's ssgi_frag_spv doc);
        // meaningless when that stage wasn't built. Every other march knob
        // (iterations/bias/thickness/start_mip) is shared with the specular trace.
        float ssgi_max_distance = 8.0f;
        // Hi-Z iteration budget for the SSGI march. Its own knob rather than a share of
        // max_iterations above: the bounce ray is short and lands in a coarse cone mip, so
        // it converges in far fewer steps than a mirror reflection does.
        int   ssgi_max_iterations = 32;
        // World-space blur sigma for the traced-SSGI denoise, when both that stage and the
        // blur stage were built. Wider than ssr_blur_radius by default and deliberately so:
        // a diffuse bounce is low-frequency, so a wide kernel costs it no real detail while
        // removing variance a single hemisphere ray per pixel cannot avoid producing.
        float ssgi_blur_radius = 1.0f;
        // Sky colours the env-specular subtraction must cancel exactly -- same
        // IndirectParams instance the lighting pass reads, same requirement as
        // sky_intensity above (see CompositePushConstants' doc).
        glm::vec3 sky_zenith   = glm::vec3(0.05f, 0.18f, 0.55f);
        glm::vec3 sky_horizon  = glm::vec3(0.25f, 0.35f, 0.45f);
        glm::vec3 sky_ground   = glm::vec3(0.05f, 0.045f, 0.04f);
        // Secondary source -- see SsrPushConstants' own doc. max_hiz_mip_b/max_color_mip_b are
        // meaningless when has_secondary is false and left at their defaults in that case.
        bool  has_secondary    = false;
        int   max_hiz_mip_b    = 0;
        int   max_color_mip_b  = 0;
        // World-space blur radius for the spatial SSR denoise (see the ctor's blur_frag_spv
        // doc) -- meaningless when the blur stage wasn't built (blur_frag_spv empty at
        // construction). Same role as SsaoPass::Params::radius plays for ssao_blur.frag.
        float ssr_blur_radius  = 0.5f;
    };

    /**
     * @param blur_vert_spv Vertex shader for the optional spatial SSR denoise stage (reuses
     *                      the same fullscreen-triangle vertex shader as resolve/composite --
     *                      pass the same path as resolve_vert_spv). Leave both this and
     *                      blur_frag_spv empty (the default) to omit the blur stage
     *                      entirely; composite then reads resolved_target_ directly.
     * @param blur_frag_spv Fragment shader (ssr_blur.frag.spv) -- bilateral, edge-aware blur
     *                      of the temporally-resolved SSR buffer, same technique as
     *                      SsaoPass's own ssao_blur.frag (5x5 footprint weighted by
     *                      G-buffer normal/position similarity), addressing hit/miss noise at
     *                      reflection boundaries that temporal accumulation alone doesn't
     *                      fully resolve, especially under continuous camera motion.
     */
    SsrPass(coopa::gfx::core::Device& device,
            coopa::gfx::memory::Allocator& allocator,
            const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
            uint32_t width,
            uint32_t height,
            const std::string& ssr_vert_spv,
            const std::string& ssr_frag_spv,
            const std::string& comp_vert_spv,
            const std::string& comp_frag_spv,
            const std::string& resolve_vert_spv,
            const std::string& resolve_frag_spv,
            bool half_res = false,
            ExtraSets composite_extra = {},
            const std::string& blur_vert_spv = "",
            const std::string& blur_frag_spv = "",
            // Optional traced-SSGI stage (ssgi.frag): one cosine-hemisphere diffuse ray
            // per pixel against the same Hi-Z/scene-colour inputs as the specular trace,
            // temporally resolved through a second instance of the resolve pipeline, and
            // consumed by the composite's diffuse-bounce term in place of its single
            // normal-offset mip tap. Empty (the default) omits the stage entirely; the
            // composite's u_ssgi_map binding then points at the permanent 1x1 zero
            // fallback and pc.ssgi_traced stays 0.
            const std::string& ssgi_frag_spv = "")
        : device_(device), allocator_(allocator), width_(width), height_(height), half_res_(half_res),
          blur_enabled_(!blur_vert_spv.empty() && !blur_frag_spv.empty()),
          ssgi_enabled_(!ssgi_frag_spv.empty()),
          composite_extra_(std::move(composite_extra))
    {
        composite_extra_.validate("SsrPass composite");

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
        nearest_sampler_ = std::make_unique<util::Sampler>(device, nearest_clamp_desc());

        // 1. Create Offscreen Targets. target_/resolved_target_/history_image_ run at trace
        // resolution (full res, or half under ssr_half_res); composite_target_ always runs at
        // full screen resolution and bilaterally upsamples the resolved SSR buffer into it.
        target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, trace_width_, trace_height_, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
        );
        composite_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, width, height, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
        );
        // Temporal resolve output: the raymarch result (target_) blended with history, read
        // by the composite pass in place of the raw raymarch output.
        resolved_target_ = std::make_unique<targets::OffscreenTarget>(
            device, allocator, trace_width_, trace_height_, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
        );

        // Spatial denoise output (see blur_frag_spv's doc) -- same shape/resolution as
        // resolved_target_, since it blurs that buffer in place, one pass later.
        if (blur_enabled_) {
            blurred_target_ = std::make_unique<targets::OffscreenTarget>(
                device, allocator, trace_width_, trace_height_, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
            );
        }

        // Traced-SSGI stage (see the ctor's ssgi_frag_spv doc): its own trace and
        // resolved targets at the same trace resolution.
        if (ssgi_enabled_) {
            ssgi_target_ = std::make_unique<targets::OffscreenTarget>(
                device, allocator, trace_width_, trace_height_, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
            );
            ssgi_resolved_target_ = std::make_unique<targets::OffscreenTarget>(
                device, allocator, trace_width_, trace_height_, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
            );
            // Spatial denoise for the bounce, when the blur stage exists. Not optional
            // polish: one cosine-hemisphere ray per pixel has far higher variance than
            // the specular trace's near-mirror ray, and the temporal resolve alone
            // leaves a residual that keeps the image visibly settling for several
            // frames after the camera stops. A diffuse bounce is low-frequency by
            // definition, so the bilateral blur costs it no real detail.
            if (blur_enabled_) {
                ssgi_blurred_target_ = std::make_unique<targets::OffscreenTarget>(
                    device, allocator, trace_width_, trace_height_, coopa::gfx::Format::RGBA16_Sfloat, coopa::gfx::SampleCount::X1
                );
            }
        }

        // History image: a copy of last frame's resolved_target_, TRANSFER_DST so
        // update_history_() can vkCmdCopyImage into it.
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device, allocator, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        if (ssgi_enabled_) {
            ssgi_history_image_ = std::make_unique<coopa::gfx::memory::Image>(
                device, allocator, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
            );
        }
        history_initialized_ = false;

        // 2. Shaders
        ssr_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, ssr_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        ssr_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, ssr_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        comp_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, comp_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        comp_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, comp_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        resolve_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, resolve_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        resolve_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, resolve_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        if (blur_enabled_) {
            blur_vert_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, blur_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
            blur_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, blur_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        }
        if (ssgi_enabled_) {
            ssgi_frag_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, ssgi_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        }

        // 3. Descriptor Set Layouts
        // G-Buffer layout (3 images: G0, G1, G2) for the SSR raymarch set. G_depth is not
        // bound here -- the Hi-Z pyramid supplies depth for the march.
        gbuf3_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        // Composite G-buffer layout: same G0-G2 plus an SSAO sampler (binding 3), so the
        // composite can attenuate the SSR/env delta by the same ao * ssao term
        // deferred_lighting.frag applies to indirect_specular (see ssr_composite.frag).
        comp_gbuf_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(3, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        // Single image sampler layouts
        hiz_layout_         = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder().combined_sampler(0, coopa::gfx::ShaderStage::Fragment).build(device));
        scene_color_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder().combined_sampler(0, coopa::gfx::ShaderStage::Fragment).build(device));
        raw_ssr_layout_     = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder().combined_sampler(0, coopa::gfx::ShaderStage::Fragment).build(device));

        // Secondary trace source (see set_secondary_source()) -- normal+position (2 bindings,
        // no albedo/AO: matches SsrPass::trace_gbuffer_layout()'s own trace-only subset, not
        // the full 3-binding G-buffer layout used elsewhere in this pass), plus its own Hi-Z
        // and scene-colour-mip single-image layouts (same shape as hiz_layout_ above, separate
        // instances since they're bound to a DIFFERENT image at runtime).
        secondary_gbuf2_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .build(device));
        hiz_b_layout_           = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder().combined_sampler(0, coopa::gfx::ShaderStage::Fragment).build(device));
        scene_color_b_layout_   = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder().combined_sampler(0, coopa::gfx::ShaderStage::Fragment).build(device));

        // Blur pass layout: matches ssr_blur.frag's set 0 exactly -- 0 = the buffer being
        // blurred (resolved_target_), 1/2 = G-buffer normal/position for the edge-aware weights.
        if (blur_enabled_) {
            blur_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
                coopa::gfx::pipeline::DescriptorLayoutBuilder()
                    .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                    .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                    .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                    .build(device));
        }

        // Composite scene-colour layout: binding 0 = raw full-res scene colour (the surface
        // being composited into), binding 1 = the prefiltered mip chain, for the SSGI diffuse
        // bounce sample. A consumer whose ssr_composite.frag doesn't implement SSGI (blendy)
        // simply never samples binding 1 -- Vulkan doesn't require every declared binding to be
        // read -- but the descriptor is still always written (see update_descriptors()), since
        // an unwritten descriptor in a bound set is undefined behaviour even if unsampled.
        // Binding 2: the traced-SSGI resolved buffer (or the permanent 1x1 zero
        // fallback when the stage is off) -- same always-written rule as binding 1.
        comp_scene_color_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        // Resolve pass layout: 0 = current raymarch output, 1 = history, 2 = the rasterized scene
        // depth the reprojection reconstructs this pixel's clip position from, 3 = the shared
        // per-pixel accumulation count (see set_temporal_count_image(), which also explains the
        // permanent neutral fallback binding 3 starts out holding).
        //
        // Depth, NOT the G-buffer world position this binding used to hold: G2 is RGBA16F, and at
        // world coordinates of a few hundred units its quantization alone is multiple pixels of
        // reprojection error. See ssr_resolve.frag's file doc.
        resolve_layout_ = std::make_unique<coopa::gfx::pipeline::DescriptorSetLayout>(
            coopa::gfx::pipeline::DescriptorLayoutBuilder()
                .combined_sampler(0, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(1, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(2, coopa::gfx::ShaderStage::Fragment)
                .combined_sampler(3, coopa::gfx::ShaderStage::Fragment)
                .build(device));

        // 4. Descriptor Pool -- sizes derived from the layouts above (10 sets, 11 when
        // blur_enabled_) rather than hand-computed headroom.
        coopa::gfx::pipeline::DescriptorPoolBuilder pool_builder;
        pool_builder.add_sets(*gbuf3_layout_, 1).add_sets(*hiz_layout_, 1).add_sets(*scene_color_layout_, 1)
            .add_sets(*resolve_layout_, 1).add_sets(*comp_gbuf_layout_, 1).add_sets(*raw_ssr_layout_, 1)
            .add_sets(*comp_scene_color_layout_, 1).add_sets(*secondary_gbuf2_layout_, 1)
            .add_sets(*hiz_b_layout_, 1).add_sets(*scene_color_b_layout_, 1);
        if (blur_enabled_) {
            pool_builder.add_sets(*blur_layout_, 1);
        }
        if (ssgi_enabled_) {
            // A second instance of the resolve layout, for the SSGI resolve chain.
            pool_builder.add_sets(*resolve_layout_, 1);
            if (blur_enabled_) {
                pool_builder.add_sets(*blur_layout_, 1);
            }
        }
        desc_pool_ = std::make_unique<coopa::gfx::pipeline::DescriptorPool>(pool_builder.build(device));

        // Allocate Descriptor Sets
        ssr_gbuf_set_    = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *gbuf3_layout_);
        hiz_set_         = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *hiz_layout_);
        scene_color_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *scene_color_layout_);
        resolve_set_     = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *resolve_layout_);

        comp_gbuf3_set_       = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *comp_gbuf_layout_);
        comp_raw_ssr_set_     = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *raw_ssr_layout_);
        comp_scene_color_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *comp_scene_color_layout_);

        secondary_gbuf2_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *secondary_gbuf2_layout_);
        hiz_b_set_           = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *hiz_b_layout_);
        scene_color_b_set_   = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *scene_color_b_layout_);

        if (blur_enabled_) {
            blur_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *blur_layout_);
        }
        if (ssgi_enabled_) {
            ssgi_resolve_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *resolve_layout_);
            if (blur_enabled_) {
                ssgi_blur_set_ = std::make_unique<coopa::gfx::pipeline::DescriptorSet>(device, *desc_pool_, *blur_layout_);
            }
        }

        // Permanent 1x1 neutral fallback for the secondary source, bound by default below and
        // rebound to a real transparent-capture target only if/when set_secondary_source() is
        // called -- so the pipeline layout is ALWAYS valid regardless of the runtime
        // has_secondary push-constant toggle (a shader statically referencing a descriptor
        // whose backing image was never transitioned out of UNDEFINED is a hard Vulkan
        // validation error even when a dynamic branch never reaches the sample instruction).
        // zero_rgba_ (all-zero) backs the normal/position/scene-colour fallbacks -- zero
        // normal is this codebase's existing "no surface here" sentinel (same one ssr.frag's
        // own dot(N,N) < 0.001 background check and skybox.frag's discard already rely on),
        // reused here so gfx_ssr_trace_secondary()'s hit test naturally treats "secondary
        // disabled" as a clean, immediate miss. one_r32_ (value 1.0 = far/background depth)
        // backs the Hi-Z fallback for the same reason: a uniformly-far Hi-Z means the ray
        // never finds a cell to penetrate, so hit_found never becomes true and the neutral
        // normal/scene-colour images are never even sampled -- the all-zero fallbacks above
        // are defense in depth, not the primary guarantee.
        {
            coopa::gfx::command::CommandPool one_shot_pool(device, device.queue_family_indices().graphics.value(), true);

            uint16_t zero_half4[4] = {0, 0, 0, 0}; // IEEE-754 half-float zero bit pattern is all-zero bytes
            zero_rgba_ = coopa::gfx::memory::upload_image_2d(
                device, allocator, one_shot_pool, zero_half4, 1, 1, coopa::gfx::Format::RGBA16_Sfloat, 8);

            float far_depth = 1.0f;
            one_r32_ = coopa::gfx::memory::upload_image_2d(
                device, allocator, one_shot_pool, &far_depth, 1, 1, coopa::gfx::Format::R32_Sfloat, 4);
        }
        neutral_sampler_ = std::make_unique<util::Sampler>(device, nearest_clamp_desc());
        secondary_gbuf2_set_->bind_image(0, zero_rgba_->view_typed(), *neutral_sampler_);
        secondary_gbuf2_set_->bind_image(1, zero_rgba_->view_typed(), *neutral_sampler_);
        hiz_b_set_->bind_image(0, one_r32_->view_typed(), *neutral_sampler_);
        scene_color_b_set_->bind_image(0, zero_rgba_->view_typed(), *neutral_sampler_);
        // Resolve binding 3 (the shared accumulation count) starts on the same all-zero fallback:
        // a zero count is ssr_resolve.frag's "no count buffer" sentinel, which selects the
        // fixed-rate blend_factor path. A consumer that wants the converging average calls
        // set_temporal_count_image().
        resolve_set_->bind_image(3, zero_rgba_->view_typed(), *neutral_sampler_);
        if (ssgi_enabled_) {
            ssgi_resolve_set_->bind_image(3, zero_rgba_->view_typed(), *neutral_sampler_);
        }

        // 5. SSR Pipeline Creation
        coopa::gfx::pipeline::PipelineDesc common_desc;
        common_desc.vertex = coopa::gfx::VertexLayout::none();
        common_desc.raster.cull = coopa::gfx::CullMode::None;
        common_desc.depth.test  = false;
        common_desc.depth.write = false;

        coopa::gfx::pipeline::PipelineDesc ssr_desc = common_desc;
        ssr_desc.shaders = {ssr_vert_.get(), ssr_frag_.get()};
        ssr_desc.descriptor_layouts = {
            &camera_layout,
            gbuf3_layout_.get(),
            hiz_layout_.get(),
            scene_color_layout_.get(),
            // Sets 4-6: secondary trace source -- matches ssr.frag's own set declarations.
            secondary_gbuf2_layout_.get(),
            hiz_b_layout_.get(),
            scene_color_b_layout_.get()
        };
        ssr_desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(SsrPushConstants)}};
        ssr_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, target_->render_pass_object(), ssr_desc);

        // 5b. Traced-SSGI pipeline: ssr.frag's own sets 0-3 (ssgi.frag declares no
        // secondary source) and the same push-constant block, so execute() reuses the
        // SsrPushConstants it already built with only the distance field swapped.
        if (ssgi_enabled_) {
            coopa::gfx::pipeline::PipelineDesc ssgi_desc = common_desc;
            ssgi_desc.shaders = {ssr_vert_.get(), ssgi_frag_.get()};
            ssgi_desc.descriptor_layouts = {
                &camera_layout,
                gbuf3_layout_.get(),
                hiz_layout_.get(),
                scene_color_layout_.get()
            };
            ssgi_desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(SsrPushConstants)}};
            ssgi_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, ssgi_target_->render_pass_object(), ssgi_desc);
        }

        // 6. SSR Composite Pipeline Creation
        std::vector<const coopa::gfx::pipeline::DescriptorSetLayout*> comp_layouts = {
            &camera_layout,
            comp_gbuf_layout_.get(),
            raw_ssr_layout_.get(),
            comp_scene_color_layout_.get()
        };
        // Never hardcode this index at the bind site -- it shifts if this pass ever gains
        // another owned set ahead of the caller's extras.
        comp_first_extra_set_ = static_cast<uint32_t>(comp_layouts.size());
        comp_layouts.insert(comp_layouts.end(), composite_extra_.layouts.begin(), composite_extra_.layouts.end());

        coopa::gfx::pipeline::PipelineDesc comp_desc = common_desc;
        comp_desc.shaders = {comp_vert_.get(), comp_frag_.get()};
        comp_desc.descriptor_layouts = comp_layouts;
        comp_desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(CompositePushConstants)}};
        comp_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, composite_target_->render_pass_object(), comp_desc);

        // 7. Temporal Resolve Pipeline Creation
        coopa::gfx::pipeline::PipelineDesc resolve_desc = common_desc;
        resolve_desc.shaders = {resolve_vert_.get(), resolve_frag_.get()};
        resolve_desc.descriptor_layouts = {resolve_layout_.get()};
        resolve_desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(ResolvePushConstants)}};
        resolve_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, resolved_target_->render_pass_object(), resolve_desc);

        // 8. Blur Pipeline Creation (optional -- see blur_frag_spv's ctor doc)
        if (blur_enabled_) {
            coopa::gfx::pipeline::PipelineDesc blur_desc = common_desc;
            blur_desc.shaders = {blur_vert_.get(), blur_frag_.get()};
            blur_desc.descriptor_layouts = {blur_layout_.get()};
            blur_desc.push_constants = {{coopa::gfx::ShaderStage::Fragment, 0, sizeof(BlurPushConstants)}};
            blur_pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(device, blurred_target_->render_pass_object(), blur_desc);
        }
    }

    void recreate(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        trace_width_  = half_res_ ? std::max(1u, width_  / 2) : width_;
        trace_height_ = half_res_ ? std::max(1u, height_ / 2) : height_;

        target_->recreate(trace_width_, trace_height_);
        composite_target_->recreate(width, height);
        resolved_target_->recreate(trace_width_, trace_height_);
        if (blur_enabled_) {
            blurred_target_->recreate(trace_width_, trace_height_);
        }
        if (ssgi_enabled_) {
            ssgi_target_->recreate(trace_width_, trace_height_);
            ssgi_resolved_target_->recreate(trace_width_, trace_height_);
            if (blur_enabled_) {
                ssgi_blurred_target_->recreate(trace_width_, trace_height_);
            }
        }

        // History no longer matches the new resolution -- drop it and start fresh.
        history_image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
        );
        if (ssgi_enabled_) {
            ssgi_history_image_ = std::make_unique<coopa::gfx::memory::Image>(
                device_, allocator_, trace_width_, trace_height_, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO
            );
        }
        history_initialized_ = false;
    }

    void update_descriptors(const targets::GBufferTarget& gbuffer,
                            coopa::gfx::TextureView hiz_view,
                            const util::Sampler& hiz_sampler,
                            coopa::gfx::TextureView scene_color_mip_view,
                            const util::Sampler& scene_color_mip_sampler,
                            coopa::gfx::TextureView scene_color_view,
                            const util::Sampler& linear_sampler)
    {
        // SSR Raymarching descriptors. Nearest sampler: see constructor comment -- these are
        // read both at the texel-centered in_uv (where nearest == linear, no change) and at
        // arbitrary marched UVs during the hit tests (where nearest is required for correctness).
        ssr_gbuf_set_->bind_image(0, gbuffer.g0_view_typed(), *nearest_sampler_);
        ssr_gbuf_set_->bind_image(1, gbuffer.g1_view_typed(), *nearest_sampler_);
        ssr_gbuf_set_->bind_image(2, gbuffer.g2_view_typed(), *nearest_sampler_);

        hiz_set_->bind_image(0, hiz_view, hiz_sampler);
        // The march samples the PREFILTERED chain (cone footprint -> textureLod), while the
        // composite below samples the raw full-res scene colour it is compositing INTO. Same
        // descriptor layout, deliberately different images.
        scene_color_set_->bind_image(0, scene_color_mip_view, scene_color_mip_sampler);

        // Temporal resolve descriptors: current frame's raw raymarch output + last frame's
        // resolved history + the rasterized scene depth the reprojection reconstructs from. The
        // current buffer is still sampled at texel-centered in_uv, so nearest_sampler_ avoids
        // implying this HDR data buffer should ever be blurred.
        resolve_set_->bind_image(0, target_->color_view_typed(), *nearest_sampler_);
        // LINEAR, not nearest: reprojected UVs are no longer texel-centred, and point-sampling
        // them makes the accumulated reflection stair-step and crawl under camera motion.
        resolve_set_->bind_image(1, history_image_->view_typed(), linear_sampler);
        // NEAREST is mandatory, not a preference: D32_SFLOAT is not guaranteed to support linear
        // filtering, and a blended depth would reproject to a point on no real surface anyway.
        resolve_set_->bind_image(2, gbuffer.depth_view_typed(), *nearest_sampler_);

        // SSR Composite descriptors
        comp_gbuf3_set_->bind_image(0, gbuffer.g0_view_typed(), linear_sampler);
        comp_gbuf3_set_->bind_image(1, gbuffer.g1_view_typed(), linear_sampler);
        comp_gbuf3_set_->bind_image(2, gbuffer.g2_view_typed(), linear_sampler);

        // Blur descriptors (optional -- see blur_frag_spv's ctor doc): the buffer being
        // blurred (resolved_target_'s temporally-resolved output, NEAREST -- same
        // texel-exact reasoning as every other in-shader G-buffer point-lookup in this
        // engine) plus the G-buffer normal/position the bilateral weights are computed from.
        if (blur_enabled_) {
            blur_set_->bind_image(0, resolved_target_->color_view_typed(), *nearest_sampler_);
            blur_set_->bind_image(1, gbuffer.g1_view_typed(), *nearest_sampler_);
            blur_set_->bind_image(2, gbuffer.g2_view_typed(), *nearest_sampler_);
        }

        // Composite reads the BLURRED buffer when the blur stage is enabled (execute() draws
        // resolved_target_ -> blurred_target_ every frame before composite runs), or the
        // temporally-resolved buffer directly otherwise -- exactly this pass's original
        // behaviour before the blur stage existed.
        comp_raw_ssr_set_->bind_image(0, blur_enabled_ ? blurred_target_->color_view_typed() : resolved_target_->color_view_typed(),
                                      linear_sampler);
        comp_scene_color_set_->bind_image(0, scene_color_view, linear_sampler);
        // Same prefiltered mip chain the raymarch's scene_color_set_ (binding 0 above) reads --
        // for the composite's SSGI diffuse-bounce sample, when the bound shader implements one.
        comp_scene_color_set_->bind_image(1, scene_color_mip_view, scene_color_mip_sampler);
        // Traced-SSGI buffer the composite reads -- the denoised one when the blur stage
        // exists, else the temporally-resolved one, else (stage off) the permanent zero
        // fallback (an unwritten descriptor in a bound set is undefined behaviour even if
        // unsampled). Same three-way choice comp_raw_ssr_set_ makes just above.
        coopa::gfx::TextureView ssgi_view = zero_rgba_->view_typed();
        if (ssgi_enabled_) {
            ssgi_view = blur_enabled_ ? ssgi_blurred_target_->color_view_typed()
                                      : ssgi_resolved_target_->color_view_typed();
        }
        comp_scene_color_set_->bind_image(2, ssgi_view, linear_sampler);

        // SSGI resolve/blur descriptors: mirror resolve_set_/blur_set_ above, over the
        // SSGI chain's own images.
        if (ssgi_enabled_) {
            ssgi_resolve_set_->bind_image(0, ssgi_target_->color_view_typed(), *nearest_sampler_);
            ssgi_resolve_set_->bind_image(1, ssgi_history_image_->view_typed(), linear_sampler);
            ssgi_resolve_set_->bind_image(2, gbuffer.depth_view_typed(), *nearest_sampler_);
            if (blur_enabled_) {
                ssgi_blur_set_->bind_image(0, ssgi_resolved_target_->color_view_typed(), *nearest_sampler_);
                ssgi_blur_set_->bind_image(1, gbuffer.g1_view_typed(), *nearest_sampler_);
                ssgi_blur_set_->bind_image(2, gbuffer.g2_view_typed(), *nearest_sampler_);
            }
        }
    }

    /// Binding 3 of the composite G-buffer set must be rebound every frame -- callers pass the
    /// SSAO pass's blurred output when enabled, or its permanent neutral (fully-unoccluded)
    /// texture when disabled/absent, mirroring DeferredLightingPass::set_ssao_image() so the two
    /// passes always attenuate indirect specular by the identical ao * ssao term.
    void set_ssao_image(VkImageView ssao_view, VkSampler ssao_sampler) {
        comp_gbuf3_set_->bind_image(3, ssao_view, ssao_sampler);
    }

    /// Points both resolve chains at the shared per-pixel accumulation count
    /// (TemporalHistoryPass's output), which is what turns their temporal blend from a fixed-rate
    /// exponential -- incapable of converging on a per-frame-rejittered trace -- into a running
    /// average over `Params::temporal_frames` draws.
    ///
    /// Optional, like set_secondary_source(): the constructor leaves binding 3 on a permanent
    /// all-zero 1x1 texture, and a zero count is ssr_resolve.frag's sentinel for "no count buffer
    /// here", which selects the Params::temporal_blend path instead. A consumer with no
    /// TemporalHistoryPass never needs to call this. Bound once at setup, like
    /// update_descriptors(): the image identity is stable, only its contents change per frame.
    void set_temporal_count_image(coopa::gfx::TextureView count_view, const util::Sampler& count_sampler) {
        resolve_set_->bind_image(3, count_view, count_sampler);
        if (ssgi_enabled_) {
            ssgi_resolve_set_->bind_image(3, count_view, count_sampler);
        }
    }

    /// Points the raymarch's secondary source (sets 4-6, see gfx/ssr_trace_secondary_body.glsl)
    /// at real views -- a second reflection source (e.g. a forward capture of transparent
    /// geometry) beyond this pass's own primary G-buffer/Hi-Z/scene-colour-mip chain. Bound
    /// once at setup, like update_descriptors(), not per frame: the source images' identity
    /// doesn't change frame to frame, only their CONTENTS do (re-rendered each frame by
    /// whatever pass produces them). Callers gate actual USE of this at runtime via
    /// Params::has_secondary in execute() -- calling this at all is optional; the constructor
    /// already leaves sets 4-6 bound to a permanent neutral fallback that produces a guaranteed
    /// miss, so a consumer with no secondary source never needs to call this.
    void set_secondary_source(coopa::gfx::TextureView normal_metallic_view, coopa::gfx::TextureView position_roughness_view,
                              coopa::gfx::TextureView hiz_view, const util::Sampler& hiz_sampler,
                              coopa::gfx::TextureView scene_color_view, const util::Sampler& scene_color_sampler)
    {
        secondary_gbuf2_set_->bind_image(0, normal_metallic_view, *nearest_sampler_);
        secondary_gbuf2_set_->bind_image(1, position_roughness_view, *nearest_sampler_);
        hiz_b_set_->bind_image(0, hiz_view, hiz_sampler);
        scene_color_b_set_->bind_image(0, scene_color_view, scene_color_sampler);
    }

    /// @name Trace-input accessors
    /// Expose the raymarch's own three input sets (bound once by update_descriptors(), never
    /// per-frame) so a forward-shaded consumer that also wants to trace gfx/ssr_trace_body.glsl
    /// -- e.g. a transparent pass, whose geometry never appears in the G-buffer the way this
    /// pass's own ssr.frag draw does -- can bind the identical descriptors rather than
    /// duplicating the images. See gfx/ssr_trace_body.glsl's required-before-include contract
    /// for the exact sampler/UBO names each set must resolve to.
    /// @{
    const coopa::gfx::pipeline::DescriptorSetLayout& trace_gbuffer_layout() const { return *gbuf3_layout_; }
    const coopa::gfx::pipeline::DescriptorSetLayout& hiz_layout() const { return *hiz_layout_; }
    const coopa::gfx::pipeline::DescriptorSetLayout& scene_color_layout() const { return *scene_color_layout_; }
    const coopa::gfx::pipeline::DescriptorSet& trace_gbuffer_set() const { return *ssr_gbuf_set_; }
    const coopa::gfx::pipeline::DescriptorSet& hiz_set() const { return *hiz_set_; }
    const coopa::gfx::pipeline::DescriptorSet& scene_color_set() const { return *scene_color_set_; }
    /// @}

    void execute(coopa::gfx::command::CommandBuffer& cmd,
                 const coopa::gfx::pipeline::DescriptorSet& camera_set,
                 const Params& params)
    {
        // On the very first frame (or right after a resize), history_image_ is UNDEFINED --
        // transition it once before it's bound as a sampled image in the resolve pass below.
        // The resolve shader doesn't read it in that case (history_valid = 0 in the push
        // constant), but the descriptor binding still needs a valid layout at draw time
        // regardless of the runtime branch.
        if (!history_initialized_) {
            VkImageMemoryBarrier barriers[2]{};
            uint32_t barrier_count = ssgi_enabled_ ? 2u : 1u;
            for (uint32_t i = 0; i < barrier_count; ++i) {
                barriers[i].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barriers[i].oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[i].newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barriers[i].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].image                           = (i == 0) ? history_image_->handle()
                                                                       : ssgi_history_image_->handle();
                barriers[i].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                barriers[i].subresourceRange.baseMipLevel   = 0;
                barriers[i].subresourceRange.levelCount     = 1;
                barriers[i].subresourceRange.baseArrayLayer = 0;
                barriers[i].subresourceRange.layerCount     = 1;
                barriers[i].srcAccessMask                   = 0;
                barriers[i].dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;
            }

            vkCmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, barrier_count, barriers);
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
        pc.jitter_strength  = params.jitter_strength;
        // 0 when there is no temporal accumulation running. Every stochastic term in the two
        // traces re-seeds from this -- ssr.frag's GGX ray jitter AND ssgi.frag's cosine
        // hemisphere direction, which samples from it unconditionally, not only when
        // jitter_strength is nonzero. Advancing it without an accumulator downstream is pure
        // per-frame noise with nothing left to integrate it, so a consumer running the resolves
        // as passthroughs (an A/B capture) would see an image that never repeats. Same rule
        // SsaoPass applies to its own noise_rotation, for the same reason.
        pc.frame_index      = params.temporal_enabled ? params.frame_index : 0;
        pc.has_secondary    = params.has_secondary ? 1.0f : 0.0f;
        pc.max_hiz_mip_b    = params.max_hiz_mip_b;
        pc.max_color_mip_b  = params.max_color_mip_b;

        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, pc);

        cmd.bind_descriptor_set(camera_set, 0);
        cmd.bind_descriptor_set(*ssr_gbuf_set_, 1);
        cmd.bind_descriptor_set(*hiz_set_, 2);
        cmd.bind_descriptor_set(*scene_color_set_, 3);
        cmd.bind_descriptor_set(*secondary_gbuf2_set_, 4);
        cmd.bind_descriptor_set(*hiz_b_set_, 5);
        cmd.bind_descriptor_set(*scene_color_b_set_, 6);

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
        rpc.reproject    = params.reproject;
        rpc.resolution_x = static_cast<float>(trace_width_);
        rpc.resolution_y = static_cast<float>(trace_height_);
        // 0 makes the resolve a passthrough of the current frame -- one code path, no branching
        // pipeline structure, matching how SsaoPass expresses the same toggle.
        rpc.max_accum    = params.temporal_enabled ? static_cast<float>(params.temporal_frames) : 0.0f;
        rpc.blend_factor = params.temporal_enabled ? params.temporal_blend : 0.0f;
        // Both a history IMAGE (history_initialized_) and a reprojection MATRIX (reproject_valid)
        // must exist -- the matrix lags a frame behind the image on a fresh start (frames 0 and
        // 1), so ANDing them is what keeps the very first reprojected sample from reading a
        // stale/identity matrix.
        rpc.history_valid = (history_initialized_ && params.reproject_valid) ? 1 : 0;
        rpc.gamma         = params.temporal_gamma;
        rpc.frozen        = params.temporal_frozen ? 1 : 0;

        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, rpc);
        cmd.bind_descriptor_set(*resolve_set_, 0);

        cmd.draw(3);
        resolved_target_->end(cmd);

        // 2b. Spatial denoise (optional -- see the ctor's blur_frag_spv doc): bilateral blur of
        // resolved_target_ into blurred_target_, which the composite pass below reads instead
        // when this stage is enabled (see update_descriptors()'s comp_raw_ssr_set_ binding).
        // Runs every frame regardless of temporal_enabled -- it addresses spatial hit/miss
        // noise the temporal resolve above doesn't remove, not a replacement for it.
        if (blur_enabled_) {
            blurred_target_->begin(cmd);
            cmd.bind_pipeline(*blur_pipeline_);
            cmd.set_viewport(0.0f, 0.0f, static_cast<float>(trace_width_), static_cast<float>(trace_height_));
            cmd.set_scissor(0, 0, trace_width_, trace_height_);

            BlurPushConstants bpc{};
            bpc.radius = params.ssr_blur_radius;
            cmd.push_constants(coopa::gfx::ShaderStage::Fragment, bpc);
            cmd.bind_descriptor_set(*blur_set_, 0);

            cmd.draw(3);
            blurred_target_->end(cmd);
        }

        // 2c. Traced-SSGI stage (optional -- see the ctor's ssgi_frag_spv doc): one
        // cosine-hemisphere ray per pixel through ssgi.frag, then a second run of the
        // resolve pipeline over the SSGI chain's own current/history pair. The trace
        // reuses this frame's SsrPushConstants with only the ray length swapped (and the
        // roughness cutoff lifted clear of ssgi.frag's fixed 0.9 trace roughness, so the
        // shared roughness fade stays 1 -- the cutoff is a specular concept).
        if (ssgi_enabled_) {
            ssgi_target_->begin(cmd);
            cmd.bind_pipeline(*ssgi_pipeline_);
            cmd.set_viewport(0.0f, 0.0f, static_cast<float>(trace_width_), static_cast<float>(trace_height_));
            cmd.set_scissor(0, 0, trace_width_, trace_height_);

            SsrPushConstants gi_pc = pc;
            gi_pc.max_distance     = params.ssgi_max_distance;
            gi_pc.max_iterations   = params.ssgi_max_iterations;
            gi_pc.roughness_cutoff = 2.0f;
            gi_pc.has_secondary    = 0.0f;
            cmd.push_constants(coopa::gfx::ShaderStage::Fragment, gi_pc);

            cmd.bind_descriptor_set(camera_set, 0);
            cmd.bind_descriptor_set(*ssr_gbuf_set_, 1);
            cmd.bind_descriptor_set(*hiz_set_, 2);
            cmd.bind_descriptor_set(*scene_color_set_, 3);

            cmd.draw(3);
            ssgi_target_->end(cmd);

            ssgi_resolved_target_->begin(cmd);
            cmd.bind_pipeline(*resolve_pipeline_);
            cmd.set_viewport(0.0f, 0.0f, static_cast<float>(trace_width_), static_cast<float>(trace_height_));
            cmd.set_scissor(0, 0, trace_width_, trace_height_);
            // Same push block as the specular resolve above with only the accumulation depth
            // swapped: the bounce averages deeper, since one cosine-hemisphere ray carries far
            // more variance than a near-mirror reflection ray. Both clamp the SAME shared count
            // buffer, which is why one buffer can serve two schedules.
            ResolvePushConstants gi_rpc = rpc;
            gi_rpc.max_accum = params.temporal_enabled
                ? static_cast<float>(params.ssgi_temporal_frames) : 0.0f;
            cmd.push_constants(coopa::gfx::ShaderStage::Fragment, gi_rpc);
            cmd.bind_descriptor_set(*ssgi_resolve_set_, 0);
            cmd.draw(3);
            ssgi_resolved_target_->end(cmd);

            // Spatial denoise -- see ssgi_blurred_target_'s construction for why this is
            // load-bearing for the bounce rather than optional polish.
            if (blur_enabled_) {
                ssgi_blurred_target_->begin(cmd);
                cmd.bind_pipeline(*blur_pipeline_);
                cmd.set_viewport(0.0f, 0.0f, static_cast<float>(trace_width_), static_cast<float>(trace_height_));
                cmd.set_scissor(0, 0, trace_width_, trace_height_);

                BlurPushConstants gi_bpc{};
                gi_bpc.radius = params.ssgi_blur_radius;
                cmd.push_constants(coopa::gfx::ShaderStage::Fragment, gi_bpc);
                cmd.bind_descriptor_set(*ssgi_blur_set_, 0);

                cmd.draw(3);
                ssgi_blurred_target_->end(cmd);
            }
        }

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
        cpc.max_color_mip     = params.max_color_mip;
        cpc.sky_intensity     = params.sky_intensity;
        cpc.ssgi_intensity    = params.ssgi_intensity;
        cpc.ssgi_distance     = params.ssgi_distance;
        cpc.ssgi_traced       = ssgi_enabled_ ? 1.0f : 0.0f;
        cpc.sky_zenith        = glm::vec4(params.sky_zenith, 0.0f);
        cpc.sky_horizon       = glm::vec4(params.sky_horizon, 0.0f);
        cpc.sky_ground        = glm::vec4(params.sky_ground, 0.0f);
        cmd.push_constants(coopa::gfx::ShaderStage::Fragment, cpc);

        cmd.bind_descriptor_set(camera_set, 0);
        cmd.bind_descriptor_set(*comp_gbuf3_set_, 1);
        cmd.bind_descriptor_set(*comp_raw_ssr_set_, 2);
        cmd.bind_descriptor_set(*comp_scene_color_set_, 3);
        if (composite_extra_.bind) {
            composite_extra_.bind(cmd, comp_first_extra_set_);
        }

        cmd.draw(3);
        composite_target_->end(cmd);

        // 4. Copy the resolved buffers into their history images for next frame's resolves.
        copy_history_(cmd, resolved_target_->color_image_object()->handle(), history_image_->handle());
        if (ssgi_enabled_) {
            copy_history_(cmd, ssgi_resolved_target_->color_image_object()->handle(),
                          ssgi_history_image_->handle());
        }
        history_initialized_ = true;
    }

    VkImageView output_view() const { return composite_target_->color_view(); }
    coopa::gfx::TextureView output_view_typed() const { return composite_target_->color_view_typed(); }
    targets::OffscreenTarget& composite_target() { return *composite_target_; }

    /// @brief The resolved reflection buffer the composite reads (post spatial-denoise
    /// blur when enabled, matching comp_raw_ssr_set_'s own choice above) -- for a caller
    /// that wants to inspect the reflection term itself rather than the finished
    /// scene-colour-plus-reflection composite output_view() returns.
    coopa::gfx::TextureView reflection_view_typed() const {
        return blur_enabled_ ? blurred_target_->color_view_typed() : resolved_target_->color_view_typed();
    }

    /// @brief The resolved traced-SSGI bounce buffer (post spatial-denoise blur when
    /// enabled), or a 1x1 neutral zero texture when this instance has no SSGI stage
    /// (ssgi_enabled_ false) -- the same fallback comp_scene_color_set_'s own SSGI
    /// binding uses, so a caller never binds a descriptor still in VK_IMAGE_LAYOUT_UNDEFINED.
    coopa::gfx::TextureView ssgi_view_typed() const {
        if (!ssgi_enabled_) return zero_rgba_->view_typed();
        return blur_enabled_ ? ssgi_blurred_target_->color_view_typed() : ssgi_resolved_target_->color_view_typed();
    }

    /// @brief The same 1x1 neutral zero texture ssgi_view_typed() falls back to when this
    /// instance has no SSGI stage -- exposed unconditionally for a caller (e.g. a debug
    /// view) that needs a safe fallback for reflection_view_typed()/ssgi_view_typed() too
    /// on a frame where execute() never runs at all (this pass is always CONSTRUCTED, per
    /// this class's own file doc, but its targets only leave VK_IMAGE_LAYOUT_UNDEFINED
    /// once execute() has run at least once).
    coopa::gfx::TextureView zero_view_typed() const { return zero_rgba_->view_typed(); }

private:
    /// NEAREST + ClampToEdge, matching the raw ctor's default mipmap_mode (NEAREST) too -- see
    /// nearest_sampler_/neutral_sampler_'s uses. All of this pass's own samplers (aside from the
    /// caller-supplied linear_sampler passed into update_descriptors()) share this exact config.
    static coopa::gfx::SamplerDesc nearest_clamp_desc() {
        coopa::gfx::SamplerDesc d;
        d.min = d.mag = coopa::gfx::Filter::Nearest;
        d.mipmap  = coopa::gfx::MipmapMode::Nearest;
        d.address = coopa::gfx::AddressMode::ClampToEdge;
        return d;
    }

    // Copies a resolved buffer's color image into its history image, so the next
    // frame's resolve pass has something to blend against. Both the SSR and the
    // traced-SSGI chains go through here; the caller sets history_initialized_
    // once after all copies.
    void copy_history_(coopa::gfx::command::CommandBuffer& cmd, VkImage src_image, VkImage dst_image) {
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

    // Spatial SSR denoise (optional -- see the ctor's blur_frag_spv doc). blur_enabled_ is set
    // once at construction from whether blur shader paths were provided; every other blur_*
    // member stays null when it's false.
    bool blur_enabled_ = false;
    std::unique_ptr<targets::OffscreenTarget> blurred_target_;

    // Traced-SSGI stage (optional -- see the ctor's ssgi_frag_spv doc). ssgi_enabled_
    // is set once at construction; every other ssgi_* member stays null when false.
    bool ssgi_enabled_ = false;
    std::unique_ptr<targets::OffscreenTarget>  ssgi_target_;
    std::unique_ptr<targets::OffscreenTarget>  ssgi_resolved_target_;
    std::unique_ptr<targets::OffscreenTarget>  ssgi_blurred_target_;
    std::unique_ptr<coopa::gfx::memory::Image> ssgi_history_image_;

    std::unique_ptr<coopa::gfx::memory::Image> history_image_;
    bool history_initialized_ = false;

    std::unique_ptr<coopa::gfx::pipeline::Shader> ssr_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> ssr_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> comp_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> comp_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> resolve_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> resolve_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> blur_vert_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> blur_frag_;
    std::unique_ptr<coopa::gfx::pipeline::Shader> ssgi_frag_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> gbuf3_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> comp_gbuf_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> hiz_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> scene_color_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> comp_scene_color_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> raw_ssr_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> resolve_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> secondary_gbuf2_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> hiz_b_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> scene_color_b_layout_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSetLayout> blur_layout_;

    ExtraSets composite_extra_;
    uint32_t  comp_first_extra_set_ = 0;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorPool> desc_pool_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> ssr_gbuf_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> hiz_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> scene_color_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> resolve_set_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> comp_gbuf3_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> comp_raw_ssr_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> comp_scene_color_set_;

    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> secondary_gbuf2_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> hiz_b_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> scene_color_b_set_;
    std::unique_ptr<util::Sampler>              neutral_sampler_;
    std::unique_ptr<coopa::gfx::memory::Image>  zero_rgba_;
    std::unique_ptr<coopa::gfx::memory::Image>  one_r32_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> blur_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> ssgi_resolve_set_;
    std::unique_ptr<coopa::gfx::pipeline::DescriptorSet> ssgi_blur_set_;

    std::unique_ptr<coopa::gfx::pipeline::Pipeline> ssr_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> comp_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> resolve_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> blur_pipeline_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> ssgi_pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SSR_PASS_H
