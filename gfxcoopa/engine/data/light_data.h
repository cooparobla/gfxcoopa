/**
 * @file light_data.h
 * @brief Per-frame directional light UBO for toon shading.
 *
 * A single directional light described by direction, color, and ambient color.
 * Uploaded as a uniform buffer updated once per frame.
 */

#ifndef GFXCOOPA_ENGINE_DATA_LIGHT_DATA_H
#define GFXCOOPA_ENGINE_DATA_LIGHT_DATA_H

#include <glm/glm.hpp>
#include <cstddef>
#include <memory>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/pipeline/descriptor.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace data {

static constexpr uint32_t MAX_POINT_LIGHTS = 16;
static constexpr uint32_t MAX_SPOT_LIGHTS  = 8;
/** Cascades the directional shadow atlas can hold. 4 fills a 2x2 tile grid exactly;
 *  the limit is the grid, not the UBO -- see LightUBO::dir_cascade_matrix. */
static constexpr uint32_t MAX_DIR_CASCADES = 4;

/**
 * @struct PointLightGPU
 * @brief std140-aligned Point Light data uploaded to GPU UBO.
 */
struct alignas(16) PointLightGPU {
    glm::vec4 position_range  = glm::vec4(0.0f);            /**< xyz = position, w = range */
    glm::vec4 color_intensity = glm::vec4(0.0f);            /**< xyz = RGB color, w = intensity */
    glm::vec4 attenuation     = glm::vec4(1.0f, 0.09f, 0.032f, 0.0f); /**< x=const, y=linear, z=quad, w=cast_shadows (1 or 0) */
};

/**
 * @struct SpotLightGPU
 * @brief std140-aligned Spot Light data uploaded to GPU UBO.
 */
struct alignas(16) SpotLightGPU {
    glm::vec4 position_range  = glm::vec4(0.0f); /**< xyz = world position, w = range */
    glm::vec4 direction_cone  = glm::vec4(0.0f, 0.0f, -1.0f, 0.7071f); /**< xyz = normalized aim (direction rays travel), w = cos(outer half-angle) */
    glm::vec4 color_intensity = glm::vec4(0.0f); /**< xyz = RGB color, w = intensity */
    glm::vec4 params          = glm::vec4(1.0f, 0.9f, 0.0f, 0.0f); /**< x=falloff sharpness, y=cos(inner half-angle), z=cast_shadows (1 or 0), w=reserved */
};

/**
 * @struct LightUBO
 * @brief std140-aligned directional and point light data uploaded to the GPU.
 */
struct alignas(16) LightUBO {
    // Directional Light
    glm::vec4 dir_direction          = glm::vec4(0.0f); /**< xyz = light direction (normalized), w = intensity */
    glm::vec4 dir_color              = glm::vec4(0.0f); /**< xyz = RGB color, w = unused */
    /**< Directional/point soft-shadow tuning. Its position in the struct is
     * fixed: the ~11 shader LightUBO blocks across gfxcoopa/toyengine/blendy
     * must byte-match this layout, so moving it shifts every later field.
     *   x = directional shadow intensity, 0..1 (DirectionalLightComponent::shadow_intensity)
     *   y = point-light PCF disk radius, as a tangent-space offset on a unit sample
     *       direction (toyengine converts this from PixelRenderConfig::point_shadow_softness
     *       texels each frame, the same way dir_shadow_params.y below converts
     *       shadow_softness from world units); 0 = single hard compare
     *   z = directional PCF tap count (float; int()-cast in the shader)
     *   w = per-frame golden-angle rotation offset, for TAA decorrelation
     * A consumer that never writes this (e.g. blendy, which transports the
     * same values via its own push constants) gets the default, which is a
     * fully dark, hard-compared shadow.
     * Spot shadows reuse z/w (PCF taps, rotation offset) rather than adding their
     * own copies -- see spot_shadow_params below for the spot-specific fields. */
    glm::vec4 dir_shadow_extra = glm::vec4(1.0f, 0.0f, 16.0f, 0.0f);
    glm::mat4 dir_light_space_matrix = glm::mat4(1.0f); /**< Light projection * view matrix for directional shadows */
    glm::vec4 dir_shadow_params      = glm::vec4(0.005f, 0.0f, 1.0f, 0.0f); /**< x=bias, y=pcf_radius_texels (0=hard), z=shadow_enabled (1 or 0), w=normal_bias */

    // Light counts, Point Lights and Spot Lights
    glm::uvec4 light_counts = glm::uvec4(0); /**< x = num_directional (0 or 1), y = num_point_lights,
                                                   z = num_spot_lights, w = index into spot_lights[] of
                                                   the one spot that owns the shadow map, or 0xFFFFFFFF
                                                   if none does (see find_first_shadow_casting_spot_light_
                                                   in pixel_render_pipeline.h) */
    PointLightGPU point_lights[MAX_POINT_LIGHTS];

    // Configurable sky/ambient colour (see IndirectParams in render_features.h).
    // Appended after point_lights so no existing member's offset moves --
    // a shader that doesn't care about sky colour can keep declaring this
    // block without these three fields; std140 only requires a matching
    // *prefix*.
    glm::vec4 sky_zenith  = glm::vec4(0.05f, 0.18f, 0.55f, 0.0f);  /**< xyz = zenith colour, straight up. */
    glm::vec4 sky_horizon = glm::vec4(0.25f, 0.35f, 0.45f, 0.0f);  /**< xyz = horizon colour. */
    glm::vec4 sky_ground  = glm::vec4(0.05f, 0.045f, 0.04f, 0.0f); /**< xyz = ground colour, straight down. */

    // Spot Lights -- appended after sky_ground for the same reason point_lights'
    // doc gives: nothing above this line moves, so a shader with no spot support
    // (gfxcoopa's pbr.frag/deferred_lighting.frag/transparent.frag/probe_capture.frag,
    // any out-of-repo consumer) keeps compiling against the shorter prefix unchanged.
    glm::mat4     spot_light_space_matrix = glm::mat4(1.0f); /**< Light projection * view matrix for the one shadow-casting spot (see light_counts.w). */
    glm::vec4     spot_shadow_params      = glm::vec4(0.005f, 0.0f, 0.0f, 0.05f); /**< x=bias, y=PCF penumbra scale K in texels*distance -- the shader divides by the fragment's light-space depth for a constant world-width penumbra; 0=hard (see calc_spot_shadow in pixel_shadow_body.glsl), z=shadow_enabled (1 or 0), w=normal_bias */
    SpotLightGPU  spot_lights[MAX_SPOT_LIGHTS];

    // Appended after spot_lights per this struct's own append-only rule: consumers
    // with no PCSS/contact-shadow support keep compiling against the shorter prefix.
    glm::vec4     pcss_params    = glm::vec4(0.0f, 0.0f, 8.0f, 8.0f); /**< PCSS contact hardening (calc_dir_shadow):
                                        x=enabled (1 or 0), y=penumbra texels per unit [0,1] light-space
                                        depth gap, for CASCADE 0 (the per-cascade values live in
                                        dir_cascade_pcss_scale below), z=blocker search radius in texels,
                                        w=blocker search tap count (the quality dial -- the search runs on
                                        every shadowed pixel). */
    glm::vec4     contact_params = glm::vec4(0.0f, 0.5f, 0.15f, 8.0f); /**< Screen-space contact shadows
                                        (pixel_lighting.frag): x=strength (0 disables), y=march length in
                                        world units, z=thickness tolerance in world units, w=step count. */
    glm::vec4     contact_soft_params = glm::vec4(0.0f); /**< Soft contact-shadow penumbra
                                        (pixel_lighting.frag): x=cone half-angle tangent (the sun's angular
                                        size, shadow_pcss_light_size, when soft_shadows is on; 0 selects the
                                        hard single-ray march). y/z/w reserved. */

    // --- Directional shadow cascades ---
    //
    // Appended last, per this struct's own append-only rule. The directional shadow map is
    // an ATLAS of up to MAX_DIR_CASCADES tiles (see ShadowMapTarget), each tile a separate
    // ortho fit to one slice of the camera's depth range, so near-camera geometry gets a
    // box a few metres across while distant geometry keeps a coarse one.
    //
    // dir_light_space_matrix / dir_shadow_params.y / dir_shadow_params.w / pcss_params.y
    // above hold CASCADE 0's values. A consumer with no cascade support (gfxcoopa's own
    // pbr.frag/deferred_lighting.frag/transparent.frag) therefore keeps shading against the
    // near cascade through the shorter prefix, rather than reading garbage.
    glm::mat4 dir_cascade_matrix[MAX_DIR_CASCADES] = {
        glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)
    };                                              /**< Per-cascade light projection * view;
                                        [0] duplicates dir_light_space_matrix. Unused slots
                                        hold the last live cascade's matrix. */
    glm::vec4 dir_cascade_pcf_texels   = glm::vec4(0.0f); /**< Per-cascade PCF radius in ATLAS
                                        texels (0 = hard compare). Per-cascade because each
                                        tile has its own world-per-texel scale, and
                                        shadow_softness is a WORLD-space width. */
    glm::vec4 dir_cascade_normal_bias  = glm::vec4(0.0f); /**< Per-cascade normal-offset bias in
                                        world units (compute_shadow_normal_bias() against that
                                        cascade's own texel size). */
    glm::vec4 dir_cascade_pcss_scale   = glm::vec4(0.0f); /**< Per-cascade PCSS penumbra texels
                                        per unit [0,1] light-space depth gap -- pcss_params.y's
                                        quantity, resolved per tile. */
    glm::vec4 dir_cascade_info = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f); /**< x = live cascade count,
                                        y = tiles per atlas row (grid_x), z = selection inset in
                                        TILE uv (keeps a PCF disk from reaching out of its tile),
                                        w = dither-transition band width in tile uv. */
    glm::vec4 dir_shadow_receiver = glm::vec4(0.0f, 4.0f, 0.0f, 0.0f); /**< Receiver-plane depth
                                        bias for the directional PCF: x = 1 to tilt every tap's
                                        compare depth along the receiver's own plane (so the
                                        normal offset no longer has to clear the PCF disk), y =
                                        steepest receiver slope honoured, as tan(angle to the
                                        light's perpendicular plane). z/w reserved. */
};

// Pins the offset of sky_zenith -- the field every pre-spot-light shader's LightUBO
// prefix ends on -- immediately after point_lights, so a future edit that inserts
// something ahead of it (rather than appending after spot_lights, like this change
// did) fails to compile instead of silently desyncing every hand-written GLSL block
// from the C++ layout.
static_assert(offsetof(LightUBO, sky_zenith) == offsetof(LightUBO, point_lights) + sizeof(PointLightGPU) * MAX_POINT_LIGHTS,
    "LightUBO::sky_zenith moved -- every shader LightUBO block (gfxcoopa/toyengine/blendy) "
    "byte-matches this prefix; see the point_lights/sky_zenith comment above.");
static_assert(offsetof(LightUBO, spot_lights) == offsetof(LightUBO, spot_light_space_matrix) + sizeof(glm::mat4) + sizeof(glm::vec4),
    "LightUBO::spot_lights must immediately follow spot_light_space_matrix/spot_shadow_params.");
static_assert(offsetof(LightUBO, contact_soft_params) == offsetof(LightUBO, contact_params) + sizeof(glm::vec4),
    "LightUBO::contact_soft_params must immediately follow contact_params.");
static_assert(offsetof(LightUBO, dir_cascade_matrix) == offsetof(LightUBO, contact_soft_params) + sizeof(glm::vec4),
    "LightUBO's cascade block must immediately follow contact_soft_params -- appended last per the "
    "append-only rule, so every shorter shader prefix still byte-matches.");
static_assert(offsetof(LightUBO, dir_cascade_info) ==
                  offsetof(LightUBO, dir_cascade_matrix) + sizeof(glm::mat4) * MAX_DIR_CASCADES + sizeof(glm::vec4) * 3,
    "LightUBO's cascade block is packed tighter or looser than the std140 GLSL block in "
    "light_ubo_body.glsl expects (mat4[4] has a 64-byte stride there too).");

/**
 * @class LightData
 * @brief Manages the per-frame directional light uniform buffer.
 *
 * Holds a host-visible uniform buffer containing LightUBO data.
 * Call upload() once per frame after modifying the light parameters.
 */
class LightData {
public:
    /**
     * @brief Creates the light uniform buffer.
     * @param device    Logical device.
     * @param allocator VMA allocator.
     */
    LightData(core::Device& device, memory::Allocator& allocator)
        : buffer_(memory::Buffer::uniform(device, allocator, sizeof(LightUBO)))
    {
        // Upload default light on creation.
        upload();
    }

    LightData(const LightData&) = delete;
    LightData& operator=(const LightData&) = delete;

    /**
     * @brief Returns the mutable light UBO data (modify then call upload()).
     */
    LightUBO& data() { return data_; }

    /**
     * @brief Returns the immutable light UBO data.
     */
    const LightUBO& data() const { return data_; }

    /**
     * @brief Uploads the current LightUBO to the GPU buffer.
     *
     * Call this once per frame (or whenever the light parameters change).
     */
    void upload() {
        buffer_.upload(&data_, sizeof(LightUBO));
    }

    /**
     * @brief Returns the underlying uniform buffer for descriptor binding.
     */
    const memory::Buffer& buffer() const { return buffer_; }

private:
    LightUBO       data_;   /**< Host-side light parameters. */
    memory::Buffer buffer_; /**< GPU-side uniform buffer. */
};

} // namespace data
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_LIGHT_DATA_H
