/**
 * @file volumetrics_data.h
 * @brief Per-frame uniform buffer for the raymarched local volume pass.
 *
 * Self-contained (own inv_view_proj/camera_pos rather than extending CameraData
 * in camera_ubo.h) for the same reason FogData is: adding it never touches the
 * std140 layout ~10 existing shaders already share via CameraUBO/LightData.
 *
 * There is deliberately NO global term here. Fog (fog_data.h) is the global,
 * analytic, config-driven atmosphere; everything in this buffer is a BOUNDED,
 * scene-placed volume. That split is why a "layer" and a "volume" are the same
 * object in this design -- each volume carries its own complete field
 * description alongside its bounds, rather than referring to a separate global
 * layer array.
 */

#ifndef GFXCOOPA_ENGINE_DATA_VOLUMETRICS_DATA_H
#define GFXCOOPA_ENGINE_DATA_VOLUMETRICS_DATA_H

#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace data {

/// Maximum simultaneous local volumes. A plain UBO array (not an SSBO)
/// deliberately -- see VolumetricsPass's doc for why this stays a uniform binding.
static constexpr uint32_t MAX_VOLUMES = 8;

/**
 * @struct VolumeGPU
 * @brief std140-aligned local volume: bounds plus a complete field description.
 *
 * `mode_params.x` selects the density function (see VolumeKind in
 * components/volume.h and gfx/volumetrics.glsl): 0 uniform, 1 ridged ribbons,
 * 2 smooth billows. Every other field means the same thing for all three, which
 * is what lets one march serve all of them.
 */
struct alignas(16) VolumeGPU {
    glm::mat4 inv_world       = glm::mat4(1.0f); /**< World -> volume local space (TRS inverse). */
    glm::vec4 extent_shape    = glm::vec4(1.0f); /**< xyz = local half-extent (Sphere uses .x as radius),
                                                   w = 0 Box / 1 Sphere. */
    glm::vec4 direction_speed = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); /**< xyz = NORMALIZED advection direction,
                                                   w = speed in world units per second. */
    glm::vec4 field_params    = glm::vec4(0.0f); /**< x = noise scale (for ridged noise this sets feature
                                                   THICKNESS as well as spacing), y = streak (elongation
                                                   along the direction; 1 = isotropic), z = coverage
                                                   (fraction of the volume with no density),
                                                   w = fbm amplitude gain per octave. */
    glm::vec4 shape_params    = glm::vec4(0.0f); /**< x = peak density (extinction per world unit),
                                                   y = height base (world Z; engine is Z-up),
                                                   z = height falloff in world units (<= 0 disables),
                                                   w = fbm octave count. */
    glm::vec4 flow_params     = glm::vec4(0.0f); /**< x = meander amplitude in WORLD UNITS (0 disables the
                                                   domain warp), y = meander frequency, z = ridge sharpness,
                                                   w = sparsity-gate frequency (<= 0 disables the gate). */
    glm::vec4 color_occlusion = glm::vec4(0.0f); /**< rgb = scatter colour, w = occlusion scale: how much of
                                                   the extinction actually veils the scene behind it.
                                                   0 = pure additive glow, 1 = a full participating medium. */
    glm::vec4 mode_params     = glm::vec4(0.0f); /**< x = kind (0 Fog / 1 Wind / 2 Haze),
                                                   y = Henyey-Greenstein sun in-scatter strength,
                                                   z = edge softness, a fraction of a dimensionless [0,1]
                                                   surface metric (0 = hard, 1 = maximally soft). w unused. */
};

/**
 * @struct VolumetricsUBO
 * @brief std140-aligned shared march state plus up to MAX_VOLUMES local volumes.
 *
 * Must match volumetrics.frag's VolumetricsUBO block field for field. Scalars are
 * packed into vec4s alongside related fields, matching FogUBO/LightUBO.
 */
struct alignas(16) VolumetricsUBO {
    glm::mat4 inv_view_proj = glm::mat4(1.0f); /**< Clip -> world, for reconstructing per-pixel view rays.
                                                 Built from the UNJITTERED projection: these fields are
                                                 sampled in world space, so TAA jitter here would make them
                                                 swim against the pixel grid on top of their own motion. */
    glm::vec4 camera_pos    = glm::vec4(0.0f); /**< xyz = world camera position, w = debug view flag
                                                 (> 0.5 outputs accumulated density alone). */
    glm::vec4 sun_direction = glm::vec4(0.0f); /**< xyz = direction the light travels (matches LightUBO). */
    glm::vec4 sun_color     = glm::vec4(0.0f); /**< rgb = sun colour * intensity, for the HG in-scatter tint. */
    glm::vec4 march_params  = glm::vec4(0.0f); /**< x = raymarch step count, y = max march distance,
                                                 z = max opacity, w = HG anisotropy g. Anisotropy is SHARED
                                                 rather than per-volume: it is a property of the light's
                                                 phase function, not of which medium a sample sits in, and
                                                 hoisting it keeps the phase evaluated once per pixel. */
    glm::vec4 time_params   = glm::vec4(0.0f); /**< x = elapsed time (seconds; drives advection),
                                                 y = delta time, z = frame index (dither shift). w unused. */
    glm::vec4 counts        = glm::vec4(0.0f); /**< x = active volume count. yzw unused. */
    VolumeGPU volumes[MAX_VOLUMES];            /**< Only the first counts.x are read. */
};

/**
 * @class VolumetricsData
 * @brief Manages the per-frame volumetrics uniform buffer.
 *
 * Mirrors FogData's shape: a host-visible, persistently-mapped uniform buffer,
 * re-uploaded once per frame. Fill data()'s fields directly, then upload().
 *
 * CAVEAT, inherited from FogData deliberately: SINGLE-buffered (one UBO, not one
 * per frame in flight), so inv_view_proj can be one frame stale under fast camera
 * motion. VolumetricsPass binds its descriptor set once at construction, so making
 * this per-slot needs an additive API change for a hazard a low-frequency
 * volumetric does not visibly exercise.
 */
class VolumetricsData {
public:
    /**
     * @brief Creates the volumetrics uniform buffer.
     * @param device    Logical device.
     * @param allocator VMA allocator.
     */
    VolumetricsData(core::Device& device, memory::Allocator& allocator)
        : buffer_(memory::Buffer::uniform(device, allocator, sizeof(VolumetricsUBO)))
    {
        buffer_.upload(&data_, sizeof(VolumetricsUBO));
    }

    VolumetricsData(const VolumetricsData&) = delete;
    VolumetricsData& operator=(const VolumetricsData&) = delete;

    /** @brief Mutable access to the host-side data, filled by the caller before upload(). */
    VolumetricsUBO& data() { return data_; }

    /** @brief Uploads the current host-side data to the GPU. Call once per frame. */
    void upload() { buffer_.upload(&data_, sizeof(VolumetricsUBO)); }

    /** @brief Returns the underlying uniform buffer for descriptor binding. */
    const memory::Buffer& buffer() const { return buffer_; }

private:
    VolumetricsUBO data_;
    memory::Buffer buffer_;
};

} // namespace data
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_DATA_VOLUMETRICS_DATA_H
