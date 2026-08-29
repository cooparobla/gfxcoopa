/**
 * @file fog_data.h
 * @brief Per-frame global + local fog uniform buffer.
 *
 * Self-contained (own inv_view_proj/camera_pos rather than extending
 * CameraData in camera_ubo.h) so adding fog never touches the std140 layout
 * ~10 existing shaders already share via CameraUBO/LightData. See FogPass
 * for the pass that consumes this.
 */

#ifndef GFXCOOPA_ENGINE_DATA_FOG_DATA_H
#define GFXCOOPA_ENGINE_DATA_FOG_DATA_H

#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace data {

/// Maximum simultaneous local fog volumes. A plain UBO array (not an SSBO)
/// deliberately -- see FogPass's doc for why this stays a uniform binding.
static constexpr uint32_t MAX_FOG_VOLUMES = 8;

/**
 * @struct FogVolumeGPU
 * @brief std140-aligned local fog volume (box or sphere), in gfx/fog.glsl's vocabulary.
 */
struct alignas(16) FogVolumeGPU {
    glm::mat4 inv_world     = glm::mat4(1.0f); /**< World -> volume local space (TRS inverse). */
    glm::vec4 extent_shape  = glm::vec4(1.0f); /**< xyz = local half-extent (Sphere uses .x as radius), w = 0 Box / 1 Sphere. */
    glm::vec4 color_density = glm::vec4(0.0f); /**< rgb = fog colour, a = density (extinction per world unit). */
    glm::vec4 falloff       = glm::vec4(0.0f); /**< x = edge softness, as a fraction of a dimensionless [0,1]
                                                 surface metric (0 = hard edge, 1 = maximally soft); NOT a
                                                 world distance. See gfx_fog_box_edge_weight/
                                                 gfx_fog_sphere_edge_weight in gfx/fog.glsl -- the fade is
                                                 evaluated at the ray's entry point on the volume's surface
                                                 so it looks the same from any viewing angle, unlike a
                                                 chord-length-based fade. yzw unused. */
};

/**
 * @struct FogUBO
 * @brief std140-aligned global fog parameters plus up to MAX_FOG_VOLUMES local volumes.
 *
 * Layout intentionally keeps every scalar packed into a vec4 alongside related fields
 * (matching PointLightGPU/LightUBO's convention in light_data.h) rather than declaring
 * bare floats, which std140 would pad individually anyway.
 */
struct alignas(16) FogUBO {
    glm::mat4 inv_view_proj  = glm::mat4(1.0f); /**< Clip -> world, for reconstructing sky-pixel rays. */
    glm::vec4 camera_pos     = glm::vec4(0.0f); /**< xyz = world-space camera position. */
    glm::vec4 fog_color      = glm::vec4(0.0f); /**< rgb = base fog colour. */
    glm::vec4 sun_direction  = glm::vec4(0.0f); /**< xyz = direction the light travels (matches LightUBO::dir_direction). */
    glm::vec4 sun_color      = glm::vec4(0.0f); /**< rgb = sun colour * intensity, for the HG in-scatter tint. */
    glm::vec4 mode_density   = glm::vec4(0.0f); /**< x = mode (0 Linear/1 Exp/2 Exp2), y = density, z = linear_start, w = linear_end. */
    glm::vec4 height_params  = glm::vec4(0.0f); /**< x = height_base, y = height_falloff (<=0 disables), z = sky_blend, w = sun_amount. */
    glm::vec4 misc_params    = glm::vec4(0.0f); /**< x = sun anisotropy g, y = max_opacity, z = volume_count,
                                                 w = fog_max_distance -- the distance the global fog term
                                                 saturates at, and the distance sky pixels are evaluated at
                                                 (rather than a hard-coded "no fog"), so a grazing ray's
                                                 near-horizon geometry and the sky converge to the same
                                                 transmittance instead of meeting at a visible seam. */
    FogVolumeGPU volumes[MAX_FOG_VOLUMES];
};

/**
 * @class FogData
 * @brief Manages the per-frame fog uniform buffer.
 *
 * Mirrors LightData's shape (light_data.h): a host-visible, persistently-mapped
 * uniform buffer, re-uploaded once per frame via update(). Call the returned
 * data() reference's fields directly then upload(), rather than a monolithic
 * setter, since the caller (PbrRenderPipeline) fills global params and a
 * variable-length volume list from two different code paths.
 */
class FogData {
public:
    /**
     * @brief Creates the fog uniform buffer.
     * @param device    Logical device.
     * @param allocator VMA allocator.
     */
    FogData(core::Device& device, memory::Allocator& allocator)
        : buffer_(memory::Buffer::uniform(device, allocator, sizeof(FogUBO)))
    {
        buffer_.upload(&data_, sizeof(FogUBO));
    }

    FogData(const FogData&) = delete;
    FogData& operator=(const FogData&) = delete;

    /** @brief Mutable access to the host-side fog data, filled by the caller before upload(). */
    FogUBO& data() { return data_; }

    /** @brief Uploads the current host-side fog data to the GPU. Call once per frame. */
    void upload() { buffer_.upload(&data_, sizeof(FogUBO)); }

    /** @brief Returns the underlying uniform buffer for descriptor binding. */
    const memory::Buffer& buffer() const { return buffer_; }

private:
    FogUBO         data_;
    memory::Buffer buffer_;
};

} // namespace data
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_DATA_FOG_DATA_H
