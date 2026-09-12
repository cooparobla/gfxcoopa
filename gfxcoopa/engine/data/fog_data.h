/**
 * @file fog_data.h
 * @brief Per-frame global fog uniform buffer.
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

/**
 * @struct FogUBO
 * @brief std140-aligned global fog parameters.
 *
 * Fog is GLOBAL ONLY and config-driven. Local volumes of every kind (including
 * static fog pockets) are raymarched by VolumetricsPass instead -- see
 * VolumeComponent -- so this carries no volume array.
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
    glm::vec4 misc_params    = glm::vec4(0.0f); /**< x = sun anisotropy g, y = max_opacity,
                                                 z = UNUSED (was the local volume count, before fog
                                                 became global-only),
                                                 w = fog_max_distance -- the distance the global fog term
                                                 saturates at, and the distance sky pixels are evaluated at
                                                 (rather than a hard-coded "no fog"), so a grazing ray's
                                                 near-horizon geometry and the sky converge to the same
                                                 transmittance instead of meeting at a visible seam. */
    // Configurable sky colour fog blends toward when sky_blend > 0 (height_params.z).
    // Must match the lighting pass's colours -- see IndirectParams' doc (render_features.h).
    glm::vec4 sky_zenith  = glm::vec4(0.05f, 0.18f, 0.55f, 0.0f);
    glm::vec4 sky_horizon = glm::vec4(0.25f, 0.35f, 0.45f, 0.0f);
    glm::vec4 sky_ground  = glm::vec4(0.05f, 0.045f, 0.04f, 0.0f);
};

/**
 * @class FogData
 * @brief Manages the per-frame fog uniform buffer.
 *
 * Mirrors LightData's shape (light_data.h): a host-visible, persistently-mapped
 * uniform buffer, re-uploaded once per frame via update(). Call the returned
 * data() reference's fields directly then upload(), rather than a monolithic
 * setter, since the caller fills these fields from several code paths.
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
