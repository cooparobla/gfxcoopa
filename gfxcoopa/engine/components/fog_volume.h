/**
 * @file fog_volume.h
 * @brief Scene component defining a local box/sphere fog pocket.
 */

#ifndef GFXCOOPA_ENGINE_COMPONENTS_FOG_VOLUME_H
#define GFXCOOPA_ENGINE_COMPONENTS_FOG_VOLUME_H

#include <coopa/scene/component.h>
#include <glm/glm.hpp>
#include <string>

namespace coopa {
namespace gfx {
namespace engine {
namespace components {

/// Shape a FogVolumeComponent is evaluated against. Matches gfx/fog.glsl's
/// extent_shape.w encoding (0 = Box, 1 = Sphere) via FogVolumeGPU.
enum class FogVolumeShape {
    Box,
    Sphere,
};

/// Defines a local, artist-placed fog pocket in world space. Density is
/// evaluated only where the containing Transform's box/sphere is crossed by
/// the view ray, on top of PbrRenderConfig's global fog term -- see FogPass.
class FogVolumeComponent : public coopa::scene::Component {
public:
    FogVolumeComponent() = default;
    std::string type_name() const override { return "FogVolumeComponent"; }
    void update(float /*dt*/) override {}

    FogVolumeShape shape   = FogVolumeShape::Box;
    glm::vec3      extent  = glm::vec3(2.0f);              ///< Local half-extent (Sphere uses .x as radius).
    glm::vec3      color   = glm::vec3(0.8f, 0.85f, 0.9f);  ///< Local fog tint.
    float          density = 0.15f;                         ///< Extinction per world unit inside the volume.
    /// Edge softness, as a fraction of a dimensionless [0,1] surface metric (0 = hard
    /// edge, 1 = maximally soft) -- NOT a world distance or a fraction of `extent`.
    /// Evaluated at the ray's entry point on the volume's surface (see
    /// gfx_fog_box_edge_weight/gfx_fog_sphere_edge_weight in gfx/fog.glsl), so the
    /// apparent softness is the same from any viewing angle.
    float          falloff = 0.2f;
};

} // namespace components
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_COMPONENTS_FOG_VOLUME_H
