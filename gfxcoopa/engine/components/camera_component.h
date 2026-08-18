/**
 * @file camera_component.h
 * @brief Component representing a scene camera with perspective or orthographic projection.
 *
 * Stores camera parameters as loaded from the scene YAML's Camera component.
 * Supports PERSP (perspective) and ORTHO (orthographic/isometric) projections.
 * The view matrix is derived from the owning object's TransformComponent.
 */

#ifndef GFXCOOPA_ENGINE_COMPONENTS_CAMERA_COMPONENT_H
#define GFXCOOPA_ENGINE_COMPONENTS_CAMERA_COMPONENT_H

#include <coopa/scene/component.h>
#include <coopa/scene/scene_object.h>
#include <coopa/scene/components/transform_component.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>

namespace coopa {
namespace gfx {
namespace engine {
namespace components {

/**
 * @enum CameraType
 * @brief Projection mode for the camera.
 */
enum class CameraType {
    Perspective,  /**< Perspective projection (PERSP in YAML). */
    Orthographic  /**< Orthographic/isometric projection (ORTHO in YAML). */
};

/**
 * @class CameraComponent
 * @brief Camera with perspective or orthographic projection.
 *
 * Parameters match the Blender YAML export format:
 *   type:         PERSP | ORTHO
 *   fov:          Vertical field of view in degrees (PERSP)
 *   ortho_scale:  Orthographic scale factor (ORTHO)
 *   clip_start:   Near clip plane
 *   clip_end:     Far clip plane
 *   lens:         Focal length in mm (informational)
 *   sensor_width: Sensor width in mm (informational)
 *   sensor_height:Sensor height in mm (informational)
 *
 * For isometric rendering, use CameraType::Orthographic. The isometric
 * viewing angle is baked into the scene object's Transform, not the projection.
 */
class CameraComponent : public coopa::scene::Component {
public:
    CameraComponent() = default;

    std::string type_name() const override { return "Camera"; }

    // --- Camera parameters ---

    CameraType type             = CameraType::Perspective; /**< Projection type. */
    float fov                   = 60.0f;   /**< Vertical FOV in degrees (Perspective mode). */
    float orthographic_size     = 3.0f;    /**< Orthographic half-height in world units (Unity Camera.orthographicSize). */
    float clip_start            = 0.1f;    /**< Near clip distance (alias: near_clip_plane). */
    float clip_end              = 1000.0f; /**< Far clip distance (alias: far_clip_plane). */
    float lens                  = 50.0f;   /**< Lens focal length (informational). */
    float sensor_width          = 36.0f;   /**< Sensor width in mm (informational). */
    float sensor_height         = 24.0f;   /**< Sensor height in mm (informational). */

    // Legacy alias for orthographic scale (full height = orthographic_size * 2)
    float get_ortho_scale() const { return orthographic_size * 2.0f; }
    void set_ortho_scale(float scale) { orthographic_size = scale * 0.5f; }

    // --- Configuration API (Unity-style) ---

    /** @brief Sets perspective projection mode with vertical FOV in degrees. */
    void set_perspective(float fov_deg, float near_clip = 0.1f, float far_clip = 1000.0f) {
        type = CameraType::Perspective;
        fov = fov_deg;
        clip_start = near_clip;
        clip_end = far_clip;
    }

    /** @brief Sets orthographic projection mode with half-height size in world units. */
    void set_orthographic(float ortho_size, float near_clip = 0.1f, float far_clip = 1000.0f) {
        type = CameraType::Orthographic;
        orthographic_size = ortho_size;
        clip_start = near_clip;
        clip_end = far_clip;
    }

    // --- Matrix computation ---

    /**
     * @brief Computes and returns the view matrix from the owner's Transform.
     *
     * The view matrix is the inverse of the owner's world matrix.
     * Returns the identity matrix if the owner has no TransformComponent.
     *
     * @return View matrix as glm::mat4.
     */
    glm::mat4 get_view_matrix() const {
        if (!owner) return glm::mat4(1.0f);
        const auto* tc = owner->get_transform();
        if (!tc) return glm::mat4(1.0f);
        // View = inverse of world transform
        return glm::inverse(tc->get_world_matrix());
    }

    /**
     * @brief Computes and returns the projection matrix.
     *
     * @param aspect_ratio Viewport width / height.
     * @return Projection matrix as glm::mat4 (Vulkan NDC: Y-flipped).
     */
    glm::mat4 get_projection_matrix(float aspect_ratio) const {
        glm::mat4 proj;
        if (type == CameraType::Perspective) {
            proj = glm::perspective(
                glm::radians(fov),
                aspect_ratio,
                clip_start,
                clip_end
            );
        } else {
            // Unity orthographic: orthographic_size is half-height.
            float half_h = orthographic_size;
            float half_w = half_h * aspect_ratio;
            proj = glm::ortho(-half_w, half_w, -half_h, half_h, clip_start, clip_end);
        }
        // Y-flip is handled by the negative viewport height (VK_KHR_maintenance1)
        // instead of flipping the projection matrix, which avoids reversing
        // triangle winding order and keeps face culling correct.
        return proj;
    }

    /**
     * @brief Returns the world-space camera position (for lighting calculations).
     */
    glm::vec3 get_world_position() const {
        if (!owner) return glm::vec3(0.0f);
        const auto* tc = owner->get_transform();
        if (!tc) return glm::vec3(0.0f);
        glm::mat4 world = tc->get_world_matrix();
        return glm::vec3(world[3]); // Translation column
    }
};

} // namespace components
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_COMPONENTS_CAMERA_COMPONENT_H
