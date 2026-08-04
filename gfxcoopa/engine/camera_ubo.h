/**
 * @file camera_ubo.h
 * @brief Per-frame camera matrices UBO for the toon and outline render passes.
 *
 * Manages a host-visible uniform buffer containing view matrix, projection matrix,
 * and camera world position. Updated once per frame from the active CameraComponent.
 *
 * Pixel-snapping: the camera position is rounded to the nearest world-space pixel
 * boundary to prevent sub-pixel jitter at low render resolutions.
 */

#ifndef COOPA_GFX_ENGINE_CAMERA_UBO_H
#define COOPA_GFX_ENGINE_CAMERA_UBO_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <cmath>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/pipeline/descriptor.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @struct CameraData
 * @brief std140-aligned camera matrices and view position.
 *
 * Layout: view (64 bytes) + proj (64 bytes) + view_pos (12 bytes) + padding (4 bytes) = 144 bytes.
 */
struct alignas(16) CameraData {
    glm::mat4 view     = glm::mat4(1.0f); /**< View matrix (world-to-camera). */
    glm::mat4 proj     = glm::mat4(1.0f); /**< Projection matrix (camera-to-clip). */
    glm::vec3 view_pos = glm::vec3(0.0f); /**< Camera world-space position (for specular). */
    float     _pad0    = 0.0f;            /**< std140 padding. */
};

/**
 * @class CameraUBO
 * @brief Manages the per-frame camera uniform buffer.
 *
 * Call update() once per frame with the view and projection matrices.
 * The buffer is host-visible and persistently mapped via VMA.
 */
class CameraUBO {
public:
    /**
     * @brief Creates the camera uniform buffer.
     * @param device    Logical device.
     * @param allocator VMA allocator.
     */
    CameraUBO(core::Device& device, memory::Allocator& allocator)
        : buffer_(memory::Buffer::uniform(device, allocator, sizeof(CameraData)))
    {
        buffer_.upload(&data_, sizeof(CameraData));
    }

    CameraUBO(const CameraUBO&) = delete;
    CameraUBO& operator=(const CameraUBO&) = delete;

    /**
     * @brief Updates the camera UBO from view/projection matrices and uploads to GPU.
     *
     * Applies pixel-snapping to the view position to prevent sub-pixel jitter.
     *
     * @param view          View matrix (world-to-camera).
     * @param proj          Projection matrix (camera-to-clip, Vulkan Y-flipped).
     * @param camera_pos    Camera world-space position.
     * @param pixel_density World-space units per render pixel (for snapping). 0 = disable snapping.
     */
    void update(const glm::mat4& view,
                const glm::mat4& proj,
                const glm::vec3& camera_pos,
                float            pixel_density = 0.0f)
    {
        data_.view = view;
        data_.proj = proj;

        // Pixel-snap: round camera translation to world-space pixel grid.
        if (pixel_density > 0.0f) {
            glm::vec3 snapped = glm::round(camera_pos / pixel_density) * pixel_density;
            data_.view_pos = snapped;
            // Also snap the view translation component.
            data_.view[3][0] = std::round(data_.view[3][0] / pixel_density) * pixel_density;
            data_.view[3][1] = std::round(data_.view[3][1] / pixel_density) * pixel_density;
            data_.view[3][2] = std::round(data_.view[3][2] / pixel_density) * pixel_density;
        } else {
            data_.view_pos = camera_pos;
        }

        buffer_.upload(&data_, sizeof(CameraData));
    }

    /** @brief Returns the underlying uniform buffer for descriptor binding. */
    const memory::Buffer& buffer() const { return buffer_; }

private:
    CameraData     data_;   /**< Host-side camera matrices. */
    memory::Buffer buffer_; /**< GPU-side uniform buffer. */
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_CAMERA_UBO_H
