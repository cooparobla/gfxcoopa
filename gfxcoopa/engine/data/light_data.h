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
 * @struct LightUBO
 * @brief std140-aligned directional and point light data uploaded to the GPU.
 */
struct alignas(16) LightUBO {
    // Directional Light
    glm::vec4 dir_direction          = glm::vec4(0.0f); /**< xyz = light direction (normalized), w = intensity */
    glm::vec4 dir_color              = glm::vec4(0.0f); /**< xyz = RGB color, w = unused */
    /**< Formerly dir_ambient (DirectionalLight::ambient, RGB). That field had
     * exactly one reader anywhere -- pbr.frag's legacy fallback branch, in
     * PbrPipeline, which no consumer ever instantiates -- so it was dead.
     * Kept as unused padding, not deleted, so this slot's offset (and every
     * later field's) stays identical across the ~11 shader LightUBO blocks
     * in gfxcoopa/toyengine/blendy that must byte-match this struct;
     * reclaiming it would require editing all of them atomically for zero
     * behavioural gain. Sky colour instead lives in the new trailing fields
     * below (see IndirectParams, which is where the config value now lives). */
    glm::vec4 _reserved_was_dir_ambient = glm::vec4(0.0f);
    glm::mat4 dir_light_space_matrix = glm::mat4(1.0f); /**< Light projection * view matrix for directional shadows */
    glm::vec4 dir_shadow_params      = glm::vec4(0.005f, 0.0f, 1.0f, 0.0f); /**< x=bias, y=pcf_samples, z=shadow_enabled (1 or 0) */

    // Light counts and Point Lights
    glm::uvec4 light_counts = glm::uvec4(0); /**< x = num_directional (0 or 1), y = num_point_lights */
    PointLightGPU point_lights[MAX_POINT_LIGHTS];

    // Configurable sky/ambient colour (see IndirectParams in render_features.h).
    // Appended after point_lights so no existing member's offset moves --
    // a shader that doesn't care about sky colour can keep declaring this
    // block without these three fields; std140 only requires a matching
    // *prefix*.
    glm::vec4 sky_zenith  = glm::vec4(0.05f, 0.18f, 0.55f, 0.0f);  /**< xyz = zenith colour, straight up. */
    glm::vec4 sky_horizon = glm::vec4(0.25f, 0.35f, 0.45f, 0.0f);  /**< xyz = horizon colour. */
    glm::vec4 sky_ground  = glm::vec4(0.05f, 0.045f, 0.04f, 0.0f); /**< xyz = ground colour, straight down. */
};

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
