/**
 * @file gi_data.h
 * @brief GPU buffers for Global Illumination (GI) probe volumes and reflection probes.
 */

#ifndef COOPA_GFX_ENGINE_GI_DATA_H
#define COOPA_GFX_ENGINE_GI_DATA_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>

namespace coopa {
namespace gfx {
namespace engine {

/// Per-probe SH data: 9 vec4s = 144 bytes per probe.
/// Each vec4 stores one SH band's RGB coefficients + padding.
/// Bands are ordered: L0(1) + L1(3) + L2(5) = 9 total.
struct alignas(16) SHProbe {
    glm::vec4 bands[9];  // bands[i].xyz = RGB SH coefficient, .w = 0
};

/// Uniform block describing the probe volume grid geometry.
/// Bound as Set 3, Binding 0 (UBO).
struct alignas(16) GiUniforms {
    glm::vec4 grid_origin  = glm::vec4(0.0f); // xyz = world-space origin of grid corner, w = unused
    glm::vec4 grid_spacing = glm::vec4(1.0f); // xyz = cell size per axis, w = unused
    glm::ivec4 grid_counts = glm::ivec4(0);   // xyz = probe counts per axis (Nx, Ny, Nz), w = total
    glm::vec4 gi_params    = glm::vec4(1.0f, 1.0f, 5.0f, 0.0f); // x = gi_intensity, y = reflection_intensity, z = max_roughness_mip, w = num_reflection_probes
};

/// Uniform block for a single reflection probe.
struct alignas(16) ReflectionProbeUniforms {
    glm::vec4 probe_position = glm::vec4(0.0f); // xyz = world position, w = unused
    glm::vec4 box_min        = glm::vec4(0.0f); // xyz = AABB min for parallax correction, w = unused
    glm::vec4 box_max        = glm::vec4(0.0f); // xyz = AABB max for parallax correction, w = unused
    glm::vec4 params         = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f); // x = blend_distance, y = importance, z = 0, w = 0
};

/// Manages the UBO and SSBO for GI data on the GPU.
class GiData {
public:
    GiData(core::Device& device, memory::Allocator& allocator, uint32_t max_probes = 4096)
        : device_(device), allocator_(allocator), max_probes_(max_probes)
    {
        uniforms_buf_ = std::make_unique<memory::Buffer>(
            memory::Buffer::uniform(device, allocator, sizeof(GiUniforms))
        );

        size_t ssbo_size = std::max(sizeof(SHProbe) * max_probes, sizeof(SHProbe));
        probes_buf_ = std::make_unique<memory::Buffer>(
            memory::Buffer::storage(device, allocator, ssbo_size)
        );

        reflection_buf_ = std::make_unique<memory::Buffer>(
            memory::Buffer::uniform(device, allocator, sizeof(ReflectionProbeUniforms))
        );

        upload_uniforms();
    }

    ~GiData() = default;

    GiData(const GiData&) = delete;
    GiData& operator=(const GiData&) = delete;

    GiUniforms& uniforms() { return uniforms_; }
    const GiUniforms& uniforms() const { return uniforms_; }

    ReflectionProbeUniforms& reflection_uniforms() { return reflection_uniforms_; }
    const ReflectionProbeUniforms& reflection_uniforms() const { return reflection_uniforms_; }

    void upload_uniforms() {
        uniforms_buf_->upload(&uniforms_, sizeof(GiUniforms));
    }

    void upload_probes(const std::vector<SHProbe>& probes) {
        if (probes.empty()) return;
        size_t upload_size = std::min(probes.size(), static_cast<size_t>(max_probes_)) * sizeof(SHProbe);
        probes_buf_->upload(probes.data(), upload_size);
    }

    void upload_reflection_uniforms(const std::vector<ReflectionProbeUniforms>& probes) {
        if (probes.empty()) {
            reflection_buf_->upload(&reflection_uniforms_, sizeof(ReflectionProbeUniforms));
        } else {
            reflection_buf_->upload(&probes[0], sizeof(ReflectionProbeUniforms));
        }
    }

    memory::Buffer& uniforms_buffer() { return *uniforms_buf_; }
    const memory::Buffer& uniforms_buffer() const { return *uniforms_buf_; }

    memory::Buffer& probes_buffer() { return *probes_buf_; }
    const memory::Buffer& probes_buffer() const { return *probes_buf_; }

    memory::Buffer& reflection_buffer() { return *reflection_buf_; }
    const memory::Buffer& reflection_buffer() const { return *reflection_buf_; }

private:
    core::Device& device_;
    memory::Allocator& allocator_;
    GiUniforms uniforms_{};
    ReflectionProbeUniforms reflection_uniforms_{};
    std::unique_ptr<memory::Buffer> uniforms_buf_;
    std::unique_ptr<memory::Buffer> probes_buf_;
    std::unique_ptr<memory::Buffer> reflection_buf_;
    uint32_t max_probes_;
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_GI_DATA_H
