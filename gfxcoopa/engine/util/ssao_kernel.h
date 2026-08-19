#include <gfxcoopa/engine/util/sampler.h>
/**
 * @file ssao_kernel.h
 * @brief SSAO hemisphere sample kernel (UBO) and tiled rotation-noise texture.
 */

#ifndef GFXCOOPA_ENGINE_UTIL_SSAO_KERNEL_H
#define GFXCOOPA_ENGINE_UTIL_SSAO_KERNEL_H

#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>
#include <memory>
#include <random>
#include <cmath>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/memory/image_upload.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace util {



/// Cosine-weighted-ish hemisphere sample kernel (tangent space, z >= 0), uploaded once into a
/// UBO sized for the maximum sample count so ssao.frag's runtime-tunable kernel_size can shrink
/// without ever reading past what was uploaded. Samples are biased toward the origin (Crysis/
/// LearnOpenGL SSAO convention) so a small kernel still looks dense near the surface it shades.
class SsaoKernel {
public:
    static constexpr uint32_t kMaxSamples = 32;

    SsaoKernel(coopa::gfx::core::Device& device, coopa::gfx::memory::Allocator& allocator)
        : buffer_(coopa::gfx::memory::Buffer::uniform(device, allocator, kMaxSamples * sizeof(glm::vec4)))
    {
        // Fixed seed: the kernel is generated once at startup and never touched again, so
        // determinism across runs matters more here than true randomness.
        std::mt19937 rng(0xA0u);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::uniform_real_distribution<float> signed_unit(-1.0f, 1.0f);

        std::vector<glm::vec4> samples(kMaxSamples);
        for (uint32_t i = 0; i < kMaxSamples; ++i) {
            glm::vec3 s(signed_unit(rng), signed_unit(rng), unit(rng));
            s = glm::normalize(s) * unit(rng);

            float t = static_cast<float>(i) / static_cast<float>(kMaxSamples);
            float scale = 0.1f + 0.9f * (t * t);
            s *= scale;

            samples[i] = glm::vec4(s, 0.0f);
        }

        buffer_.upload(samples.data(), kMaxSamples * sizeof(glm::vec4));
    }

    const coopa::gfx::memory::Buffer& buffer() const { return buffer_; }

private:
    coopa::gfx::memory::Buffer buffer_;
};

/// 4x4 tiled rotation-noise texture (RG8_UNORM, one unit XY vector per texel). Generated
/// procedurally (no disk asset) and uploaded through a one-time staging buffer, mirroring
/// SmaaTextures::upload_texture_'s shape.
class SsaoNoiseTexture {
public:
    static constexpr uint32_t kSize = 4;

    SsaoNoiseTexture(coopa::gfx::core::Device& device,
                     coopa::gfx::memory::Allocator& allocator,
                     coopa::gfx::command::CommandPool& cmd_pool)
        : device_(device), allocator_(allocator)
    {
        std::vector<uint8_t> texels(kSize * kSize * 2);
        for (uint32_t i = 0; i < kSize * kSize; ++i) {
            float theta = 2.0f * 3.14159265359f * static_cast<float>(i) / static_cast<float>(kSize * kSize);
            float x = std::cos(theta);
            float y = std::sin(theta);
            texels[i * 2 + 0] = static_cast<uint8_t>((x * 0.5f + 0.5f) * 255.0f);
            texels[i * 2 + 1] = static_cast<uint8_t>((y * 0.5f + 0.5f) * 255.0f);
        }

        upload_texture_(cmd_pool, texels.data());

        // REPEAT so the 4x4 tile covers the full screen (ssao.frag samples it at
        // in_uv * (screen_size / 4)); NEAREST because each texel is a discrete rotation vector,
        // not something to blend between.
        sampler_ = std::make_unique<Sampler>(
            device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT
        );
    }

    VkImageView view() const { return image_->view(); }
    const Sampler& sampler() const { return *sampler_; }

private:
    void upload_texture_(coopa::gfx::command::CommandPool& cmd_pool, const void* data) {
        image_ = coopa::gfx::memory::upload_image_2d(
            device_, allocator_, cmd_pool, data, kSize, kSize, VK_FORMAT_R8G8_UNORM, 2);
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    std::unique_ptr<coopa::gfx::memory::Image> image_;
    std::unique_ptr<Sampler>                   sampler_;
};

} // namespace util
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_UTIL_SSAO_KERNEL_H
