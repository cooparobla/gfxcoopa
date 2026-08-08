/**
 * @file sampler.h
 * @brief RAII VkSampler wrapper with nearest-neighbour and linear factory methods.
 *
 * Sampler::nearest() is the critical one for pixel-art upscaling — it ensures
 * no bilinear interpolation blurs the retro low-res render.
 */

#ifndef COOPA_GFX_ENGINE_SAMPLER_H
#define COOPA_GFX_ENGINE_SAMPLER_H

#include <volk/volk.h>
#include <stdexcept>
#include <gfxcoopa/core/device.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @class Sampler
 * @brief RAII VkSampler wrapper.
 *
 * Factory methods:
 *   - Sampler::nearest() — pixel-perfect, no interpolation (for upscale pass)
 *   - Sampler::linear()  — bilinear filtering (for post-processing intermediates)
 */
class Sampler {
public:
    /**
     * @brief Creates a VkSampler with the given filter and address mode.
     * @param device        Logical device.
     * @param filter        Min/mag filter (VK_FILTER_NEAREST or VK_FILTER_LINEAR).
     * @param address_mode  UV wrap mode (default: clamp to edge).
     * @param max_lod       Maximum LOD clamp for explicit-LOD sampling (e.g. textureLod) as well
     *                      as implicit mip selection. Defaults to 0.0f (single-mip images, the
     *                      behavior every existing caller relies on). Pass the image's mip count
     *                      to sample a real mip chain (e.g. a Hi-Z pyramid).
     * @param mipmap_mode   Mip selection filter (VK_SAMPLER_MIPMAP_MODE_NEAREST or _LINEAR).
     *                      Defaults to NEAREST, matching every existing caller's expectations.
     *                      Pass LINEAR for smooth roughness-driven mip blending (e.g. a
     *                      prefiltered reflection cubemap).
     */
    Sampler(core::Device& device,
            VkFilter filter,
            VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            float max_lod = 0.0f,
            VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST)
        : device_(device)
    {
        VkSamplerCreateInfo info{};
        info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter               = filter;
        info.minFilter               = filter;
        info.mipmapMode              = mipmap_mode;
        info.addressModeU            = address_mode;
        info.addressModeV            = address_mode;
        info.addressModeW            = address_mode;
        info.mipLodBias              = 0.0f;
        info.anisotropyEnable        = VK_FALSE;
        info.compareEnable           = VK_FALSE;
        info.minLod                  = 0.0f;
        info.maxLod                  = max_lod;
        info.borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        GFX_VK_CHECK(vkCreateSampler(device_.handle(), &info, nullptr, &sampler_));
    }

    ~Sampler() {
        if (sampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_.handle(), sampler_, nullptr);
        }
    }

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    /**
     * @brief Move constructor: transfers VkSampler ownership.
     * @param other The Sampler to move from (left in a null state).
     */
    Sampler(Sampler&& other) noexcept
        : device_(other.device_), sampler_(other.sampler_)
    {
        other.sampler_ = VK_NULL_HANDLE;
    }

    /** @brief Returns the raw VkSampler handle. */
    VkSampler handle() const { return sampler_; }

    // --- Factory methods ---

    /**
     * @brief Creates a nearest-neighbour sampler (pixel-perfect, no blurring).
     *
     * Use this for the upscale pass to preserve the retro pixel aesthetic.
     *
     * @param device Logical device.
     * @return Nearest-neighbour Sampler.
     */
    static Sampler nearest(core::Device& device) {
        return Sampler(device, VK_FILTER_NEAREST);
    }

    /**
     * @brief Creates a linear (bilinear) sampler.
     *
     * Use this for post-processing intermediate steps where smooth sampling is acceptable.
     *
     * @param device Logical device.
     * @return Linear Sampler.
     */
    static Sampler linear(core::Device& device) {
        return Sampler(device, VK_FILTER_LINEAR);
    }

private:
    core::Device& device_;                    /**< Owning logical device (not owned). */
    VkSampler     sampler_ = VK_NULL_HANDLE; /**< The Vulkan sampler. */
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_SAMPLER_H
