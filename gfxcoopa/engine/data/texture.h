/**
 * @file texture.h
 * @brief GPU-resident 2D texture: an Image plus its own Sampler, decoded from an image file.
 *
 * Introduced alongside coopa::asset::TextureLoader to close a long-standing
 * gap: gfxcoopa had no image decoder at all (stbi_load only ever lived in
 * uicoopa, for UI sprites/fonts), so PBRMaterial's texture_albedo/
 * texture_normal/texture_metallic_roughness fields were parsed from scene
 * YAML and then read by nothing. This class + its loader are what those
 * fields now resolve to.
 */

#ifndef GFXCOOPA_ENGINE_DATA_TEXTURE_H
#define GFXCOOPA_ENGINE_DATA_TEXTURE_H

#include <volk/volk.h>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/memory/image_upload.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/engine/util/sampler.h>

#include <memory>
#include <cstdint>
#include <vector>

namespace coopa {
namespace gfx {
namespace engine {
namespace data {

/**
 * @class Texture
 * @brief GPU-resident 2D texture: an Image (RGBA8) plus a Sampler.
 *
 * Move-only, like every other GPU-owning type in gfxcoopa. Constructed via
 * upload() from already-decoded RGBA8 pixel bytes — decoding itself
 * (stb_image) happens in coopa::asset::TextureLoader::decode_typed(), off
 * the main thread when loaded via AssetManager::load_async(); this class
 * only ever touches the GPU.
 */
class Texture {
public:
    /**
     * @brief Uploads decoded RGBA8 pixel data into a new GPU texture.
     *
     * @param device     Logical device.
     * @param allocator  VMA allocator.
     * @param cmd_pool   Command pool for the one-shot upload command buffer.
     * @param pixels     Tightly packed RGBA8 pixel data, width * height * 4 bytes.
     * @param width      Texture width in pixels.
     * @param height     Texture height in pixels.
     * @param srgb       True to interpret the data as sRGB-encoded (albedo/base color maps);
     *                   false for linear data (normal maps, metallic/roughness, masks).
     * @return A new GPU-resident Texture, ready to sample.
     */
    static Texture upload(core::Device& device,
                          memory::Allocator& allocator,
                          command::CommandPool& cmd_pool,
                          const uint8_t* pixels,
                          uint32_t width, uint32_t height,
                          bool srgb)
    {
        VkFormat format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        auto image = memory::upload_image_2d(device, allocator, cmd_pool, pixels, width, height, format, 4);
        auto sampler = std::make_unique<util::Sampler>(device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
        return Texture(std::move(image), std::move(sampler));
    }

    /** @brief The image view to bind into a descriptor set. */
    VkImageView view() const { return image_->view(); }

    /** @brief The sampler to bind alongside view(). */
    VkSampler sampler() const { return sampler_->handle(); }

    uint32_t width()  const { return image_->width(); }
    uint32_t height() const { return image_->height(); }

    // Move only (Image/Sampler are not copyable).
    Texture(Texture&&) = default;
    Texture& operator=(Texture&&) = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

private:
    Texture(std::unique_ptr<memory::Image> image, std::unique_ptr<util::Sampler> sampler)
        : image_(std::move(image)), sampler_(std::move(sampler)) {}

    std::unique_ptr<memory::Image>   image_;
    std::unique_ptr<util::Sampler>   sampler_;
};

} // namespace data
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_DATA_TEXTURE_H
