/**
 * @file texture_loader.h
 * @brief coopa::asset loader for gfx::data::Texture — decodes PNG/JPG/etc via stb_image.
 *
 * decode_typed() runs off-thread (when loaded via AssetManager::load_async())
 * and only ever touches CPU memory: it reads the file's bytes and decodes
 * pixels with stb_image. finalize_typed() runs on the main thread and is
 * the only place that touches the GPU (uploads the decoded pixels via
 * gfx::memory::upload_image_2d).
 *
 * stb_image's implementation lives in gfxcoopa now (STB_IMAGE_IMPLEMENTATION
 * is defined once, in gfxcoopa's own CMakeLists.txt) — moved down from
 * uicoopa, which had the only image decoder in the whole workspace despite
 * gfxcoopa being where texture-mapped materials actually need one.
 */

#ifndef GFXCOOPA_ENGINE_LOADERS_TEXTURE_LOADER_H
#define GFXCOOPA_ENGINE_LOADERS_TEXTURE_LOADER_H

#include <coopa/asset/asset_loader.h>
#include <coopa/asset/asset_source.h>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/engine/data/texture.h>

#include <stb/stb_image.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

namespace coopa {
namespace gfx {
namespace engine {
namespace loaders {

/**
 * @struct DecodedImage
 * @brief CPU-side decoded RGBA8 pixel buffer — the intermediate between decode_typed() and finalize_typed().
 */
struct DecodedImage {
    std::vector<uint8_t> pixels; /**< Tightly packed RGBA8, width * height * 4 bytes. */
    uint32_t width  = 0;
    uint32_t height = 0;
};

/**
 * @class TextureLoader
 * @brief Registers as the coopa::asset loader for gfx::data::Texture.
 *
 * Always decodes to RGBA8 and uploads as VK_FORMAT_R8G8B8A8_UNORM. sRGB
 * vs. linear interpretation (needed once material textures are actually
 * bound into the G-buffer pipeline's descriptor sets — a separate, not-yet-
 * implemented change, see gfxcoopa's asset-system integration plan) is left
 * for that follow-up rather than guessed at here.
 *
 * @code
 * assets.register_loader<gfx::data::Texture>(
 *     std::make_unique<TextureLoader>(device, allocator, cmd_pool));
 * auto tex = assets.load<gfx::data::Texture>("textures/brick.png");
 * @endcode
 */
class TextureLoader : public coopa::asset::TypedAssetLoader<data::Texture, DecodedImage> {
public:
    TextureLoader(core::Device& device, memory::Allocator& allocator, command::CommandPool& cmd_pool)
        : device_(device), allocator_(allocator), cmd_pool_(cmd_pool) {}

    std::shared_ptr<DecodedImage> decode_typed(const coopa::asset::AssetId& id,
                                               const coopa::asset::LoadContext& ctx) override {
        int w = 0, h = 0, channels = 0;
        stbi_uc* pixels = stbi_load(ctx.resolved_path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
        if (!pixels) {
            throw std::runtime_error("[TextureLoader] Failed to decode '" + id.path() + "': " +
                                     (stbi_failure_reason() ? stbi_failure_reason() : "unknown error"));
        }

        auto decoded = std::make_shared<DecodedImage>();
        decoded->width  = static_cast<uint32_t>(w);
        decoded->height = static_cast<uint32_t>(h);
        decoded->pixels.assign(pixels, pixels + (static_cast<size_t>(w) * h * 4));
        stbi_image_free(pixels);
        return decoded;
    }

    std::shared_ptr<data::Texture> finalize_typed(std::shared_ptr<DecodedImage> decoded,
                                                  const coopa::asset::AssetId&,
                                                  const coopa::asset::LoadContext&) override {
        return std::make_shared<data::Texture>(
            data::Texture::upload(device_, allocator_, cmd_pool_,
                                  decoded->pixels.data(), decoded->width, decoded->height,
                                  /*srgb=*/false));
    }

    const char* type_name() const override { return "Texture"; }

private:
    core::Device&         device_;
    memory::Allocator&     allocator_;
    command::CommandPool&  cmd_pool_;
};

} // namespace loaders
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_LOADERS_TEXTURE_LOADER_H
