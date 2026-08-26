/**
 * @file smaa_textures.h
 * @brief Official Jimenez 2013 SMAA 1x Area and Search textures loader.
 */

#ifndef GFXCOOPA_ENGINE_UTIL_SMAA_TEXTURES_H
#define GFXCOOPA_ENGINE_UTIL_SMAA_TEXTURES_H

#include <volk/volk.h>
#include <vector>
#include <memory>
#include <cstdint>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/memory/image_upload.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/util/sampler.h>

// Include official Jimenez SMAA texture headers
#include "/home/coopa/third-party/smaa/Textures/SearchTex.h"
#include "/home/coopa/third-party/smaa/Textures/AreaTex.h"

namespace coopa {
namespace gfx {
namespace engine {
namespace util {

class SmaaTextures {
public:
    SmaaTextures(coopa::gfx::core::Device& device,
                 coopa::gfx::memory::Allocator& allocator,
                 coopa::gfx::command::CommandPool& cmd_pool)
        : device_(device), allocator_(allocator)
    {
        create_search_texture_(cmd_pool);
        create_area_texture_(cmd_pool);

        // Linear + clamp-to-edge, matching the raw ctor this replaces (its default mipmap_mode
        // was NEAREST, moot here since both textures are single-mip -- max_lod=0).
        coopa::gfx::SamplerDesc desc;
        desc.min = desc.mag = coopa::gfx::Filter::Linear;
        desc.mipmap  = coopa::gfx::MipmapMode::Nearest;
        desc.address = coopa::gfx::AddressMode::ClampToEdge;
        sampler_ = std::make_unique<Sampler>(device, desc);
    }

    VkImageView search_view() const { return search_image_->view(); }
    coopa::gfx::TextureView search_view_typed() const { return search_image_->view_typed(); }
    VkImageView area_view()   const { return area_image_->view(); }
    coopa::gfx::TextureView area_view_typed() const { return area_image_->view_typed(); }
    const Sampler& sampler() const { return *sampler_; }

private:
    void create_search_texture_(coopa::gfx::command::CommandPool& cmd_pool) {
        search_image_ = coopa::gfx::memory::upload_image_2d(
            device_, allocator_, cmd_pool, searchTexBytes,
            SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, coopa::gfx::Format::R8_Unorm, 1);
    }

    void create_area_texture_(coopa::gfx::command::CommandPool& cmd_pool) {
        area_image_ = coopa::gfx::memory::upload_image_2d(
            device_, allocator_, cmd_pool, areaTexBytes,
            AREATEX_WIDTH, AREATEX_HEIGHT, coopa::gfx::Format::RG8_Unorm, 2);
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    std::unique_ptr<coopa::gfx::memory::Image>   search_image_;
    std::unique_ptr<coopa::gfx::memory::Image>   area_image_;
    std::unique_ptr<Sampler> sampler_;
};

} // namespace util
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_UTIL_SMAA_TEXTURES_H
