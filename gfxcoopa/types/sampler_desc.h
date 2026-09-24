/**
 * @file sampler_desc.h
 * @brief Sampler configuration, with named presets for the common cases so
 * most call sites never need to name a Filter/AddressMode/CompareOp at all.
 */

#ifndef COOPA_GFX_TYPES_SAMPLER_DESC_H
#define COOPA_GFX_TYPES_SAMPLER_DESC_H

#include <gfxcoopa/types/enums.h>

namespace coopa {
namespace gfx {

/**
 * @struct SamplerDesc
 * @brief Full sampler configuration.
 *
 * `compare == CompareOp::Never` means comparison sampling is disabled — see
 * CompareOp's docs for why this doubles as a bool. `max_anisotropy == 0.0f`
 * means anisotropic filtering is disabled.
 */
struct SamplerDesc {
    Filter      min = Filter::Linear;   ///< Minification filter.
    Filter      mag = Filter::Linear;   ///< Magnification filter.
    MipmapMode  mipmap = MipmapMode::Linear;
    AddressMode address = AddressMode::Repeat;
    CompareOp   compare = CompareOp::Never;   ///< Never == comparison disabled.
    float       max_anisotropy = 0.0f;        ///< 0 == anisotropic filtering disabled.

    /// @brief Bilinear filtering, tiling UVs -- the default SamplerDesc,
    /// named for call sites that want the choice to read explicitly.
    static SamplerDesc linear_repeat() { return SamplerDesc{}; }

    /// @brief Point sampling, clamped UVs. For pixel-art sprite/tile atlases,
    /// where bilinear filtering blurs texel edges and wrapping across
    /// unrelated packed frames is never wanted.
    static SamplerDesc pixel_art() {
        SamplerDesc d;
        d.min = Filter::Nearest;
        d.mag = Filter::Nearest;
        d.mipmap = MipmapMode::Nearest;
        d.address = AddressMode::ClampToEdge;
        return d;
    }

    /// @brief Bilinear filtering, clamped UVs -- the sampler half of texel-AA
    /// ("anti-aliased point sampling") for pixel-art textures rendered by a free
    /// 3D camera. Pair it with gfx_texel_aa_uv() (gfx/texel_aa.glsl), which
    /// sharpens UVs so the bilinear blend spans exactly one screen pixel at each
    /// texel boundary: a resting frame stays visually identical to pixel_art()'s
    /// point sampling, while camera motion glides texel edges sub-pixel instead
    /// of snapping them -- the snap is what reads as full-surface shimmer on
    /// magnified texels.
    static SamplerDesc pixel_art_smooth() {
        SamplerDesc d;
        d.min = Filter::Linear;
        d.mag = Filter::Linear;
        d.mipmap = MipmapMode::Nearest;
        d.address = AddressMode::ClampToEdge;
        return d;
    }

    /// @brief Linear filtering with depth-compare enabled, for shadow map
    /// sampling (hardware PCF via sampler2DShadow).
    static SamplerDesc shadow() {
        SamplerDesc d;
        d.address = AddressMode::ClampToBorder;
        d.compare = CompareOp::LessOrEqual;
        return d;
    }
};

} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_TYPES_SAMPLER_DESC_H
