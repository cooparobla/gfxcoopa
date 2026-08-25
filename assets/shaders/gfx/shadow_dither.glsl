#ifndef GFX_SHADOW_DITHER_GLSL
#define GFX_SHADOW_DITHER_GLSL

// gfx/shadow_dither.glsl -- helpers shared by the depth-only shadow passes.
// (Renamed from shadow_common.glsl when promoted into the shared base
// library, to leave room for gfx/shadow_sampling.glsl alongside it.)

// Deterministic per-shadow-texel/per-world-point hash (NOT time-varying) --
// stable across frames as long as the shadow-casting geometry doesn't move,
// since these inputs are the shadow pass's own render-target/world position,
// not the main camera's. Used to stochastically discard a BLEND occluder's
// fragment so it blocks light only `alpha` of the time; averaged over a PCF
// kernel at sample time (see gfx/shadow_sampling.glsl), this converges to a
// shadow exactly `alpha` dark, with no second render target needed.
float shadow_alpha_dither(vec2 p) {
    return fract(sin(dot(p, vec2(41.469, 289.917))) * 28001.879);
}

float shadow_alpha_dither(vec3 p) {
    return fract(sin(dot(p, vec3(41.469, 289.917, 127.311))) * 28001.879);
}

#endif // GFX_SHADOW_DITHER_GLSL
