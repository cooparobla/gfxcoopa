#version 450

// shadow_depth.frag — Directional light depth map rendering.
//
// Mostly still "empty": depth is written automatically by the
// fixed-function rasterizer. The one thing this stage does is let a BLEND
// occluder cast a lighter shadow proportional to its material alpha, via a
// stochastic discard -- see gfx/shadow_dither.glsl for how that converges to
// a shadow exactly `alpha` dark once averaged over calc_dir_shadow's PCF
// taps in deferred_lighting.frag.

#include <gfx/shadow_dither.glsl>

// light_space_matrix is unused here -- it's the vertex stage's field -- but
// must be declared to match that stage's block byte-for-byte (shared
// VkPushConstantRange).
layout(push_constant) uniform ShadowPC {
    mat4 light_space_matrix;
    float alpha;
} pc;

void main() {
    // alpha == 1.0 (OPAQUE/MASK casters, and the overwhelming common case)
    // skips this entirely -- no cost added to the pre-existing path.
    if (pc.alpha < 1.0 && shadow_alpha_dither(gl_FragCoord.xy) > pc.alpha) {
        discard;
    }
}
