#version 450

// shadow_cube.frag — Point light cubemap face fragment shader.
// Calculates linear depth normalized by the light's range. A BLEND occluder
// also gets a chance to stochastically discard here, proportional to its
// material alpha -- see gfx/shadow_dither.glsl.

#include <gfx/shadow_dither.glsl>

layout(location = 0) in vec3 frag_world_pos;

// model moved to the per-instance vertex stream (see shadow_cube.vert); not
// read here regardless, but the block layout must still match the vertex
// stage's declaration byte-for-byte.
layout(push_constant) uniform CubeShadowPC {
    mat4 light_space_matrix;
    vec4 light_pos_range; // xyz = light pos, w = range
    float alpha;
} pc;

void main() {
    // alpha == 1.0 (OPAQUE/MASK casters, and the overwhelming common case)
    // skips this entirely -- no cost added to the pre-existing path. Dithers
    // on frag_world_pos rather than gl_FragCoord.xy so the decision doesn't
    // repeat identically across the cube's 6 faces (each face is its own
    // render target sharing the same texel coordinates).
    if (pc.alpha < 1.0 && shadow_alpha_dither(frag_world_pos) > pc.alpha) {
        discard;
    }

    float light_dist = length(frag_world_pos - pc.light_pos_range.xyz);
    // Normalize distance by light range to map to [0, 1]
    gl_FragDepth = clamp(light_dist / pc.light_pos_range.w, 0.0, 1.0);
}
