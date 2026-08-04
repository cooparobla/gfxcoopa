#version 450

// shadow_cube.frag — Point light cubemap face fragment shader.
// Calculates linear depth normalized by the light's range.

layout(location = 0) in vec3 frag_world_pos;

layout(push_constant) uniform CubeShadowPC {
    mat4 light_space_matrix;
    mat4 model;
    vec4 light_pos_range; // xyz = light pos, w = range
} pc;

void main() {
    float light_dist = length(frag_world_pos - pc.light_pos_range.xyz);
    // Normalize distance by light range to map to [0, 1]
    gl_FragDepth = clamp(light_dist / pc.light_pos_range.w, 0.0, 1.0);
}
