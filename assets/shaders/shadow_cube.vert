#version 450

// shadow_cube.vert — Point light cubemap face vertex shader.

layout(location = 0) in vec3 in_position;
layout(location = 4) in mat4 in_model; // per-instance (locations 4-7)

layout(location = 0) out vec3 frag_world_pos;

// model moved to the per-instance stream above. alpha is unused here -- it
// drives shadow_cube.frag's stochastic discard -- but must be declared to
// match that stage's block byte-for-byte (shared VkPushConstantRange).
layout(push_constant) uniform CubeShadowPC {
    mat4 light_space_matrix;
    vec4 light_pos_range; // xyz = light pos, w = range
    float alpha;
} pc;

void main() {
    vec4 world_pos = in_model * vec4(in_position, 1.0);
    frag_world_pos = world_pos.xyz;
    gl_Position    = pc.light_space_matrix * world_pos;
}
