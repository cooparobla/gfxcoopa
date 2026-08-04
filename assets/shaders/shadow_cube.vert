#version 450

// shadow_cube.vert — Point light cubemap face vertex shader.

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 frag_world_pos;

layout(push_constant) uniform CubeShadowPC {
    mat4 light_space_matrix;
    mat4 model;
    vec4 light_pos_range; // xyz = light pos, w = range
} pc;

void main() {
    vec4 world_pos = pc.model * vec4(in_position, 1.0);
    frag_world_pos = world_pos.xyz;
    gl_Position    = pc.light_space_matrix * world_pos;
}
