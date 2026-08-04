#version 450

// outline.vert — Inverted-hull outline vertex shader.
// Extrudes back-face vertices along world-space normals.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// set 0: camera UBO
layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 view_pos;
} camera;

// Single push constant block (GLSL allows one push_constant block per stage).
layout(push_constant) uniform PC {
    mat4 model;
    mat4 normal_matrix;
    vec4 outline_params; // xyz=color, w=extrusion_width
} pc;

void main() {
    float outline_width = pc.outline_params.w;
    vec3 world_normal   = normalize(mat3(pc.normal_matrix) * in_normal);
    vec4 world_pos      = pc.model * vec4(in_position, 1.0);
    world_pos.xyz      += world_normal * outline_width;

    gl_Position = camera.proj * camera.view * world_pos;
}
