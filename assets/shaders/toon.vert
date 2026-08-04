#version 450

// toon.vert — Toon shading vertex shader.
// Transforms vertex position by the MVP matrices.
// Passes world-space normal and world-space position to the fragment shader
// for per-fragment toon lighting calculations.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// set 0: camera UBO
layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 view_pos;
} camera;

// Push constants: per-object model + normal matrix (128 bytes)
layout(push_constant) uniform ModelPC {
    mat4 model;
    mat4 normal_matrix;
} model_pc;

layout(location = 0) out vec3 frag_world_normal;
layout(location = 1) out vec3 frag_world_pos;
layout(location = 2) out vec2 frag_uv;

void main() {
    vec4 world_pos = model_pc.model * vec4(in_position, 1.0);
    gl_Position    = camera.proj * camera.view * world_pos;

    // Transform normal into world space using the normal matrix.
    frag_world_normal = normalize(mat3(model_pc.normal_matrix) * in_normal);
    frag_world_pos    = world_pos.xyz;
    frag_uv           = in_uv;
}
