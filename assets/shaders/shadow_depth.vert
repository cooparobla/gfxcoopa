#version 450

// shadow_depth.vert — Directional light depth map rendering.
// Transforms vertex positions into directional light space.

layout(location = 0) in vec3 in_position;

// Push constants: light_space_matrix (64 bytes) + model (64 bytes) = 128 bytes.
layout(push_constant) uniform ShadowPC {
    mat4 light_space_matrix;
    mat4 model;
} pc;

void main() {
    gl_Position = pc.light_space_matrix * pc.model * vec4(in_position, 1.0);
}
