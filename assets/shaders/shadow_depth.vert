#version 450

// shadow_depth.vert — Directional light depth map rendering.
// Transforms vertex positions into directional light space.

layout(location = 0) in vec3 in_position;
layout(location = 4) in mat4 in_model; // per-instance (locations 4-7)

// Push constants: light_space_matrix (64 bytes) + alpha (4 bytes). model
// moved to the per-instance stream above. alpha is unused here -- it drives
// shadow_depth.frag's stochastic discard -- but must be declared to match
// that stage's block byte-for-byte (shared VkPushConstantRange).
layout(push_constant) uniform ShadowPC {
    mat4 light_space_matrix;
    float alpha;
} pc;

void main() {
    gl_Position = pc.light_space_matrix * in_model * vec4(in_position, 1.0);
}
