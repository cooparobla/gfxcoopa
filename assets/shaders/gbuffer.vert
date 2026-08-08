#version 450

// Inputs matching Vertex struct (location 0: pos, 1: norm, 2: uv, 3: tangent)
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_tangent; // xyz = tangent, w = handedness

// Set 0: Camera UBO
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Push constants: ModelPushConstants (128 bytes)
layout(push_constant) uniform ModelPushConstants {
    mat4 model;
    mat4 normal_matrix;
} model_push;

// Outputs to G-Buffer fragment shader
layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_world_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out mat3 frag_TBN;

void main() {
    vec4 world_pos = model_push.model * vec4(in_position, 1.0);
    frag_world_pos = world_pos.xyz;

    mat3 norm_mat = mat3(model_push.normal_matrix);
    vec3 N = normalize(norm_mat * in_normal);
    vec3 T = normalize(norm_mat * in_tangent.xyz);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * in_tangent.w;

    frag_world_normal = N;
    frag_uv = in_uv;
    frag_TBN = mat3(T, B, N);

    gl_Position = camera.proj * camera.view * world_pos;
}
