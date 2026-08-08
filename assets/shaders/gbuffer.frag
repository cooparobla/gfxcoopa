#version 450

layout(location = 0) in vec3 frag_world_pos;
layout(location = 1) in vec3 frag_world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in mat3 frag_TBN;

// Push constants: Model (128) + Material (32) = 160 bytes total
layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 normal_matrix;
    vec4  albedo;     // xyz = albedo
    float metallic;
    float roughness;
    float ao;
    float flags;
} material;

// G-Buffer Render Targets
layout(location = 0) out vec4 out_albedo_ao;          // RGB = Albedo, A = AO
layout(location = 1) out vec4 out_normal_metallic;    // RGB = World Normal, A = Metallic
layout(location = 2) out vec4 out_position_roughness; // RGB = World Pos, A = Roughness

void main() {
    vec3 albedo   = material.albedo.rgb;
    float metallic  = material.metallic;
    float roughness = max(material.roughness, 0.045);
    float ao        = material.ao;

    vec3 N = normalize(frag_world_normal);

    out_albedo_ao          = vec4(albedo, ao);
    out_normal_metallic    = vec4(N, metallic);
    out_position_roughness = vec4(frag_world_pos, roughness);
}
