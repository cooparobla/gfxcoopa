#version 450

// Direct-lit forward shading for reflection-probe geometry capture. Pairs
// with the existing pbr.vert (reused unmodified -- it declares only Set 0
// camera + a 128-byte model push block, which this shader's full 160-byte
// push constant is a superset of, exactly like gbuffer.vert already relies
// on).
//
// v1 limitations, deliberate:
//   - No shadow sampling. GiSystem::bake() runs once, early, before the main
//     pipeline's shadow maps have ever been rendered (still UNDEFINED at that
//     point), so this shader declares no shadow samplers at all.
//   - No reflection-cubemap read anywhere in this file. Sampling the probe's
//     own cubemap while it's mid-capture would be a read/write hazard against
//     the very image being written, and would recurse. Indirect specular here
//     uses the same non-recursive sky_gradient() fallback the main shaders
//     already fall back to when no probe exists.

#include "sky.glsl"
#include "ibl.glsl"   // fresnel_schlick_roughness only

layout(location = 0) in vec3 frag_world_pos;
layout(location = 1) in vec3 frag_world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in mat3 frag_TBN;   // matches pbr.vert's outputs; unused here

// Set 0: Camera UBO (probe's per-face view/proj/position)
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Set 1: Light UBO -- byte-identical layout to pbr.frag/deferred_lighting.frag,
// populated directly from the scene by GiSystem (not the main per-frame
// LightData), with dir_shadow_params.z always 0 and no point-light shadow flags.
struct PointLight {
    vec4 position_range;  // xyz = pos, w = range
    vec4 color_intensity; // xyz = color, w = intensity
    vec4 attenuation;     // x=const, y=lin, z=quad, w=cast_shadows (always 0 here)
};

layout(set = 1, binding = 0) uniform LightUBO {
    vec4 dir_direction;
    vec4 dir_color;
    vec4 dir_ambient;
    mat4 dir_light_space_matrix; // unused (no shadows)
    vec4 dir_shadow_params;      // z forced 0 by GiSystem

    uvec4 light_counts;
    PointLight point_lights[16];
} lights;

// Set 2: BRDF LUT (reused from GiSystem's already-built BRDFLUT)
layout(set = 2, binding = 0) uniform sampler2D brdfLUT;

// Push constants: identical 160-byte layout to GBufferPipeline/PbrPipeline.
layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 normal_matrix;
    vec4  albedo;     // xyz = albedo
    float metallic;
    float roughness;
    float ao;
    float flags;
} material;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

// Cook-Torrance BRDF components -- copied verbatim from deferred_lighting.frag.
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / max(denom, 0.000001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3  albedo    = material.albedo.rgb;
    float metallic  = material.metallic;
    float roughness = material.roughness;
    float ao        = material.ao;

    vec3 N = normalize(frag_world_normal);
    vec3 V = normalize(camera.camera_pos - frag_world_pos); // camera_pos == probe position

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);

    // Direct directional light -- unshadowed.
    if (lights.light_counts.x > 0) {
        vec3 L = normalize(-lights.dir_direction.xyz);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float NDF = DistributionGGX(N, H, roughness);
            float G   = GeometrySmith(N, V, L, roughness);
            vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

            vec3 numerator    = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
            vec3 specular     = numerator / denominator;

            vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

            vec3 radiance = lights.dir_color.rgb * lights.dir_direction.w;
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
        }
    }

    // Direct point lights -- unshadowed.
    uint num_points = min(lights.light_counts.y, 16u);
    for (uint i = 0u; i < num_points; ++i) {
        PointLight pl = lights.point_lights[i];
        vec3 light_pos = pl.position_range.xyz;
        float range = pl.position_range.w;

        vec3 frag_to_light = frag_world_pos - light_pos;
        float dist = length(frag_to_light);
        if (dist > range) continue;

        vec3 L = normalize(-frag_to_light);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        float dist2 = dist * dist;
        float factor = clamp(dist / range, 0.0, 1.0);
        float smooth_falloff = clamp(1.0 - factor * factor * factor * factor, 0.0, 1.0);
        smooth_falloff = smooth_falloff * smooth_falloff;
        float attenuation = (1.0 / (4.0 * PI * (dist2 + 1.0))) * smooth_falloff;
        vec3 radiance = pl.color_intensity.rgb * (pl.color_intensity.w * 0.08) * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
        vec3 specular     = numerator / denominator;

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Indirect: the same non-recursive sky fallback the main shaders use when
    // no probe/GI volume exists.
    float NdotV = max(dot(N, V), 0.0);
    vec3  F_ind = fresnel_schlick_roughness(NdotV, F0, roughness);
    vec2  brdf  = texture(brdfLUT, vec2(NdotV, roughness)).rg;

    vec3 indirect_diffuse  = sky_gradient(N);
    vec3 indirect_specular = sky_gradient(reflect(-V, N)) * (F_ind * brdf.x + brdf.y);

    vec3 kD_indirect = (vec3(1.0) - F_ind) * (1.0 - metallic);
    vec3 ambient = (kD_indirect * albedo * indirect_diffuse + indirect_specular) * ao;

    out_color = vec4(ambient + Lo, 1.0);
}
