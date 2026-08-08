#version 450

#include "sky.glsl"
#include "ibl.glsl"

layout(location = 0) in vec2 in_uv;

// Set 0: Camera UBO
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Set 1: Light UBO
struct PointLight {
    vec4 position_range;  // xyz = pos, w = range
    vec4 color_intensity; // xyz = color, w = intensity
    vec4 attenuation;     // x=const, y=lin, z=quad, w=cast_shadows (1 or 0)
};

layout(set = 1, binding = 0) uniform LightUBO {
    vec4 dir_direction;
    vec4 dir_color;
    vec4 dir_ambient;
    mat4 dir_light_space_matrix;
    vec4 dir_shadow_params; // x=bias, y=pcf_samples, z=shadow_enabled

    uvec4 light_counts; // x=num_dir, y=num_point
    PointLight point_lights[16];
} lights;

// Set 2: Shadow maps
layout(set = 2, binding = 0) uniform sampler2D dir_shadow_map;
layout(set = 2, binding = 1) uniform samplerCube point_shadow_map_0;
layout(set = 2, binding = 2) uniform samplerCube point_shadow_map_1;
layout(set = 2, binding = 3) uniform samplerCube point_shadow_map_2;
layout(set = 2, binding = 4) uniform samplerCube point_shadow_map_3;

// Set 3: GI Data
layout(set = 3, binding = 0) uniform GiUniforms {
    vec4  grid_origin;     // xyz = world origin of probe grid
    vec4  grid_spacing;    // xyz = cell size per axis
    ivec4 grid_counts;     // xyz = probe counts, w = total
    vec4  gi_params;       // x = gi_intensity, y = refl_intensity, z = max_mip, w = num_refl
} gi;

struct SHProbe {
    vec4 bands[9];         // Each .xyz = RGB SH coefficient
};

layout(std430, set = 3, binding = 1) readonly buffer GiProbeBuffer {
    SHProbe probes[];
} gi_probes;

layout(set = 3, binding = 2) uniform sampler2D brdfLUT;

// Bindings 3-6: up to MAX_REFLECTION_PROBES reflection probe cubemaps.
// Separately-named (not a samplerCube[] array): this device does not enable
// shaderSampledImageArrayDynamicIndexing, so a dynamically-indexed sampler
// array would be illegal here -- mirrors point_shadow_map_0..3 above.
layout(set = 3, binding = 3) uniform samplerCube reflectionMap_0;
layout(set = 3, binding = 4) uniform samplerCube reflectionMap_1;
layout(set = 3, binding = 5) uniform samplerCube reflectionMap_2;
layout(set = 3, binding = 6) uniform samplerCube reflectionMap_3;

layout(set = 3, binding = 7) uniform ReflectionProbeUBO {
    ReflectionProbeData probes[MAX_REFLECTION_PROBES];
} reflection;

// Set 4: G-Buffer textures
layout(set = 4, binding = 0) uniform sampler2D g_albedo_ao;          // RGB = Albedo, A = AO
layout(set = 4, binding = 1) uniform sampler2D g_normal_metallic;    // RGB = Normal, A = Metallic
layout(set = 4, binding = 2) uniform sampler2D g_position_roughness; // RGB = World Pos, A = Roughness

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

// Cook-Torrance BRDF components
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

const vec2 POISSON_DISK_16[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648370),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090160),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

float random_angle(vec2 seed) {
    vec4 floor_seed = vec4(seed, 0.0, 0.0);
    return fract(sin(dot(floor_seed.xy, vec2(12.9898, 78.233))) * 43758.5453) * 6.28318530718;
}

float calc_dir_shadow(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
    if (lights.dir_shadow_params.z < 0.5) return 0.0;

    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }

    float current_depth = proj_coords.z;
    float NdotL = max(dot(normal, light_dir), 0.0);
    float bias = max(lights.dir_shadow_params.x * (1.0 - NdotL), 0.0002);

    float angle = random_angle(gl_FragCoord.xy);
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    mat2 rot = mat2(cos_a, -sin_a, sin_a, cos_a);

    vec2 texel_size = 1.75 / textureSize(dir_shadow_map, 0);

    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 offset = rot * POISSON_DISK_16[i] * texel_size;
        float pcf_depth = texture(dir_shadow_map, proj_coords.xy + offset).r;
        shadow += (current_depth - bias > pcf_depth) ? 1.0 : 0.0;
    }
    return shadow / 16.0;
}

float sample_point_shadow_map(int index, vec3 frag_to_light, float range) {
    float shadow = 0.0;
    vec3 dir = normalize(frag_to_light);
    float current_dist = length(frag_to_light) / range;
    float bias = 0.05 / range;

    float sampled_depth = 1.0;
    if (index == 0) sampled_depth = texture(point_shadow_map_0, dir).r;
    else if (index == 1) sampled_depth = texture(point_shadow_map_1, dir).r;
    else if (index == 2) sampled_depth = texture(point_shadow_map_2, dir).r;
    else if (index == 3) sampled_depth = texture(point_shadow_map_3, dir).r;

    if (current_dist - bias > sampled_depth) {
        shadow = 1.0;
    }
    return shadow;
}

// --- SH Irradiance Evaluation ---

void sh_basis(vec3 d, out float basis[9]) {
    basis[0] = 0.282094791;
    basis[1] = 0.488602512 * d.y;
    basis[2] = 0.488602512 * d.z;
    basis[3] = 0.488602512 * d.x;
    basis[4] = 1.092548431 * d.x * d.y;
    basis[5] = 1.092548431 * d.y * d.z;
    basis[6] = 0.315391565 * (3.0 * d.z*d.z - 1.0);
    basis[7] = 1.092548431 * d.x * d.z;
    basis[8] = 0.546274215 * (d.x*d.x - d.y*d.y);
}

vec3 evaluate_sh_irradiance(int probeIdx, vec3 N) {
    float basis[9];
    sh_basis(N, basis);
    const float A[9] = float[9](
        3.14159265, 2.09439510, 2.09439510, 2.09439510,
        0.78539816, 0.78539816, 0.78539816, 0.78539816, 0.78539816
    );
    vec3 irr = vec3(0.0);
    for (int i = 0; i < 9; ++i) {
        irr += gi_probes.probes[probeIdx].bands[i].xyz * basis[i] * A[i];
    }
    return max(irr, vec3(0.0));
}

vec3 sample_gi_probes(vec3 P, vec3 N) {
    if (gi.grid_counts.w == 0) return vec3(0.0);

    vec3 local = (P - gi.grid_origin.xyz) / gi.grid_spacing.xyz;
    vec3 clamped = clamp(local, vec3(0.0), vec3(gi.grid_counts.xyz - 1));

    ivec3 lo = ivec3(floor(clamped));
    ivec3 hi = min(lo + 1, gi.grid_counts.xyz - 1);
    vec3 frac = clamped - vec3(lo);

    int Nx = gi.grid_counts.x;
    int Ny = gi.grid_counts.y;

    vec3 c000 = evaluate_sh_irradiance(lo.z*Ny*Nx + lo.y*Nx + lo.x, N);
    vec3 c100 = evaluate_sh_irradiance(lo.z*Ny*Nx + lo.y*Nx + hi.x, N);
    vec3 c010 = evaluate_sh_irradiance(lo.z*Ny*Nx + hi.y*Nx + lo.x, N);
    vec3 c110 = evaluate_sh_irradiance(lo.z*Ny*Nx + hi.y*Nx + hi.x, N);
    vec3 c001 = evaluate_sh_irradiance(hi.z*Ny*Nx + lo.y*Nx + lo.x, N);
    vec3 c101 = evaluate_sh_irradiance(hi.z*Ny*Nx + lo.y*Nx + hi.x, N);
    vec3 c011 = evaluate_sh_irradiance(hi.z*Ny*Nx + hi.y*Nx + lo.x, N);
    vec3 c111 = evaluate_sh_irradiance(hi.z*Ny*Nx + hi.y*Nx + hi.x, N);

    vec3 c00 = mix(c000, c100, frac.x);
    vec3 c10 = mix(c010, c110, frac.x);
    vec3 c01 = mix(c001, c101, frac.x);
    vec3 c11 = mix(c011, c111, frac.x);

    vec3 c0 = mix(c00, c10, frac.y);
    vec3 c1 = mix(c01, c11, frac.y);

    return mix(c0, c1, frac.z);
}

void main() {
    vec4 g0 = texture(g_albedo_ao, in_uv);
    vec4 g1 = texture(g_normal_metallic, in_uv);
    vec4 g2 = texture(g_position_roughness, in_uv);

    vec3 N = g1.rgb;
    if (length(N) < 0.001) {
        out_color = vec4(0.05, 0.05, 0.05, 1.0);
        return;
    }
    N = normalize(N);

    vec3 albedo   = g0.rgb;
    float ao      = g0.a;
    float metallic  = g1.a;
    vec3 frag_world_pos = g2.rgb;
    float roughness = g2.a;

    vec3 V = normalize(camera.camera_pos - frag_world_pos);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);

    // Direct Directional Light contribution
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

            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - metallic;

            float normal_bias_scale = clamp(1.0 - dot(N, L), 0.0, 1.0);
            vec3 biased_world_pos = frag_world_pos + N * (lights.dir_shadow_params.w * (0.5 + 0.5 * normal_bias_scale));
            vec4 light_space_pos = lights.dir_light_space_matrix * vec4(biased_world_pos, 1.0);
            float shadow = calc_dir_shadow(light_space_pos, N, L);

            vec3 radiance = lights.dir_color.rgb * lights.dir_direction.w;
            Lo += (kD * albedo / PI + specular) * radiance * NdotL * (1.0 - shadow);
        }
    }

    // Direct Point Lights contribution
    uint num_points = min(lights.light_counts.y, 16u);
    int shadow_casting_point_idx = 0;

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

        float shadow = 0.0;
        if (pl.attenuation.w > 0.5) {
            if (shadow_casting_point_idx < 4) {
                shadow = sample_point_shadow_map(shadow_casting_point_idx, frag_to_light, range);
            }
            shadow_casting_point_idx++;
        }

        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
        vec3 specular     = numerator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        Lo += (kD * albedo / PI + specular) * radiance * NdotL * (1.0 - shadow);
    }

    // --- Indirect Lighting (GI) ---
    vec3 indirect_diffuse = vec3(0.0);
    vec3 indirect_specular = vec3(0.0);
    float NdotV_indirect = max(dot(N, V), 0.0);

    if (gi.grid_counts.w > 0) {
        indirect_diffuse = sample_gi_probes(frag_world_pos, N) * gi.gi_params.x;
    } else {
        // No GI volume baked: fall back to sky-lit diffuse irradiance.
        indirect_diffuse = sky_gradient(N);
    }

    // One Fresnel value, shared by the diffuse split, the probe branch and the
    // sky fallback -- ssr_composite.frag reproduces this exact expression so
    // its env_specular subtraction cancels what this pass added.
    vec3 F_indirect = fresnel_schlick_roughness(NdotV_indirect, F0, roughness);
    vec2 brdf_indirect = texture(brdfLUT, vec2(NdotV_indirect, roughness)).rg;

    // Sky is always the fallback/base term now -- ibl_specular_probes_blended
    // crossfades smoothly into it wherever probe coverage is partial or absent,
    // so metallic surfaces are never left with zero indirect specular.
    vec3 R_indirect = reflect(-V, N);
    vec3 sky_specular = sky_gradient(R_indirect) * (F_indirect * brdf_indirect.x + brdf_indirect.y);

    indirect_specular = ibl_specular_probes_blended(
        reflectionMap_0, reflectionMap_1, reflectionMap_2, reflectionMap_3, brdfLUT,
        frag_world_pos, N, V, roughness, F_indirect,
        reflection.probes, int(gi.gi_params.w), sky_specular);

    vec3 kD_indirect = (vec3(1.0) - F_indirect) * (1.0 - metallic);
    vec3 ambient = (kD_indirect * albedo * indirect_diffuse + indirect_specular) * ao;

    vec3 color = ambient + Lo;
    out_color = vec4(color, 1.0);
}
