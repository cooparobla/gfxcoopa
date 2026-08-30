#version 450

#include <gfx/sky.glsl>
#include <gfx/ibl.glsl>
#include <gfx/brdf.glsl>
#include <gfx/shadow_sampling.glsl>
#include <gfx/indirect_specular.glsl>

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

// Set 2: Shadow maps -- *Shadow sampler types: hardware compareEnable
// (util::Sampler::shadow()), see gfx/shadow_sampling.glsl's *Shadow-family doc.
layout(set = 2, binding = 0) uniform sampler2DShadow dir_shadow_map;
layout(set = 2, binding = 1) uniform samplerCubeShadow point_shadow_map_0;
layout(set = 2, binding = 2) uniform samplerCubeShadow point_shadow_map_1;
layout(set = 2, binding = 3) uniform samplerCubeShadow point_shadow_map_2;
layout(set = 2, binding = 4) uniform samplerCubeShadow point_shadow_map_3;

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

layout(set = 3, binding = 2) uniform sampler2D u_brdf_lut;

// Bindings 3-6: up to MAX_REFLECTION_PROBES reflection probe cubemaps.
// Separately-named (not a samplerCube[] array): this device does not enable
// shaderSampledImageArrayDynamicIndexing, so a dynamically-indexed sampler
// array would be illegal here -- mirrors point_shadow_map_0..3 above.
layout(set = 3, binding = 3) uniform samplerCube u_reflection_map_0;
layout(set = 3, binding = 4) uniform samplerCube u_reflection_map_1;
layout(set = 3, binding = 5) uniform samplerCube u_reflection_map_2;
layout(set = 3, binding = 6) uniform samplerCube u_reflection_map_3;

layout(set = 3, binding = 7) uniform ReflectionProbeUBO {
    ReflectionProbeData probes[MAX_REFLECTION_PROBES];
} reflection;

// Set 4: G-Buffer textures
layout(set = 4, binding = 0) uniform sampler2D g_albedo_ao;          // RGB = Albedo, A = AO
layout(set = 4, binding = 1) uniform sampler2D g_normal_metallic;    // RGB = Normal, A = Metallic
layout(set = 4, binding = 2) uniform sampler2D g_position_roughness; // RGB = World Pos, A = Roughness
layout(set = 4, binding = 3) uniform sampler2D g_ssao;                // R = screen-space AO (SsaoPass output)
layout(set = 4, binding = 4) uniform sampler2D g_emissive;            // RGB = emissive radiance (HDR)

layout(location = 0) out vec4 out_color;

#include "indirect_hooks.glsl"

// Directional shadow: rotated-Vogel-disk PCF, hardware depth-compare taps (kernel in
// gfx/shadow_sampling.glsl). IGN, not gfx_random_angle: this pass has no temporal resolve
// downstream (no TAA in the base library -- a real consumer's own frame counter can be
// folded into the angle the same way blendy's deferred_lighting.frag does, via
// shadow_frame_offset, once it has one), so the rotation itself must not read as grain.
float calc_dir_shadow(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
    if (lights.dir_shadow_params.z < 0.5) return 0.0;

    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }

    float NdotL = max(dot(normal, light_dir), 0.0);
    float bias = max(lights.dir_shadow_params.x * (1.0 - NdotL), 0.0002);

    float angle = gfx_ign_angle(gl_FragCoord.xy);
    vec2 texel_size = lights.dir_shadow_params.y / textureSize(dir_shadow_map, 0);
    return gfx_shadow_dir_pcf_vogel(dir_shadow_map, proj_coords, bias, texel_size, angle, 24);
}

// Point shadow: 8-tap Rodrigues-rotated cube-corner PCF, hardware depth-compare taps (kernel
// in gfx/shadow_sampling.glsl). point_shadow_map_0..3 are separately-named bindings, not a
// dynamically-indexed array (see the binding comment above), so the sampler selection stays
// a compile-time-constant if/else chain here.
float sample_point_shadow_map(int index, vec3 frag_to_light, float range) {
    vec3 dir = normalize(frag_to_light);
    float current_dist = length(frag_to_light) / range;
    float bias = 0.05 / range;
    // disk_radius is an angular spread applied to a unit direction, so a fixed constant makes
    // the world-space penumbra grow with distance from the light; dividing by current_dist
    // (light-to-surface distance normalized by range) holds it roughly constant instead.
    float disk_radius = 0.03 / max(current_dist, 0.15);
    float angle = gfx_ign_angle(gl_FragCoord.xy);

    if (index == 0) return gfx_shadow_cube_pcf(point_shadow_map_0, dir, current_dist, bias, disk_radius, angle);
    else if (index == 1) return gfx_shadow_cube_pcf(point_shadow_map_1, dir, current_dist, bias, disk_radius, angle);
    else if (index == 2) return gfx_shadow_cube_pcf(point_shadow_map_2, dir, current_dist, bias, disk_radius, angle);
    else if (index == 3) return gfx_shadow_cube_pcf(point_shadow_map_3, dir, current_dist, bias, disk_radius, angle);
    return 0.0;
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
    // 1.0 (fully unoccluded) whenever SSAO is disabled -- DeferredLightingPass::set_ssao_image()
    // binds SsaoPass::neutral_view() in that case, so this read needs no separate enabled flag.
    float ssao = texture(g_ssao, in_uv).r;
    vec3 emissive = texture(g_emissive, in_uv).rgb;

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
            float NDF = distribution_ggx(N, H, roughness);
            float G   = geometry_smith(N, V, L, roughness);
            vec3 F    = fresnel_schlick(max(dot(H, V), 0.0), F0);

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
            Lo += (kD * albedo / BRDF_PI + specular) * radiance * NdotL * (1.0 - shadow);
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
        // pl.attenuation.x -- formerly an unused classical "constant attenuation" term -- is
        // repurposed as a per-light falloff sharpness exponent: ~1 (the component's own default,
        // used when a scene doesn't set this) gives a gradual, realistic fade to the light's
        // range; ~4-8 gives a crisper, more cel-shaded-style cutoff (4 reproduces this pass's
        // original hardcoded curve exactly).
        float sharpness = max(pl.attenuation.x, 0.1);
        float factor = clamp(dist / range, 0.0, 1.0);
        float smooth_falloff = clamp(1.0 - pow(factor, sharpness), 0.0, 1.0);
        smooth_falloff = smooth_falloff * smooth_falloff;
        float attenuation = (1.0 / (4.0 * BRDF_PI * (dist2 + 1.0))) * smooth_falloff;
        vec3 radiance = pl.color_intensity.rgb * (pl.color_intensity.w * 0.08) * attenuation;

        float shadow = 0.0;
        if (pl.attenuation.w > 0.5) {
            if (shadow_casting_point_idx < 4) {
                shadow = sample_point_shadow_map(shadow_casting_point_idx, frag_to_light, range);
            }
            shadow_casting_point_idx++;
        }

        float NDF = distribution_ggx(N, H, roughness);
        float G   = geometry_smith(N, V, L, roughness);
        vec3 F    = fresnel_schlick(max(dot(H, V), 0.0), F0);

        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
        vec3 specular     = numerator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        Lo += (kD * albedo / BRDF_PI + specular) * radiance * NdotL * (1.0 - shadow);
    }

    // --- Indirect Lighting (GI) ---
    vec3 indirect_diffuse = vec3(0.0);

    if (gi.grid_counts.w > 0) {
        indirect_diffuse = sample_gi_probes(frag_world_pos, N) * gi.gi_params.x;
    } else {
        // No GI volume baked: fall back to sky-lit diffuse irradiance.
        indirect_diffuse = sky_gradient(N);
    }

    // The one call that guarantees this cancels against ssr_composite.frag's subtraction --
    // see gfx/indirect_specular.glsl.
    GfxIndirectSpecular ind = gfx_indirect_specular(frag_world_pos, N, V, F0, roughness, 1.0);

    vec3 kD_indirect = (vec3(1.0) - ind.F) * (1.0 - metallic);
    vec3 ambient = (kD_indirect * albedo * indirect_diffuse + ind.value) * ao * ssao;

    // emissive is added last, after ambient's * ao * ssao -- an emissive surface glows
    // even in a fully occluded/dark crevice, unlike the lit terms above it.
    vec3 color = ambient + Lo + emissive;
    out_color = vec4(color, 1.0);
}
