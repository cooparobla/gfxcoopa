#version 450

#include "sky.glsl"
#include "ibl.glsl"
#include "ssr_common.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

// Set 0: Camera UBO
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Set 1: G-Buffer
layout(set = 1, binding = 0) uniform sampler2D g_albedo_ao;
layout(set = 1, binding = 1) uniform sampler2D g_normal_metallic;
layout(set = 1, binding = 2) uniform sampler2D g_position_roughness;

// Set 2: Resolved SSR Map (trace resolution -- full res, or half under ssr_half_res)
layout(set = 2, binding = 0) uniform sampler2D u_ssr_map;

// Set 3: Deferred Lit Scene Color
layout(set = 3, binding = 0) uniform sampler2D u_scene_color;

// Set 4: GI Data & Reflection Probe
layout(set = 4, binding = 0) uniform GiUniforms {
    vec4  grid_origin;
    vec4  grid_spacing;
    ivec4 grid_counts;
    vec4  gi_params; // x = gi_intensity, y = refl_intensity, z = max_mip, w = num_refl
} gi;

layout(set = 4, binding = 2) uniform sampler2D u_brdf_lut;

// Bindings 3-6: up to MAX_REFLECTION_PROBES reflection probe cubemaps.
// Separately-named (not a samplerCube[] array): this device does not enable
// shaderSampledImageArrayDynamicIndexing, so a dynamically-indexed sampler
// array would be illegal here -- mirrors deferred_lighting.frag/pbr.frag.
layout(set = 4, binding = 3) uniform samplerCube u_reflection_map_0;
layout(set = 4, binding = 4) uniform samplerCube u_reflection_map_1;
layout(set = 4, binding = 5) uniform samplerCube u_reflection_map_2;
layout(set = 4, binding = 6) uniform samplerCube u_reflection_map_3;

layout(set = 4, binding = 7) uniform ReflectionProbeUBO {
    ReflectionProbeData probes[MAX_REFLECTION_PROBES];
} reflection;

layout(push_constant) uniform CompositePushConstants {
    vec2 ssr_resolution;      // resolution of u_ssr_map (trace res under half-res)
    vec2 screen_resolution;   // always full screen res
    int  half_res;
} pc;

/// Depth/normal-aware upsample of the (possibly half-resolution) resolved SSR buffer.
///
/// A plain bilinear tap would bleed a reflection across every silhouette in the scene, which is
/// exactly where a half-res trace has no data. At full resolution this degenerates to a single
/// texel fetch, so both paths share one shader.
vec4 ssr_fetch(vec2 uv, vec3 P, vec3 N, float px_world) {
    if (pc.half_res == 0) {
        return texture(u_ssr_map, uv);
    }

    // Each half-res texel hc was traced from full-res G-buffer texel 2*hc + 1: that is what
    // ssr.frag's origin_px snapping resolves to when the half-res texel centre
    // (hc + 0.5) / half_size is scaled by the full-res size. So each tap's surface can be
    // looked up exactly rather than guessed.
    vec2  hs   = pc.ssr_resolution;
    vec2  f    = uv * hs - 0.5;
    ivec2 base = ivec2(floor(f));
    vec2  frac = f - vec2(base);

    vec4  sum  = vec4(0.0);
    float wsum = 0.0;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            ivec2 hc = clamp(base + ivec2(dx, dy), ivec2(0), ivec2(hs) - 1);
            ivec2 fc = clamp(hc * 2 + 1, ivec2(0), ivec2(pc.screen_resolution) - 1);

            vec3 Pt = texelFetch(g_position_roughness, fc, 0).rgb;
            vec3 Nt = texelFetch(g_normal_metallic,    fc, 0).rgb;

            float bw = (dx == 0 ? 1.0 - frac.x : frac.x) * (dy == 0 ? 1.0 - frac.y : frac.y);

            // Plane distance rather than raw position distance, so a tap sliding along this
            // pixel's own surface is not penalised. Normalised by a texel's world size, which
            // makes the falloff scale-free (same reasoning as ssr.frag's depth-scaled bias).
            float dw = exp(-abs(dot(Pt - P, N)) / max(2.0 * px_world, 1e-6));
            float nw = pow(max(dot(normalize(Nt + vec3(1e-6)), N), 0.0), 8.0);

            float w = bw * dw * nw;
            sum  += texelFetch(u_ssr_map, hc, 0) * w;
            wsum += w;
        }
    }

    // Every tap rejected (a lone pixel of thin geometry, where no half-res sample shares this
    // surface): fall back to plain bilinear rather than to black, which would punch a hole in
    // the reflection instead of merely softening it.
    return (wsum > 1e-5) ? sum / wsum : texture(u_ssr_map, uv);
}

void main() {
    vec3 scene_color = texture(u_scene_color, in_uv).rgb;

    ivec2 gcoord = ivec2(gl_FragCoord.xy);
    vec4 norm_met = texelFetch(g_normal_metallic, gcoord, 0);
    vec3 N = norm_met.rgb;
    if (dot(N, N) < 0.001) {
        out_color = vec4(scene_color, 1.0);
        return;
    }
    N = normalize(N);
    float metallic = norm_met.a;

    vec4 albedo_ao = texelFetch(g_albedo_ao, gcoord, 0);
    vec3 albedo = albedo_ao.rgb;
    float ao    = albedo_ao.a;

    vec4 pos_rough = texelFetch(g_position_roughness, gcoord, 0);
    vec3 P = pos_rough.rgb;
    float roughness = pos_rough.a;

    float view_z   = (camera.view * vec4(P, 1.0)).z;
    float px_world = ssr_texel_world_size(view_z, abs(camera.proj[1][1]), pc.screen_resolution.y);

    vec4 ssr_sample = ssr_fetch(in_uv, P, N, px_world);
    vec3 ssr_color = ssr_sample.rgb;
    float confidence = ssr_sample.a;

    vec3 V = normalize(camera.camera_pos - P);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    // Must be the identical expression deferred_lighting.frag uses for
    // F_indirect, or the subtraction below does not cancel.
    vec3 F = fresnel_schlick_roughness(NdotV, F0, roughness);

    vec2 brdf = texture(u_brdf_lut, vec2(NdotV, roughness)).rg;

    // scene_color already contains deferred_lighting.frag's
    //     indirect_specular * ao
    // Reconstruct exactly that, then swap it for the SSR term in proportion to
    // confidence. Both terms are indirect specular under the same occlusion,
    // so ao is applied to the whole delta below rather than to env_specular
    // alone.
    vec3 sky_specular = sky_gradient(reflect(-V, N)) * (F * brdf.x + brdf.y);
    vec3 env_specular = ibl_specular_probes_blended(
        u_reflection_map_0, u_reflection_map_1, u_reflection_map_2, u_reflection_map_3, u_brdf_lut,
        P, N, V, roughness, F,
        reflection.probes, int(gi.gi_params.w), sky_specular);

    // u_ssr_map.rgb is already premultiplied by confidence (ssr.frag), so this delta is
    // algebraically identical to the previous confidence * (ssr - env), just with the weight
    // folded into the buffer where the temporal filter can average it correctly.
    vec3 ssr_specular = ssr_color * (F * brdf.x + brdf.y);
    // Clamp: a false-positive SSR hit against nearby dark geometry can leave ssr_specular
    // near zero, making the unclamped delta go negative -- a visible black speckle rather
    // than "no reflection here".
    vec3 final_color = max(scene_color + (ssr_specular - confidence * env_specular) * ao, 0.0);
    out_color = vec4(final_color, 1.0);
}
