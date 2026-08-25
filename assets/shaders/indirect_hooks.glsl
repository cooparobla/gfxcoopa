// indirect_hooks.glsl -- default hooks for gfx/indirect_specular.glsl:
// real BRDF LUT, up to 4 blended reflection probes. Shared by this base
// library's deferred_lighting.frag and ssr_composite.frag.
//
// Requires, already declared by the includer with these exact names:
//   sampler2D   u_brdf_lut
//   samplerCube u_reflection_map_0..3
//   ReflectionProbeUBO { ReflectionProbeData probes[]; } reflection
//   GiUniforms  { ... vec4 gi_params; } gi
// The set INDEX differs between the two includers -- only the names are
// fixed.

vec2 hook_env_brdf(float NdotV, float roughness) {
    return texture(u_brdf_lut, vec2(NdotV, roughness)).rg;
}

vec3 hook_env_specular(vec3 P, vec3 N, vec3 V, float roughness,
                       vec3 F, vec3 sky_specular) {
    return ibl_specular_probes_blended(
        u_reflection_map_0, u_reflection_map_1, u_reflection_map_2, u_reflection_map_3, u_brdf_lut,
        P, N, V, roughness, F,
        reflection.probes, int(gi.gi_params.w), sky_specular);
}
