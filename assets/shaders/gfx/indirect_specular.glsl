#ifndef GFX_INDIRECT_SPECULAR_GLSL
#define GFX_INDIRECT_SPECULAR_GLSL

// gfx/indirect_specular.glsl -- the indirect-specular term, shared verbatim
// by a consumer's lighting pass and its SSR composite.
//
// The SSR composite's delta is a SUBTRACTION of exactly what the lighting
// pass already added (see gfx/ssr_composite_body.glsl). Historically both
// sites open-coded the same F/brdf/sky_specular/env_specular expression and
// a comment begged the next editor to keep them in sync -- any drift left a
// faint reflection-shaped residue. Calling one function from both sites
// makes agreement structural instead of a matter of convention.
//
// Declares no bindings (same rule as ibl.glsl/ssr_common.glsl): every
// resource the hooks need reaches them through the includer's own
// declarations, so two callers may place their descriptor sets at
// different indices and give their resources different names.

// --- Hooks. Define both BEFORE including this file. ---
vec2 hook_env_brdf(float NdotV, float roughness);
vec3 hook_env_specular(vec3 P, vec3 N, vec3 V, float roughness,
                       vec3 F, vec3 sky_specular);

struct GfxIndirectSpecular {
    vec3 F;     // Fresnel-Schlick-roughness
    vec2 brdf;  // split-sum (scale, bias)
    vec3 sky;   // analytic sky term, pre-hook
    vec3 value; // what the lighting pass adds / the composite subtracts
};

GfxIndirectSpecular gfx_indirect_specular(vec3 P, vec3 N, vec3 V, vec3 F0,
                                          float roughness, float sky_intensity)
{
    GfxIndirectSpecular o;
    float NdotV = max(dot(N, V), 0.0);
    o.F    = fresnel_schlick_roughness(NdotV, F0, roughness);
    o.brdf = hook_env_brdf(NdotV, roughness);
    o.sky  = sky_gradient(reflect(-V, N)) * (o.F * o.brdf.x + o.brdf.y) * sky_intensity;
    o.value = hook_env_specular(P, N, V, roughness, o.F, o.sky);
    return o;
}

#endif // GFX_INDIRECT_SPECULAR_GLSL
