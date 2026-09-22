#ifndef GFX_AO_COMPOSITE_GLSL
#define GFX_AO_COMPOSITE_GLSL

// Ambient-occlusion compositing helpers shared by the deferred lighting pass, the SSR
// composite (whose indirect-specular subtraction must cancel lighting's term exactly) and
// the forward surface captures. Both functions follow Unity HDRP's AO application
// (core RP AmbientOcclusion.hlsl), so a scene tuned against HDRP's look transfers.

/// Multi-bounce ambient occlusion (Jimenez et al. 2016, sec. 10.2): a cubic fit to
/// path-traced interreflection that tints occluded areas toward the surface albedo
/// instead of flat gray -- bright surfaces bounce light into their own creases.
///
/// @param visibility Scalar AO visibility in [0, 1] (1 = unoccluded).
/// @param albedo     Surface diffuse color the bounce light picks up.
/// @return Per-channel occlusion factor, >= visibility in every channel.
vec3 gfx_gtao_multi_bounce(float visibility, vec3 albedo) {
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

/// Specular occlusion from ambient occlusion (Lagarde & de Rousiers, "Moving Frostbite to
/// PBR" -- HDRP's GetSpecularOcclusionFromAmbientOcclusion): narrows the occlusion cone
/// for the specular lobe, so smooth surfaces at grazing view keep their reflections while
/// rough, deeply occluded ones lose them.
///
/// @param ndotv     Clamped N.V of the shading point.
/// @param ao        Scalar AO visibility in [0, 1].
/// @param roughness Perceptual roughness in [0, 1].
/// @return Scalar occlusion factor for indirect specular, in [0, 1].
float gfx_specular_occlusion(float ndotv, float ao, float roughness) {
    return clamp(pow(ndotv + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

#endif // GFX_AO_COMPOSITE_GLSL
