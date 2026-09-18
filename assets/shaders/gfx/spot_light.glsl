#ifndef GFX_SPOT_LIGHT_GLSL
#define GFX_SPOT_LIGHT_GLSL

// gfx/spot_light.glsl -- shared SpotLight struct and cone-attenuation term.
//
// Holds the two things every LightUBO consumer's spot loop must byte-match:
// the struct itself (see light_data.h's SpotLightGPU for the C++ side of this
// std140 layout) and the cone falloff, gfx_spot_cone(). Deliberately does NOT
// include the distance falloff a caller multiplies gfx_spot_cone() by --
// toyengine's point-light loops normalize distance by range (a
// `factor = dist/range` curve) while gfxcoopa's own point-light loops use raw
// world-space dist^2; a shared distance helper here would change one of those
// two point-light looks. Each spot loop instead reuses whichever distance
// curve its own file's point-light loop already uses immediately above it, so
// a spot light always falls off consistently with the point lights in the
// same pass -- see pixel_forward_shading.glsl's spot loop for the toyengine
// curve, deferred_lighting.frag's for gfxcoopa's.

struct SpotLight {
    vec4 position_range;   // xyz = world position, w = range
    vec4 direction_cone;   // xyz = normalized aim (direction rays travel), w = cos(outer half-angle)
    vec4 color_intensity;  // xyz = RGB color, w = intensity
    vec4 params;           // x = falloff sharpness, y = cos(inner half-angle), z = cast_shadows (1 or 0), w = reserved
};

// Cone attenuation, 0 outside the outer angle ramping to 1 inside the inner angle.
// `L` is the surface-to-light unit vector (frag_to_light / dist, same convention every
// point-light loop already uses here), so `-L` is the direction the light's own rays
// travel, which is what compares against the light's aim (direction_cone.xyz).
//
// Squared (not linear) so the ramp reads as a soft-but-bounded ring under this engine's
// banded/cel shading (band() in pixel_lighting.frag) rather than a visible linear gradient
// competing with the discrete N.L steps.
float gfx_spot_cone(vec3 L, vec3 spot_dir, float cos_outer, float cos_inner) {
    float cd = dot(-L, spot_dir);
    float t  = clamp((cd - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
    return t * t;
}

#endif // GFX_SPOT_LIGHT_GLSL
