#ifndef GFX_SSR_COMPOSITE_PC_GLSL
#define GFX_SSR_COMPOSITE_PC_GLSL

// gfx/ssr_composite_pc.glsl -- push-constant block shared by every
// ssr_composite.frag, matching SsrPass::CompositePushConstants (ssr_pass.h)
// field-for-field. A consumer that doesn't use a field (blendy has no SSGI
// bounce term) simply never reads it; the C++ side always pushes the full
// union struct regardless of which shader is bound.
layout(push_constant) uniform CompositePushConstants {
    vec2  ssr_resolution;      // offset 0  -- resolution of u_ssr_map (trace res under half-res)
    vec2  screen_resolution;   // offset 8  -- always full screen res
    int   half_res;            // offset 16
    int   max_color_mip;       // offset 20 -- top mip of the prefiltered scene-colour chain
    float sky_intensity;       // offset 24 -- MUST match the lighting pass's sky_intensity
    float ssgi_intensity;      // offset 28 -- 0 disables the diffuse-bounce term
    float ssgi_distance;       // offset 32 -- world-space offset along N for the bounce sample
} pc;

#endif // GFX_SSR_COMPOSITE_PC_GLSL
