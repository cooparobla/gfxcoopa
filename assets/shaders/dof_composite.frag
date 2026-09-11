#version 450

// Depth-of-field stage 3/3: composite dof_bokeh.frag's half-res gather back over
// the full-res sharp image. Recomputes CoC from FULL-res depth here rather than
// upsampling dof_coc.frag's half-res CoC channel -- sharper at the exact silhouette
// a half-res value would have already lost half a pixel of precision on, and this
// shader is cheap enough (one more dof_signed_coc() call) that reusing the coarser
// value would only trade accuracy for a saving that isn't needed.
//
// Also doubles as the CoC debug view (pc.bokeh.w >= 0.5), in the same spirit as
// ssao_debug_view -- but as a branch in this shader rather than a separate pass
// like GBufferVisualizePass/ssao_debug.frag: SSAO's debug view REPLACES the
// lighting draw it can't otherwise inspect standalone, whereas this one only
// annotates a value (signed CoC) this pass was already computing for the blend.

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color; // full-res, linear HDR, NEAREST
layout(set = 0, binding = 1) uniform sampler2D u_bokeh;     // dof_bokeh.frag's output, half-res, LINEAR
layout(set = 0, binding = 2) uniform sampler2D scene_depth; // full-res gbuffer depth, NEAREST

#include <gfx/dof_common.glsl>

void main() {
    float d = texture(scene_depth, in_uv).r;
    float view_depth = gfx_linear_depth(d, pc.camera.x, pc.camera.y, pc.camera.z);
    float coc = dof_signed_coc(view_depth);

    if (pc.bokeh.w >= 0.5) {
        // Red = near field (in front of focus), blue = far field, black = in
        // focus, brightness = |CoC| / max_radius.
        float t = clamp(abs(coc) / max(pc.camera.w, 1e-4), 0.0, 1.0);
        out_color = coc < 0.0 ? vec4(t, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, t, 1.0);
        return;
    }

    vec3 sharp = texture(scene_color, in_uv).rgb;

    // |coc| < 1 FULL-res px stays bit-exact sharp -- matches dof_bokeh.frag's own
    // early-out (`abs(coc) < 0.5` there, since that pass's CoC is in HALF-res
    // px), so the fully-sharp region reported there and the region that skips
    // blending here agree exactly, with no seam between them. These two
    // thresholds must be kept in the same units by construction (1 == 2*0.5) --
    // they previously read as the same literal `1.0` in both files despite one
    // being half-res and the other full-res, which is not the same threshold and
    // produced a ~1px-wide dim ring at every focus-plane silhouette.
    if (abs(coc) < 1.0) {
        out_color = vec4(sharp, 1.0);
        return;
    }

    // 4-tap bilinear tent over u_bokeh's own half-res texels, offset a half
    // texel off-centre in each diagonal direction. dof_bokeh.frag's spiral
    // gather is already the blur; this only smooths the half-res grid it was
    // written on before that grid gets stretched back up to full res, the same
    // role Unity URP's dedicated PostFilter pass plays between its gather and
    // its composite. Cheaper to fold into this existing sample than to add a
    // fourth DofPass stage/target for it.
    vec2 bokeh_texel = pc.inv_size * 2.0;
    vec3 bokeh = (
        texture(u_bokeh, in_uv + vec2(-0.5,  0.5) * bokeh_texel).rgb +
        texture(u_bokeh, in_uv + vec2( 0.5,  0.5) * bokeh_texel).rgb +
        texture(u_bokeh, in_uv + vec2(-0.5, -0.5) * bokeh_texel).rgb +
        texture(u_bokeh, in_uv + vec2( 0.5, -0.5) * bokeh_texel).rgb
    ) * 0.25;

    // Blend over a fixed, narrow band in FULL-res pixels -- NOT normalized by
    // max_radius. max_radius is a clamp ceiling (see DofPush's own doc and
    // DofPass::Params::max_radius); dividing by it here used to turn it into a
    // global blur-OPACITY dial instead, so every pixel below the ceiling was a
    // partial cross-fade of sharp and (half-res, undersampled) bokeh rather than
    // a clean defocus -- visibly a translucent, doubled ghost on any object near
    // the focal plane, and an under-blurred background everywhere else, since
    // only pixels AT the clamp ever reached blend == 1. dof_bokeh.frag's gather
    // radius already IS the physical blur amount; this composite's only job is
    // to hide the half-res upsample seam in a short band just past the sharp
    // cutoff above, then hand off to the gather at full strength.
    float blend = smoothstep(1.0, 2.0, abs(coc));
    out_color = vec4(mix(sharp, bokeh, blend), 1.0);
}
