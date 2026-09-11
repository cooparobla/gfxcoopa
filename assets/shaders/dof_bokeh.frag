#version 450

// Depth-of-field stage 2/3: golden-angle spiral gather over dof_coc.frag's
// half-res (colour, signed CoC) output. See gfx/dof_common.glsl for the tap
// generator (and its optional N-blade iris snap) and gfxcoopa's dof_pass.h file
// doc for why this pass exists instead of a separable Gaussian: the CoC here is
// depth-driven, so it has hard discontinuities at silhouette edges that a
// horizontal-then-vertical blur (tilt_shift.frag's approach) would leak across.
// A single gather pass has no such ordering to leak across -- every tap either
// contributes or doesn't, independent of every other tap.

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_bokeh_coc;

layout(set = 0, binding = 0) uniform sampler2D u_coc; // dof_coc.frag's output, half-res, LINEAR

#include <gfx/dof_common.glsl>

void main() {
    vec4  center = texture(u_coc, in_uv);
    float coc    = center.a; // half-res pixels, signed

    // Sharp/near-sharp texel: a single tap is exact (nothing within 1px of centre
    // can be resolved as blur anyway), and this is the common case across most of
    // a typical frame -- skip the loop entirely rather than spending
    // sample_count taps converging on the same answer. Threshold is 0.5 HALF-res
    // px, i.e. 1 FULL-res px -- must match dof_composite.frag's own `abs(coc) <
    // 1.0` full-res early-out exactly (that shader recomputes CoC at full res),
    // or the two disagree on which band is "still sharp" and a dim seam/ring
    // appears at the mismatch.
    if (abs(coc) < 0.5) {
        out_bokeh_coc = center;
        return;
    }

    vec2 half_texel = pc.inv_size * 2.0; // this pass's own resolution is half of pc.inv_size's
    int n = clamp(int(pc.bokeh.x), 8, MAX_DOF_TAPS);
    // Per-pixel spiral rotation -- see dof_spiral_tap()'s own doc for why an
    // unrotated spiral shows up as a fixed dot lattice wherever CoC saturates.
    float rotation = dof_ign_angle(gl_FragCoord.xy);

    vec3  sum  = center.rgb;
    float wsum = 1.0;

    for (int i = 0; i < n; ++i) {
        vec2 o_unit = dof_spiral_tap(i, n, rotation);
        vec2 o_px   = o_unit * abs(coc);
        vec4 tap    = texture(u_coc, in_uv + o_px * half_texel);

        // Occlusion-aware weight: a tap contributes only if ITS OWN circle of
        // confusion reaches out far enough to cover this pixel (length(o_px) away).
        // This is the asymmetry a depth-blind or separable blur cannot express: a
        // blurred foreground tap (large |tap.a|) bleeds over a sharp background
        // centre, but a sharp foreground tap (small |tap.a|) does NOT get pulled
        // into a blurred background's gather, because it fails this test at any
        // offset beyond its own tiny CoC.
        float w = clamp(abs(tap.a) - length(o_px) + 1.0, 0.0, 1.0);
        sum  += tap.rgb * w;
        wsum += w;
    }

    out_bokeh_coc = vec4(sum / max(wsum, 1e-4), coc);
}
