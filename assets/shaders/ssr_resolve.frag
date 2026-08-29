#version 450

#include <gfx/ssr_common.glsl>

// Temporal resolve for the SSR raymarch output (rgb = confidence-premultiplied hit radiance,
// a = confidence).
//
// Unlike taa.frag -- and unlike this shader's previous revision, which copied taa.frag's
// "no velocity, static reprojection" simplification -- history is now reprojected. G2 already
// stores world position, so all this needs is the previous frame's view-projection.

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex_current;            // nearest
layout(set = 0, binding = 1) uniform sampler2D tex_history;            // LINEAR (reprojected UV)
layout(set = 0, binding = 2) uniform sampler2D g_position_roughness;   // nearest

layout(push_constant) uniform PushConstants {
    mat4  prev_view_proj;  // previous frame's JITTERED proj * view
    vec2  resolution;      // TRACE resolution (half of the screen under ssr_half_res)
    float blend_factor;
    int   history_valid;   // 0 until both a history image and a previous matrix exist
    float gamma;           // variance-clipping width, in std deviations (ssr_temporal_gamma)
} pc;

void main() {
    vec4 current = texture(tex_current, in_uv);

    if (pc.history_valid == 0) {
        out_color = current;
        return;
    }

    // Reprojection. G2 holds the world position of the surface this pixel's reflection
    // ORIGINATES from, so this reprojects the reflecting surface, not the reflected image. A
    // reflection parallaxes faster than its host surface -- as a mirror sweeps past a fixed
    // object the virtual image moves at roughly twice the surface's screen velocity -- so this
    // is exact only for points on the mirror plane itself. It removes the bulk of the smear;
    // the residual is what the neighbourhood clamp below is still for. That is why the clamp
    // stays even now that reprojection is real.
    vec3 P = texture(g_position_roughness, in_uv).rgb;
    vec4 prev_clip = pc.prev_view_proj * vec4(P, 1.0);

    // Behind the previous frame's eye: no history exists for this point at all.
    if (prev_clip.w <= 0.0) {
        out_color = current;
        return;
    }

    vec2 prev_uv = ssr_ndc_to_uv(prev_clip.xy / prev_clip.w);

    // Off-screen last frame -- the cheapest and most common disocclusion case. CLAMP_TO_EDGE on
    // the history sampler would otherwise smear the border texel inward along every screen edge
    // the camera is turning toward.
    if (any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)))) {
        out_color = current;
        return;
    }

    // 3x3 neighbourhood AABB of the CURRENT frame's trace. Computed after the early-outs above,
    // which the previous revision did not do.
    vec2 texel_size = 1.0 / pc.resolution;
    vec4 s0 = texture(tex_current, in_uv + vec2(-1.0,  1.0) * texel_size);
    vec4 s1 = texture(tex_current, in_uv + vec2( 0.0,  1.0) * texel_size);
    vec4 s2 = texture(tex_current, in_uv + vec2( 1.0,  1.0) * texel_size);
    vec4 s3 = texture(tex_current, in_uv + vec2(-1.0,  0.0) * texel_size);
    vec4 s4 = current;
    vec4 s5 = texture(tex_current, in_uv + vec2( 1.0,  0.0) * texel_size);
    vec4 s6 = texture(tex_current, in_uv + vec2(-1.0, -1.0) * texel_size);
    vec4 s7 = texture(tex_current, in_uv + vec2( 0.0, -1.0) * texel_size);
    vec4 s8 = texture(tex_current, in_uv + vec2( 1.0, -1.0) * texel_size);

    // Variance clipping (Salvi 2016 / Marrs et al.) instead of a raw min/max AABB. A min/max
    // box over just 9 samples is maximally sensitive to a single outlier -- exactly what the
    // stochastic ray jitter in gfx_ssr_trace() now produces every frame at a silhouette, since
    // a jittered ray occasionally hits/misses independently of its neighbours even when the
    // underlying surface hasn't changed. Clipping to the neighbourhood's mean +/- gamma standard
    // deviations instead accepts that per-pixel noise as normal variation and only rejects
    // genuine disocclusion, which is what lets the temporal blend below actually integrate the
    // jitter into a soft edge instead of rejecting history every single frame.
    vec4 sum  = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
    vec4 sum2 = s0*s0 + s1*s1 + s2*s2 + s3*s3 + s4*s4 + s5*s5 + s6*s6 + s7*s7 + s8*s8;
    vec4 mean = sum / 9.0;
    vec4 sigma = sqrt(max(sum2 / 9.0 - mean * mean, vec4(0.0)));

    // Neighbourhood maximum confidence, kept separately from the moments above -- needed to
    // tell "the whole neighbourhood genuinely missed" (disocclusion) apart from "confidence is
    // just noisy" (the jitter), which the decay branch below depends on.
    float max_a = max(s0.a, max(s1.a, max(s2.a, max(s3.a, max(s4.a, max(s5.a, max(s6.a, max(s7.a, s8.a))))))));

    vec4 history = texture(tex_history, prev_uv);

    // Colour: two-sided clip against the variance ellipsoid, as the AABB clamp used to be.
    vec3 clamped_rgb = clamp(history.rgb, mean.rgb - pc.gamma * sigma.rgb, mean.rgb + pc.gamma * sigma.rgb);

    // Confidence: same one-sided INTENT as the old AABB-max clamp -- accumulated history should
    // never be pulled ABOVE what this frame's neighbourhood supports -- but a hard collapse to
    // (near) zero the instant the neighbourhood misses is exactly the pulse the jitter is meant
    // to avoid: with a boundary that moves every frame, "the whole neighbourhood missed" now
    // happens often even where a reflection genuinely belongs, not just at real disocclusion. So
    // a genuine whole-neighbourhood miss DECAYS history instead of zeroing it outright; a
    // neighbourhood that has at least one hit is clipped to its mean + gamma*sigma as before.
    //
    // The two branches are blended by a smoothstep of max_a rather than a hard ?: -- a discrete
    // switch AT max_a == 1e-4 is itself a flicker source: a neighbourhood sitting right at that
    // threshold flips between "decay" and "clip" behaviour from one frame to the next as max_a
    // drifts a hair either side of it, on top of whatever moved max_a in the first place. Both
    // branches still evaluate identically well below/above the old threshold; only the
    // transition between them is now continuous.
    float decayed  = history.a * 0.6;
    float clipped  = min(history.a, mean.a + pc.gamma * sigma.a);
    float clamped_a = mix(decayed, clipped, smoothstep(0.0, 1e-3, max_a));

    out_color = vec4(mix(current.rgb, clamped_rgb, pc.blend_factor),
                     mix(current.a,   clamped_a,   pc.blend_factor));
}
