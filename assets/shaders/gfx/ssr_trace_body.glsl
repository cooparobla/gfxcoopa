#ifndef GFX_SSR_TRACE_BODY_GLSL
#define GFX_SSR_TRACE_BODY_GLSL

// gfx/ssr_trace_body.glsl -- shared Hi-Z screen-space reflection raymarch.
//
// Factored out of ssr.frag so a forward-shaded consumer (toyengine's
// transparent.frag) can trace the same reflection an opaque G-buffer pixel
// would get, from an origin that never appears in the G-buffer itself. Only
// the ORIGIN surface (P, N, roughness) is taken as a parameter -- every other
// G-buffer read below happens at the ray's HIT point, which by construction
// is opaque geometry, so it is valid to sample regardless of what kind of
// surface the ray started from.
//
// REQUIRED BEFORE INCLUDE (name-for-name; set indices are the includer's
// choice, but every name below must resolve to exactly this type):
//   uniform CameraUBO { mat4 view; mat4 proj; vec3 camera_pos; } camera;
//   sampler2D g_normal_metallic, g_position_roughness;
//   sampler2D u_hiz_map;
//   sampler2D u_scene_color;
//
// Also requires <gfx/ssr_common.glsl> to have been included first.

/// Tunable knobs for one gfx_ssr_trace() call. Split out of a push-constant
/// block (rather than read directly from one) so ssr.frag and a consumer
/// with its own differently-laid-out push block can both populate this the
/// same way -- see transparent.frag's PushConstants for the latter.
struct GfxSsrParams {
    float max_distance;      // world-space ray length (ssr_max_distance)
    float bias_texels;       // normal bias, in full-res screen texels (ssr_bias_texels)
    float thickness_min;     // world-space floor for the thickness test (ssr_thickness)
    float thickness_scale;   // thickness as a fraction of |view z| (ssr_thickness_scale)
    float roughness_cutoff;
    int   max_iterations;
    int   max_hiz_mip;
    int   start_mip;         // Hi-Z mip the march starts at (ssr_start_mip)
    int   min_mip0_steps;    // self-reflection gate (ssr_min_mip0_steps)
    int   max_color_mip;     // top mip of the prefiltered scene-colour chain
    float jitter_strength;   // 0 = old deterministic mirror-ray trace, exactly (ssr_jitter)
    int   frame_index;       // decorrelates ssr_ign2() noise frame to frame; meaningless at
                              // jitter_strength == 0
};

float gfx_ssr_get_view_z(float depth_ndc, mat4 inv_proj) {
    vec4 clip = inv_proj * vec4(0.0, 0.0, depth_ndc, 1.0);
    return clip.z / clip.w;
}

/// Result of one gfx_ssr_trace() call. `color` is premultiplied by `confidence` -- same
/// convention ssr.frag's out_ssr_color always used, still true here (vec4(color, confidence)
/// reconstructs exactly what that used to be directly). `travel` (world-space distance from
/// the ray's origin P to the hit point) and `hit` are new: a caller juggling more than one
/// independent trace against different geometry sources (e.g. an opaque reflector also
/// checking a second, transparent-only Hi-Z -- see gfx_ssr_trace_secondary() in
/// gfx/ssr_trace_secondary_body.glsl) needs a way to pick whichever hit is physically NEARER
/// along the shared ray, and `confidence` alone can't answer that: it encodes screen-edge/
/// roughness/grazing/distance FADES, not proximity. `hit` makes "no hit" unambiguous rather
/// than inferred from `confidence <= 0.0` (a found-but-fully-faded hit is a real distinction
/// from no hit at all, even though both currently zero out `color`).
struct GfxSsrHit {
    vec3  color;
    float confidence;
    float travel;
    bool  hit;
};

/// Traces one reflection ray from world-space origin (P, N, roughness) against the Hi-Z
/// map. Caller is responsible for the background/roughness-cutoff early-out and the
/// dot(reflect(V,N), N) <= 0 check -- both are about the ORIGIN, which only the caller knows
/// how to fetch (G-buffer texelFetch for ssr.frag, forward interpolants for a transparent
/// pass).
GfxSsrHit gfx_ssr_trace(vec3 P, vec3 N, float roughness, mat4 inv_proj, GfxSsrParams sp) {
    vec3 V = normalize(P - camera.camera_pos);
    vec3 R = reflect(V, N);

    // Stochastic ray jitter. The far end of a reflection -- where the ray either clears the
    // reflected object's silhouette or doesn't -- is a hard binary hit/miss decision on a
    // perfectly deterministic ray; as the camera moves that decision flips in lockstep across
    // the whole boundary, which is what reads as shimmer no amount of temporal/spatial
    // filtering downstream can fully absorb. Sampling a fresh point inside the GGX lobe every
    // pixel, every frame turns that hard edge into per-pixel noise instead, which the
    // resolve/blur stages already exist to integrate into a soft edge. jitter_strength == 0
    // skips this entirely and reproduces the old single deterministic ray exactly.
    if (sp.jitter_strength > 0.0) {
        vec3 up = (abs(R.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent   = normalize(cross(up, R));
        vec3 bitangent = cross(R, tangent);

        vec2 xi = ssr_ign2(gl_FragCoord.xy, sp.frame_index);
        float lobe_radius = sqrt(xi.x) * sp.jitter_strength * ssr_ggx_cone_tan(roughness);
        float phi = 6.28318530718 * xi.y;

        vec3 candidate = normalize(R + tangent * (lobe_radius * cos(phi))
                                      + bitangent * (lobe_radius * sin(phi)));
        // The caller already gated on dot(R, N) <= 0 using this SAME unjittered R (computed
        // identically, before this branch runs) -- a jittered ray that dips below the origin
        // surface must fall back to the mirror ray rather than silently trace garbage.
        if (dot(candidate, N) > 0.0) R = candidate;
    }

    // Depth-scaled bias. The self-reflection this exists to prevent is a screen-space
    // phenomenon -- the ray must clear roughly one texel's worth of the source surface -- so
    // the correct model is a bias constant in TEXELS, not in metres.
    ivec2 gsize    = textureSize(g_position_roughness, 0);
    float view_z   = (camera.view * vec4(P, 1.0)).z;  // negative: RH + DEPTH_ZERO_TO_ONE
    float p11      = abs(camera.proj[1][1]);          // untouched by TAA jitter, which only
                                                       // perturbs proj[2][0] / proj[2][1]
    float px_world = ssr_texel_world_size(view_z, p11, float(gsize.y));

    // 0.002 floor: px_world -> 0 near the near plane, and a zero bias reintroduces the
    // silhouette ring outright. This is the one case where depth-scaling is strictly WORSE
    // than the old constant, so the floor is not optional.
    float normal_bias = max(sp.bias_texels * px_world, 0.002);
    float ray_bias    = normal_bias * 1.35;
    float self_hit_r  = normal_bias * 1.0;    // relaxed from 1.35x in Phase 2 tuning

    // Ray start and end in world space, biased along N and R to prevent self-reflection.
    vec3 P0 = P + N * normal_bias + R * ray_bias;
    vec3 P1 = P + N * normal_bias + R * sp.max_distance;

    vec4 clip0 = camera.proj * camera.view * vec4(P0, 1.0);
    vec4 clip1 = camera.proj * camera.view * vec4(P1, 1.0);

    if (clip0.w <= 0.0 || clip1.w <= 0.0) return GfxSsrHit(vec3(0.0), 0.0, 0.0, false);

    vec3 ndc0 = clip0.xyz / clip0.w;
    vec3 ndc1 = clip1.xyz / clip1.w;

    vec3 ray_start = vec3(ssr_ndc_to_uv(ndc0.xy), ndc0.z);
    vec3 ray_end   = vec3(ssr_ndc_to_uv(ndc1.xy), ndc1.z);
    vec3 ray_dir   = ray_end - ray_start;

    if (length(ray_dir.xy) < 0.0001) return GfxSsrHit(vec3(0.0), 0.0, 0.0, false);

    // Sub-texel start offset: shifts the ray's first sample point a fraction of one mip-0
    // texel along its own direction, decorrelating which Hi-Z cell each frame's march first
    // tests. Without this, a static sub-pixel camera offset makes every frame snap to
    // identical cell boundaries even with the direction jitter above -- the hit/miss boundary
    // would still sit at the same on-screen location every frame.
    if (sp.jitter_strength > 0.0) {
        vec2 mip0_size = vec2(textureSize(u_hiz_map, 0));
        float texel_size = 1.0 / max(mip0_size.x, mip0_size.y);
        float t_offset = (ssr_ign2(gl_FragCoord.xy + vec2(13.0, 7.0), sp.frame_index).x - 0.5) * texel_size;
        ray_start += (ray_dir / max(length(ray_dir.xy), 1e-5)) * t_offset;
    }

    vec3 current_pos = ray_start;
    // Starting at a coarse mip is safe: cell_min_depth is a conservative MIN over the cell, so
    // "ray is in front of the cell" at a coarse mip provably means no hit anywhere in that
    // cell. The only cost is that a ray starting behind its own cell's min burns start_mip
    // iterations descending, which is why this is a knob rather than a fixed 3 or 4.
    int current_mip = clamp(sp.start_mip, 0, sp.max_hiz_mip);
    bool hit_found = false;
    vec2 hit_uv = vec2(0.0);
    vec3 hit_P = vec3(0.0);
    float hit_view_z = -1.0;
    int mip0_steps = 0;

    for (int i = 0; i < sp.max_iterations; ++i) {
        if (current_pos.x < 0.0 || current_pos.x > 1.0 ||
            current_pos.y < 0.0 || current_pos.y > 1.0 ||
            current_pos.z < 0.0 || current_pos.z > 1.0) {
            break;
        }

        vec2 mip_size = vec2(textureSize(u_hiz_map, current_mip));
        vec2 cell_idx = floor(current_pos.xy * mip_size);

        float cell_min_depth = textureLod(u_hiz_map, (cell_idx + 0.5) / mip_size, float(current_mip)).r;

        if (current_pos.z < cell_min_depth) {
            // Ray is in front of surface geometry in cell -> step to cell boundary & step up mip
            vec2 cell_min = cell_idx / mip_size;
            vec2 cell_max = (cell_idx + 1.0) / mip_size;

            vec2 t_planes;
            t_planes.x = (ray_dir.x > 0.0) ? (cell_max.x - current_pos.x) / ray_dir.x : (cell_min.x - current_pos.x) / ray_dir.x;
            t_planes.y = (ray_dir.y > 0.0) ? (cell_max.y - current_pos.y) / ray_dir.y : (cell_min.y - current_pos.y) / ray_dir.y;

            // min_t_step: a UV-space nudge, converted into t-space by dividing by the ray's own
            // length -- ray_dir is the full un-normalized on-screen span (see the mip-0 texel
            // advance below, which normalizes for exactly this reason), so a bare constant here
            // would be a wildly different absolute distance for a screen-spanning ray than for a
            // short one, drifting continuously as the camera moves and each pixel's ray length
            // changes. That produced a genuine, deterministic, camera-rotation-keyed instability
            // in which Hi-Z cell boundary got crossed frame to frame. Capped at 0.01 of the t
            // range: for a very short on-screen ray (small length(ray_dir.xy)) the raw quotient
            // can exceed 1.0 -- an epsilon meant to be a tiny nudge would instead jump past the
            // entire remaining ray in one step and terminate the march immediately, turning
            // every such ray into a guaranteed miss. The cap keeps this a small nudge (at most
            // 1% of the ray) in that regime instead, while leaving normal-length rays (where the
            // quotient is already far below the cap) unaffected.
            float min_t_step = min(0.0001 / max(length(ray_dir.xy), 1e-5), 0.01);
            float t_step = max(min(t_planes.x, t_planes.y), min_t_step) + min_t_step;
            current_pos += ray_dir * t_step;

            current_mip = min(current_mip + 1, sp.max_hiz_mip);
        } else {
            // Ray penetrated cell surface
            if (current_mip == 0) {
                // Require at least a couple of mip-0 steps before honoring a hit. Right at a
                // silhouette (grazing angle, N nearly perpendicular to R), the normal bias barely
                // projects along the ray, so the very first texel(s) sampled after the bias can
                // still land back on the *same* reflecting surface just past the self-hit radius
                // -- a false self-reflection ring traced right along every silhouette edge.
                ivec2 hit_px = clamp(ivec2(current_pos.xy * vec2(gsize)), ivec2(0), gsize - 1);
                vec3 hit_world_pos = texelFetch(g_position_roughness, hit_px, 0).rgb;
                if (mip0_steps >= sp.min_mip0_steps && length(hit_world_pos - P) >= self_hit_r) {
                    float ray_z  = gfx_ssr_get_view_z(current_pos.z, inv_proj);
                    float surf_z = gfx_ssr_get_view_z(cell_min_depth, inv_proj);
                    float depth_diff_m = surf_z - ray_z;

                    // Thickness as a fraction of view depth with a world-space floor. A flat
                    // 0.1 m tolerance is far too tight for distant geometry (one Hi-Z texel
                    // already spans more than that in depth) and needlessly loose up close.
                    float thickness = max(sp.thickness_min, sp.thickness_scale * abs(ray_z));

                    if (depth_diff_m >= 0.0 && depth_diff_m <= thickness) {
                        vec3 hit_normal = texelFetch(g_normal_metallic, hit_px, 0).rgb;
                        if (dot(hit_normal, R) < -0.05) {
                            hit_found  = true;
                            hit_uv     = current_pos.xy;
                            hit_P      = hit_world_pos;
                            hit_view_z = ray_z;
                            break;
                        }
                    }
                }
                // Advance exactly one mip-0 texel along the ray direction. ray_dir is the
                // full un-normalized span of the whole ray (P0 -> P1 in UV space), not a
                // unit vector, so it must be normalized here -- otherwise this step is
                // "1/1920 of however long the ray happens to be on screen", which is far
                // too small for short on-screen rays (stalling the iteration budget) and
                // far too large for long ones (skipping over thin geometry).
                vec2 mip0_size = vec2(textureSize(u_hiz_map, 0));
                float texel_size = 1.0 / max(mip0_size.x, mip0_size.y);
                current_pos += (ray_dir / max(length(ray_dir.xy), 1e-5)) * texel_size;
                mip0_steps++;
            } else {
                current_mip = current_mip - 1;
            }
        }
    }

    if (!hit_found) return GfxSsrHit(vec3(0.0), 0.0, 0.0, false);

    // Confidence / Fading factors
    vec2 edge = smoothstep(vec2(0.0), vec2(0.08), hit_uv) * smoothstep(vec2(1.0), vec2(0.92), hit_uv);
    float screen_fade = edge.x * edge.y;

    // Fade out reflections whose ray points back toward the camera -- these are the ones
    // most prone to grazing-angle stretching/parallax error near the viewer's own reflection.
    // dot(-V, R) approaches 1 as R points back at the camera, so this fade must go toward
    // 0 (not 1) as that dot product increases.
    float dir_fade = 1.0 - smoothstep(0.25, 0.85, dot(-V, R));
    // Widened band (was 0.2). With cone tracing there is no longer a sharp-to-nothing pop to
    // hide, so the fade's job changes: it now hands over to the probe/sky prefilter, which
    // above ~0.8 roughness is genuinely the better answer anyway -- a single screen-space ray
    // with a cone that wide is sampling a mip so coarse that it has stopped being a reflection
    // of anything local, and the screen-space blur ignores the depth discontinuities the probe
    // does not have.
    float roughness_fade = 1.0 - smoothstep(sp.roughness_cutoff - 0.3, sp.roughness_cutoff, roughness);

    // Fade out reflections originating right at a silhouette (view direction nearly tangent
    // to the surface, NdotV near 0). Hit data there is the least reliable: the normal bias
    // barely projects along the ray at grazing angles, so even with the minimum-step gate
    // above, occasional false self-hits still show up as a thin ring right on every silhouette.
    float NdotV_origin = max(dot(N, -V), 0.0);
    float grazing_fade = smoothstep(0.0, 0.05, NdotV_origin);

    // Distance fade. Rays that run to max_distance are the ones whose screen-space error is
    // largest, and without this they pop out abruptly as the camera moves.
    float travel    = length(hit_P - P);
    float dist_fade = 1.0 - smoothstep(sp.max_distance * 0.7, sp.max_distance, travel);

    float confidence = screen_fade * dir_fade * roughness_fade * grazing_fade * dist_fade;

    // Roughness-aware cone footprint -> mip level. The specular cone has half-angle theta at
    // the origin, so at the hit it has spread to a world-space radius of tan(theta) * travel.
    // Project that to full-res texels at the HIT's depth (not the origin's -- the footprint is
    // a feature of the reflected image, which lives at the hit); a footprint N texels across is
    // exactly mip log2(N).
    float px_world_hit  = ssr_texel_world_size(hit_view_z, p11, float(gsize.y));
    float cone_diameter = 2.0 * ssr_ggx_cone_tan(roughness) * travel;
    float lod = clamp(log2(max(cone_diameter / max(px_world_hit, 1e-6), 1.0)),
                      0.0, float(sp.max_color_mip));

    vec3 hit_color = textureLod(u_scene_color, hit_uv, lod).rgb;

    // Premultiplied by confidence -- see GfxSsrHit's own doc above for why.
    return GfxSsrHit(hit_color * confidence, confidence, travel, true);
}

#endif // GFX_SSR_TRACE_BODY_GLSL
