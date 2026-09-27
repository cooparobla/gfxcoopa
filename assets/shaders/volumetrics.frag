#version 450

// Fullscreen raymarched composite for LOCAL volumes (fog pockets, wind ribbons,
// drifting haze -- see gfx/volumetrics.glsl for the three kinds).
//
// The counterpart to fog.frag, which handles the GLOBAL atmosphere analytically.
// Everything MARCHED here is bounded and scene-placed: a volume carries its own
// complete field description alongside its bounds.
//
// The global term is still analytic, but this shader applies it itself (through
// the same gfx_fog_apply() call fog.frag makes) whenever the caller sets the
// merged flag -- because with both effects on, fog.frag's only job would be to
// write a full-resolution HDR image this pass immediately reads back. See
// `merged_fog` in main() and PixelRenderPipeline::fog_merged_into_volumetrics_().
//
// Vertex stage is the shared fullscreen triangle (fullscreen.vert). Writes into
// its own HDR target -- pipeline::RenderPass hardcodes LOAD_OP_CLEAR, so this
// cannot composite in place onto the image it reads from.

layout(location = 0) in  vec2 in_uv;
layout(location = 0) out vec4 out_color;

// Set 0: scene colour (linear sampler) + G-buffer normal/position (nearest --
// linear filtering would blend world positions across silhouette edges and
// produce a wrong ray-termination distance on every object outline).
layout(set = 0, binding = 0) uniform sampler2D scene_color;
layout(set = 0, binding = 1) uniform sampler2D g_normal_metallic;
layout(set = 0, binding = 2) uniform sampler2D g_position_roughness;

struct Volume {
    mat4 inv_world;
    vec4 extent_shape;     // xyz = half-extent (sphere uses .x), w = 0 Box / 1 Sphere
    vec4 direction_speed;  // xyz = normalized advection direction, w = speed
    vec4 field_params;     // x = noise scale, y = streak, z = coverage, w = fbm gain
    vec4 shape_params;     // x = density, y = height base, z = height falloff, w = octaves
    vec4 flow_params;      // x = meander amp, y = meander freq, z = sharpness, w = gate freq
    vec4 color_occlusion;  // rgb = scatter colour, w = occlusion scale
    vec4 mode_params;      // x = kind, y = sun amount, z = edge softness
};

// A point or spot light that in-scatters into the march -- a trimmed copy of the
// lighting pass's per-light data (see ScatterLightGPU, volumetrics_data.h).
struct ScatterLight {
    vec4 position_range;   // xyz = world position, w = range
    vec4 color_intensity;  // rgb = colour, w = intensity
    vec4 direction_cone;   // xyz = spot direction, w = cos(outer); ignored for points
    vec4 params;           // x = falloff sharpness, y = cos(inner), z = 1 spot / 0 point,
                           // w = 1 -> shadow with the spot map
};

layout(set = 1, binding = 0) uniform VolumetricsUBO {
    mat4 inv_view_proj;
    vec4 camera_pos;     // xyz = world camera position, w = debug view flag
    vec4 sun_direction;  // xyz = direction the light travels
    vec4 sun_color;      // rgb = sun colour * intensity
    vec4 march_params;   // x = step count, y = max distance, z = max opacity, w = sun anisotropy
    vec4 time_params;    // x = elapsed time, y = delta time, z = frame index
    vec4 counts;         // x = active volume count, y = scatter light count,
                         // z = light-scatter strength (0 skips the light loop)
    Volume volumes[8];
    mat4 dir_light_space_matrix;   // world -> cascade 0's shadow clip; the sun term reads
                                   // dir_cascade_matrix below instead, but the field holds
                                   // this block's place in the std140 layout
    mat4 spot_light_space_matrix;  // world -> spot shadow clip
    vec4 shadow_params;  // x = shadow the sun term (0/1), y = shadow strength,
                         // z = dir depth bias, w = spot depth bias
    ScatterLight scatter_lights[4];
    // Directional shadow cascades -- see VolumetricsUBO (volumetrics_data.h). The
    // directional map is a tile atlas, so dir_light_space_matrix above only reaches
    // cascade 0; a shaft marching the full distance needs the whole set.
    mat4 dir_cascade_matrix[4];
    vec4 dir_cascade_info; // x = cascade count, y = tiles per atlas row, z = selection
                           // inset in tile uv, w = dither band (unused here)
} u_vol;

// Set 2: shadow maps for the in-scatter terms. Compare-enabled samplers
// (util::Sampler::shadow(), VK_COMPARE_OP_GREATER), so one texture() call is a
// hardware-filtered depth compare returning 1 = in shadow -- the same convention
// gfx/shadow_sampling.glsl's *Shadow family documents.
layout(set = 2, binding = 0) uniform sampler2DShadow dir_shadow_map;
layout(set = 2, binding = 1) uniform sampler2DShadow spot_shadow_map;

// Set 3: the GLOBAL fog description (FogUBO's layout, field for field -- see
// fog.frag). Read only when shadow_params.z selects the merged path, where this
// pass applies the global fog term itself instead of reading an image FogPass
// already wrote it into -- one fullscreen HDR pass per frame instead of two. The
// binding is always declared and always bound (the fog UBO buffer exists
// regardless of the toggle), so the pipeline layout never depends on the flag.
layout(set = 3, binding = 0) uniform FogUBO {
    mat4 inv_view_proj;
    vec4 camera_pos;
    vec4 fog_color;
    vec4 sun_direction;
    vec4 sun_color;
    vec4 mode_density;
    vec4 height_params;
    vec4 misc_params;
    vec4 sky_zenith;
    vec4 sky_horizon;
    vec4 sky_ground;
} u_fog;

#include <gfx/volumetrics.glsl>
#include <gfx/fog.glsl>        // gfx_fog_hg + the box/sphere containment weights
#include <gfx/spot_light.glsl> // gfx_spot_cone for the scatter-light loop
#include <gfx/shadow_sampling.glsl> // gfx_csm_select/gfx_csm_atlas_coords for the sun term

// One hardware-PCF visibility tap of a 2D shadow map at world point p, for the
// march's in-scatter terms. Out of bounds / behind the map => fully lit: a march
// sample outside the fitted shadow frustum carries no occlusion information, and
// darkening it would draw the frustum's edges into the fog as a visible box.
// One tap per march step is deliberate -- the IGN start-offset dither already
// decorrelates neighbouring pixels' sample positions, so the tap noise reads as
// the same fine dither the rest of the march produces and TAA integrates it.
float vol_shadow_vis(mat4 light_space, sampler2DShadow map, vec3 p, float bias, float strength) {
    vec4 lsp = light_space * vec4(p, 1.0);
    if (lsp.w <= 0.0) return 1.0;
    vec3 proj = lsp.xyz / lsp.w;
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.z <= 0.0 || proj.z > 1.0 ||
        proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
        return 1.0;
    }
    return 1.0 - texture(map, vec3(proj.xy, proj.z - bias)) * strength;
}

// The same tap against the CASCADED directional map, whose tiles each cover one slice of
// the camera's depth range. vol_shadow_vis() above still serves the spot map, which is a
// single frustum; the sun needs this one, because a shaft marches out to
// volumetrics_max_distance and most of it lands past the near cascade's tile.
//
// Dither is passed as 0: gfx_csm_select()'s transition dither exists for TAA to average
// away, and there is no temporal filter behind this march to do it -- an undithered hard
// switch between two cascades is invisible in fog, where a shadow tap only modulates
// in-scatter, while the dither would read as grain.
float vol_shadow_vis_cascaded(sampler2DShadow map, vec3 p, float bias, float strength) {
    int c = gfx_csm_select(u_vol.dir_cascade_matrix, u_vol.dir_cascade_info, p, 0.0);
    // Past the last cascade: fully lit, the same reasoning vol_shadow_vis()'s
    // out-of-bounds early-out documents.
    if (c < 0) return 1.0;
    vec3 atlas = gfx_csm_atlas_coords(u_vol.dir_cascade_matrix, u_vol.dir_cascade_info, p, c);
    return 1.0 - texture(map, vec3(atlas.xy, atlas.z - bias)) * strength;
}

void main() {
    vec3 color = texture(scene_color, in_uv).rgb;

    int  volume_count = int(u_vol.counts.x);
    // Merged path: this pass also applies the GLOBAL fog term, so FogPass doesn't run
    // and the frame pays for one fullscreen HDR pass instead of two. The composite
    // itself is gfx_fog_apply(), shared with fog.frag, and it is applied to the scene
    // colour BEFORE the local march -- exactly the order the two separate passes
    // produce (fog writes an image, volumetrics marches over it), so the merge is
    // arithmetically identical rather than an approximation.
    bool merged_fog = u_vol.counts.w > 0.5;

    if (volume_count <= 0 && !merged_fog) {
        out_color = vec4(color, 1.0);   // nothing placed -- costs one fetch, not a march
        return;
    }

    vec3 N       = texture(g_normal_metallic, in_uv).rgb;
    vec3 cam_pos = u_vol.camera_pos.xyz;
    bool is_sky  = dot(N, N) < 0.001;   // G1 normal is zero on sky, same test fog.frag uses

    vec3 A = cam_pos;

    // Reconstruct the view ray from inv_view_proj for EVERY pixel rather than by
    // subtracting the G2 world position on geometry pixels -- fog.frag's comment
    // explains the precision reasoning. NDC.y is negated because this fullscreen
    // triangle uses a positive-height viewport while inv_view_proj follows the
    // Y-up convention every other unprojection here shares.
    vec3 ndc      = vec3(in_uv.x * 2.0 - 1.0, 1.0 - in_uv.y * 2.0, 1.0);
    vec4 world    = u_vol.inv_view_proj * vec4(ndc, 1.0);
    vec3 view_dir = normalize(world.xyz / world.w - cam_pos);

    float d_geo = is_sky ? 1e6 : distance(A, texture(g_position_roughness, in_uv).rgb);

    if (merged_fog) {
        color = gfx_fog_apply(color, A, view_dir, d_geo, is_sky,
                              u_fog.mode_density, u_fog.height_params, u_fog.misc_params,
                              u_fog.fog_color.rgb, u_fog.sun_direction.xyz, u_fog.sun_color.rgb,
                              u_fog.sky_zenith.rgb, u_fog.sky_horizon.rgb, u_fog.sky_ground.rgb);
        if (volume_count <= 0) {
            out_color = vec4(color, 1.0);
            return;
        }
    }

    float ray_end = min(d_geo, max(u_vol.march_params.y, 0.0));

    int steps = int(u_vol.march_params.x);
    if (ray_end <= 0.0 || steps <= 0) {
        out_color = vec4(color, 1.0);
        return;
    }

    // Clip the march to the UNION of the volumes' bounds. Every VOLUME is bounded by
    // definition -- the global term is analytic and was already applied above, not
    // marched -- so marching a fixed distance would spend most steps in empty air.
    // Two wins: pixels looking at
    // nothing skip the march entirely, and pixels that do hit a volume spend the whole
    // step budget inside it, sampling it far more finely than a fixed span would.
    //
    // A union span still crosses the gap between disjoint volumes; the per-sample
    // containment test early-outs there cheaply, and per-volume marching is not worth
    // the complexity.
    float t_near = 1e30;
    float t_far  = -1e30;
    for (int v = 0; v < volume_count; ++v) {
        if (u_vol.volumes[v].shape_params.x <= 0.0) {
            continue;
        }
        // inv_world is rotation + translation only (volumes assume unit scale), so the
        // returned t stays in world-space distance units on both sides of the transform
        // and needs no renormalisation -- the same contract fog's volume loop relied on.
        vec3 ro = (u_vol.volumes[v].inv_world * vec4(A, 1.0)).xyz;
        vec3 rd = (u_vol.volumes[v].inv_world * vec4(view_dir, 0.0)).xyz;
        vec3 ext = u_vol.volumes[v].extent_shape.xyz;
        vec2 hit = (u_vol.volumes[v].extent_shape.w > 0.5)
                 ? gfx_fog_sphere_intersect(ro, rd, ext.x)
                 : gfx_fog_box_intersect(ro, rd, ext);
        if (hit.y <= hit.x) {
            continue;                     // miss: both helpers return an empty interval
        }
        t_near = min(t_near, max(hit.x, 0.0));
        t_far  = max(t_far,  min(hit.y, ray_end));
    }

    if (t_far <= t_near) {
        out_color = vec4(color, 1.0);     // no volume in front of this pixel
        return;
    }

    float dt = (t_far - t_near) / float(steps);

    // Dither the START offset within one step. Without it a low step count lays
    // visible concentric bands over every feature. The per-frame R2 shift means a
    // temporal accumulator downstream averages a DIFFERENT pattern each frame,
    // resolving the banding instead of re-resolving one frozen one.
    float t = t_near + dt * gfx_volume_ign(gl_FragCoord.xy, int(u_vol.time_params.z));

    // Hoisted: the phase function depends only on view_dir and the sun, neither of
    // which varies along the ray or between volumes. Negated sun_direction because
    // that field stores the direction light TRAVELS, not the direction to the sun.
    float phase    = gfx_fog_hg(dot(view_dir, -u_vol.sun_direction.xyz), u_vol.march_params.w);
    vec3  sun_base = u_vol.sun_color.rgb * phase;

    float T        = 1.0;
    vec3  scatter  = vec3(0.0);
    float coverage = 0.0;

    for (int i = 0; i < steps; ++i) {
        vec3 p = A + view_dir * t;

        float sigma     = 0.0;
        vec3  emit      = vec3(0.0);
        float light_w   = 0.0;   // density-weighted in-scatter response (mode_params.y)
        float occl_sum  = 0.0;

        for (int v = 0; v < volume_count; ++v) {
            float density = u_vol.volumes[v].shape_params.x;
            if (density <= 0.0) {
                continue;   // a disabled volume costs nothing
            }

            // Containment first: it is far cheaper than the field, and outside the
            // bounds the field is irrelevant. gfx_fog_box_edge_weight is already a
            // POINT-based soft containment test (0 outside), so it drops straight in.
            vec3  lp  = (u_vol.volumes[v].inv_world * vec4(p, 1.0)).xyz;
            vec3  ext = u_vol.volumes[v].extent_shape.xyz;
            float soft = u_vol.volumes[v].mode_params.z;
            float vw  = (u_vol.volumes[v].extent_shape.w > 0.5)
                      ? gfx_fog_sphere_point_weight(lp, ext.x, soft)
                      : gfx_fog_box_edge_weight(lp, ext, soft);
            if (vw <= 1e-4) {
                continue;
            }

            float f = gfx_volume_field(p, u_vol.time_params.x,
                                       int(u_vol.volumes[v].mode_params.x),
                                       u_vol.volumes[v].direction_speed,
                                       u_vol.volumes[v].field_params,
                                       u_vol.volumes[v].shape_params,
                                       u_vol.volumes[v].flow_params);
            float s = f * vw * density;
            if (s <= 1e-5) {
                continue;
            }

            sigma    += s;
            emit     += s * u_vol.volumes[v].color_occlusion.rgb;
            light_w  += s * u_vol.volumes[v].mode_params.y;
            occl_sum += s * u_vol.volumes[v].color_occlusion.w;
        }

        // Light in-scatter, evaluated once per STEP (not per volume: it depends
        // only on the sample point) and only where density responded to light at
        // all -- shadow taps and the light loop cost nothing over empty air.
        // Each volume's own response is its density-weighted mode_params.y
        // (light_w), so sun and local lights share one per-volume dial.
        if (light_w > 0.0) {
            float sun_vis = (u_vol.shadow_params.x > 0.5)
                ? vol_shadow_vis_cascaded(dir_shadow_map, p,
                                          u_vol.shadow_params.z, u_vol.shadow_params.y)
                : 1.0;
            vec3 in_scatter = sun_base * sun_vis;

            float light_strength = u_vol.counts.z;
            int   light_count    = int(u_vol.counts.y);
            for (int li = 0; li < light_count; ++li) {
                if (light_strength <= 0.0) break;
                vec3  to_light = u_vol.scatter_lights[li].position_range.xyz - p;
                float dist     = length(to_light);
                float range    = u_vol.scatter_lights[li].position_range.w;
                if (dist > range || dist < 1e-4) continue;
                vec3 L = to_light / dist;

                float cone = 1.0;
                if (u_vol.scatter_lights[li].params.z > 0.5) {
                    cone = gfx_spot_cone(L, u_vol.scatter_lights[li].direction_cone.xyz,
                                         u_vol.scatter_lights[li].direction_cone.w,
                                         u_vol.scatter_lights[li].params.y);
                    if (cone <= 0.0) continue;
                }

                // Same distance curve as the lighting pass's point/spot loops
                // (pixel_lighting.frag), so a light's glow in a volume matches
                // its glow on the surfaces around it.
                float sharpness = max(u_vol.scatter_lights[li].params.x, 0.1);
                float factor    = clamp(dist / range, 0.0, 1.0);
                float falloff   = clamp(1.0 - pow(factor, sharpness), 0.0, 1.0);
                falloff *= falloff;
                float attenuation = falloff / (12.566370614 * (factor * factor + 1.0)); // 4*pi
                vec3 radiance = u_vol.scatter_lights[li].color_intensity.rgb
                              * (u_vol.scatter_lights[li].color_intensity.w * 0.08)
                              * attenuation * cone;

                float vis = (u_vol.scatter_lights[li].params.w > 0.5)
                    ? vol_shadow_vis(u_vol.spot_light_space_matrix, spot_shadow_map, p,
                                     u_vol.shadow_params.w, u_vol.shadow_params.y)
                    : 1.0;

                // Same HG phase as the sun term: the anisotropy is a property of
                // the medium's phase function, whichever light the energy came from.
                float phase_l = gfx_fog_hg(dot(view_dir, L), u_vol.march_params.w);
                in_scatter += radiance * (phase_l * light_strength * vis);
            }

            emit += light_w * in_scatter;
        }

        if (sigma > 1e-4) {
            // Emission and extinction are DECOUPLED. With one coefficient driving
            // both, a feature bright enough to see necessarily also darkens what is
            // behind it -- which is what makes thin bright strands read as grey
            // smudges rather than as light. The occlusion scale is density-weighted
            // across the volumes contributing here, so overlapping volumes composite
            // as one medium instead of double-darkening.
            float a   = 1.0 - exp(-sigma * dt);
            scatter  += T * a * (emit / sigma);
            coverage += T * a;

            float ext = 1.0 - exp(-sigma * dt * (occl_sum / sigma));
            T        *= (1.0 - ext);
            if (T < 0.01) {
                break;   // saturated -- no remaining step can contribute visibly
            }
        }
        t += dt;
    }

    T = clamp(T, 1.0 - clamp(u_vol.march_params.z, 0.0, 1.0), 1.0);

    // Debug view: accumulated emission coverage alone, scene colour suppressed.
    // Coverage, not 1 - T: the occlusion scale is independent, so transmittance is
    // near 1 by design and says almost nothing about the field's shape.
    if (u_vol.camera_pos.w > 0.5) {
        out_color = vec4(vec3(clamp(coverage, 0.0, 1.0)), 1.0);
        return;
    }

    out_color = vec4(color * T + scatter, 1.0);
}
