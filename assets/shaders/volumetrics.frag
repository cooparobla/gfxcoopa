#version 450

// Fullscreen raymarched composite for LOCAL volumes (fog pockets, wind ribbons,
// drifting haze -- see gfx/volumetrics.glsl for the three kinds).
//
// The counterpart to fog.frag, which handles the GLOBAL atmosphere analytically.
// Everything here is bounded and scene-placed, so there is no global term: a
// volume carries its own complete field description alongside its bounds.
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

layout(set = 1, binding = 0) uniform VolumetricsUBO {
    mat4 inv_view_proj;
    vec4 camera_pos;     // xyz = world camera position, w = debug view flag
    vec4 sun_direction;  // xyz = direction the light travels
    vec4 sun_color;      // rgb = sun colour * intensity
    vec4 march_params;   // x = step count, y = max distance, z = max opacity, w = sun anisotropy
    vec4 time_params;    // x = elapsed time, y = delta time, z = frame index
    vec4 counts;         // x = active volume count
    Volume volumes[8];
} u_vol;

#include <gfx/volumetrics.glsl>
#include <gfx/fog.glsl>   // gfx_fog_hg + the box/sphere containment weights

void main() {
    vec3 color = texture(scene_color, in_uv).rgb;

    int volume_count = int(u_vol.counts.x);
    if (volume_count <= 0) {
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
    float ray_end = min(d_geo, max(u_vol.march_params.y, 0.0));

    int steps = int(u_vol.march_params.x);
    if (ray_end <= 0.0 || steps <= 0) {
        out_color = vec4(color, 1.0);
        return;
    }

    // Clip the march to the UNION of the volumes' bounds. Everything here is bounded
    // by definition (fog is the global term and lives in fog.frag), so marching a
    // fixed distance would spend most steps in empty air. Two wins: pixels looking at
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
            emit     += s * (u_vol.volumes[v].color_occlusion.rgb
                             + sun_base * u_vol.volumes[v].mode_params.y);
            occl_sum += s * u_vol.volumes[v].color_occlusion.w;
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
