#ifndef GFX_SSR_TRACE_SECONDARY_BODY_GLSL
#define GFX_SSR_TRACE_SECONDARY_BODY_GLSL

// gfx/ssr_trace_secondary_body.glsl -- second, independent Hi-Z SSR raymarch source.
//
// gfx_ssr_trace_secondary() is a LITERAL DUPLICATE of gfx_ssr_trace()
// (gfx/ssr_trace_body.glsl), algorithm-for-algorithm identical, reading a SECOND fixed set of
// sampler names instead. It exists so an opaque reflector can test its ray against a second
// geometry source -- e.g. a forward capture of transparent objects, which (like the opaque
// reflector's own primary source) never appears in the SAME G-buffer/Hi-Z the primary trace
// reads -- and pick whichever of the two independent hits is physically nearer (smaller
// `GfxSsrHit.travel`) along the shared ray. See ssr.frag's dual-trace call site.
//
// Duplicated rather than parameterized by a sampler array or an index: gfxcoopa's
// DescriptorSet::bind_image() (pipeline/descriptor.h) hardcodes descriptorCount=1/
// dstArrayElement=0 with zero existing precedent anywhere in this codebase for
// descriptorCount > 1, so two fixed single-image sets is the lower-risk, more idiomatic
// choice here -- matches this codebase's own established precedent for this exact tradeoff
// (pixel_lighting.frag's and transparent.frag's duplicated band()/shade_light()).
//
// KEEP IN SYNC WITH gfx_ssr_trace() IN gfx/ssr_trace_body.glsl IF THAT ALGORITHM EVER CHANGES.
//
// REQUIRED BEFORE INCLUDE (name-for-name; set indices are the includer's choice, but every
// name below must resolve to exactly this type). Note this is IN ADDITION to
// gfx/ssr_trace_body.glsl's own contract (camera UBO, GfxSsrParams, GfxSsrHit,
// gfx_ssr_get_view_z -- all reused from there, not redeclared here):
//   sampler2D g_normal_metallic_b, g_position_roughness_b;
//   sampler2D u_hiz_map_b;
//   sampler2D u_scene_color_b;
//
// Also requires <gfx/ssr_common.glsl> and <gfx/ssr_trace_body.glsl> to have been included first.

GfxSsrHit gfx_ssr_trace_secondary(vec3 P, vec3 N, float roughness, mat4 inv_proj, GfxSsrParams sp) {
    vec3 V = normalize(P - camera.camera_pos);
    vec3 R = reflect(V, N);

    ivec2 gsize    = textureSize(g_position_roughness_b, 0);
    float view_z   = (camera.view * vec4(P, 1.0)).z;
    float p11      = abs(camera.proj[1][1]);
    float px_world = ssr_texel_world_size(view_z, p11, float(gsize.y));

    float normal_bias = max(sp.bias_texels * px_world, 0.002);
    float ray_bias    = normal_bias * 1.35;
    float self_hit_r  = normal_bias * 1.0;

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

    vec3 current_pos = ray_start;
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

        vec2 mip_size = vec2(textureSize(u_hiz_map_b, current_mip));
        vec2 cell_idx = floor(current_pos.xy * mip_size);

        float cell_min_depth = textureLod(u_hiz_map_b, (cell_idx + 0.5) / mip_size, float(current_mip)).r;

        if (current_pos.z < cell_min_depth) {
            vec2 cell_min = cell_idx / mip_size;
            vec2 cell_max = (cell_idx + 1.0) / mip_size;

            vec2 t_planes;
            t_planes.x = (ray_dir.x > 0.0) ? (cell_max.x - current_pos.x) / ray_dir.x : (cell_min.x - current_pos.x) / ray_dir.x;
            t_planes.y = (ray_dir.y > 0.0) ? (cell_max.y - current_pos.y) / ray_dir.y : (cell_min.y - current_pos.y) / ray_dir.y;

            float t_step = max(min(t_planes.x, t_planes.y), 0.0001) + 0.0001;
            current_pos += ray_dir * t_step;

            current_mip = min(current_mip + 1, sp.max_hiz_mip);
        } else {
            if (current_mip == 0) {
                ivec2 hit_px = clamp(ivec2(current_pos.xy * vec2(gsize)), ivec2(0), gsize - 1);
                vec3 hit_world_pos = texelFetch(g_position_roughness_b, hit_px, 0).rgb;
                if (mip0_steps >= sp.min_mip0_steps && length(hit_world_pos - P) >= self_hit_r) {
                    float ray_z  = gfx_ssr_get_view_z(current_pos.z, inv_proj);
                    float surf_z = gfx_ssr_get_view_z(cell_min_depth, inv_proj);
                    float depth_diff_m = surf_z - ray_z;

                    float thickness = max(sp.thickness_min, sp.thickness_scale * abs(ray_z));

                    if (depth_diff_m >= 0.0 && depth_diff_m <= thickness) {
                        vec3 hit_normal = texelFetch(g_normal_metallic_b, hit_px, 0).rgb;
                        if (dot(hit_normal, R) < -0.05) {
                            hit_found  = true;
                            hit_uv     = current_pos.xy;
                            hit_P      = hit_world_pos;
                            hit_view_z = ray_z;
                            break;
                        }
                    }
                }
                vec2 mip0_size = vec2(textureSize(u_hiz_map_b, 0));
                float texel_size = 1.0 / max(mip0_size.x, mip0_size.y);
                current_pos += (ray_dir / max(length(ray_dir.xy), 1e-5)) * texel_size;
                mip0_steps++;
            } else {
                current_mip = current_mip - 1;
            }
        }
    }

    if (!hit_found) return GfxSsrHit(vec3(0.0), 0.0, 0.0, false);

    vec2 edge = smoothstep(vec2(0.0), vec2(0.08), hit_uv) * smoothstep(vec2(1.0), vec2(0.92), hit_uv);
    float screen_fade = edge.x * edge.y;

    float dir_fade = 1.0 - smoothstep(0.25, 0.85, dot(-V, R));
    float roughness_fade = 1.0 - smoothstep(sp.roughness_cutoff - 0.3, sp.roughness_cutoff, roughness);

    float NdotV_origin = max(dot(N, -V), 0.0);
    float grazing_fade = smoothstep(0.0, 0.05, NdotV_origin);

    float travel    = length(hit_P - P);
    float dist_fade = 1.0 - smoothstep(sp.max_distance * 0.7, sp.max_distance, travel);

    float confidence = screen_fade * dir_fade * roughness_fade * grazing_fade * dist_fade;

    float px_world_hit  = ssr_texel_world_size(hit_view_z, p11, float(gsize.y));
    float cone_diameter = 2.0 * ssr_ggx_cone_tan(roughness) * travel;
    float lod = clamp(log2(max(cone_diameter / max(px_world_hit, 1e-6), 1.0)),
                      0.0, float(sp.max_color_mip));

    vec3 hit_color = textureLod(u_scene_color_b, hit_uv, lod).rgb;

    return GfxSsrHit(hit_color * confidence, confidence, travel, true);
}

#endif // GFX_SSR_TRACE_SECONDARY_BODY_GLSL
