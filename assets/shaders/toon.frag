#version 450

// toon.frag — Cel/toon shading with Directional & Point lights and Shadow Mapping.

layout(location = 0) in vec3 frag_world_normal;
layout(location = 1) in vec3 frag_world_pos;
layout(location = 2) in vec2 frag_uv;

// set 0: camera UBO
layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 view_pos;
} camera;

struct PointLightGPU {
    vec4 position_range;  // xyz = pos, w = range
    vec4 color_intensity; // xyz = color, w = intensity
    vec4 attenuation;     // x=const, y=linear, z=quad, w=cast_shadows
};

// set 1: light UBO
layout(set = 1, binding = 0) uniform LightData {
    vec4 dir_direction;
    vec4 dir_color;
    vec4 dir_ambient;
    mat4 dir_light_space_matrix;
    vec4 dir_shadow_params; // x=bias, y=pcf_samples, z=shadow_enabled

    uvec4 light_counts; // x=num_dir, y=num_point_lights
    PointLightGPU point_lights[16];
} light;

// set 2: shadow maps
layout(set = 2, binding = 0) uniform sampler2D dir_shadow_map;
layout(set = 2, binding = 1) uniform samplerCube point_shadow_map;

layout(location = 0) out vec4 out_color;

const int   TOON_BANDS      = 4;
const float SPECULAR_POWER  = 32.0;
const float SPECULAR_CUTOFF = 0.92;

float calculate_dir_shadow(vec4 light_space_pos) {
    if (light.dir_shadow_params.z < 0.5) return 1.0; // Shadows disabled

    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // Map NDC [-1, 1] to UV [0, 1]
    proj_coords.x = proj_coords.x * 0.5 + 0.5;
    proj_coords.y = proj_coords.y * 0.5 + 0.5;

    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }

    float bias = max(light.dir_shadow_params.x * (1.0 - dot(normalize(frag_world_normal), normalize(-light.dir_direction.xyz))), 0.001);
    float closest_depth = texture(dir_shadow_map, proj_coords.xy).r;
    float current_depth = proj_coords.z;

    return (current_depth - bias > closest_depth) ? 0.0 : 1.0;
}

float calculate_point_shadow(vec3 light_pos, float range) {
    vec3 frag_to_light = frag_world_pos - light_pos;
    float current_depth = length(frag_to_light) / range;
    if (current_depth > 1.0) return 1.0;

    float bias = 0.005;
    float closest_depth = texture(point_shadow_map, frag_to_light).r;

    return (current_depth - bias > closest_depth) ? 0.0 : 1.0;
}

void main() {
    vec3 N = normalize(frag_world_normal);
    vec3 V = normalize(camera.view_pos - frag_world_pos);

    vec3 base_color = vec3(0.85, 0.72, 0.48); // default warm ochre palette
    vec3 total_diffuse = vec3(0.0);
    vec3 total_specular = vec3(0.0);

    // --- 1. Directional Light ---
    if (light.light_counts.x > 0) {
        vec3 L = normalize(-light.dir_direction.xyz);
        vec3 H = normalize(L + V);

        float NdotL = max(dot(N, L), 0.0);
        float band  = floor(NdotL * float(TOON_BANDS)) / float(TOON_BANDS);

        float NdotH = max(dot(N, H), 0.0);
        float spec  = (NdotH > SPECULAR_CUTOFF) ? 1.0 : 0.0;

        vec4 frag_light_space = light.dir_light_space_matrix * vec4(frag_world_pos, 1.0);
        float shadow = calculate_dir_shadow(frag_light_space);

        vec3 dir_light_color = light.dir_color.xyz * light.dir_color.w;
        total_diffuse  += band * shadow * dir_light_color;
        total_specular += spec * shadow * dir_light_color * 0.6;
    }

    // --- 2. Point Lights ---
    uint num_points = min(light.light_counts.y, 16);
    for (uint i = 0u; i < num_points; ++i) {
        PointLightGPU pl = light.point_lights[i];
        vec3 light_pos = pl.position_range.xyz;
        float range    = pl.position_range.w;

        vec3 light_vec = light_pos - frag_world_pos;
        float dist     = length(light_vec);
        if (dist > range) continue;

        vec3 L = normalize(light_vec);
        vec3 H = normalize(L + V);

        float NdotL = max(dot(N, L), 0.0);
        float band  = floor(NdotL * float(TOON_BANDS)) / float(TOON_BANDS);

        float NdotH = max(dot(N, H), 0.0);
        float spec  = (NdotH > SPECULAR_CUTOFF) ? 1.0 : 0.0;

        // Attenuation
        float att = 1.0 / (pl.attenuation.x + pl.attenuation.y * dist + pl.attenuation.z * (dist * dist));
        // Soft edge cutoff at max range
        float range_factor = clamp(1.0 - (dist / range), 0.0, 1.0);
        att *= range_factor;

        float shadow = 1.0;
        if (pl.attenuation.w > 0.5) { // cast_shadows enabled
            shadow = calculate_point_shadow(light_pos, range);
        }

        vec3 pt_color = pl.color_intensity.xyz * pl.color_intensity.w;
        total_diffuse  += band * att * shadow * pt_color;
        total_specular += spec * att * shadow * pt_color * 0.6;
    }

    vec3 ambient_color = light.dir_ambient.xyz;
    vec3 final_color   = base_color * (ambient_color + total_diffuse) + total_specular;

    out_color = vec4(final_color, 1.0);
}
