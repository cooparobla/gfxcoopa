#version 450

layout(location = 0) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform sampler2D tex_scene;
layout(set = 0, binding = 1) uniform sampler2D tex_history;

layout(push_constant) uniform PushConstants {
    vec2  resolution;    // width, height
    float blend_factor;  // temporal blend weight toward history (0.9 default)
    float weight_scale;  // reprojection weight scale (30.0 default, maps to SMAA_REPROJECTION_WEIGHT_SCALE)
} pc;

layout(location = 0) out vec4 out_color;

// ---------------------------------------------------------------------------
// Color-space conversions: RGB <-> YCoCg
// YCoCg is more perceptually uniform than RGB for neighborhood clamping,
// following the industry-standard TAA practice (UE4, CryEngine, etc.)
// ---------------------------------------------------------------------------

vec3 RGB_to_YCoCg(vec3 rgb) {
    return vec3(
         0.25 * rgb.r + 0.50 * rgb.g + 0.25 * rgb.b,   // Y  (luma)
         0.50 * rgb.r                 - 0.50 * rgb.b,    // Co (chroma orange)
        -0.25 * rgb.r + 0.50 * rgb.g - 0.25 * rgb.b     // Cg (chroma green)
    );
}

vec3 YCoCg_to_RGB(vec3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return vec3(
        Y + Co - Cg,
        Y      + Cg,
        Y - Co - Cg
    );
}

void main() {
    vec2 texel_size = 1.0 / pc.resolution;

    // -----------------------------------------------------------------------
    // Step 1: Sample current frame center + 3x3 neighborhood
    // -----------------------------------------------------------------------
    vec3 current_rgb = texture(tex_scene, frag_uv).rgb;

    // 3x3 neighborhood samples (current frame) for AABB computation
    vec3 s0 = texture(tex_scene, frag_uv + vec2(-1.0,  1.0) * texel_size).rgb;
    vec3 s1 = texture(tex_scene, frag_uv + vec2( 0.0,  1.0) * texel_size).rgb;
    vec3 s2 = texture(tex_scene, frag_uv + vec2( 1.0,  1.0) * texel_size).rgb;
    vec3 s3 = texture(tex_scene, frag_uv + vec2(-1.0,  0.0) * texel_size).rgb;
    vec3 s4 = current_rgb; // center
    vec3 s5 = texture(tex_scene, frag_uv + vec2( 1.0,  0.0) * texel_size).rgb;
    vec3 s6 = texture(tex_scene, frag_uv + vec2(-1.0, -1.0) * texel_size).rgb;
    vec3 s7 = texture(tex_scene, frag_uv + vec2( 0.0, -1.0) * texel_size).rgb;
    vec3 s8 = texture(tex_scene, frag_uv + vec2( 1.0, -1.0) * texel_size).rgb;

    // -----------------------------------------------------------------------
    // Step 2: Compute neighborhood AABB in YCoCg space
    // -----------------------------------------------------------------------
    vec3 y0 = RGB_to_YCoCg(s0);
    vec3 y1 = RGB_to_YCoCg(s1);
    vec3 y2 = RGB_to_YCoCg(s2);
    vec3 y3 = RGB_to_YCoCg(s3);
    vec3 y4 = RGB_to_YCoCg(s4);
    vec3 y5 = RGB_to_YCoCg(s5);
    vec3 y6 = RGB_to_YCoCg(s6);
    vec3 y7 = RGB_to_YCoCg(s7);
    vec3 y8 = RGB_to_YCoCg(s8);

    vec3 aabb_min = min(y0, min(y1, min(y2, min(y3, min(y4, min(y5, min(y6, min(y7, y8))))))));
    vec3 aabb_max = max(y0, max(y1, max(y2, max(y3, max(y4, max(y5, max(y6, max(y7, y8))))))));

    // -----------------------------------------------------------------------
    // Step 3: Sample history at frag_uv (no velocity — static reprojection)
    // This matches the smaa-cpp reference non-reprojection path:
    //   previousColorImage->getPixel(x, y, previous);
    // -----------------------------------------------------------------------
    vec3 history_rgb = texture(tex_history, frag_uv).rgb;

    // -----------------------------------------------------------------------
    // Step 4: Clamp history to current neighborhood AABB in YCoCg
    // This prevents ghosting artifacts, serving the same purpose as the
    // velocity-based weight attenuation in the smaa-cpp reference:
    //   delta = abs(current.a^2 - previous.a^2) / 5.0
    //   weight = 0.5 * saturate(1.0 - sqrt(delta) * WEIGHT_SCALE)
    // -----------------------------------------------------------------------
    vec3 history_ycocg = RGB_to_YCoCg(history_rgb);
    vec3 clamped_ycocg = clamp(history_ycocg, aabb_min, aabb_max);
    vec3 clamped_history = YCoCg_to_RGB(clamped_ycocg);

    // -----------------------------------------------------------------------
    // Step 5: Blend current with clamped history
    // Reference: lerp(current, previous, weight) where weight ∈ [0, 0.5]
    // We use a configurable blend_factor (default 0.9) toward history for
    // maximum temporal stability, relying on the AABB clamp for ghosting.
    // -----------------------------------------------------------------------
    out_color = vec4(mix(current_rgb, clamped_history, pc.blend_factor), 1.0);
}
