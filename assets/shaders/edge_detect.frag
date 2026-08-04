#version 450

// edge_detect.frag — Depth-based edge detection (post-process pass).
//
// Applies a Sobel operator on the depth buffer to detect depth discontinuities
// (silhouettes and edges between surfaces). Detected edges are composited as
// dark outlines over the source color image.
//
// This catches interior edges that the inverted-hull technique misses
// (e.g. edges between two touching objects or concave surfaces).

layout(location = 0) in vec2 frag_uv;

// Source color image (from the offscreen toon+outline pass).
layout(set = 0, binding = 0) uniform sampler2D color_image;
// Depth image from the offscreen pass.
layout(set = 0, binding = 1) uniform sampler2D depth_image;

layout(location = 0) out vec4 out_color;

// Push constant: edge detection threshold.
layout(push_constant) uniform EdgePC {
    float edge_threshold; // Depth difference threshold (default ~0.005)
    float edge_strength;  // How dark the edge is (0 = off, 1 = full black)
} edge_pc;

float sample_depth(vec2 uv) {
    return texture(depth_image, uv).r;
}

void main() {
    vec2 texel_size = 1.0 / vec2(textureSize(color_image, 0));

    // Sobel kernel samples (3x3 neighbourhood).
    float d00 = sample_depth(frag_uv + texel_size * vec2(-1.0, -1.0));
    float d10 = sample_depth(frag_uv + texel_size * vec2( 0.0, -1.0));
    float d20 = sample_depth(frag_uv + texel_size * vec2( 1.0, -1.0));
    float d01 = sample_depth(frag_uv + texel_size * vec2(-1.0,  0.0));
    float d21 = sample_depth(frag_uv + texel_size * vec2( 1.0,  0.0));
    float d02 = sample_depth(frag_uv + texel_size * vec2(-1.0,  1.0));
    float d12 = sample_depth(frag_uv + texel_size * vec2( 0.0,  1.0));
    float d22 = sample_depth(frag_uv + texel_size * vec2( 1.0,  1.0));

    // Sobel in X and Y.
    float gx = -d00 - 2.0 * d01 - d02 + d20 + 2.0 * d21 + d22;
    float gy = -d00 - 2.0 * d10 - d20 + d02 + 2.0 * d12 + d22;
    float edge = sqrt(gx * gx + gy * gy);

    // Threshold and composite.
    float edge_mask = step(edge_pc.edge_threshold, edge);
    vec4  src_color = texture(color_image, frag_uv);

    // Darken the source color by edge_strength where edges are detected.
    vec3 final_color = src_color.rgb * (1.0 - edge_mask * edge_pc.edge_strength);
    out_color = vec4(final_color, src_color.a);
}
