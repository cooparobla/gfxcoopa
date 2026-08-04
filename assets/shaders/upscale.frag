#version 450

// upscale.frag — Nearest-neighbour upscale fragment shader.
//
// Samples the low-resolution offscreen render target using the nearest-neighbour
// sampler (configured in C++ as VK_FILTER_NEAREST) to produce the hard-pixel
// retro aesthetic. No filtering, no blurring — raw pixel colours.

layout(location = 0) in vec2 frag_uv;

// The low-res offscreen color image (sampled with VK_FILTER_NEAREST on C++ side).
layout(set = 0, binding = 0) uniform sampler2D low_res_image;

layout(location = 0) out vec4 out_color;

void main() {
    // UV is in [0, 1]. Clamp to avoid any edge bleeding.
    vec2 uv     = clamp(frag_uv, vec2(0.0), vec2(1.0));
    out_color   = texture(low_res_image, uv);
}
