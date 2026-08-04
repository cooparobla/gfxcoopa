#version 450

// color_quantize.frag — Optional retro color palette quantization.
//
// Reduces the number of distinct colors to simulate the limited palette
// of retro hardware (e.g. Game Boy, NES, palette-constrained systems).
//
// Implementation: posterization — round each channel to N discrete levels.
// Toggle by setting color_levels = 0 (pass-through) or color_levels >= 2.

layout(location = 0) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform sampler2D color_image;

layout(location = 0) out vec4 out_color;

// Push constant.
layout(push_constant) uniform QuantizePC {
    int   color_levels;     // Number of discrete levels per channel (0 = disabled, typical: 8-16)
    float brightness;       // Overall brightness multiplier (default 1.0)
    float saturation;       // Saturation boost (default 1.0, >1 increases saturation)
} quantize_pc;

vec3 adjust_saturation(vec3 color, float sat) {
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luminance), color, sat);
}

void main() {
    vec4 src = texture(color_image, frag_uv);
    vec3 color = src.rgb * quantize_pc.brightness;

    // Saturation adjustment.
    color = adjust_saturation(color, quantize_pc.saturation);

    // Color quantization (posterization).
    if (quantize_pc.color_levels >= 2) {
        float levels = float(quantize_pc.color_levels);
        color = floor(color * levels) / (levels - 1.0);
    }

    out_color = vec4(clamp(color, 0.0, 1.0), src.a);
}
