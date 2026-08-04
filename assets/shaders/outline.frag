#version 450

// outline.frag — Inverted-hull outline fragment shader.
// Outputs the outline color from push constants.
// outline_params: xyz = color, w = extrusion width (not used here).

layout(push_constant) uniform OutlinePC {
    layout(offset = 128) vec4 outline_params; // xyz=color, w=width
} outline_pc;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(outline_pc.outline_params.xyz, 1.0);
}
