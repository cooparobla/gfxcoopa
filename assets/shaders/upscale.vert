#version 450

// upscale.vert — Fullscreen triangle vertex shader (no vertex buffer).
//
// Standard fullscreen triangle trick: three vertices with hard-coded
// clip-space positions and UVs covering the entire screen.
// The vertex index drives the position calculation.

layout(location = 0) out vec2 frag_uv;

void main() {
    // Generate a triangle that covers the full NDC screen [-1, 1]:
    //   vertex 0: bottom-left  (-1, -1, uv 0,0)
    //   vertex 1: bottom-right ( 3, -1, uv 2,0)
    //   vertex 2: top-left     (-1,  3, uv 0,2)
    // The overshoot clips naturally. This avoids a vertex buffer entirely.
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 uvs[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 2.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    frag_uv     = uvs[gl_VertexIndex];
}
