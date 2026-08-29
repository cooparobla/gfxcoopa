#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D u_src;

layout(push_constant) uniform PushConstants {
    ivec2 src_size;
    int   is_first_pass;
} u_push;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);

    if (u_push.is_first_pass != 0) {
        // Mip 0 is a straight copy of the deferred-lit HDR scene colour. gl_FragCoord.xy is a
        // 1:1 texel index here regardless of the source having been rendered with a negative-
        // height viewport, so this is an exact identity copy and the mip chain shares
        // ssr.frag's UV convention for free.
        out_color = texelFetch(u_src, min(c, u_push.src_size - 1), 0);
        return;
    }

    // 2x2 box average of the previous mip. Deliberately NOT hiz_downsample.frag's conservative
    // 3x3 tap: dropping the last row/column at an odd source size costs a sliver of blur
    // accuracy here, whereas dropping it from a MIN reduction would break the Hi-Z pyramid's
    // conservativeness outright.
    //
    // Coverage-weighted by alpha, not a plain vec4 average: this chain also serves
    // TransparentCapturePass's colour (see SceneColorMipPass's doc), which is cleared to
    // (0,0,0,0) everywhere except the small area actual transparent geometry covers, with
    // covered pixels written at alpha=1 (see transparent_capture.frag). An unweighted average
    // mixes real colour with cleared-black at every mip, roughly halving a small object's
    // effective radiance per mip level -- by mip 6 (SceneColorMipPass::kMaxMips - 1) that is
    // over a 1/16 dilution, enough to invert an otherwise-valid SSR composite delta from
    // slightly positive to slightly negative and make a real reflection vanish. Weighting by
    // alpha keeps the colour of a covered mip texel representative of only the geometry that's
    // actually there. For the OPAQUE scene-colour chain (this same shader's other caller) every
    // source texel already has alpha=1 (skybox.frag:37, pixel_lighting.frag:260, both literal),
    // so w0..w3 are always 1 here and this is an exact no-op vs. the old plain average.
    ivec2 b = c * 2;
    vec4 s0 = texelFetch(u_src, min(b + ivec2(0, 0), u_push.src_size - 1), 0);
    vec4 s1 = texelFetch(u_src, min(b + ivec2(1, 0), u_push.src_size - 1), 0);
    vec4 s2 = texelFetch(u_src, min(b + ivec2(0, 1), u_push.src_size - 1), 0);
    vec4 s3 = texelFetch(u_src, min(b + ivec2(1, 1), u_push.src_size - 1), 0);

    float wsum = s0.a + s1.a + s2.a + s3.a;
    vec3 rgb = (s0.rgb * s0.a + s1.rgb * s1.a + s2.rgb * s2.a + s3.rgb * s3.a) / max(wsum, 1e-5);
    out_color = vec4(rgb, wsum * 0.25);
}
