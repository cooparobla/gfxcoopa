#ifndef GFX_SHADOW_SAMPLING_GLSL
#define GFX_SHADOW_SAMPLING_GLSL

// gfx/shadow_sampling.glsl -- shared rotated-tap PCF kernels for directional
// and point/spot shadow maps.
//
// Declares no uniforms or samplers (same rule as ibl.glsl/ssr_common.glsl):
// every shadow map is passed in as a parameter, so callers can bind it at
// whatever set/binding index suits them, and (for point lights, which use
// several separately-named cube samplers rather than a dynamically-indexed
// array -- see ibl.glsl's identical constraint on reflection probes) pick
// which sampler to pass in before calling.
//
// Every kernel takes its direction/angle/bias arguments already derived,
// rather than deriving them itself, so adopting this file from an existing
// inline implementation is a pure code-motion: pass in the same expressions
// that were computed inline before and the result is bit-identical. In
// particular the two apps that use this seed their rotation angle from
// different (sign-flipped) vectors today; gfx_random_angle()/
// gfx_random_angle_3d() are offered as shared hash helpers, but callers may
// use their own seed expression and pass the resulting angle straight into
// the PCF kernels below.
//
// Two families of kernel are declared: a `sampler2D`/`samplerCube` family
// (the original, doing its own manual depth compare against a plain
// R32_SFLOAT-style filtered read) and a `sampler2DShadow`/`samplerCubeShadow`
// family, which requires the sampler bound at that binding to have been
// created with hardware compareEnable (see util::Sampler::shadow()). Prefer
// the *Shadow family for anything user-facing: the GPU compares each tap
// BEFORE bilinearly filtering, so a single texture() call already averages a
// 2x2 neighbourhood of pass/fail results instead of returning one binary
// value per tap -- see each *_shadow() kernel's doc below for why this
// matters for how noisy the softened penumbra looks. The plain family is
// kept only for callers not yet migrated to a compare-enabled sampler.

const vec2 GFX_POISSON_DISK_16[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648370),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090160),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

// Fixed cube-corner offset directions for cubemap PCF -- a standard
// technique for softening cube shadow-map edges. Rotating this pattern per
// shadow point (gfx_shadow_cube_pcf) turns the 9 discrete shadow values 8
// fixed unrotated taps would otherwise produce into noise, which reads as
// soft instead of banded.
const vec3 GFX_CUBE_PCF_OFFSETS_8[8] = vec3[](
    vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
    vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1)
);

float gfx_random_angle(vec2 seed) {
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453) * 6.28318530718;
}

float gfx_random_angle_3d(vec3 seed) {
    return fract(sin(dot(seed, vec3(12.9898, 78.233, 37.719))) * 43758.5453) * 6.28318530718;
}

/// Interleaved gradient noise (Jimenez 2014), as a rotation angle in [0, 2pi).
/// Unlike gfx_random_angle(), whose fract(sin(...)) hash is uncorrelated between
/// neighbouring pixels, IGN's coefficients are chosen so the output forms a
/// smooth ramp across any 3x5 tile of pixels. The per-tap PCF error this
/// rotation produces therefore reads as fine, stable dither rather than white
/// grain -- important because (unlike gfx_random_angle's use in AO/SSR, which
/// get denoised by a dedicated temporal resolve pass) a shadow's rotation
/// angle is read straight into the final shaded color with nothing after it
/// to average the noise back out.
float gfx_ign_angle(vec2 px) {
    return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715)))) * 6.28318530718;
}

/// Rodrigues' rotation formula: rotates v by `angle` radians around unit axis `axis`.
vec3 gfx_rotate_around_axis(vec3 v, vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

/// Single hard compare against a directional/spot 2D depth map. `proj_coords`
/// is the light-space position after the perspective divide and [0,1] remap
/// (the caller has already early-out'd on out-of-bounds/behind-map).
float gfx_shadow_dir_hard(sampler2D map, vec3 proj_coords, float bias) {
    float closest_depth = texture(map, proj_coords.xy).r;
    return (proj_coords.z - bias > closest_depth) ? 1.0 : 0.0;
}

/// 16-tap rotated-Poisson-disk PCF against a directional/spot 2D depth map.
/// `texel_size` is the (already radius-scaled) sample offset in UV space;
/// `rotation_angle` is normally gfx_random_angle(gl_FragCoord.xy).
float gfx_shadow_dir_pcf(sampler2D map, vec3 proj_coords, float bias,
                         vec2 texel_size, float rotation_angle) {
    float cos_a = cos(rotation_angle);
    float sin_a = sin(rotation_angle);
    mat2 rot = mat2(cos_a, -sin_a, sin_a, cos_a);

    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 offset = rot * GFX_POISSON_DISK_16[i] * texel_size;
        float pcf_depth = texture(map, proj_coords.xy + offset).r;
        shadow += (proj_coords.z - bias > pcf_depth) ? 1.0 : 0.0;
    }
    return shadow / 16.0;
}

/// Single hard compare against a point/spot light's depth cube. `dir` is the
/// (already normalized) sample direction the cube was rendered looking
/// outward along; `current_dist`/`bias` are in the cube's own
/// [0,1]-normalized-by-range depth space (see shadow_cube.frag).
float gfx_shadow_cube_hard(samplerCube map, vec3 dir, float current_dist, float bias) {
    float sampled_depth = texture(map, dir).r;
    return (current_dist - bias > sampled_depth) ? 1.0 : 0.0;
}

/// 8-tap Rodrigues-rotated cube-corner PCF against a point/spot light's depth
/// cube. `rotation_angle` is normally gfx_random_angle_3d() of whatever
/// per-shadow-point vector the caller already has on hand.
float gfx_shadow_cube_pcf(samplerCube map, vec3 dir, float current_dist, float bias,
                          float disk_radius, float rotation_angle) {
    float shadow = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec3 offset = gfx_rotate_around_axis(GFX_CUBE_PCF_OFFSETS_8[i], dir, rotation_angle);
        vec3 sample_dir = dir + offset * disk_radius;
        float sampled_depth = texture(map, sample_dir).r;
        shadow += (current_dist - bias > sampled_depth) ? 1.0 : 0.0;
    }
    return shadow / 8.0;
}

// --- Hardware depth-compare overloads (sampler2DShadow/samplerCubeShadow) ---
//
// Same names, same argument order, same 0=lit/1=in-shadow convention as the
// plain family above -- only the sampler type (and therefore what the driver
// does per tap) changes, so a caller switches families by re-binding its
// descriptor to a util::Sampler::shadow() sampler and changing its GLSL
// binding's type, with no call-site edits.
//
// The compare op the sampler was created with (VK_COMPARE_OP_GREATER, see
// util::Sampler::shadow()) computes exactly `ref > stored`, matching the
// manual `proj_coords.z - bias > pcf_depth` compare the plain family does
// itself -- so `ref` here is always `depth - bias`, never plain `depth`.
// Critically, hardware compare happens BEFORE bilinear filtering: one
// texture() call already blends the 2x2-neighbourhood compare result, so
// each of these taps carries ~4x the effective samples of its plain-family
// counterpart at the same tap count -- this is what actually fixes the
// grain, not just a rename.

/// Single hard compare against a directional/spot 2D depth map, hardware PCF.
float gfx_shadow_dir_hard(sampler2DShadow map, vec3 proj_coords, float bias) {
    return texture(map, vec3(proj_coords.xy, proj_coords.z - bias));
}

/// 16-tap rotated-Poisson-disk PCF against a directional/spot 2D depth map,
/// hardware-compare taps (see the family doc above).
float gfx_shadow_dir_pcf(sampler2DShadow map, vec3 proj_coords, float bias,
                         vec2 texel_size, float rotation_angle) {
    float cos_a = cos(rotation_angle);
    float sin_a = sin(rotation_angle);
    mat2 rot = mat2(cos_a, -sin_a, sin_a, cos_a);

    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 offset = rot * GFX_POISSON_DISK_16[i] * texel_size;
        shadow += texture(map, vec3(proj_coords.xy + offset, proj_coords.z - bias));
    }
    return shadow / 16.0;
}

/// Rotated Vogel-disk PCF against a directional/spot 2D depth map, hardware-
/// compare taps, with a caller-chosen tap count instead of a fixed 16.
///
/// A Vogel spiral (radius_i = sqrt((i+0.5)/n), angle_i = i * golden_angle) is
/// stratified by construction -- consecutive taps never cluster the way a
/// fixed Poisson set can when undersampled -- so it degrades gracefully as
/// `sample_count` is lowered to fit a large penumbra into a fixed budget
/// (see pixel_render_pipeline.h's dir_pcf_radius_texels: the pattern this
/// exists to replace, GFX_POISSON_DISK_16, was tuned for a ~2-3 texel radius
/// and thins out badly once shadow_softness pushes the radius into the tens
/// of texels). `sample_count` must not exceed 32.
float gfx_shadow_dir_pcf_vogel(sampler2DShadow map, vec3 proj_coords, float bias,
                               vec2 texel_size, float rotation_angle, int sample_count) {
    const float GOLDEN_ANGLE = 2.39996323;
    float shadow = 0.0;
    for (int i = 0; i < 32; ++i) {
        if (i >= sample_count) break;
        float r = sqrt((float(i) + 0.5) / float(sample_count));
        float a = float(i) * GOLDEN_ANGLE + rotation_angle;
        vec2 offset = r * vec2(cos(a), sin(a)) * texel_size;
        shadow += texture(map, vec3(proj_coords.xy + offset, proj_coords.z - bias));
    }
    return shadow / float(sample_count);
}

/// Single hard compare against a point/spot light's depth cube, hardware PCF.
float gfx_shadow_cube_hard(samplerCubeShadow map, vec3 dir, float current_dist, float bias) {
    return texture(map, vec4(dir, current_dist - bias));
}

/// 8-tap Rodrigues-rotated cube-corner PCF against a point/spot light's depth
/// cube, hardware-compare taps (see the family doc above).
///
/// CAUTION: `gfx_rotate_around_axis` rotates each `GFX_CUBE_PCF_OFFSETS_8`
/// corner AROUND `dir`, which by construction preserves that corner's
/// component ALONG `dir` -- a no-op for a direction-only cubemap lookup. So
/// each tap's effective spread ranges from 0 to `sqrt(3) * disk_radius`
/// depending purely on the angle between that corner and `dir`, and the
/// realized penumbra width is direction-dependent rather than a clean
/// function of `disk_radius`. Kept only for existing callers
/// (deferred_lighting.frag, transparent.frag, pbr.frag) that already tune
/// their radius around this behavior; new callers should use
/// gfx_shadow_cube_pcf_vogel below instead.
float gfx_shadow_cube_pcf(samplerCubeShadow map, vec3 dir, float current_dist, float bias,
                          float disk_radius, float rotation_angle) {
    float shadow = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec3 offset = gfx_rotate_around_axis(GFX_CUBE_PCF_OFFSETS_8[i], dir, rotation_angle);
        vec3 sample_dir = dir + offset * disk_radius;
        shadow += texture(map, vec4(sample_dir, current_dist - bias));
    }
    return shadow / 8.0;
}

/// Rotated Vogel-disk PCF against a point/spot light's depth cube,
/// hardware-compare taps, with a caller-chosen tap count instead of a fixed 8.
///
/// Unlike gfx_shadow_cube_pcf above, taps are placed in the tangent PLANE
/// perpendicular to `dir` (a `tx`/`ty` basis built off `dir`, the same shape
/// as a normal-mapping TBN), on the same Vogel spiral
/// (radius_i = sqrt((i+0.5)/n), angle_i = i * golden_angle) that
/// gfx_shadow_dir_pcf_vogel uses. `disk_radius` is therefore EXACTLY the
/// tangent-space offset applied to a unit `dir` -- no sqrt(3) inflation, no
/// direction-dependent collapse -- so it degrades gracefully with
/// `sample_count` the same way the directional Vogel kernel does.
///
/// `disk_radius` is in the same unit as `dir`: a tangent offset on the unit
/// sphere. To express it as a cube-face texel count, note a cube face spans
/// [-1, 1] in face-local tangent coordinates across `resolution` texels, so
/// texels -> tangent offset is `texels * (2.0 / resolution)`. `sample_count`
/// must not exceed 32.
float gfx_shadow_cube_pcf_vogel(samplerCubeShadow map, vec3 dir, float current_dist, float bias,
                                float disk_radius, float rotation_angle, int sample_count) {
    const float GOLDEN_ANGLE = 2.39996323;
    vec3 up = abs(dir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tx = normalize(cross(up, dir));
    vec3 ty = cross(dir, tx);

    float shadow = 0.0;
    for (int i = 0; i < 32; ++i) {
        if (i >= sample_count) break;
        float r = sqrt((float(i) + 0.5) / float(sample_count));
        float a = float(i) * GOLDEN_ANGLE + rotation_angle;
        vec3 offset = (tx * cos(a) + ty * sin(a)) * (r * disk_radius);
        shadow += texture(map, vec4(dir + offset, current_dist - bias));
    }
    return shadow / float(sample_count);
}

#endif // GFX_SHADOW_SAMPLING_GLSL
