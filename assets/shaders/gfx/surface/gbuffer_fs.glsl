#ifndef GFX_SURFACE_GBUFFER_FS_GLSL
#define GFX_SURFACE_GBUFFER_FS_GLSL

// gfx/surface/gbuffer_fs.glsl -- opaque/mask G-Buffer fragment backbone.
// See gbuffer_vs.glsl for the include-order contract; GFX_SURFACE_FRAGMENT
// gates gfx_surface_fragment() the same way GFX_SURFACE_VERTEX gates
// gfx_surface_vertex() there. The backbone -- not the hook -- owns the
// CUTOUT alpha test and the MRT writeout, so every derived shader keeps
// exactly one alpha-test policy and one G-Buffer layout.

layout(location = 0) in vec3 frag_world_pos;
layout(location = 1) in vec3 frag_world_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in mat3 frag_TBN;

layout(push_constant) uniform PushConstants {
    vec4  albedo;     // xyz = albedo, w = alpha
    float metallic;
    float roughness;
    float ao;
    float alpha_cutoff; // 0.0 disables the alpha test below
    vec4  emissive;     // xyz = pre-multiplied emissive radiance, w reserved
    vec4  gfx_time;     // x=time, y=delta_time, z=frame_index, w=spare
    vec4  gfx_params;   // four author-defined floats; see the surface shader's own doc
} material;

// See gbuffer_vs.glsl's identical aliases for why these exist: a surface file's
// gfx_surface_fragment() reads gfx_time/gfx_params without knowing this backbone's push
// block is named `material` rather than `pc` (shadow_fs.glsl/shadow_cube_fs.glsl's name).
vec4 gfx_time   = material.gfx_time;
vec4 gfx_params = material.gfx_params;

// Set 1: material texture(s) -- currently just the CUTOUT alpha mask. A consumer that passes
// GBufferPipeline a non-null material_layout must bind a combined sampler here (a 1x1 white
// fallback for every non-masked material -- see toyengine's MaterialTextureCache -- so
// texture(...).a == 1.0 and the test below collapses back to the plain constant-alpha MASK test
// it replaces). A consumer that never passes a material_layout never reaches set 1 at all.
layout(set = 1, binding = 0) uniform sampler2D u_alpha_mask;

// G-Buffer Render Targets
layout(location = 0) out vec4 out_albedo_ao;          // RGB = Albedo, A = AO
layout(location = 1) out vec4 out_normal_metallic;    // RGB = World Normal, A = Metallic
layout(location = 2) out vec4 out_position_roughness; // RGB = World Pos, A = Roughness
layout(location = 3) out vec4 out_emissive;           // RGB = emissive radiance (HDR), A = unused

/// What a fragment-shading hook receives and may edit, seeded from the
/// material push block and the interpolated vertex outputs. Editing
/// normal_ws re-lights the surface (e.g. animated ripple normals); the
/// backbone still runs the cutout test before this struct is built and
/// owns the MRT writeout after the hook returns.
struct GfxSurface {
    vec3  albedo;
    float metallic;
    float roughness;
    float ao;
    vec3  emissive;
    vec3  normal_ws;
    vec3  position_ws;
    vec2  uv;
};

#ifdef GFX_SURFACE_FRAGMENT
void gfx_surface_fragment(inout GfxSurface s);
#else
void gfx_surface_fragment(inout GfxSurface s) {}
#endif

void main() {
    // CUTOUT/MASK materials: alpha_cutoff > 0 arms the test. albedo.a (the constant per-material
    // alpha) multiplied by the sampled mask's alpha gives a real per-texel silhouette test when a
    // texture_alpha_mask is authored, and collapses to the old constant-only test otherwise.
    float alpha = material.albedo.a * texture(u_alpha_mask, frag_uv).a;
    if (material.alpha_cutoff > 0.0 && alpha < material.alpha_cutoff) discard;

    GfxSurface s;
    s.albedo      = material.albedo.rgb;
    s.metallic    = material.metallic;
    s.roughness   = max(material.roughness, 0.045);
    s.ao          = material.ao;
    s.emissive    = material.emissive.rgb;
    s.normal_ws   = normalize(frag_world_normal);
    s.position_ws = frag_world_pos;
    s.uv          = frag_uv;

    gfx_surface_fragment(s);

    out_albedo_ao          = vec4(s.albedo, s.ao);
    out_normal_metallic    = vec4(s.normal_ws, s.metallic);
    out_position_roughness = vec4(s.position_ws, s.roughness);
    out_emissive            = vec4(s.emissive, 0.0);
}

#endif // GFX_SURFACE_GBUFFER_FS_GLSL
