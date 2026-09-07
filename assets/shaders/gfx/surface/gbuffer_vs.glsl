#ifndef GFX_SURFACE_GBUFFER_VS_GLSL
#define GFX_SURFACE_GBUFFER_VS_GLSL

// gfx/surface/gbuffer_vs.glsl -- opaque/mask G-Buffer vertex backbone.
//
// A derived shader (e.g. toyengine's foliage) overrides displacement by
// following this include order exactly (see gbuffer.vert for the canonical
// no-override example):
//
//   #version 450
//   #define GFX_SURFACE_VERTEX
//   #include <gfx/surface/gbuffer_vs.glsl>
//   #include "my_surface.glsl"   // defines gfx_surface_vertex()
//
// GFX_SURFACE_VERTEX must be #defined BEFORE this include -- that's what
// turns the hook below from a no-op default into a forward declaration the
// includer must satisfy. The file defining gfx_surface_vertex() must be
// #included AFTER this one, since it needs GfxSurfaceVertex in scope.
// Getting either backwards is a glslc redefinition/undeclared-identifier
// error at compile time, never a silent miscompile.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_tangent; // xyz = tangent, w = handedness
layout(location = 4) in mat4 in_model;   // per-instance (locations 4-7)

// Set 0: Camera UBO
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

// Shared with gbuffer_fs.glsl -- see that file for the fragment-stage-only
// fields (albedo/metallic/.../emissive are unread here but must stay
// byte-identical to the fragment stage's declaration; they share one
// VkPushConstantRange). gfx_time/gfx_params are the standard trailing
// "surface" block every gfxcoopa/toyengine surface backbone appends -- see
// the push-constant budget table in the layered-shaders plan for why it's
// fixed at 32 bytes (the point-light cube shadow pass is the tightest fit).
layout(push_constant) uniform PushConstants {
    vec4  albedo;
    float metallic;
    float roughness;
    float ao;
    float alpha_cutoff;
    vec4  emissive;
    vec4  gfx_time;    // x=time, y=delta_time, z=frame_index, w=spare
    vec4  gfx_params;  // four author-defined floats; see the surface shader's own doc
} material;

// Plain-named aliases so a surface file's gfx_surface_vertex()/gfx_surface_fragment() can
// read gfx_time/gfx_params without knowing which backbone it was compiled against -- the
// push block's own instance name differs per backbone (`material` here, `pc` in
// gfx/surface/shadow_vs.glsl and shadow_cube_vs.glsl), but every backbone assigns these two
// globals from it, so a hook body is portable across all of them. GLSL evaluates
// non-constant global initializers once per shader invocation, before main() runs, so this
// correctly reads each draw's own push-constant values rather than some shared/stale copy.
vec4 gfx_time   = material.gfx_time;
vec4 gfx_params = material.gfx_params;

// Outputs to G-Buffer fragment shader
layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_world_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out mat3 frag_TBN;

/// What a vertex-displacement hook receives and may edit. The _os fields are
/// as-authored (pre-model-matrix); everything else is already in world
/// space by the time the hook runs, since most displacement (wind, waves)
/// is easiest to reason about in world space. Edit position_ws/normal_ws/
/// tangent_ws in place; the backbone re-derives the TBN and clip position
/// from whatever the hook leaves behind.
struct GfxSurfaceVertex {
    vec3 position_os;
    vec3 normal_os;
    vec4 tangent_os;
    vec2 uv;
    mat4 model;
    mat3 normal_matrix;
    vec3 position_ws;
    vec3 normal_ws;
    vec3 tangent_ws;
};

#ifdef GFX_SURFACE_VERTEX
void gfx_surface_vertex(inout GfxSurfaceVertex v);
#else
void gfx_surface_vertex(inout GfxSurfaceVertex v) {}
#endif

void main() {
    GfxSurfaceVertex v;
    v.position_os = in_position;
    v.normal_os   = in_normal;
    v.tangent_os  = in_tangent;
    v.uv          = in_uv;
    v.model       = in_model;

    // normal_matrix used to be CPU-computed (glm::transpose(glm::inverse(world)))
    // and streamed alongside model; now derived here instead, since scenes use
    // non-uniform scale (e.g. Cornell box walls) so mat3(in_model) alone is wrong.
    v.normal_matrix = transpose(inverse(mat3(in_model)));

    vec4 world_pos = in_model * vec4(in_position, 1.0);
    v.position_ws = world_pos.xyz;
    v.normal_ws   = normalize(v.normal_matrix * in_normal);
    v.tangent_ws  = normalize(v.normal_matrix * in_tangent.xyz);

    gfx_surface_vertex(v);

    vec3 T = normalize(v.tangent_ws - dot(v.tangent_ws, v.normal_ws) * v.normal_ws);
    vec3 B = cross(v.normal_ws, T) * in_tangent.w;

    frag_world_pos    = v.position_ws;
    frag_world_normal = v.normal_ws;
    frag_uv           = v.uv;
    frag_TBN          = mat3(T, B, v.normal_ws);

    gl_Position = camera.proj * camera.view * vec4(v.position_ws, 1.0);
}

#endif // GFX_SURFACE_GBUFFER_VS_GLSL
