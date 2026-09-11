#ifndef GFX_DEPTH_GLSL
#define GFX_DEPTH_GLSL

// gfx/depth.glsl -- shared depth-linearization helper, hoisted out of
// pixel_stylize.frag (where it started life as a local `linear_depth()`) so
// dof_coc.frag/dof_composite.frag can share the exact same formula rather than
// carrying their own byte-identical copy. See toyengine's pixel_shadow_body.glsl
// doc for this codebase's general policy on de-duplicating shared shader bodies.

// True view-space distance from the camera, from raw Vulkan [0,1] post-projection
// depth. Perspective depth is hyperbolic (glm::perspective -> perspectiveRH_ZO);
// orthographic depth is already linear in the raw value (glm::ortho ->
// orthoRH_ZO), hence the branch.
float gfx_linear_depth(float d, float near_z, float far_z, float is_perspective) {
    if (is_perspective < 0.5) {
        return mix(near_z, far_z, d);
    }
    return near_z * far_z / (far_z - d * (far_z - near_z));
}

#endif // GFX_DEPTH_GLSL
