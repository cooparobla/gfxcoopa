#version 450

// Stock opaque/mask G-Buffer vertex shader -- identity displacement. See
// gfx/surface/gbuffer_vs.glsl for the backbone this includes and the
// include-order contract a derived shader (e.g. toyengine's foliage) must
// follow to override displacement while keeping everything else.
#include <gfx/surface/gbuffer_vs.glsl>
