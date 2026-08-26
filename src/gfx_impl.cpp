/**
 * @file gfx_impl.cpp
 * @brief The single translation unit that compiles the vendored implementation
 * blocks (volk, VMA, stb_image, stb_image_write) for the entire dependency
 * graph.
 *
 * volk's loader and VMA's allocator body are single-header libraries that
 * must have their implementation compiled in EXACTLY ONE translation unit
 * per final binary. Previously every consumer application defined
 * VOLK_IMPLEMENTATION / VMA_IMPLEMENTATION (and the two stb macros) itself,
 * which meant four repos duplicated this file nearly verbatim. Now the
 * `gfxcoopa_impl` STATIC target owns it once, and consumers link against
 * `coopa::gfx` (an INTERFACE target depending on `gfxcoopa_impl`) without
 * ever naming these macros themselves.
 *
 * The macros themselves are supplied via target_compile_definitions() on
 * `gfxcoopa_impl` in CMakeLists.txt, not defined here, so this file stays a
 * pure list of includes.
 */

#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
