cmake_minimum_required(VERSION 3.20)
cmake_policy(VERSION 3.20)

project(gfxcoopa)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(ROOT_DIR_PARENT "${CMAKE_SOURCE_DIR}" DIRECTORY)

# Vulkan headers only — volk provides the loader at runtime
find_package(Vulkan REQUIRED)

# Window system (GLFW)
find_package(glfw3 REQUIRED)

# --- Sibling library ---
# engine/components/ includes <coopa/scene/...> and <coopa/asset/...>, so
# gfxcoopa depends on libcoopa's coopa::lib target for those headers plus
# glm/fkYAML/phmap.
set(LIBCOOPA_DIR "${ROOT_DIR_PARENT}/libcoopa")
if(NOT TARGET coopa::lib)
    add_subdirectory(${LIBCOOPA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/libcoopa-build)
endif()

# Configure and include root directory header
configure_file(configuration/root_directory.h.in configuration/root_directory.h)

# --- Shader compilation ---
# gfxcoopa's own assets/shaders/ is the base shader library every other repo
# points -I at, so compiling it here (INCLUDE_DIR == SHADER_DIR) is both
# gfxcoopa's own shader build AND a continuously-run compile test that every
# base entry point still works standalone against the gfx/ headers it ships.
# INCLUDE_DIR defaults to gfxcoopa's own shaders (see cmake/GfxShaders.cmake),
# so a downstream consumer does not pass it explicitly.
include(cmake/GfxShaders.cmake)
gfx_add_shader_target(shaders
    SHADER_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders"
)

# --- gfxcoopa_impl: the ONE translation unit in the whole dependency graph
# that compiles VOLK_IMPLEMENTATION / VMA_IMPLEMENTATION / the two stb
# implementation blocks. See src/gfx_impl.cpp for why this must be singular.
add_library(gfxcoopa_impl STATIC src/gfx_impl.cpp)
target_compile_features(gfxcoopa_impl PRIVATE cxx_std_20)
target_compile_definitions(gfxcoopa_impl PRIVATE
    VOLK_IMPLEMENTATION
    VMA_IMPLEMENTATION
    STB_IMAGE_IMPLEMENTATION
    STB_IMAGE_WRITE_IMPLEMENTATION)
target_include_directories(gfxcoopa_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes)
# glfw must be PUBLIC here (not PRIVATE): Window's inline glfw*() calls
# compile into *consumer* TUs via the header-only gfxcoopa target below, so
# glfw symbols must resolve at the consumer's link step too. This leaks the
# glfw *target name* into a consumer's link line but not the glfw *API* --
# consumers still call zero find_package(glfw3) and name zero GLFW symbols.
# Making this PRIVATE would require pimpl'ing Window, at the cost of a .cpp,
# a heap allocation and lost inlining, for no seal benefit beyond what the
# header/type seal and the leak-check script already provide.
target_link_libraries(gfxcoopa_impl PUBLIC glfw Vulkan::Headers)
if(APPLE)
    # Window::set_icon() sets the Dock icon through the Objective-C runtime (AppKit).
    target_link_libraries(gfxcoopa_impl PUBLIC "-framework AppKit" objc)
endif()

# --- coopa::gfx: the public, header-only consumable target. ---
# Named gfxcoopa_lib (not `gfxcoopa`) because `cplay` resolves the run
# binary as `./build/<project name>` -- the executable below must keep the
# bare `gfxcoopa` target name, matching project(gfxcoopa) above.
add_library(gfxcoopa_lib INTERFACE)
add_library(coopa::gfx ALIAS gfxcoopa_lib)
target_compile_features(gfxcoopa_lib INTERFACE cxx_std_20)

# Location of the official Jimenez SMAA reference implementation's Textures/ directory
# (SearchTex.h / AreaTex.h), consumed by gfxcoopa/engine/util/smaa_textures.h. Overridable
# with -DSMAA_TEXTURES_DIR=... on a machine where the SMAA repo lives somewhere else; not
# vendored into gfxcoopa because AreaTex.h alone is a 1.1 MB generated C array.
set(SMAA_TEXTURES_DIR "/home/coopa/third-party/smaa/Textures" CACHE PATH
    "Directory containing the SMAA reference implementation's SearchTex.h/AreaTex.h")

target_include_directories(gfxcoopa_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}                    # gfxcoopa/...
    ${CMAKE_CURRENT_SOURCE_DIR}/includes           # volk/, vma/, stb/
    ${CMAKE_CURRENT_BINARY_DIR}/configuration
    ${SMAA_TEXTURES_DIR})
target_link_libraries(gfxcoopa_lib INTERFACE gfxcoopa_impl coopa::lib ${CMAKE_DL_LIBS})

# meshoptimizer (MIT): vertex welding, vertex-cache ordering and LOD simplification for
# Mesh's load path (gfxcoopa/engine/data/mesh_build.h). A compiled library rather than
# header-only, so it is fetched and built here and linked through the interface target --
# consumers include <meshoptimizer.h> only via mesh_build.h. Pinned to a release tag.
if(NOT TARGET meshoptimizer)
    include(FetchContent)
    FetchContent_Declare(meshoptimizer
        GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
        GIT_TAG        v0.25
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(meshoptimizer)
endif()
target_link_libraries(gfxcoopa_lib INTERFACE meshoptimizer)

# --- coopa::gfx_pure: types/ only, with NO Vulkan/GLFW include path
# available at all. A stray `#include <gfxcoopa/memory/buffer.h>` in a
# target linked against this fails to compile -- this is what makes the
# "headless tests never touch Vulkan" invariant structural rather than
# aspirational (see tools/check_no_vulkan.sh for the complementary grep gate
# over source text). Keyboard/mouse input is Vulkan-free for the same reason
# but lives in coopa/input/ in libcoopa, which coopa::lib below brings in --
# see coopa/input/README.md.
add_library(gfxcoopa_pure INTERFACE)
add_library(coopa::gfx_pure ALIAS gfxcoopa_pure)
target_compile_features(gfxcoopa_pure INTERFACE cxx_std_20)
target_include_directories(gfxcoopa_pure INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gfxcoopa_pure INTERFACE coopa::lib)

# --- gfxcoopa's own dogfood/test executable. Must be named `gfxcoopa`,
# matching project() above, because `cplay` runs `./build/<project name>`. ---
add_executable(gfxcoopa test.cpp)
add_dependencies(gfxcoopa shaders)
target_link_libraries(gfxcoopa PRIVATE coopa::gfx)

enable_testing()
# WORKING_DIRECTORY: test.cpp resolves shader paths relative to the repo
# root (matching how `cplay` invokes the binary, cwd=repo root), not
# CTest's default of CMAKE_CURRENT_BINARY_DIR.
add_test(NAME gfxcoopa_render COMMAND gfxcoopa WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
