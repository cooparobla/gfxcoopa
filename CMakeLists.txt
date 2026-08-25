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

# --- Sibling library paths (header-only, no build targets required) ---
# engine/components/ already includes <coopa/scene/...> and (as of the
# coopa::asset integration) <coopa/asset/...>; without this, gfxcoopa's own
# test target can't compile its own engine/ layer -- only a downstream app
# that separately adds these paths (blendy, uicoopa) could.
set(LIBCOOPA_DIR "${ROOT_DIR_PARENT}/libcoopa")
include_directories(${LIBCOOPA_DIR}/includes/)        # fkYAML, glm, phmap
include_directories(${LIBCOOPA_DIR})                  # coopa/

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/includes/)   # volk/, vma/
include_directories(${CMAKE_SOURCE_DIR})

# Configure and include root directory header
configure_file(configuration/root_directory.h.in configuration/root_directory.h)
include_directories(${CMAKE_BINARY_DIR}/configuration)

# volk implementation compiled once in this translation unit.
# All library headers use volk.h with VK_NO_PROTOTYPES already set by volk.
add_definitions(-DVOLK_IMPLEMENTATION)

# VMA implementation compiled once here.
add_definitions(-DVMA_IMPLEMENTATION)

# --- Shader compilation ---
# gfxcoopa's own assets/shaders/ is the base shader library every other repo
# points -I at, so compiling it here (INCLUDE_DIR == SHADER_DIR) is both
# gfxcoopa's own shader build AND a continuously-run compile test that every
# base entry point still works standalone against the gfx/ headers it ships.
include(cmake/GfxShaders.cmake)
gfx_add_shader_target(shaders
    SHADER_DIR   "${CMAKE_SOURCE_DIR}/assets/shaders"
    INCLUDE_DIR  "${CMAKE_SOURCE_DIR}/assets/shaders"
)

# Add executable
add_executable(gfxcoopa test.cpp)
add_dependencies(gfxcoopa shaders)
target_link_libraries(gfxcoopa Vulkan::Vulkan glfw dl)
