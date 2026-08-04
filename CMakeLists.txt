cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.0)

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

# Add executable
add_executable(gfxcoopa test.cpp)
target_link_libraries(gfxcoopa Vulkan::Vulkan glfw dl)
