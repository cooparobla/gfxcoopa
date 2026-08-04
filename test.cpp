// volk and VMA implementations are compiled once via -DVOLK_IMPLEMENTATION
// and -DVMA_IMPLEMENTATION defined in CMakeLists.txt.
#include <volk/volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vma/vk_mem_alloc.h>

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>

// --- gfxcoopa headers ---
#include <gfxcoopa/util/error.h>
#include <gfxcoopa/util/volk_init.h>
#include <gfxcoopa/util/debug_messenger.h>
#include <gfxcoopa/util/format.h>
#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/presentation/window.h>
#include <gfxcoopa/core/surface.h>
#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/command/sync.h>
#include <gfxcoopa/presentation/renderer.h>

// ANSI colors — identical to libcoopa test.cpp
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_RESET   "\x1b[0m"

static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define RUN_TEST(test_func) \
    do { \
        std::cout << ANSI_COLOR_BLUE << "[ RUN      ] " << ANSI_COLOR_RESET << #test_func << std::endl; \
        g_tests_run++; \
        try { \
            test_func(); \
            std::cout << ANSI_COLOR_GREEN << "[       OK ] " << ANSI_COLOR_RESET << #test_func << std::endl; \
        } catch (const std::exception& e) { \
            std::cerr << ANSI_COLOR_RED << "[  FAILED  ] " << ANSI_COLOR_RESET << #test_func << " (Exception: " << e.what() << ")" << std::endl; \
            g_tests_failed++; \
        } catch (...) { \
            std::cerr << ANSI_COLOR_RED << "[  FAILED  ] " << ANSI_COLOR_RESET << #test_func << " (Unknown Exception)" << std::endl; \
            g_tests_failed++; \
        } \
    } while (0)

#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << ANSI_COLOR_RED << "  Assertion failed: " << #condition \
                      << " at " << __FILE__ << ":" << __LINE__ << ANSI_COLOR_RESET << std::endl; \
            throw std::runtime_error("Assertion failed: " #condition); \
        } \
    } while (0)

#define ASSERT_EQ(val1, val2) \
    do { \
        if ((val1) != (val2)) { \
            std::cerr << ANSI_COLOR_RED << "  Assertion failed: " << #val1 << " == " << #val2 \
                      << " (Actual: " << (val1) << ", Expected: " << (val2) << ") at " \
                      << __FILE__ << ":" << __LINE__ << ANSI_COLOR_RESET << std::endl; \
            throw std::runtime_error("Assertion failed: " #val1 " == " #val2); \
        } \
    } while (0)

// ==========================================================================
// Fixtures — shared Vulkan objects used across multiple tests.
// Created once at the start of main() and destroyed at the end.
// ==========================================================================

// Forward declarations.
static coopa::gfx::core::Instance*               g_instance   = nullptr;
static coopa::gfx::presentation::Window*         g_window     = nullptr;
static coopa::gfx::core::Surface*                g_surface    = nullptr;
static coopa::gfx::core::Device*                 g_device     = nullptr;
static coopa::gfx::core::Swapchain*              g_swapchain  = nullptr;
static coopa::gfx::memory::Allocator*            g_allocator  = nullptr;
static coopa::gfx::command::CommandPool*         g_cmd_pool   = nullptr;
static coopa::gfx::pipeline::RenderPass*         g_render_pass = nullptr;

// Shader .spv paths (relative to the binary, resolved at test time).
static const char* VERT_SPV = "assets/shaders/test.vert.spv";
static const char* FRAG_SPV = "assets/shaders/test.frag.spv";

// ==========================================================================
// Test cases
// ==========================================================================

// --- util/error.h ---

void test_error_check() {
    // GFX_VK_CHECK should pass on VK_SUCCESS.
    GFX_VK_CHECK(VK_SUCCESS);

    // GFX_VK_CHECK should throw on any error code.
    bool threw = false;
    try {
        GFX_VK_CHECK(VK_ERROR_INITIALIZATION_FAILED);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // vk_result_string should return meaningful text.
    std::string s = coopa::gfx::util::vk_result_string(VK_SUCCESS);
    ASSERT_EQ(s, std::string("VK_SUCCESS"));

    std::string e = coopa::gfx::util::vk_result_string(VK_ERROR_OUT_OF_DATE_KHR);
    ASSERT_EQ(e, std::string("VK_ERROR_OUT_OF_DATE_KHR"));
}

// --- util/format.h ---

void test_format_helpers() {
    // format_from_channels
    ASSERT_EQ(coopa::gfx::util::format_from_channels(4, false), VK_FORMAT_R8G8B8A8_UNORM);
    ASSERT_EQ(coopa::gfx::util::format_from_channels(4, true),  VK_FORMAT_R8G8B8A8_SRGB);
    ASSERT_EQ(coopa::gfx::util::format_from_channels(1, false), VK_FORMAT_R8_UNORM);

    // format_byte_size
    ASSERT_EQ(coopa::gfx::util::format_byte_size(VK_FORMAT_R8G8B8A8_UNORM), 4u);
    ASSERT_EQ(coopa::gfx::util::format_byte_size(VK_FORMAT_R16G16B16A16_SFLOAT), 8u);
    ASSERT_EQ(coopa::gfx::util::format_byte_size(VK_FORMAT_R8_UNORM), 1u);

    // format_has_depth / format_has_stencil
    ASSERT_TRUE(coopa::gfx::util::format_has_depth(VK_FORMAT_D32_SFLOAT));
    ASSERT_TRUE(!coopa::gfx::util::format_has_depth(VK_FORMAT_R8G8B8A8_UNORM));
    ASSERT_TRUE(coopa::gfx::util::format_has_stencil(VK_FORMAT_D24_UNORM_S8_UINT));
    ASSERT_TRUE(!coopa::gfx::util::format_has_stencil(VK_FORMAT_D32_SFLOAT));

    // format_from_channels should throw for unsupported channel counts.
    bool threw = false;
    try { coopa::gfx::util::format_from_channels(3, false); } catch (...) { threw = true; }
    ASSERT_TRUE(threw);
}

// --- presentation/window.h ---

void test_window_creation() {
    ASSERT_TRUE(g_window != nullptr);
    ASSERT_TRUE(g_window->handle() != nullptr);

    auto [w, h] = g_window->framebuffer_size();
    ASSERT_TRUE(w > 0);
    ASSERT_TRUE(h > 0);

    ASSERT_TRUE(!g_window->should_close());
    ASSERT_TRUE(!g_window->was_resized());
}

// --- core/instance.h ---

void test_instance_creation() {
    ASSERT_TRUE(g_instance != nullptr);
    ASSERT_TRUE(g_instance->handle() != VK_NULL_HANDLE);
}

// --- core/surface.h ---

void test_surface_creation() {
    ASSERT_TRUE(g_surface != nullptr);
    ASSERT_TRUE(g_surface->handle() != VK_NULL_HANDLE);

    // Support query must return at least one format and present mode.
    auto support = g_surface->query_support(g_device->physical());
    ASSERT_TRUE(!support.formats.empty());
    ASSERT_TRUE(!support.present_modes.empty());
}

// --- core/device.h ---

void test_device_selection() {
    ASSERT_TRUE(g_device != nullptr);
    ASSERT_TRUE(g_device->handle() != VK_NULL_HANDLE);
    ASSERT_TRUE(g_device->physical() != VK_NULL_HANDLE);
    ASSERT_TRUE(g_device->graphics_queue() != VK_NULL_HANDLE);
    ASSERT_TRUE(g_device->present_queue()  != VK_NULL_HANDLE);
}

// --- core/swapchain.h ---

void test_swapchain_creation() {
    ASSERT_TRUE(g_swapchain != nullptr);
    ASSERT_TRUE(g_swapchain->handle() != VK_NULL_HANDLE);
    ASSERT_TRUE(g_swapchain->image_count() > 0);
    ASSERT_TRUE(!g_swapchain->image_views().empty());
    ASSERT_TRUE(g_swapchain->extent().width > 0);
    ASSERT_TRUE(g_swapchain->extent().height > 0);
    ASSERT_TRUE(g_swapchain->image_format() != VK_FORMAT_UNDEFINED);
}

// --- memory/buffer.h ---

void test_buffer_vertex() {
    float verts[] = { 0.0f, -0.5f,  0.5f, 0.5f,  -0.5f, 0.5f };
    VkDeviceSize sz = sizeof(verts);

    auto buf = coopa::gfx::memory::Buffer::vertex(*g_device, *g_allocator, sz);
    ASSERT_TRUE(buf.handle() != VK_NULL_HANDLE);
    ASSERT_EQ(buf.size(), sz);

    buf.upload(verts, sz);
    // If upload() doesn't throw, the map/memcpy/unmap cycle succeeded.
}

void test_buffer_uniform() {
    // Upload a 4x4 float matrix (16 floats = 64 bytes).
    float mat[16] = {};
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f; // Identity.

    VkDeviceSize sz = sizeof(mat);
    auto buf = coopa::gfx::memory::Buffer::uniform(*g_device, *g_allocator, sz);
    ASSERT_TRUE(buf.handle() != VK_NULL_HANDLE);
    ASSERT_EQ(buf.size(), sz);

    buf.upload(mat, sz);
}

// --- memory/image.h ---

void test_image_creation() {
    coopa::gfx::memory::Image img(
        *g_device, *g_allocator,
        64, 64,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ASSERT_TRUE(img.handle() != VK_NULL_HANDLE);
    ASSERT_TRUE(img.view()   != VK_NULL_HANDLE);
    ASSERT_EQ(img.width(),  64u);
    ASSERT_EQ(img.height(), 64u);
    ASSERT_EQ(img.format(), VK_FORMAT_R8G8B8A8_UNORM);
}

// --- pipeline/shader.h ---

void test_shader_loading() {
    coopa::gfx::pipeline::Shader vert(*g_device, VERT_SPV, VK_SHADER_STAGE_VERTEX_BIT);
    ASSERT_TRUE(vert.handle() != VK_NULL_HANDLE);

    VkPipelineShaderStageCreateInfo info = vert.stage_info();
    ASSERT_EQ(info.stage, VK_SHADER_STAGE_VERTEX_BIT);
    ASSERT_TRUE(info.module != VK_NULL_HANDLE);
    ASSERT_TRUE(std::strcmp(info.pName, "main") == 0);

    coopa::gfx::pipeline::Shader frag(*g_device, FRAG_SPV, VK_SHADER_STAGE_FRAGMENT_BIT);
    ASSERT_TRUE(frag.handle() != VK_NULL_HANDLE);
}

// --- pipeline/render_pass.h ---

void test_render_pass() {
    ASSERT_TRUE(g_render_pass != nullptr);
    ASSERT_TRUE(g_render_pass->handle() != VK_NULL_HANDLE);
}

// --- pipeline/pipeline.h ---

void test_pipeline_creation() {
    coopa::gfx::pipeline::Shader vert(*g_device, VERT_SPV, VK_SHADER_STAGE_VERTEX_BIT);
    coopa::gfx::pipeline::Shader frag(*g_device, FRAG_SPV, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<coopa::gfx::pipeline::Shader*> shaders = { &vert, &frag };

    // No vertex input bindings — the vertex shader hard-codes the triangle.
    coopa::gfx::pipeline::Pipeline pipeline(
        *g_device, *g_render_pass, shaders, {}, {});

    ASSERT_TRUE(pipeline.handle() != VK_NULL_HANDLE);
    ASSERT_TRUE(pipeline.layout() != VK_NULL_HANDLE);
}

// --- command/command_pool.h ---

void test_command_pool() {
    ASSERT_TRUE(g_cmd_pool != nullptr);
    ASSERT_TRUE(g_cmd_pool->handle() != VK_NULL_HANDLE);

    auto buffers = g_cmd_pool->allocate(3);
    ASSERT_EQ(buffers.size(), 3u);
    for (auto buf : buffers) {
        ASSERT_TRUE(buf != VK_NULL_HANDLE);
    }
    // Free the allocated buffers back to the pool.
    vkFreeCommandBuffers(g_device->handle(), g_cmd_pool->handle(),
                         static_cast<uint32_t>(buffers.size()), buffers.data());
}

// --- command/sync.h ---

void test_sync_primitives() {
    coopa::gfx::command::Fence fence(*g_device, /*signaled=*/true);
    ASSERT_TRUE(fence.handle() != VK_NULL_HANDLE);

    // Fence starts signaled — wait should return immediately.
    fence.wait(0); // Zero timeout; won't block since it's already signaled.
    fence.reset();

    coopa::gfx::command::Semaphore sem(*g_device);
    ASSERT_TRUE(sem.handle() != VK_NULL_HANDLE);
}

// --- pipeline/descriptor.h ---

void test_descriptor_set() {
    // Create a pool with capacity for one uniform buffer descriptor.
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = 1;

    coopa::gfx::pipeline::DescriptorPool pool(*g_device, 1, { pool_size });
    ASSERT_TRUE(pool.handle() != VK_NULL_HANDLE);

    // Create a layout with one uniform buffer binding at binding 0.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    coopa::gfx::pipeline::DescriptorSetLayout layout(*g_device, { binding });
    ASSERT_TRUE(layout.handle() != VK_NULL_HANDLE);

    // Allocate a descriptor set.
    coopa::gfx::pipeline::DescriptorSet set(*g_device, pool, layout);
    ASSERT_TRUE(set.handle() != VK_NULL_HANDLE);

    // Bind a uniform buffer into slot 0.
    float mat[16] = {};
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
    auto buf = coopa::gfx::memory::Buffer::uniform(*g_device, *g_allocator, sizeof(mat));
    buf.upload(mat, sizeof(mat));

    set.bind_buffer(0, buf); // Should not throw.
}

// --- presentation/renderer.h (frame cycle) ---

void test_renderer_frame() {
    coopa::gfx::pipeline::Shader vert(*g_device, VERT_SPV, VK_SHADER_STAGE_VERTEX_BIT);
    coopa::gfx::pipeline::Shader frag(*g_device, FRAG_SPV, VK_SHADER_STAGE_FRAGMENT_BIT);

    coopa::gfx::pipeline::Pipeline pipeline(
        *g_device, *g_render_pass, { &vert, &frag }, {}, {});

    coopa::gfx::presentation::Renderer renderer(
        *g_device, *g_swapchain, *g_render_pass, *g_cmd_pool);

    VkExtent2D ext = g_swapchain->extent();

    bool frame_ok = renderer.begin_frame([&](coopa::gfx::command::CommandBuffer& cmd) {
        cmd.bind_pipeline(pipeline);
        cmd.set_viewport(0.0f, 0.0f,
                         static_cast<float>(ext.width),
                         static_cast<float>(ext.height));
        cmd.set_scissor(0, 0, ext.width, ext.height);
        cmd.draw(3); // Hard-coded triangle in the vertex shader.
    });

    // begin_frame() returns false only on minimized window; otherwise it should succeed.
    // We do not assert frame_ok == true because a headless CI could return false.
    (void)frame_ok;

    g_device->wait_idle();
}

// --- core/swapchain.h (resize) ---

void test_swapchain_resize() {
    auto [w, h] = g_window->framebuffer_size();
    // Recreate at the same size — validates the destroy/create cycle.
    g_swapchain->recreate(w, h);
    ASSERT_TRUE(g_swapchain->handle() != VK_NULL_HANDLE);
    ASSERT_TRUE(g_swapchain->image_count() > 0);
    ASSERT_EQ(g_swapchain->extent().width,  w);
    ASSERT_EQ(g_swapchain->extent().height, h);
}

// ==========================================================================
// main — fixture setup, test runner, teardown
// ==========================================================================

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "        Running gfxcoopa Test Suite       " << std::endl;
    std::cout << "===========================================" << std::endl;

    // --- Stand-alone tests (no Vulkan required) ---
    RUN_TEST(test_error_check);
    RUN_TEST(test_format_helpers);

    // --- Vulkan fixture setup ---
    std::cout << std::endl;
    std::cout << ANSI_COLOR_YELLOW << "[ SETUP    ] " << ANSI_COLOR_RESET
              << "Initializing Vulkan fixtures..." << std::endl;

    try {
        g_window      = new coopa::gfx::presentation::Window("gfxcoopa test", 800, 600, false);
        g_instance    = new coopa::gfx::core::Instance("gfxcoopa_test", /*validation=*/true);
        g_surface     = new coopa::gfx::core::Surface(*g_instance, g_window->handle());
        g_device      = new coopa::gfx::core::Device(*g_instance, *g_surface);
        g_allocator   = new coopa::gfx::memory::Allocator(*g_instance, *g_device);

        auto [w, h]   = g_window->framebuffer_size();
        g_swapchain   = new coopa::gfx::core::Swapchain(*g_device, *g_surface, w, h, /*vsync=*/true);

        g_cmd_pool    = new coopa::gfx::command::CommandPool(
            *g_device, g_device->graphics_family());

        g_render_pass = new coopa::gfx::pipeline::RenderPass(
            *g_device, g_swapchain->image_format(), VK_FORMAT_UNDEFINED);

        std::cout << ANSI_COLOR_GREEN << "[ SETUP    ] " << ANSI_COLOR_RESET
                  << "Fixtures ready." << std::endl << std::endl;
    } catch (const std::exception& e) {
        std::cerr << ANSI_COLOR_RED << "[ SETUP FAILED ] " << ANSI_COLOR_RESET
                  << e.what() << std::endl;
        return 1;
    }

    // --- Vulkan-dependent tests ---
    RUN_TEST(test_window_creation);
    RUN_TEST(test_instance_creation);
    RUN_TEST(test_surface_creation);
    RUN_TEST(test_device_selection);
    RUN_TEST(test_swapchain_creation);
    RUN_TEST(test_buffer_vertex);
    RUN_TEST(test_buffer_uniform);
    RUN_TEST(test_image_creation);
    RUN_TEST(test_shader_loading);
    RUN_TEST(test_render_pass);
    RUN_TEST(test_pipeline_creation);
    RUN_TEST(test_command_pool);
    RUN_TEST(test_sync_primitives);
    RUN_TEST(test_descriptor_set);
    RUN_TEST(test_renderer_frame);
    RUN_TEST(test_swapchain_resize);

    // --- Fixture teardown ---
    std::cout << std::endl;
    std::cout << ANSI_COLOR_YELLOW << "[ TEARDOWN ] " << ANSI_COLOR_RESET
              << "Destroying Vulkan fixtures..." << std::endl;

    g_device->wait_idle();

    delete g_render_pass;
    delete g_cmd_pool;
    delete g_swapchain;
    delete g_allocator;
    delete g_device;
    delete g_surface;
    delete g_instance;
    delete g_window;

    std::cout << ANSI_COLOR_GREEN << "[ TEARDOWN ] " << ANSI_COLOR_RESET
              << "Done." << std::endl << std::endl;

    // --- Summary ---
    std::cout << "===========================================" << std::endl;
    std::cout << "Test Summary: " << g_tests_run - g_tests_failed
              << " / " << g_tests_run << " Passed." << std::endl;
    if (g_tests_failed > 0) {
        std::cout << ANSI_COLOR_RED << "Some tests failed!" << ANSI_COLOR_RESET << std::endl;
        return 1;
    } else {
        std::cout << ANSI_COLOR_GREEN << "All tests passed successfully!" << ANSI_COLOR_RESET << std::endl;
        return 0;
    }
}
