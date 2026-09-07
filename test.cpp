// volk and VMA implementations are compiled once via -DVOLK_IMPLEMENTATION
// and -DVMA_IMPLEMENTATION defined in CMakeLists.txt.
#include <volk/volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vma/vk_mem_alloc.h>

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// Declarations only -- STB_IMAGE_IMPLEMENTATION is compiled once in src/gfx_impl.cpp.
#include <stb/stb_image.h>

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
#include <gfxcoopa/pipeline/surface_shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/command/sync.h>
#include <gfxcoopa/presentation/renderer.h>
#include <gfxcoopa/util/image_readback.h>
#include <gfxcoopa/app/context.h>
#include <gfxcoopa/engine/components/camera_component.h>
#include <gfxcoopa/engine/components/mesh_renderer.h>
#include <gfxcoopa/engine/components/register.h>

#include <coopa/scene/scene_object.h>
#include <coopa/scene/components/transform_component.h>

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

// --- engine/components/camera_component.h ---
// Stand-alone: CameraComponent only needs a SceneObject/TransformComponent,
// no Vulkan device -- so these run in the no-Vulkan-required group above.

void test_camera_main_singleton_explicit() {
    using coopa::gfx::engine::components::CameraComponent;
    using coopa::scene::SceneObject;

    SceneObject a("cam_a"), b("cam_b");
    auto* cam_a = a.add_component<CameraComponent>();
    auto* cam_b = b.add_component<CameraComponent>();
    cam_b->is_main = true;

    // b starts first despite a existing first -- is_main should still win regardless of order.
    cam_b->start();
    cam_a->start();
    ASSERT_TRUE(CameraComponent::main() == cam_b);
}

void test_camera_main_singleton_defaults_to_first() {
    using coopa::gfx::engine::components::CameraComponent;
    using coopa::scene::SceneObject;

    SceneObject a("cam_a"), b("cam_b");
    auto* cam_a = a.add_component<CameraComponent>();
    auto* cam_b = b.add_component<CameraComponent>();

    // Neither declares is_main -- the first to start() claims the singleton,
    // so a loaded scene always has a main camera.
    cam_a->start();
    cam_b->start();
    ASSERT_TRUE(CameraComponent::main() == cam_a);
}

void test_camera_main_singleton_clears_on_destroy() {
    using coopa::gfx::engine::components::CameraComponent;
    using coopa::scene::SceneObject;

    SceneObject holder("cam_temp");
    auto* cam = holder.add_component<CameraComponent>();
    cam->make_main();
    ASSERT_TRUE(CameraComponent::main() == cam);

    holder.remove_component<CameraComponent>();
    ASSERT_TRUE(CameraComponent::main() == nullptr);
}

void test_camera_projection_runtime_switch() {
    using coopa::gfx::engine::components::CameraComponent;
    using coopa::gfx::engine::components::CameraType;
    using coopa::scene::SceneObject;

    SceneObject obj("cam");
    auto* cam = obj.add_component<CameraComponent>();

    cam->set_orthographic(5.0f);
    ASSERT_TRUE(cam->type == CameraType::Orthographic);
    glm::mat4 ortho_proj = cam->get_projection_matrix(1.0f);
    ASSERT_TRUE(ortho_proj[3][3] == 1.0f); // orthographic: w stays 1

    cam->set_perspective(60.0f);
    ASSERT_TRUE(cam->type == CameraType::Perspective);
    glm::mat4 persp_proj = cam->get_projection_matrix(1.0f);
    ASSERT_TRUE(persp_proj[3][3] == 0.0f); // perspective: w comes from -z
}

// --- engine/components/register.h + mesh_renderer.h ---
// CUTOUT support: "CUTOUT" is a new alpha_mode alias for the pre-existing AlphaMode::Mask (see
// mesh_renderer.h), and PBRMaterial::gpu_alpha_cutoff() is what both the G-buffer and shadow
// shaders read to arm their alpha-mask discard. Neither needs a Device/AssetManager, so these
// run in the no-Vulkan-required group above.

void test_parse_alpha_mode() {
    using coopa::gfx::engine::components::AlphaMode;
    using coopa::gfx::engine::components::parse_alpha_mode_;

    ASSERT_TRUE(parse_alpha_mode_("BLEND") == AlphaMode::Blend);
    ASSERT_TRUE(parse_alpha_mode_("MASK") == AlphaMode::Mask);
    ASSERT_TRUE(parse_alpha_mode_("CLIP") == AlphaMode::Mask);
    // CUTOUT is the Unity-facing name for the same alpha-tested behaviour MASK/CLIP (the glTF
    // names) already select -- all three must collapse onto AlphaMode::Mask.
    ASSERT_TRUE(parse_alpha_mode_("CUTOUT") == AlphaMode::Mask);
    // Unrecognised (or misspelled) strings silently fall back to Opaque, matching this parser's
    // existing behaviour for every other unrecognised enum-like value.
    ASSERT_TRUE(parse_alpha_mode_("") == AlphaMode::Opaque);
    ASSERT_TRUE(parse_alpha_mode_("OPAQUE") == AlphaMode::Opaque);
    ASSERT_TRUE(parse_alpha_mode_("cutout") == AlphaMode::Opaque); // case-sensitive by design
}

void test_pbr_material_gpu_alpha_cutoff() {
    using coopa::gfx::engine::components::AlphaMode;
    using coopa::gfx::engine::components::PBRMaterial;

    PBRMaterial opaque;
    opaque.alpha_mode = AlphaMode::Opaque;
    opaque.alpha_cutoff = 0.7f;
    ASSERT_TRUE(opaque.gpu_alpha_cutoff() == 0.0f); // OPAQUE never arms the discard

    PBRMaterial blended;
    blended.alpha_mode = AlphaMode::Blend;
    blended.alpha_cutoff = 0.7f;
    ASSERT_TRUE(blended.gpu_alpha_cutoff() == 0.0f); // BLEND never arms the discard either

    PBRMaterial masked;
    masked.alpha_mode = AlphaMode::Mask;
    masked.alpha_cutoff = 0.7f;
    ASSERT_TRUE(masked.gpu_alpha_cutoff() == 0.7f);

    // Clamped to (0, 1] -- 0.0 would otherwise collide with the "disabled" sentinel itself.
    PBRMaterial masked_zero;
    masked_zero.alpha_mode = AlphaMode::Mask;
    masked_zero.alpha_cutoff = 0.0f;
    ASSERT_TRUE(masked_zero.gpu_alpha_cutoff() > 0.0f);

    // has_alpha_mask() requires BOTH AlphaMode::Mask AND a loaded texture -- an unset
    // alpha_mask_handle (the default for every material, masked or not) must read false, so
    // MaterialTextureCache::set_for() knows to bind the white fallback instead.
    ASSERT_TRUE(!masked.has_alpha_mask());
    ASSERT_TRUE(!opaque.has_alpha_mask());
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

// --- pipeline/surface_shader.h ---

void test_surface_shader_registry() {
    using coopa::gfx::pipeline::SurfaceShaderDesc;
    using coopa::gfx::pipeline::SurfaceShaderDomain;
    using coopa::gfx::pipeline::SurfaceShaderRegistry;

    SurfaceShaderRegistry registry;

    // Empty name -- find()/require() treat it as "the stock shader", never registered.
    ASSERT_TRUE(registry.find("") == nullptr);
    registry.require(""); // must not throw

    // An unregistered name is a hard failure via require(), not a silent fall-back.
    bool threw = false;
    try {
        registry.require("nonexistent");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    ASSERT_TRUE(registry.find("nonexistent") == nullptr);

    // A registered name resolves and validates.
    SurfaceShaderDesc water;
    water.name   = "water";
    water.domain = SurfaceShaderDomain::Transparent;
    water.vert   = "water.vert";
    water.frag   = "water.frag";
    registry.add(water);

    const SurfaceShaderDesc* found = registry.find("water");
    ASSERT_TRUE(found != nullptr);
    ASSERT_TRUE(found->vert == "water.vert");
    ASSERT_TRUE(found->domain == SurfaceShaderDomain::Transparent);
    registry.require("water"); // must not throw
    ASSERT_EQ(registry.all().size(), 1u);

    // Duplicate names are rejected -- a silent overwrite would mean two materials
    // referencing the same name draw with different pipelines depending on
    // registration order (see SurfaceShaderRegistry::add()'s own doc).
    threw = false;
    try {
        SurfaceShaderDesc dup;
        dup.name = "water";
        registry.add(dup);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    ASSERT_EQ(registry.all().size(), 1u); // the failed add() must not have appended anyway

    // An empty name is rejected too.
    threw = false;
    try {
        registry.add(SurfaceShaderDesc{});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
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

// --- Sealed API (gfxcoopa/types/*, gfxcoopa/pipeline builders,
//     gfxcoopa/command/command_buffer.h's transition/copy additions) ---
//
// The tests above exercise the raw Vk*-typed API surface, which every
// pre-seal downstream consumer had to speak directly. This test exercises
// the sealed replacement end-to-end against a live device+validation
// layers, not just compiling it -- proving the barrier access/stage masks
// in detail::barrier_masks_for() and the descriptor pool sizing in
// DescriptorPoolBuilder are actually correct at runtime, not merely
// type-correct.
void test_sealed_api() {
    using namespace coopa::gfx;

    // --- Sealed Buffer/Image construction ---
    memory::Buffer sealed_buf(*g_device, *g_allocator, 256,
                              BufferUsage::Uniform, MemoryResidency::CpuToGpu);
    ASSERT_TRUE(sealed_buf.handle() != VK_NULL_HANDLE);

    memory::Image sealed_img(*g_device, *g_allocator, 4, 4, Format::RGBA8_Unorm,
                             ImageUsage::Sampled | ImageUsage::TransferDst | ImageUsage::TransferSrc);
    ASSERT_TRUE(sealed_img.handle() != VK_NULL_HANDLE);
    ASSERT_TRUE(sealed_img.view_typed().valid());
    ASSERT_TRUE(sealed_img.current_usage() == TextureUsage::Undefined);
    ASSERT_EQ(static_cast<int>(sealed_img.format_typed()), static_cast<int>(Format::RGBA8_Unorm));

    // --- TextureView identity: null vs. real, hashable, test-fabricable ---
    ASSERT_TRUE(!TextureView::null().valid());
    ASSERT_TRUE(TextureView{0x1234}.valid());
    ASSERT_TRUE(TextureView{0x1234} == TextureView{0x1234});
    ASSERT_TRUE(TextureView{0x1234} != TextureView{0x5678});
    std::hash<TextureView> hasher;
    ASSERT_EQ(hasher(TextureView{42}), hasher(TextureView{42}));

    // --- DescriptorLayoutBuilder / DescriptorPoolBuilder ---
    pipeline::DescriptorSetLayout sealed_layout =
        pipeline::DescriptorLayoutBuilder()
            .uniform_buffer(0, ShaderStage::Vertex | ShaderStage::Fragment)
            .build(*g_device);
    ASSERT_TRUE(sealed_layout.handle() != VK_NULL_HANDLE);
    ASSERT_EQ(sealed_layout.bindings().size(), 1u);

    pipeline::DescriptorPool sealed_pool =
        pipeline::DescriptorPoolBuilder()
            .add_sets(sealed_layout, 1)
            .build(*g_device);
    ASSERT_TRUE(sealed_pool.handle() != VK_NULL_HANDLE);

    pipeline::DescriptorSet sealed_set(*g_device, sealed_pool, sealed_layout);
    sealed_set.bind_buffer(0, sealed_buf); // Should not throw.

    // --- Sealed PipelineDesc + CommandBuffer's layout-caching overloads ---
    pipeline::Shader vert(*g_device, VERT_SPV, ShaderStage::Vertex);
    pipeline::Shader frag(*g_device, FRAG_SPV, ShaderStage::Fragment);

    pipeline::PipelineDesc desc;
    desc.shaders = { &vert, &frag };
    desc.descriptor_layouts = { &sealed_layout };
    desc.push_constants = { { ShaderStage::Vertex, 0, sizeof(float) * 4 } };
    desc.raster.cull = CullMode::None;

    pipeline::Pipeline sealed_pipeline(*g_device, *g_render_pass, desc);
    ASSERT_TRUE(sealed_pipeline.handle() != VK_NULL_HANDLE);

    // bind_descriptor_set(uint32_t, set) / push_constants(ShaderStage, value)
    // both need a bound pipeline -- verify the "no pipeline bound" guard
    // throws, then verify the happy path records without throwing.
    {
        VkCommandBuffer raw = g_cmd_pool->begin_single_use();
        command::CommandBuffer cmd(raw);
        bool threw = false;
        try {
            cmd.bind_descriptor_set(sealed_set);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        ASSERT_TRUE(threw);

        cmd.bind_pipeline(sealed_pipeline);
        cmd.bind_descriptor_set(sealed_set); // Should not throw now.
        float push_data[4] = {1, 2, 3, 4};
        cmd.push_constants(ShaderStage::Vertex, push_data); // Template overload.
        g_cmd_pool->end_single_use(raw, g_device->graphics_queue());
    }

    // --- transition()/copy_buffer_to_image()/copy_image_to_buffer() round trip ---
    // Uploads a known 4x4 RGBA8 pattern into sealed_img via a staging buffer,
    // transitions it to ShaderRead (the steady state a texture lives in),
    // then transitions back to TransferSrc and reads it back into a second
    // buffer -- verifying the bytes survive both barrier directions intact.
    const uint32_t w = 4, h = 4, byte_size = w * h * 4;
    std::vector<uint8_t> pattern(byte_size);
    for (uint32_t i = 0; i < byte_size; ++i) pattern[i] = static_cast<uint8_t>(i * 7 + 3);

    memory::Buffer staging(*g_device, *g_allocator, byte_size,
                           BufferUsage::TransferSrc, MemoryResidency::CpuToGpu);
    staging.upload(pattern.data(), byte_size);

    memory::Buffer readback(*g_device, *g_allocator, byte_size,
                            BufferUsage::TransferDst, MemoryResidency::GpuToCpu);

    g_cmd_pool->submit_once([&](command::CommandBuffer& cmd) {
        cmd.transition(sealed_img, TextureUsage::TransferDst);
        cmd.copy_buffer_to_image(staging, sealed_img, Extent2D{w, h});
        cmd.transition(sealed_img, TextureUsage::ShaderRead);
        ASSERT_TRUE(sealed_img.current_usage() == TextureUsage::ShaderRead);
        cmd.transition(sealed_img, TextureUsage::TransferSrc);
        cmd.copy_image_to_buffer(sealed_img, readback, Extent2D{w, h});
    });

    void* mapped = nullptr;
    GFX_VK_CHECK(vmaMapMemory(g_allocator->handle(), readback.allocation(), &mapped));
    bool bytes_match = std::memcmp(mapped, pattern.data(), byte_size) == 0;
    vmaUnmapMemory(g_allocator->handle(), readback.allocation());
    ASSERT_TRUE(bytes_match);
}

// --- util/image_readback.h ---
//
// Verifies the readback path end-to-end and independently of any consumer:
// clears an offscreen image to a known color, writes it to disk via
// save_image_png(), then reads the PNG back with stb_image (a completely
// separate code path from the write side) and checks every pixel matches.
// This is the "render a triangle, screenshot it, and byte-compare" proof
// the seal plan calls for before any consumer adopts this API.
void test_image_readback() {
    using namespace coopa::gfx;

    // ColorAttachment is included because memory::Image unconditionally
    // creates a VkImageView, and the Vulkan spec requires at least one
    // view-compatible usage bit (Sampled/Storage/ColorAttachment/...) --
    // a real render target being screenshotted always has this anyway.
    memory::Image target(*g_device, *g_allocator, 8, 8, Format::RGBA8_Unorm,
                         ImageUsage::ColorAttachment | ImageUsage::TransferSrc | ImageUsage::TransferDst);

    ClearColor clear{0.25f, 0.5f, 0.75f, 1.0f};
    g_cmd_pool->submit_once([&](command::CommandBuffer& cmd) {
        cmd.transition(target, TextureUsage::TransferDst);
        cmd.clear_color(target, clear);
    });
    ASSERT_TRUE(target.current_usage() == TextureUsage::TransferDst);

    const std::string out_path = "test_image_readback_output.png";
    util::save_image_png(*g_device, *g_allocator, *g_cmd_pool, target, out_path);

    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(out_path.c_str(), &w, &h, &channels, 4);
    ASSERT_TRUE(pixels != nullptr);
    ASSERT_EQ(w, 8);
    ASSERT_EQ(h, 8);

    auto to_u8 = [](float f) { return static_cast<int>(f * 255.0f + 0.5f); };
    int expected_r = to_u8(clear.r), expected_g = to_u8(clear.g), expected_b = to_u8(clear.b);

    bool all_match = true;
    for (int i = 0; i < w * h && all_match; ++i) {
        int r = pixels[i * 4 + 0], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
        // +/-2 tolerance for float->unorm8 rounding through the GPU clear.
        if (std::abs(r - expected_r) > 2 || std::abs(g - expected_g) > 2 || std::abs(b - expected_b) > 2) {
            all_match = false;
        }
    }
    stbi_image_free(pixels);
    std::remove(out_path.c_str());
    ASSERT_TRUE(all_match);

    // read_image()'s "restore original usage" behavior: TransferDst is
    // restorable, so after read_image() runs internally inside
    // save_image_png(), the image should be back in TransferDst, not
    // stranded in TransferSrc.
    ASSERT_TRUE(target.current_usage() == TextureUsage::TransferDst);
}

// --- app/context.h ---
//
// Verifies gfx::app::Context independently, before any consumer adopts it:
// full bring-up (its own Window/Instance/Surface/Device/Allocator/
// Swapchain/CommandPool/RenderPass/Renderer, entirely separate from this
// file's g_* fixtures), a driven run() loop bounded by max_frames, and that
// frame timing/derived accessors report sane values.
void test_context() {
    using namespace coopa::gfx;

    app::ContextConfig config;
    config.title      = "gfxcoopa_context_test";
    config.width      = 320;
    config.height     = 240;
    config.validation = true;
    config.max_frames = 3; // Bound the loop regardless of ONESHOT/MAX_FRAMES env.

    app::Context ctx(config);

    ASSERT_TRUE(ctx.extent().width == 320);
    ASSERT_TRUE(ctx.extent().height == 240);
    ASSERT_TRUE(ctx.color_format() != Format::Undefined);
    ASSERT_EQ(ctx.frames_in_flight(), presentation::MAX_FRAMES_IN_FLIGHT);
    ASSERT_TRUE(!ctx.should_close());

    uint32_t frames_recorded = 0;
    float last_dt = -1.0f;

    ctx.run(
        [&](float dt) { last_dt = dt; },
        app::FrameCallbacks{
            [&](command::CommandBuffer& cmd) { (void)cmd; ++frames_recorded; },
            nullptr,
            nullptr,
            ClearColor{0.1f, 0.2f, 0.3f, 1.0f},
        });

    ASSERT_EQ(frames_recorded, 3u);
    ASSERT_EQ(ctx.frame_index(), 3u); // poll() calls time_.update() once per iteration.
    ASSERT_TRUE(last_dt >= 0.0f);

    // A single frame() call driven manually (not via run()) should also work.
    ctx.poll();
    bool presented = ctx.frame([](command::CommandBuffer&) {});
    ASSERT_TRUE(presented);
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
    RUN_TEST(test_camera_main_singleton_explicit);
    RUN_TEST(test_camera_main_singleton_defaults_to_first);
    RUN_TEST(test_camera_main_singleton_clears_on_destroy);
    RUN_TEST(test_camera_projection_runtime_switch);
    RUN_TEST(test_parse_alpha_mode);
    RUN_TEST(test_pbr_material_gpu_alpha_cutoff);

    // --- Vulkan fixture setup ---
    std::cout << std::endl;
    std::cout << ANSI_COLOR_YELLOW << "[ SETUP    ] " << ANSI_COLOR_RESET
              << "Initializing Vulkan fixtures..." << std::endl;

    try {
        g_window      = new coopa::gfx::presentation::Window("gfxcoopa test", 800, 600, false);
        g_instance    = new coopa::gfx::core::Instance("gfxcoopa_test", /*validation=*/true);
        g_surface     = new coopa::gfx::core::Surface(*g_instance, *g_window);
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
    RUN_TEST(test_surface_shader_registry);
    RUN_TEST(test_render_pass);
    RUN_TEST(test_pipeline_creation);
    RUN_TEST(test_command_pool);
    RUN_TEST(test_sync_primitives);
    RUN_TEST(test_descriptor_set);
    RUN_TEST(test_sealed_api);
    RUN_TEST(test_image_readback);
    RUN_TEST(test_context);
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
