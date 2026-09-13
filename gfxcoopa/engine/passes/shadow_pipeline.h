/**
 * @file shadow_pipeline.h
 * @brief Vulkan pipelines for directional and cubemap shadow map depth passes.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SHADOW_PIPELINE_H
#define GFXCOOPA_ENGINE_PASSES_SHADOW_PIPELINE_H

#include <volk/volk.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/data/mesh.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

/**
 * @struct DirectionalShadowPushConstants
 * @brief Push constant block for directional shadow depth pass (68 bytes).
 *
 * model is streamed per-instance via data::InstanceData rather than pushed
 * here. light_space_matrix is shared across the whole pass call; alpha is
 * per-BATCH (all instances in one draw share one
 * mesh AND, for BLEND casters, one alpha -- see InstanceBatcher's shadow
 * batch key) and drives shadow_depth.frag's stochastic alpha-dither discard.
 * 1.0 (the default, and always what OPAQUE/MASK casters get) means "fully
 * opaque, no dithering" -- see shadow_common.glsl.
 */
struct alignas(16) DirectionalShadowPushConstants {
    glm::mat4 light_space_matrix;
    float     alpha = 1.0f;

    /// CUTOUT (AlphaMode::Mask) support: 0.0 disables the alpha-mask discard in
    /// shadow_depth.frag entirely (the default, and what OPAQUE/BLEND casters get -- see
    /// PBRMaterial::gpu_alpha_cutoff()); a masked caster's own material.gpu_alpha_cutoff()
    /// otherwise. Only meaningful when this ShadowPipeline was built with a non-null
    /// material_layout (see the ctor) -- shadow_depth.frag has no mask sampler to test
    /// against otherwise, and every caller that predates CUTOUT never sets this field, so it
    /// stays at its 0.0 default and behaves exactly as before.
    float     alpha_cutoff = 0.0f;

    /// Explicit std430 padding: glm::vec4 is NOT guaranteed 16-byte aligned in this build
    /// (alignof(glm::vec4) == 4 unless GLM's SIMD/aligned-gentype flags are on), but GLSL's
    /// push_constant blocks always align vec4 to 16 bytes. Without this pad, gfx_time below
    /// would land at C++ offset 72 while the GLSL side puts it at 80 -- exactly the
    /// mismatch Vulkan's validation layer catches as "block range outside push constant
    /// range". Same convention as data::CameraData's _pad0 (see camera_ubo.h).
    float     _pad0 = 0.0f;
    float     _pad1 = 0.0f;

    /// Standard trailing "surface" block (see gfx/surface/shadow_vs.glsl) -- 32 bytes,
    /// zero-initialized by default so a stock caster is unaffected. A derived shader's
    /// caller fills these from the same object's shader_params/elapsed time it already
    /// pushes to GBufferPipeline, so displacement stays in sync between the G-buffer and
    /// this shadow pass. This is the tightest-fitting pass: with these fields,
    /// CubeShadowPushConstants below (not this one) lands at exactly 128 bytes, Vulkan's
    /// guaranteed minimum maxPushConstantsSize -- see the layered-shaders plan's budget
    /// table before growing either struct.
    glm::vec4 gfx_time   = {0.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 gfx_params = {0.0f, 0.0f, 0.0f, 0.0f};
};
static_assert(sizeof(DirectionalShadowPushConstants) <= 128,
             "DirectionalShadowPushConstants exceeds Vulkan's guaranteed "
             "maxPushConstantsSize (128 bytes) -- see the layered-shaders plan's "
             "push-constant budget table before growing this struct.");

/**
 * @struct CubeShadowPushConstants
 * @brief Push constant block for point light cubemap shadow pass (84 bytes).
 *
 * Same model-removal as DirectionalShadowPushConstants above; alpha has the
 * same per-batch, stochastic-dither meaning too.
 */
struct alignas(16) CubeShadowPushConstants {
    glm::mat4 light_space_matrix;
    glm::vec4 light_pos_range; // xyz = light pos, w = range
    float     alpha = 1.0f;

    /// See DirectionalShadowPushConstants::alpha_cutoff -- same CUTOUT meaning, same
    /// backward-compatible 0.0 default.
    float     alpha_cutoff = 0.0f;

    /// Explicit std430 padding -- see DirectionalShadowPushConstants::_pad0/_pad1 for why
    /// this is required rather than relying on glm::vec4's C++ alignment.
    float     _pad0 = 0.0f;
    float     _pad1 = 0.0f;

    /// See DirectionalShadowPushConstants::gfx_time/gfx_params -- same surface-block
    /// meaning. With mat4 + vec4 + 2 floats + 2 pad floats already at 96 bytes, these
    /// 32 bytes land this struct at exactly 128 -- Vulkan's guaranteed minimum
    /// maxPushConstantsSize. Do not add fields to this struct without shrinking something
    /// else first (the documented growth path is moving light_space_matrix into a small
    /// per-face UBO); a size above 128 is a portability fault, not just a bigger push.
    glm::vec4 gfx_time   = {0.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 gfx_params = {0.0f, 0.0f, 0.0f, 0.0f};
};
static_assert(sizeof(CubeShadowPushConstants) <= 128,
             "CubeShadowPushConstants exceeds Vulkan's guaranteed maxPushConstantsSize "
             "(128 bytes) -- this is the tightest-fitting surface-shader pass, so this is "
             "the first struct to break; see the layered-shaders plan's push-constant "
             "budget table for the documented growth path (shrink light_space_matrix into "
             "a per-face UBO) before adding fields here.");

/**
 * @class ShadowPipeline
 * @brief Manages graphics pipelines for directional & point light shadow map rendering.
 */
class ShadowPipeline {
public:
    ShadowPipeline(core::Device&         device,
                   pipeline::RenderPass& dir_pass,
                   pipeline::RenderPass& cube_pass,
                   const std::string&    dir_vert_spv,
                   const std::string&    dir_frag_spv,
                   const std::string&    cube_vert_spv,
                   const std::string&    cube_frag_spv,
                   const pipeline::DescriptorSetLayout* material_layout = nullptr)
        : device_(device), dir_pass_(dir_pass), cube_pass_(cube_pass)
    {
        // Shadow depth shaders only consume position (location 0) from the per-vertex stream,
        // plus the per-instance model matrix (locations 4-7) -- a position-only binding-0 layout
        // (built here, not via data::Vertex::layout(), which declares all four of its
        // position/normal/uv/tangent attributes) eliminates validation warnings about unconsumed
        // locations 1/2/3 for normal, uv, and tangent. When material_layout is non-null (CUTOUT
        // support requested), location 2 (uv) is added too, so shadow_depth.frag/shadow_cube.frag
        // can alpha-test against the same mask texture the G-buffer pass uses; location 1
        // (normal) and 3 (tangent) stay unconsumed either way -- depth-only shading needs neither.
        //
        // Cached as members (not locals): add_variant() below builds additional dir/cube
        // pipelines against this exact vertex layout and descriptor set -- a derived
        // shader's shadow entry points read the same attributes and mask sampler as the
        // stock ones, only the shader modules (and hence displacement) differ.
        vertex_layout_.binding(0, sizeof(data::Vertex));
        vertex_layout_.attribute(0, coopa::gfx::Format::RGB32_Sfloat,
                                 static_cast<uint32_t>(offsetof(data::Vertex, position)));
        if (material_layout != nullptr) {
            vertex_layout_.attribute(2, coopa::gfx::Format::RG32_Sfloat,
                                     static_cast<uint32_t>(offsetof(data::Vertex, uv)));
        }
        vertex_layout_.append(data::InstanceData::layout());
        material_layout_ = material_layout;

        // 1. Directional Shadow Pipeline
        dir_vert_ = std::make_unique<pipeline::Shader>(device, dir_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        dir_frag_ = std::make_unique<pipeline::Shader>(device, dir_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        dir_pipeline_ = create_dir_pipeline_(*dir_vert_, *dir_frag_);

        // 2. Cube Shadow Pipeline
        cube_vert_ = std::make_unique<pipeline::Shader>(device, cube_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        cube_frag_ = std::make_unique<pipeline::Shader>(device, cube_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        cube_pipeline_ = create_cube_pipeline_(*cube_vert_, *cube_frag_);
    }

    /**
     * @brief Registers a derived shader's directional + cube shadow entry points as a
     *        named variant, reusing this ShadowPipeline's vertex layout and (if CUTOUT
     *        support was requested) material descriptor set layout.
     *
     * Unlike GBufferPipeline::add_variant(), each variant here gets its own
     * pipeline::Pipeline (and hence its own VkPipelineLayout) rather than sharing one --
     * ShadowPipeline was already built on the sealed Pipeline/PipelineDesc API, which owns
     * its layout internally, so duplicating a functionally-identical layout per variant is
     * the lower-friction choice here; the extra VkPipelineLayout objects are negligible
     * next to N pipelines already being created.
     *
     * @param name           The SurfaceShaderDesc's name.
     * @param dir_vert_spv   Resolved .spv path for this shader's directional shadow vertex entry point.
     * @param dir_frag_spv   Resolved .spv path for this shader's directional shadow fragment entry point.
     * @param cube_vert_spv  Resolved .spv path for this shader's cube shadow vertex entry point.
     * @param cube_frag_spv  Resolved .spv path for this shader's cube shadow fragment entry point.
     */
    void add_variant(const std::string& name,
                     const std::string& dir_vert_spv, const std::string& dir_frag_spv,
                     const std::string& cube_vert_spv, const std::string& cube_frag_spv) {
        Variant v;
        v.dir_vert  = std::make_unique<pipeline::Shader>(device_, dir_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        v.dir_frag  = std::make_unique<pipeline::Shader>(device_, dir_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        v.dir_pipeline = create_dir_pipeline_(*v.dir_vert, *v.dir_frag);

        v.cube_vert = std::make_unique<pipeline::Shader>(device_, cube_vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        v.cube_frag = std::make_unique<pipeline::Shader>(device_, cube_frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);
        v.cube_pipeline = create_cube_pipeline_(*v.cube_vert, *v.cube_frag);

        variants_.emplace(name, std::move(v));
    }

    /** @brief True if a variant named `name` was registered via add_variant(). */
    bool has_variant(const std::string& name) const {
        return variants_.find(name) != variants_.end();
    }

    void bind_directional(command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*dir_pipeline_);
    }

    /// Binds a named variant's directional pipeline, or the stock one if `name` is empty
    /// or unregistered (see PBRMaterial::shader's doc: empty means "stock").
    void bind_directional(command::CommandBuffer& cmd, const std::string& name) const {
        auto it = variants_.find(name);
        cmd.bind_pipeline(it != variants_.end() ? *it->second.dir_pipeline : *dir_pipeline_);
    }

    void push_directional(command::CommandBuffer& cmd, const DirectionalShadowPushConstants& pc) const {
        cmd.push_constants(coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment, pc);
    }

    void bind_cube(command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*cube_pipeline_);
    }

    /// Binds a named variant's cube pipeline, or the stock one if `name` is empty or
    /// unregistered.
    void bind_cube(command::CommandBuffer& cmd, const std::string& name) const {
        auto it = variants_.find(name);
        cmd.bind_pipeline(it != variants_.end() ? *it->second.cube_pipeline : *cube_pipeline_);
    }

    void push_cube(command::CommandBuffer& cmd, const CubeShadowPushConstants& pc) const {
        cmd.push_constants(coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment, pc);
    }

private:
    struct Variant {
        std::unique_ptr<pipeline::Shader>   dir_vert, dir_frag;
        std::unique_ptr<pipeline::Pipeline> dir_pipeline;
        std::unique_ptr<pipeline::Shader>   cube_vert, cube_frag;
        std::unique_ptr<pipeline::Pipeline> cube_pipeline;
    };

    std::unique_ptr<pipeline::Pipeline> create_dir_pipeline_(pipeline::Shader& vert, pipeline::Shader& frag) {
        pipeline::PipelineDesc desc;
        desc.vertex       = vertex_layout_;
        desc.raster.cull  = coopa::gfx::CullMode::None;
        desc.depth.test   = true;
        desc.depth.write  = true;
        if (material_layout_ != nullptr) {
            desc.descriptor_layouts = {material_layout_};
        }
        desc.shaders = {&vert, &frag};
        desc.push_constants = {{coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment,
                                0, sizeof(DirectionalShadowPushConstants)}};
        return std::make_unique<pipeline::Pipeline>(device_, dir_pass_, desc);
    }

    std::unique_ptr<pipeline::Pipeline> create_cube_pipeline_(pipeline::Shader& vert, pipeline::Shader& frag) {
        pipeline::PipelineDesc desc;
        desc.vertex       = vertex_layout_;
        desc.raster.cull  = coopa::gfx::CullMode::None;
        desc.depth.test   = true;
        desc.depth.write  = true;
        if (material_layout_ != nullptr) {
            desc.descriptor_layouts = {material_layout_};
        }
        desc.shaders = {&vert, &frag};
        desc.push_constants = {{coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment,
                                0, sizeof(CubeShadowPushConstants)}};
        return std::make_unique<pipeline::Pipeline>(device_, cube_pass_, desc);
    }

    core::Device&                       device_;
    pipeline::RenderPass&               dir_pass_;
    pipeline::RenderPass&               cube_pass_;
    coopa::gfx::VertexLayout            vertex_layout_;
    const pipeline::DescriptorSetLayout* material_layout_ = nullptr;

    std::unique_ptr<pipeline::Shader>   dir_vert_;
    std::unique_ptr<pipeline::Shader>   dir_frag_;
    std::unique_ptr<pipeline::Pipeline> dir_pipeline_;

    std::unique_ptr<pipeline::Shader>   cube_vert_;
    std::unique_ptr<pipeline::Shader>   cube_frag_;
    std::unique_ptr<pipeline::Pipeline> cube_pipeline_;

    std::map<std::string, Variant> variants_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_SHADOW_PIPELINE_H
