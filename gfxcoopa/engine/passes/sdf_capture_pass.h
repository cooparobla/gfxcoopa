/**
 * @file sdf_capture_pass.h
 * @brief Raymarches every BLEND SdfRenderer's directly-lit appearance into
 *        TransparentCaptureTarget -- a second reflection SOURCE for opaque
 *        reflectors' SSR raymarch, not a visible draw of its own.
 *
 * Drawn inside TransparentCaptureTarget's existing begin()/end() bracket,
 * right after TransparentCapturePass's own mesh loop -- see
 * PixelRenderPipeline::record_transparent_capture_(). Mirrors
 * TransparentCapturePass exactly: depth-tested AND written (front-most
 * transparent surface wins per pixel, among transparent objects only -- not
 * cross-tested against the opaque G-buffer, see TransparentCaptureTarget's
 * own doc for why), no blending on any of the three colour outputs, no SSR
 * trace of its own (this capture must not itself reflect anything, to avoid
 * a glass-reflects-its-own-reflection recursion -- see
 * transparent_capture.frag's doc, which toyengine's sdf_capture.frag mirrors).
 */

#ifndef GFXCOOPA_ENGINE_PASSES_SDF_CAPTURE_PASS_H
#define GFXCOOPA_ENGINE_PASSES_SDF_CAPTURE_PASS_H

#include <volk/volk.h>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/detail/vk_convert.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {

class SdfCapturePass {
public:
    struct PushConstants {
        uint32_t renderer_index = 0;
    };

    /**
     * @param device        Logical device.
     * @param capture_render_pass TransparentCaptureTarget's raw render pass (3 color + depth).
     * @param camera_layout Set 0.
     * @param light_layout  Set 1.
     * @param shadow_layout Set 2.
     * @param sdf_layout    Set 3.
     * @param vert_spv      sdf_quad.vert.
     * @param frag_spv      sdf_capture.frag.
     */
    SdfCapturePass(coopa::gfx::core::Device& device,
                  VkRenderPass capture_render_pass,
                  const coopa::gfx::pipeline::DescriptorSetLayout& camera_layout,
                  const coopa::gfx::pipeline::DescriptorSetLayout& light_layout,
                  const coopa::gfx::pipeline::DescriptorSetLayout& shadow_layout,
                  const coopa::gfx::pipeline::DescriptorSetLayout& sdf_layout,
                  const std::string& vert_spv,
                  const std::string& frag_spv)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        coopa::gfx::pipeline::PipelineDesc desc;
        desc.shaders = {vert_shader_.get(), frag_shader_.get()};
        desc.vertex  = coopa::gfx::VertexLayout::none();
        desc.descriptor_layouts = {&camera_layout, &light_layout, &shadow_layout, &sdf_layout};
        desc.push_constants = {{coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment,
                                0, sizeof(PushConstants)}};
        desc.raster.cull   = coopa::gfx::CullMode::None;
        desc.depth.test    = true;
        desc.depth.write   = true;
        desc.depth.compare = coopa::gfx::CompareOp::Less;
        // See file doc / TransparentCapturePass's: 3 unblended colour attachments.
        desc.blend.color_attachment_count = 3;

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, coopa::gfx::detail::RawRenderPass{capture_render_pass}, desc);
    }

    SdfCapturePass(const SdfCapturePass&) = delete;
    SdfCapturePass& operator=(const SdfCapturePass&) = delete;

    void bind(coopa::gfx::command::CommandBuffer& cmd) const {
        cmd.bind_pipeline(*pipeline_);
    }

    void push(coopa::gfx::command::CommandBuffer& cmd, uint32_t renderer_index) const {
        PushConstants pc{renderer_index};
        cmd.push_constants(coopa::gfx::ShaderStage::Vertex | coopa::gfx::ShaderStage::Fragment, pc);
    }

private:
    std::unique_ptr<coopa::gfx::pipeline::Shader>   vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>   frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_SDF_CAPTURE_PASS_H
