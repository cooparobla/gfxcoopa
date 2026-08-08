#include <gfxcoopa/engine/util/fullscreen_quad.h>
#include <gfxcoopa/engine/util/sampler.h>
/**
 * @file env_prefilter_pass.h
 * @brief GGX-prefiltered analytic-sky bake into a reflection probe cubemap.
 */

#ifndef GFXCOOPA_ENGINE_PASSES_ENV_PREFILTER_PASS_H
#define GFXCOOPA_ENGINE_PASSES_ENV_PREFILTER_PASS_H

#include <volk/volk.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/pipeline/pipeline.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/pipeline/shader.h>
#include <gfxcoopa/pipeline/descriptor.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/engine/targets/cubemap_target.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace passes {



/// GGX-prefilters mips 1..N-1 of a targets::CubemapTarget from its real mip-0 capture
/// (see probe_capture.frag / probe_sky_background.frag, which write mip 0).
/// Reads the source via a samplerCube bound to targets::CubemapTarget::mip0_cube_view()
/// -- mip 0 is never written by this pass.
class EnvPrefilterPass {
public:
    struct PushConstants {
        int32_t face;
        float   roughness;
    };

    EnvPrefilterPass(coopa::gfx::core::Device& device,
                     coopa::gfx::pipeline::RenderPass& prefilter_pass,
                     const std::string& vert_spv,
                     const std::string& frag_spv,
                     VkDescriptorSetLayout source_layout)
        : device_(device)
    {
        vert_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, vert_spv, VK_SHADER_STAGE_VERTEX_BIT);
        frag_shader_ = std::make_unique<coopa::gfx::pipeline::Shader>(device, frag_spv, VK_SHADER_STAGE_FRAGMENT_BIT);

        coopa::gfx::pipeline::PipelineConfig cfg{};
        cfg.cull_mode   = VK_CULL_MODE_NONE;
        cfg.depth_test  = false;
        cfg.depth_write = false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = sizeof(PushConstants);

        pipeline_ = std::make_unique<coopa::gfx::pipeline::Pipeline>(
            device, prefilter_pass,
            std::vector<coopa::gfx::pipeline::Shader*>{vert_shader_.get(), frag_shader_.get()},
            std::vector<VkVertexInputBindingDescription>{},
            std::vector<VkVertexInputAttributeDescription>{},
            std::vector<VkDescriptorSetLayout>{source_layout},
            cfg,
            std::vector<VkPushConstantRange>{pc_range}
        );
    }

    /// GGX-prefilters mips 1..N-1 of target from source_set (a samplerCube
    /// bound to target's mip0_cube_view()). Mip 0 is left untouched -- it's
    /// expected to already hold a real capture. Caller owns the command
    /// buffer and the final shader-read layout transition.
    void execute(coopa::gfx::command::CommandBuffer& cmd,
                 targets::CubemapTarget& target,
                 const coopa::gfx::pipeline::DescriptorSet& source_set) const
    {
        const uint32_t mips = target.mip_levels();
        if (mips < 2) return; // nothing to prefilter

        const float denom = static_cast<float>(mips - 1);

        cmd.bind_pipeline(*pipeline_);
        cmd.bind_descriptor_set(pipeline_->layout(), source_set, 0);
        for (uint32_t f = 0; f < 6; ++f) {
            for (uint32_t m = 1; m < mips; ++m) { // mip 0 is the geometry capture
                target.begin_face_mip_pass(cmd, f, m);

                PushConstants pc{};
                pc.face      = static_cast<int32_t>(f);
                pc.roughness = static_cast<float>(m) / denom;
                cmd.push_constants(pipeline_->layout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(PushConstants), &pc);

                cmd.draw(3);
                target.end_face_pass(cmd);
            }
        }
    }

private:
    coopa::gfx::core::Device& device_;

    std::unique_ptr<coopa::gfx::pipeline::Shader>   vert_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Shader>   frag_shader_;
    std::unique_ptr<coopa::gfx::pipeline::Pipeline> pipeline_;
};

} // namespace passes
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_PASSES_ENV_PREFILTER_PASS_H
