#include <gfxcoopa/engine/util/sampler.h>
/**
 * @file ssao_kernel.h
 * @brief SSAO hemisphere sample kernel (UBO) and tiled rotation-noise texture.
 */

#ifndef GFXCOOPA_ENGINE_UTIL_SSAO_KERNEL_H
#define GFXCOOPA_ENGINE_UTIL_SSAO_KERNEL_H

#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>
#include <memory>
#include <random>
#include <cmath>
#include <glm/glm.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/memory/image.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/engine/util/sampler.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace util {



/// Cosine-weighted-ish hemisphere sample kernel (tangent space, z >= 0), uploaded once into a
/// UBO sized for the maximum sample count so ssao.frag's runtime-tunable kernel_size can shrink
/// without ever reading past what was uploaded. Samples are biased toward the origin (Crysis/
/// LearnOpenGL SSAO convention) so a small kernel still looks dense near the surface it shades.
class SsaoKernel {
public:
    static constexpr uint32_t kMaxSamples = 32;

    SsaoKernel(coopa::gfx::core::Device& device, coopa::gfx::memory::Allocator& allocator)
        : buffer_(coopa::gfx::memory::Buffer::uniform(device, allocator, kMaxSamples * sizeof(glm::vec4)))
    {
        // Fixed seed: the kernel is generated once at startup and never touched again, so
        // determinism across runs matters more here than true randomness.
        std::mt19937 rng(0xA0u);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::uniform_real_distribution<float> signed_unit(-1.0f, 1.0f);

        std::vector<glm::vec4> samples(kMaxSamples);
        for (uint32_t i = 0; i < kMaxSamples; ++i) {
            glm::vec3 s(signed_unit(rng), signed_unit(rng), unit(rng));
            s = glm::normalize(s) * unit(rng);

            float t = static_cast<float>(i) / static_cast<float>(kMaxSamples);
            float scale = 0.1f + 0.9f * (t * t);
            s *= scale;

            samples[i] = glm::vec4(s, 0.0f);
        }

        buffer_.upload(samples.data(), kMaxSamples * sizeof(glm::vec4));
    }

    const coopa::gfx::memory::Buffer& buffer() const { return buffer_; }

private:
    coopa::gfx::memory::Buffer buffer_;
};

/// 4x4 tiled rotation-noise texture (RG8_UNORM, one unit XY vector per texel). Generated
/// procedurally (no disk asset) and uploaded through a one-time staging buffer, mirroring
/// SmaaTextures::upload_texture_'s shape.
class SsaoNoiseTexture {
public:
    static constexpr uint32_t kSize = 4;

    SsaoNoiseTexture(coopa::gfx::core::Device& device,
                     coopa::gfx::memory::Allocator& allocator,
                     coopa::gfx::command::CommandPool& cmd_pool)
        : device_(device), allocator_(allocator)
    {
        std::vector<uint8_t> texels(kSize * kSize * 2);
        for (uint32_t i = 0; i < kSize * kSize; ++i) {
            float theta = 2.0f * 3.14159265359f * static_cast<float>(i) / static_cast<float>(kSize * kSize);
            float x = std::cos(theta);
            float y = std::sin(theta);
            texels[i * 2 + 0] = static_cast<uint8_t>((x * 0.5f + 0.5f) * 255.0f);
            texels[i * 2 + 1] = static_cast<uint8_t>((y * 0.5f + 0.5f) * 255.0f);
        }

        upload_texture_(cmd_pool, texels.data());

        // REPEAT so the 4x4 tile covers the full screen (ssao.frag samples it at
        // in_uv * (screen_size / 4)); NEAREST because each texel is a discrete rotation vector,
        // not something to blend between.
        sampler_ = std::make_unique<Sampler>(
            device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT
        );
    }

    VkImageView view() const { return image_->view(); }
    const Sampler& sampler() const { return *sampler_; }

private:
    void upload_texture_(coopa::gfx::command::CommandPool& cmd_pool, const void* data) {
        VkDeviceSize size = static_cast<VkDeviceSize>(kSize) * kSize * 2;

        coopa::gfx::memory::Buffer staging(
            device_, allocator_, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
        );
        staging.upload(data, size);

        image_ = std::make_unique<coopa::gfx::memory::Image>(
            device_, allocator_, kSize, kSize, VK_FORMAT_R8G8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, VMA_MEMORY_USAGE_AUTO, VK_SAMPLE_COUNT_1_BIT
        );

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool        = cmd_pool.handle();
        alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer cmd_handle = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_.handle(), &alloc_info, &cmd_handle);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_handle, &begin_info);

        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image_->handle();
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd_handle, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset      = 0;
        copy_region.bufferRowLength   = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel       = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount     = 1;
        copy_region.imageOffset                     = {0, 0, 0};
        copy_region.imageExtent                     = {kSize, kSize, 1};

        vkCmdCopyBufferToImage(cmd_handle, staging.handle(), image_->handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd_handle, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        vkEndCommandBuffer(cmd_handle);

        VkSubmitInfo submit{};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &cmd_handle;
        vkQueueSubmit(device_.graphics_queue(), 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(device_.graphics_queue());

        vkFreeCommandBuffers(device_.handle(), cmd_pool.handle(), 1, &cmd_handle);
    }

    coopa::gfx::core::Device&      device_;
    coopa::gfx::memory::Allocator& allocator_;

    std::unique_ptr<coopa::gfx::memory::Image> image_;
    std::unique_ptr<Sampler>                   sampler_;
};

} // namespace util
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_UTIL_SSAO_KERNEL_H
