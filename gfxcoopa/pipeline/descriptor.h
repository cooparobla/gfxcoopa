/**
 * @file descriptor.h
 * @brief Vulkan descriptor pool, layout, and set management.
 *
 * Provides three RAII classes mirroring OpenGL's uniform/texture binding model:
 * - DescriptorPool: allocates descriptor sets.
 * - DescriptorSetLayout: describes binding points (like an interface block layout).
 * - DescriptorSet: a single set of bound resources (bind_buffer = glBindBufferBase,
 *   bind_image = glBindTexture).
 */

#ifndef COOPA_GFX_PIPELINE_DESCRIPTOR_H
#define COOPA_GFX_PIPELINE_DESCRIPTOR_H

#include <volk/volk.h>
#include <vector>
#include <stdexcept>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/util/error.h>

namespace coopa {
namespace gfx {
namespace pipeline {

// ---------------------------------------------------------
// DescriptorPool
// ---------------------------------------------------------

/**
 * @class DescriptorPool
 * @brief RAII wrapper around VkDescriptorPool.
 *
 * Allocates a pool with the given pool sizes. All DescriptorSet objects
 * allocated from this pool are freed when the pool is destroyed.
 */
class DescriptorPool {
public:
    /**
     * @brief Creates a VkDescriptorPool.
     * @param device     The logical device.
     * @param max_sets   Maximum number of descriptor sets that can be allocated.
     * @param pool_sizes Per-type descriptor counts for the pool.
     */
    DescriptorPool(core::Device&                            device,
                   uint32_t                                 max_sets,
                   const std::vector<VkDescriptorPoolSize>& pool_sizes)
        : device_(device)
    {
        VkDescriptorPoolCreateInfo info{};
        info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets       = max_sets;
        info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        info.pPoolSizes    = pool_sizes.data();
        info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        GFX_VK_CHECK(vkCreateDescriptorPool(device_.handle(), &info, nullptr, &pool_));
    }

    /**
     * @brief Destroys the descriptor pool.
     */
    ~DescriptorPool() {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_.handle(), pool_, nullptr);
        }
    }

    /// @brief Non-copyable.
    DescriptorPool(const DescriptorPool&) = delete;
    /// @brief Non-copyable.
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    /**
     * @brief Returns the underlying VkDescriptorPool handle.
     * @return Raw VkDescriptorPool.
     */
    VkDescriptorPool handle() const { return pool_; }

private:
    core::Device&    device_;              /**< Owning logical device (not owned). */
    VkDescriptorPool pool_ = VK_NULL_HANDLE;/**< The Vulkan descriptor pool. */
};

// ---------------------------------------------------------
// DescriptorSetLayout
// ---------------------------------------------------------

/**
 * @class DescriptorSetLayout
 * @brief RAII wrapper around VkDescriptorSetLayout.
 *
 * Describes the binding points used by a pipeline stage, analogous to
 * declaring a uniform block or sampler in GLSL.
 */
class DescriptorSetLayout {
public:
    /**
     * @brief Creates a VkDescriptorSetLayout from a list of binding descriptions.
     * @param device   The logical device.
     * @param bindings Vector of VkDescriptorSetLayoutBinding structs.
     */
    DescriptorSetLayout(core::Device&                                    device,
                        const std::vector<VkDescriptorSetLayoutBinding>& bindings)
        : device_(device)
    {
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings    = bindings.data();

        GFX_VK_CHECK(vkCreateDescriptorSetLayout(device_.handle(), &info, nullptr, &layout_));
    }

    /**
     * @brief Destroys the descriptor set layout.
     */
    ~DescriptorSetLayout() {
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_.handle(), layout_, nullptr);
        }
    }

    /// @brief Non-copyable.
    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    /// @brief Non-copyable.
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

    /**
     * @brief Returns the underlying VkDescriptorSetLayout handle.
     * @return Raw VkDescriptorSetLayout.
     */
    VkDescriptorSetLayout handle() const { return layout_; }

private:
    core::Device&         device_;               /**< Owning logical device (not owned). */
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;/**< The Vulkan descriptor set layout. */
};

// ---------------------------------------------------------
// DescriptorSet
// ---------------------------------------------------------

/**
 * @class DescriptorSet
 * @brief A single allocated and updatable descriptor set.
 *
 * Wraps VkDescriptorSet allocation from a pool. Provides bind_buffer()
 * and bind_image() for updating bindings, mirroring glBindBufferBase and
 * glBindTexture in the OpenGL API.
 */
class DescriptorSet {
public:
    /**
     * @brief Allocates a VkDescriptorSet from the given pool and layout.
     * @param device The logical device.
     * @param pool   The descriptor pool to allocate from.
     * @param layout The layout describing the set's bindings.
     */
    DescriptorSet(core::Device&         device,
                  DescriptorPool&       pool,
                  DescriptorSetLayout&  layout)
        : device_(device), pool_(pool.handle())
    {
        VkDescriptorSetLayout raw_layout = layout.handle();

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool     = pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts        = &raw_layout;

        GFX_VK_CHECK(vkAllocateDescriptorSets(device_.handle(), &alloc_info, &set_));
    }

    /**
     * @brief Frees the descriptor set back into the pool.
     */
    ~DescriptorSet() {
        if (set_ != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(device_.handle(), pool_, 1, &set_);
        }
    }

    /// @brief Non-copyable.
    DescriptorSet(const DescriptorSet&) = delete;
    /// @brief Non-copyable.
    DescriptorSet& operator=(const DescriptorSet&) = delete;

    /**
     * @brief Binds a uniform buffer to the given binding point (like glBindBufferBase).
     * @param binding The binding index within the set.
     * @param buffer  The Buffer containing uniform data.
     */
    void bind_buffer(uint32_t binding, const memory::Buffer& buffer) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = buffer.handle();
        buffer_info.offset = 0;
        buffer_info.range  = buffer.size();

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = set_;
        write.dstBinding      = binding;
        write.dstArrayElement = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &buffer_info;

        vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
    }

    /**
     * @brief Binds a storage buffer (SSBO) to the given binding point.
     * @param binding The binding index within the set.
     * @param buffer  The Buffer containing storage data.
     */
    void bind_storage_buffer(uint32_t binding, const memory::Buffer& buffer) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = buffer.handle();
        buffer_info.offset = 0;
        buffer_info.range  = buffer.size();

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = set_;
        write.dstBinding      = binding;
        write.dstArrayElement = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &buffer_info;

        vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
    }

    /**
     * @brief Binds a combined image sampler to the given binding point (like glBindTexture).
     * @param binding    The binding index within the set.
     * @param image_view The VkImageView to expose to the shader.
     * @param sampler    The VkSampler controlling filtering and addressing.
     */
    void bind_image(uint32_t binding, VkImageView image_view, VkSampler sampler) {
        VkDescriptorImageInfo image_info{};
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView   = image_view;
        image_info.sampler     = sampler;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = set_;
        write.dstBinding      = binding;
        write.dstArrayElement = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &image_info;

        vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
    }

    /**
     * @brief Returns the underlying VkDescriptorSet handle.
     * @return Raw VkDescriptorSet.
     */
    VkDescriptorSet handle() const { return set_; }

private:
    core::Device&    device_;               /**< Owning logical device (not owned). */
    VkDescriptorPool pool_  = VK_NULL_HANDLE;/**< Pool this set was allocated from (not owned). */
    VkDescriptorSet  set_   = VK_NULL_HANDLE;/**< The allocated descriptor set. */
};

} // namespace pipeline
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_PIPELINE_DESCRIPTOR_H
