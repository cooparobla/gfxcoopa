/**
 * @file mesh.h
 * @brief GPU-resident mesh data loaded from the Blender-exported YAML format.
 *
 * Reads the mesh YAML structure:
 *   vertices: [[x,y,z], ...]
 *   normals:  [[x,y,z], ...]
 *   uvs:      [[u,v], ...]
 *   faces:    [[i0,i1,i2,i3], ...] (quads, triangulated on load)
 *
 * Data is uploaded into host-visible vertex + index buffers via the
 * Buffer::vertex() and Buffer::index() factory methods. Since these are
 * already host-accessible (mapped), no staging copy is needed.
 *
 * A Mesh can also be built from in-memory arrays (from_arrays()) and, when
 * created with more than one vertex buffer, rewritten every frame
 * (update_vertices()) -- the path a CPU-simulated surface such as cloth needs.
 * See from_arrays()'s doc for why that takes a buffer COUNT rather than
 * flipping a "dynamic" bool.
 */

#ifndef GFXCOOPA_ENGINE_DATA_MESH_H
#define GFXCOOPA_ENGINE_DATA_MESH_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <fkYAML/node.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/types/vertex_layout.h>
#include <gfxcoopa/types/format.h>

#include <vector>
#include <array>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace coopa {
namespace gfx {
namespace engine {
namespace data {

/**
 * @struct Vertex
 * @brief Interleaved per-vertex data for the toon and outline pipelines.
 */
struct Vertex {
    glm::vec3 position; /**< Object-space vertex position. */
    glm::vec3 normal;   /**< Object-space vertex normal (normalized). */
    glm::vec2 uv;       /**< Texture UV coordinate. */
    glm::vec4 tangent;  /**< Object-space vertex tangent (xyz) and handedness sign (w). */

    /**
     * @brief Returns the VkVertexInputBindingDescription for a Vertex stream.
     */
    static VkVertexInputBindingDescription binding_description() {
        VkVertexInputBindingDescription desc{};
        desc.binding   = 0;
        desc.stride    = sizeof(Vertex);
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    /**
     * @brief Returns the VkVertexInputAttributeDescriptions for position, normal, uv, tangent.
     */
    static std::array<VkVertexInputAttributeDescription, 4> attribute_descriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attrs{};

        // location = 0: position (vec3)
        attrs[0].binding  = 0;
        attrs[0].location = 0;
        attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset   = offsetof(Vertex, position);

        // location = 1: normal (vec3)
        attrs[1].binding  = 0;
        attrs[1].location = 1;
        attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset   = offsetof(Vertex, normal);

        // location = 2: uv (vec2)
        attrs[2].binding  = 0;
        attrs[2].location = 2;
        attrs[2].format   = VK_FORMAT_R32G32_SFLOAT;
        attrs[2].offset   = offsetof(Vertex, uv);

        // location = 3: tangent (vec4)
        attrs[3].binding  = 0;
        attrs[3].location = 3;
        attrs[3].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs[3].offset   = offsetof(Vertex, tangent);

        return attrs;
    }

    /**
     * @brief Sealed vertex input layout for binding 0 (position/normal/uv/tangent),
     * replacing binding_description()/attribute_descriptions() above.
     */
    static VertexLayout layout() {
        return VertexLayout{}
            .binding(0, sizeof(Vertex))
            .attribute(0, Format::RGB32_Sfloat,  static_cast<uint32_t>(offsetof(Vertex, position)))
            .attribute(1, Format::RGB32_Sfloat,  static_cast<uint32_t>(offsetof(Vertex, normal)))
            .attribute(2, Format::RG32_Sfloat,   static_cast<uint32_t>(offsetof(Vertex, uv)))
            .attribute(3, Format::RGBA32_Sfloat, static_cast<uint32_t>(offsetof(Vertex, tangent)));
    }
};

/**
 * @struct InstanceData
 * @brief Per-instance vertex stream at binding 1: one world matrix per instance.
 *
 * normal_matrix is deliberately NOT streamed — every consuming shader derives
 * it as transpose(inverse(mat3(in_model))) in-shader instead, halving the
 * per-instance payload (64B vs 128B) and removing a glm::inverse() call per
 * object per frame from the CPU side of every geometry pass. Scenes here use
 * non-uniform scale (e.g. Cornell box walls), so the shader must do the full
 * 3x3 inverse-transpose, not just mat3(in_model) directly.
 */
struct InstanceData {
    glm::mat4 model = glm::mat4(1.0f); /**< Object-to-world; consumed as locations 4..7, one vec4 per column. */

    /** @brief Binding 1, per-instance rate. Binding 0 stays Vertex's per-vertex stream. */
    static VkVertexInputBindingDescription binding_description() {
        VkVertexInputBindingDescription desc{};
        desc.binding   = 1;
        desc.stride    = sizeof(InstanceData);
        desc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        return desc;
    }

    /**
     * @brief Returns the VkVertexInputAttributeDescriptions for locations 4-7 (one mat4).
     *
     * A mat4 attribute occupies four consecutive vec4 locations. These
     * locations are used uniformly by every pipeline that consumes instance
     * data (including the shadow pipelines, whose binding-0 attributes only
     * use location 0), so this one array serves all of them.
     */
    static std::array<VkVertexInputAttributeDescription, 4> attribute_descriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attrs{};
        for (uint32_t i = 0; i < 4; ++i) {
            attrs[i].binding  = 1;
            attrs[i].location = 4 + i;
            attrs[i].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
            attrs[i].offset   = static_cast<uint32_t>(offsetof(InstanceData, model) + i * sizeof(glm::vec4));
        }
        return attrs;
    }

    /**
     * @brief Sealed vertex input layout for binding 1 (per-instance model matrix,
     * locations 4-7), replacing binding_description()/attribute_descriptions() above.
     * Combine with Vertex::layout() via `Vertex::layout().append(InstanceData::layout())`
     * for a pipeline that reads both streams.
     */
    static VertexLayout layout() {
        VertexLayout vl;
        vl.binding(1, sizeof(InstanceData), VertexRate::Instance);
        for (uint32_t i = 0; i < 4; ++i) {
            vl.attribute(4 + i, Format::RGBA32_Sfloat,
                        static_cast<uint32_t>(offsetof(InstanceData, model) + i * sizeof(glm::vec4)));
        }
        return vl;
    }
};

/**
 * @class Mesh
 * @brief GPU-resident interleaved vertex + index buffer, loaded from YAML.
 *
 * Quads in the YAML faces list are triangulated on load (split into 2 triangles).
 *
 * Usage:
 * @code
 * caml::CAMLMap map = caml::CAMLMap::load_yaml("cube.000.yaml");
 * Mesh mesh = Mesh::from_node(device, allocator, cmd_pool, map.get_raw_node());
 * // Per draw call:
 * mesh.bind(cmd);
 * mesh.draw(cmd);
 * @endcode
 */
class Mesh {
public:
    /**
     * @brief Constructs a Mesh by loading from a fkYAML node.
     *
     * The YAML node is expected at the root level with fields:
     *   vertices, normals, uvs, faces
     *
     * @param device    Vulkan logical device.
     * @param allocator VMA allocator.
     * @param cmd_pool  Command pool (used for submit, not staging — data is host-visible).
     * @param node      fkYAML root node of the mesh YAML file.
     * @return A new GPU-resident Mesh.
     * @throws std::runtime_error on parse failures.
     */
    static Mesh from_node(core::Device&             device,
                          memory::Allocator&        allocator,
                          command::CommandPool&     cmd_pool,
                          const fkyaml::node&       node)
    {
        (void)cmd_pool; // Host-visible buffers; staging not required.

        // --- Parse positions ---
        std::vector<glm::vec3> positions;
        if (node.contains("vertices")) {
            for (const auto& v : node.at("vertices")) {
                positions.push_back({
                    v.at(0).get_value<float>(),
                    v.at(1).get_value<float>(),
                    v.at(2).get_value<float>()
                });
            }
        }

        // --- Parse normals ---
        std::vector<glm::vec3> normals;
        if (node.contains("normals")) {
            for (const auto& n : node.at("normals")) {
                normals.push_back({
                    n.at(0).get_value<float>(),
                    n.at(1).get_value<float>(),
                    n.at(2).get_value<float>()
                });
            }
        }

        // --- Parse UVs ---
        std::vector<glm::vec2> uvs;
        if (node.contains("uvs")) {
            for (const auto& uv : node.at("uvs")) {
                uvs.push_back({
                    uv.at(0).get_value<float>(),
                    uv.at(1).get_value<float>()
                });
            }
        }

        // --- Parse tangents ---
        std::vector<glm::vec4> tangents;
        if (node.contains("tangents")) {
            for (const auto& tan : node.at("tangents")) {
                float w = (tan.size() > 3) ? tan.at(3).get_value<float>() : 1.0f;
                tangents.push_back({
                    tan.at(0).get_value<float>(),
                    tan.at(1).get_value<float>(),
                    tan.at(2).get_value<float>(),
                    w
                });
            }
        }

        // --- Parse faces and build interleaved vertices + indices ---
        // Faces are quads [i0,i1,i2,i3]; split into two triangles:
        //   tri1: [i0, i1, i2]   tri2: [i0, i2, i3]
        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;

        if (node.contains("faces")) {
            for (const auto& face : node.at("faces")) {
                // Each face element is a list of vertex indices.
                std::vector<uint32_t> face_indices;
                for (const auto& idx_node : face) {
                    face_indices.push_back(idx_node.get_value<uint32_t>());
                }

                if (face_indices.size() < 3) continue;

                // Fan triangulation from the first vertex.
                for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                    uint32_t v_indices[3] = {
                        face_indices[0],
                        face_indices[i],
                        face_indices[i + 1]
                    };

                    for (uint32_t vi : v_indices) {
                        Vertex vert{};
                        vert.position = (vi < positions.size()) ? positions[vi] : glm::vec3(0.0f);
                        vert.normal   = (vi < normals.size())   ? normals[vi]   : glm::vec3(0.0f, 1.0f, 0.0f);
                        vert.uv       = (vi < uvs.size())       ? uvs[vi]       : glm::vec2(0.0f);
                        vert.tangent  = (vi < tangents.size())  ? tangents[vi]  : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

                        indices.push_back(static_cast<uint32_t>(vertices.size()));
                        vertices.push_back(vert);
                    }
                }
            }
        }

        if (vertices.empty()) {
            throw std::runtime_error("[Mesh] No vertices parsed — empty or invalid mesh YAML.");
        }

        // --- Fallback tangent computation if tangents were absent in YAML ---
        if (tangents.empty()) {
            std::vector<glm::vec3> tan_sum(vertices.size(), glm::vec3(0.0f));
            for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
                const auto& v0 = vertices[i];
                const auto& v1 = vertices[i + 1];
                const auto& v2 = vertices[i + 2];

                glm::vec3 edge1 = v1.position - v0.position;
                glm::vec3 edge2 = v2.position - v0.position;
                glm::vec2 deltaUV1 = v1.uv - v0.uv;
                glm::vec2 deltaUV2 = v2.uv - v0.uv;

                float f = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
                glm::vec3 t;
                if (std::abs(f) > 1e-6f) {
                    float r = 1.0f / f;
                    t = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) * r;
                } else {
                    t = glm::vec3(1.0f, 0.0f, 0.0f);
                }
                tan_sum[i]     += t;
                tan_sum[i + 1] += t;
                tan_sum[i + 2] += t;
            }

            for (size_t i = 0; i < vertices.size(); ++i) {
                glm::vec3 t = glm::length(tan_sum[i]) > 1e-5f ? glm::normalize(tan_sum[i]) : glm::vec3(1.0f, 0.0f, 0.0f);
                glm::vec3 n = vertices[i].normal;
                t = glm::normalize(t - n * glm::dot(n, t));
                vertices[i].tangent = glm::vec4(t, 1.0f);
            }
        }

        // --- Upload to GPU (host-visible buffers, no staging required) ---
        VkDeviceSize vb_size = sizeof(Vertex)   * vertices.size();
        VkDeviceSize ib_size = sizeof(uint32_t) * indices.size();

        auto vb = memory::Buffer::vertex(device, allocator, vb_size);
        auto ib = memory::Buffer::index (device, allocator, ib_size);

        vb.upload(vertices.data(), vb_size);
        ib.upload(indices.data(),  ib_size);

        std::vector<memory::Buffer> vbs;
        vbs.push_back(std::move(vb)); // a YAML-loaded mesh is static: exactly one buffer

        // Object-space AABB over the triangulated vertex stream (not the raw
        // `positions` array, which can contain entries no face references) --
        // this is exactly the geometry that gets drawn. Used by callers that
        // need to fit a projection (e.g. a directional shadow-map ortho box)
        // to what a mesh instance actually occupies.
        glm::vec3 bounds_min(std::numeric_limits<float>::max());
        glm::vec3 bounds_max(std::numeric_limits<float>::lowest());
        for (const auto& v : vertices) {
            bounds_min = glm::min(bounds_min, v.position);
            bounds_max = glm::max(bounds_max, v.position);
        }

        return Mesh(std::move(vbs), std::move(ib),
                    static_cast<uint32_t>(indices.size()),
                    static_cast<uint32_t>(vertices.size()),
                    bounds_min, bounds_max);
    }

    /**
     * @brief Constructs a Mesh from already-interleaved vertex and index arrays.
     *
     * Unlike from_node(), which unwelds every triangle corner into its own Vertex, this keeps the
     * caller's indexed topology exactly as given. That is the whole point for a simulated surface:
     * a cloth with one Vertex per particle rewrites `particle_count` vertices per frame, where an
     * unwelded copy would rewrite six times that and then have to average the duplicates' normals
     * back together to avoid faceting.
     *
     * `buffer_count` is a COUNT, not a `bool dynamic`, because the correct number is a property of
     * the presentation loop (how many frames it keeps in flight), not of the mesh -- and the
     * caller is the only one who knows it. Pass 1 for a mesh that is uploaded once; pass
     * `Context::frames_in_flight()` for one that is rewritten per frame. A single shared buffer
     * would be wrong for the latter: this engine's pipeline never waits per frame, so a rewrite
     * would race a still-in-flight GPU read of the previous frame -- the same reason
     * toyengine's DebugLinePass keeps per-frame-in-flight buffers.
     *
     * @param device       Vulkan logical device.
     * @param allocator    VMA allocator.
     * @param vertices     Interleaved vertex data; must be non-empty.
     * @param indices      32-bit index data; must be non-empty.
     * @param buffer_count Number of vertex buffers to allocate (clamped to at least 1).
     * @return A new GPU-resident Mesh.
     * @throws std::runtime_error if either array is empty.
     */
    static Mesh from_arrays(core::Device&               device,
                            memory::Allocator&          allocator,
                            const std::vector<Vertex>&  vertices,
                            const std::vector<uint32_t>& indices,
                            uint32_t                    buffer_count = 1)
    {
        if (vertices.empty() || indices.empty()) {
            throw std::runtime_error("[Mesh] from_arrays() needs non-empty vertex and index arrays.");
        }

        const VkDeviceSize vb_size = sizeof(Vertex) * vertices.size();
        const VkDeviceSize ib_size = sizeof(uint32_t) * indices.size();
        const uint32_t count = (buffer_count < 1u) ? 1u : buffer_count;

        std::vector<memory::Buffer> vbs;
        vbs.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto vb = memory::Buffer::vertex(device, allocator, vb_size);
            vb.upload(vertices.data(), vb_size);   // seed every slot, so frame 0 draws correctly
            vbs.push_back(std::move(vb));           // whichever slot it happens to land on
        }

        auto ib = memory::Buffer::index(device, allocator, ib_size);
        ib.upload(indices.data(), ib_size);

        glm::vec3 bounds_min(std::numeric_limits<float>::max());
        glm::vec3 bounds_max(std::numeric_limits<float>::lowest());
        for (const auto& v : vertices) {
            bounds_min = glm::min(bounds_min, v.position);
            bounds_max = glm::max(bounds_max, v.position);
        }

        return Mesh(std::move(vbs), std::move(ib),
                    static_cast<uint32_t>(indices.size()),
                    static_cast<uint32_t>(vertices.size()),
                    bounds_min, bounds_max);
    }

    /**
     * @brief Rewrites this frame's vertex buffer and makes it the one bind() will use.
     *
     * Call exactly once per frame, before the frame's command buffer is recorded, passing that
     * frame's in-flight slot (Context::current_frame()). Writing into the slot the GPU is not
     * currently reading is what makes this safe without a per-frame fence wait.
     *
     * The object-space bounds are recomputed here rather than left at their creation values: the
     * directional shadow pass fits its ortho box to bounds_min()/bounds_max(), so a cloth that has
     * drooped well outside its rest-pose box would have its shadow clipped.
     *
     * @param data       Vertices to upload; must hold at least `count` entries.
     * @param count      Number of vertices to write. Must not exceed vertex_count().
     * @param frame_slot In-flight frame index; taken modulo the buffer count, so passing a
     *                   monotonically increasing frame counter also works.
     * @throws std::runtime_error if `count` exceeds the allocated vertex count.
     */
    void update_vertices(const Vertex* data, std::size_t count, uint32_t frame_slot) {
        if (!data || count == 0) return;
        if (count > vertex_count_) {
            throw std::runtime_error("[Mesh] update_vertices() exceeds the allocated vertex count.");
        }
        const uint32_t slot = frame_slot % static_cast<uint32_t>(vertex_buffers_.size());
        vertex_buffers_[slot].upload(data, sizeof(Vertex) * count);
        active_slot_ = slot;

        glm::vec3 lo(std::numeric_limits<float>::max());
        glm::vec3 hi(std::numeric_limits<float>::lowest());
        for (std::size_t i = 0; i < count; ++i) {
            lo = glm::min(lo, data[i].position);
            hi = glm::max(hi, data[i].position);
        }
        bounds_min_ = lo;
        bounds_max_ = hi;
    }

    /** @brief True if this mesh has more than one vertex buffer, i.e. is safe to rewrite per frame. */
    bool is_dynamic() const { return vertex_buffers_.size() > 1; }

    /** @brief Number of vertices the vertex buffers were allocated for. */
    uint32_t vertex_count() const { return vertex_count_; }

    // --- Draw calls ---

    /**
     * @brief Binds the vertex and index buffers into the command buffer.
     * @param cmd Command buffer to record into.
     */
    void bind(command::CommandBuffer& cmd) const {
        // A static mesh has exactly one slot and active_slot_ never leaves 0, so this is the same
        // single-buffer bind it always was; only a dynamic mesh ever advances the slot.
        cmd.bind_vertex_buffer(vertex_buffers_[active_slot_]);
        cmd.bind_index_buffer(index_buffer_);
    }

    /**
     * @brief Records an indexed draw call for this mesh.
     * @param cmd Command buffer to record into.
     */
    void draw(command::CommandBuffer& cmd) const {
        cmd.draw_indexed(index_count_);
    }

    /**
     * @brief Records an instanced indexed draw call for this mesh.
     *
     * Callers bind the shared per-instance stream (see InstanceData,
     * engine::util::InstanceBatcher::bind()) once per pass before looping
     * batches — bind() above only binds slot 0 (this mesh's own vertex/index
     * buffers), so per-batch mesh rebinding never disturbs the instance
     * stream at slot 1.
     *
     * @param cmd            Command buffer to record into.
     * @param instance_count Number of instances to draw.
     * @param first_instance Offset into the bound instance-rate stream.
     */
    void draw(command::CommandBuffer& cmd, uint32_t instance_count, uint32_t first_instance) const {
        cmd.draw_indexed(index_count_, 0, 0, instance_count, first_instance);
    }

    /** @brief Returns the number of indices in the mesh. */
    uint32_t index_count() const { return index_count_; }

    /** @brief Returns the minimum corner of the object-space bounding box. */
    const glm::vec3& bounds_min() const { return bounds_min_; }

    /** @brief Returns the maximum corner of the object-space bounding box. */
    const glm::vec3& bounds_max() const { return bounds_max_; }

    // Move only (buffers are not copyable).
    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

private:
    Mesh(std::vector<memory::Buffer> vbs, memory::Buffer ib, uint32_t index_count,
         uint32_t vertex_count, const glm::vec3& bounds_min, const glm::vec3& bounds_max)
        : vertex_buffers_(std::move(vbs)),
          index_buffer_(std::move(ib)),
          index_count_(index_count),
          vertex_count_(vertex_count),
          bounds_min_(bounds_min),
          bounds_max_(bounds_max)
    {}

    /** @brief One entry for a static mesh; one per frame-in-flight for a dynamic one. A vector
     *         rather than a fixed array so a static mesh pays for exactly one buffer -- most
     *         meshes in a scene are static, and triple-buffering all of them would waste GPU
     *         memory proportional to the whole scene. */
    std::vector<memory::Buffer> vertex_buffers_;
    memory::Buffer index_buffer_;  /**< 32-bit index data; topology is fixed, so never per-frame. */
    uint32_t       index_count_;   /**< Total number of indices to draw. */
    uint32_t       vertex_count_;  /**< Vertices each buffer was allocated for. */
    glm::vec3      bounds_min_;    /**< Object-space AABB minimum corner. */
    glm::vec3      bounds_max_;    /**< Object-space AABB maximum corner. */

    /** @brief Which slot bind() uses. Mutable-free: only update_vertices() advances it, and that
     *         runs before the frame is recorded, never during. */
    uint32_t       active_slot_ = 0;
};

} // namespace data
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_MESH_H
