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
 */

#ifndef COOPA_GFX_ENGINE_MESH_H
#define COOPA_GFX_ENGINE_MESH_H

#include <volk/volk.h>
#include <glm/glm.hpp>
#include <fkYAML/node.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/memory/buffer.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/command/command_pool.h>

#include <vector>
#include <array>
#include <stdexcept>
#include <cstdint>

namespace coopa {
namespace gfx {
namespace engine {

/**
 * @struct Vertex
 * @brief Interleaved per-vertex data for the toon and outline pipelines.
 */
struct Vertex {
    glm::vec3 position; /**< Object-space vertex position. */
    glm::vec3 normal;   /**< Object-space vertex normal (normalized). */
    glm::vec2 uv;       /**< Texture UV coordinate. */

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
     * @brief Returns the VkVertexInputAttributeDescriptions for position, normal, uv.
     */
    static std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attrs{};

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

        return attrs;
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

                        indices.push_back(static_cast<uint32_t>(vertices.size()));
                        vertices.push_back(vert);
                    }
                }
            }
        }

        if (vertices.empty()) {
            throw std::runtime_error("[Mesh] No vertices parsed — empty or invalid mesh YAML.");
        }

        // --- Upload to GPU (host-visible buffers, no staging required) ---
        VkDeviceSize vb_size = sizeof(Vertex)   * vertices.size();
        VkDeviceSize ib_size = sizeof(uint32_t) * indices.size();

        auto vb = memory::Buffer::vertex(device, allocator, vb_size);
        auto ib = memory::Buffer::index (device, allocator, ib_size);

        vb.upload(vertices.data(), vb_size);
        ib.upload(indices.data(),  ib_size);

        return Mesh(std::move(vb), std::move(ib),
                    static_cast<uint32_t>(indices.size()));
    }

    // --- Draw calls ---

    /**
     * @brief Binds the vertex and index buffers into the command buffer.
     * @param cmd Command buffer to record into.
     */
    void bind(command::CommandBuffer& cmd) const {
        cmd.bind_vertex_buffer(vertex_buffer_);
        cmd.bind_index_buffer(index_buffer_);
    }

    /**
     * @brief Records an indexed draw call for this mesh.
     * @param cmd Command buffer to record into.
     */
    void draw(command::CommandBuffer& cmd) const {
        cmd.draw_indexed(index_count_);
    }

    /** @brief Returns the number of indices in the mesh. */
    uint32_t index_count() const { return index_count_; }

    // Move only (buffers are not copyable).
    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

private:
    Mesh(memory::Buffer vb, memory::Buffer ib, uint32_t index_count)
        : vertex_buffer_(std::move(vb)),
          index_buffer_(std::move(ib)),
          index_count_(index_count)
    {}

    memory::Buffer vertex_buffer_; /**< Interleaved vertex data. */
    memory::Buffer index_buffer_;  /**< 32-bit index data. */
    uint32_t       index_count_;   /**< Total number of indices to draw. */
};

} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_MESH_H
