/**
 * @file mesh_loader.h
 * @brief coopa::asset loader for gfx::data::Mesh — reads and parses a Blender-exported mesh YAML.
 *
 * decode_typed() runs off-thread and only reads + parses the YAML file (pure
 * CPU work via fkYAML); finalize_typed() runs on the main thread and is
 * where the GPU vertex/index buffers actually get created and uploaded
 * (via data::Mesh::from_node).
 *
 * Replaces the ad-hoc `mesh_cache` unordered_map that used to live in a
 * lambda closure inside register_render_components() — that cache keyed on
 * the *logical* mesh name (e.g. "cube.000"), so two different scenes with
 * different meshes/cube.000.yaml files would silently collide and share one
 * upload. AssetManager keys on the *resolved* filesystem path instead (see
 * coopa/asset/asset_id.h), so this can't happen.
 */

#ifndef GFXCOOPA_ENGINE_LOADERS_MESH_LOADER_H
#define GFXCOOPA_ENGINE_LOADERS_MESH_LOADER_H

#include <coopa/asset/asset_loader.h>
#include <coopa/asset/asset_id.h>

#include <fkYAML/node.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/engine/data/mesh.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace coopa {
namespace gfx {
namespace engine {
namespace loaders {

/**
 * @class MeshLoader
 * @brief Registers as the coopa::asset loader for gfx::data::Mesh.
 *
 * @code
 * assets.register_loader<gfx::data::Mesh>(
 *     std::make_unique<MeshLoader>(device, allocator, cmd_pool));
 * auto mesh = assets.load<gfx::data::Mesh>("meshes/cube.000.yaml", ctx.scene_dir);
 * @endcode
 */
class MeshLoader : public coopa::asset::TypedAssetLoader<data::Mesh, fkyaml::node> {
public:
    MeshLoader(core::Device& device, memory::Allocator& allocator, command::CommandPool& cmd_pool)
        : device_(device), allocator_(allocator), cmd_pool_(cmd_pool) {}

    std::shared_ptr<fkyaml::node> decode_typed(const coopa::asset::AssetId& id,
                                               const coopa::asset::LoadContext& ctx) override {
        std::ifstream ifs(ctx.resolved_path);
        if (!ifs) {
            throw std::runtime_error("[MeshLoader] Failed to open '" + id.path() + "'");
        }
        return std::make_shared<fkyaml::node>(fkyaml::node::deserialize(ifs));
    }

    std::shared_ptr<data::Mesh> finalize_typed(std::shared_ptr<fkyaml::node> node,
                                               const coopa::asset::AssetId&,
                                               const coopa::asset::LoadContext&) override {
        return std::make_shared<data::Mesh>(data::Mesh::from_node(device_, allocator_, cmd_pool_, *node));
    }

    const char* type_name() const override { return "Mesh"; }

private:
    core::Device&         device_;
    memory::Allocator&     allocator_;
    command::CommandPool&  cmd_pool_;
};

} // namespace loaders
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_LOADERS_MESH_LOADER_H
