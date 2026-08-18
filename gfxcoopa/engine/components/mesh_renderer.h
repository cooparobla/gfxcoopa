/**
 * @file mesh_renderer.h
 * @brief Component that references a GPU mesh for rendering.
 *
 * Holds a shared_ptr to a coopa::gfx::engine::data::Mesh (GPU-resident).
 * The mesh is loaded and assigned by register_render_components()'s
 * "MeshRenderer" parser after parsing the scene YAML's MeshRenderer
 * component. Multiple objects can share the same Mesh instance.
 */

#ifndef GFXCOOPA_ENGINE_COMPONENTS_MESH_RENDERER_H
#define GFXCOOPA_ENGINE_COMPONENTS_MESH_RENDERER_H

#include <coopa/scene/component.h>
#include <glm/glm.hpp>
#include <string>
#include <memory>

#include <gfxcoopa/engine/data/mesh.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace components {

/**
 * @brief How a material's alpha channel is interpreted by the renderer.
 */
enum class AlphaMode {
    Opaque, /**< Alpha ignored; the material is fully opaque. */
    Mask,   /**< Alpha-tested: discarded below alpha_cutoff, otherwise fully opaque. */
    Blend   /**< Alpha-blended by the forward transparent pass. */
};

struct PBRMaterial {
    glm::vec3 albedo    = {0.8f, 0.8f, 0.8f};
    float     metallic  = 0.0f;
    float     roughness = 0.5f;
    float     ao        = 1.0f;

    float     alpha        = 1.0f;              /**< Straight (non-premultiplied) opacity. */
    AlphaMode alpha_mode   = AlphaMode::Opaque; /**< Selects the draw list this material joins. */
    float     alpha_cutoff = 0.5f;              /**< Only meaningful for AlphaMode::Mask. */

    std::string texture_albedo             = "";
    std::string texture_normal             = "";
    std::string texture_metallic_roughness = "";

    /** @brief Returns true when this material must be drawn by the forward transparent pass. */
    bool is_blended() const { return alpha_mode == AlphaMode::Blend; }

    /**
     * @brief Returns the alpha cutoff as the GPU shader sees it.
     *
     * A value of 0.0 disables the discard entirely, which is what OPAQUE and BLEND
     * materials need since only MASK performs an alpha test in the G-buffer pass.
     *
     * @return 0.0 for non-Mask materials, otherwise alpha_cutoff clamped to (0, 1].
     */
    float gpu_alpha_cutoff() const {
        return alpha_mode == AlphaMode::Mask ? glm::clamp(alpha_cutoff, 0.0001f, 1.0f) : 0.0f;
    }
};

/**
 * @class MeshRenderer
 * @brief References a GPU Mesh and marks the owning object as renderable.
 *
 * The mesh_path field stores the logical path from the scene YAML (e.g. "cube.000").
 * register_render_components()'s "MeshRenderer" parser resolves this to a full
 * filesystem path, loads the mesh YAML, creates a coopa::gfx::engine::data::Mesh,
 * and sets it here.
 */
class MeshRenderer : public coopa::scene::Component {
public:
    MeshRenderer() = default;

    std::string type_name() const override { return "MeshRenderer"; }

    /**
     * @brief Sets the logical mesh path (e.g. "cube.000").
     *
     * Called by the registered parser when parsing the YAML component data.
     *
     * @param path Logical mesh path relative to the scene's meshes/ directory.
     */
    void set_mesh_path(const std::string& path) { mesh_path_ = path; }

    /** @brief Returns the logical mesh path. */
    const std::string& mesh_path() const { return mesh_path_; }

    /**
     * @brief Sets the GPU mesh (shared ownership).
     *
     * Called once the Mesh has been uploaded to the GPU.
     *
     * @param mesh Shared pointer to the GPU-resident Mesh.
     */
    void set_mesh(std::shared_ptr<coopa::gfx::engine::data::Mesh> mesh) {
        mesh_ = std::move(mesh);
    }

    /**
     * @brief Returns the GPU mesh, or nullptr if not yet loaded.
     */
    std::shared_ptr<coopa::gfx::engine::data::Mesh> get_mesh() const { return mesh_; }

    /**
     * @brief Returns true if this renderer has a valid GPU mesh ready to draw.
     */
    bool is_ready() const { return mesh_ != nullptr; }

    PBRMaterial material;

    /// Whether reflection probes may bake this renderer into their captured
    /// cubemaps. Defaults to true (static scenery). Set false on animated /
    /// otherwise-moving objects: probe bakes are static one-shot captures, so
    /// a moving object frozen into a cubemap goes stale the instant it moves
    /// -- and for an object reflecting itself, shows up as a visible
    /// mismatch (probe capture uses a simplified, non-recursive shading path
    /// that looks different from the main render). Matches standard engine
    /// practice (e.g. Unity's per-renderer "Reflection Probes" toggle):
    /// static geometry contributes real captured detail to reflections,
    /// dynamic objects rely on SSR alone.
    bool affects_reflection_probes = true;

private:
    std::string mesh_path_;                                    /**< Logical mesh path from YAML. */
    std::shared_ptr<coopa::gfx::engine::data::Mesh> mesh_;          /**< GPU-resident mesh (shared). */
};

} // namespace components
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_COMPONENTS_MESH_RENDERER_H
