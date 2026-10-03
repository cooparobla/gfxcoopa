/**
 * @file mesh_renderer.h
 * @brief Component that references a GPU mesh for rendering.
 *
 * Holds a coopa::asset::AssetHandle to a coopa::gfx::engine::data::Mesh
 * (GPU-resident), obtained from register_render_components()'s
 * "MeshRenderer" parser after parsing the scene YAML's MeshRenderer
 * component. Multiple objects loading the same resolved mesh path share
 * one underlying asset slot (see coopa::asset::AssetManager); the handle
 * itself is a lightweight, refcounted, non-owning reference.
 */

#ifndef GFXCOOPA_ENGINE_COMPONENTS_MESH_RENDERER_H
#define GFXCOOPA_ENGINE_COMPONENTS_MESH_RENDERER_H

#include <coopa/scene/component.h>
#include <coopa/asset/asset_handle.h>
#include <glm/glm.hpp>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include <gfxcoopa/engine/data/mesh.h>
#include <gfxcoopa/engine/data/texture.h>

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

/**
 * @struct PBRMaterial
 * @brief A mesh's surface parameters, as authored in scene YAML.
 *
 * Covers the Cook-Torrance inputs (albedo, metallic, roughness, emissive), the
 * alpha mode that decides which draw list the mesh joins, optional texture map
 * paths, and the name of a SurfaceShaderDesc variant when the material overrides
 * the stock pipeline.
 */
struct PBRMaterial {
    glm::vec3 albedo    = {0.8f, 0.8f, 0.8f};
    float     metallic  = 0.0f;
    float     roughness = 0.5f;
    float     ao        = 1.0f;

    float     alpha        = 1.0f;              /**< Straight (non-premultiplied) opacity. */
    AlphaMode alpha_mode   = AlphaMode::Opaque; /**< Selects the draw list this material joins. */
    float     alpha_cutoff = 0.5f;              /**< Only meaningful for AlphaMode::Mask. */

    /// Backface culling for this material, in the stock (no `shader:` override) G-buffer/Mask
    /// pipeline only. false renders both winding orders -- for a mesh that is not a closed solid, or
    /// where the CUTOUT/Mask alpha test intentionally exposes the interior (see
    /// assets/scenes/pixel_demo/scene.yaml's cutout_sphere.000). Ignored when `shader` is set
    /// (a named SurfaceShaderDesc's own `cull` always wins -- see GBufferPipeline::add_variant())
    /// and by the forward BLEND transparent pass (TransparentPass hardcodes CullMode::Back for
    /// every BLEND material -- see that pass's add_variant() doc). NOT consumed by
    /// SdfRenderer, same treatment as refraction/ior below -- parses and stores without effect.
    bool      cull_backfaces = true;

    /// Emissive radiance colour, linear space, before emissive_strength. Added to the
    /// shaded result *after* AO/SSAO attenuation -- an emissive surface glows even in a
    /// dark crevice. Not a real-time light source: no pass gathers emissive into direct
    /// lighting, so it reaches neighbouring surfaces only via the screen-space SSR/SSGI
    /// terms that sample lit scene colour (and, for baked probes, via GiBaker). Deferred
    /// (opaque G-buffer) path only -- the forward transparent/probe-capture paths don't
    /// carry this field.
    glm::vec3 emissive = {0.0f, 0.0f, 0.0f};

    /// Multiplier on `emissive`, kept separate so a scene can drive an HDR intensity (>1)
    /// without denormalising the authored colour. Only ever multiplied together on the
    /// way to the GPU -- see gpu_emissive().
    float     emissive_strength = 1.0f;

    /// Screen-space refraction (bent, blurred, absorption-tinted background sample) applied
    /// by the forward MESH transparent pass -- see toyengine's transparent.frag/refraction.glsl.
    /// Only meaningful when is_blended() is also true; a Mask/Opaque material with
    /// refraction=true is simply never drawn by the pass that reads it. NOT consumed by
    /// SdfRenderer/the SDF forward pass (sdf_forward.frag) even though PBRMaterial is shared
    /// with it -- these fields parse and store on an SdfRenderer's material without effect;
    /// see toyengine's refraction plan for why SDF glass was deliberately excluded.
    bool      refraction           = false;

    /// Negative (the default) means "not set by this material" -- the consumer (toyengine's
    /// record_transparent_()) falls back to its own engine-wide default (PixelRenderConfig::
    /// refraction_ior) in that case. A real IOR is never negative, so this sentinel can't
    /// collide with an authored value. Kept negative rather than defaulting to, say, 1.45
    /// directly so a scene author's choice and "the engine picked something" stay
    /// distinguishable -- changing the engine-wide default in config.yaml then actually
    /// changes every object that didn't override it, instead of only new ones.
    float     ior                  = -1.0f; /**< < 0 => use PixelRenderConfig::refraction_ior; else the IOR (glass ~1.45, water ~1.33). */
    float     refraction_thickness = -1.0f; /**< < 0 => use PixelRenderConfig::refraction_thickness; else world-space ray distance through the object. */
    glm::vec3 refraction_tint      = {-1.0f, -1.0f, -1.0f}; /**< Any component < 0 => use PixelRenderConfig::refraction_tint; else the Beer-Lambert absorption tint. */

    std::string texture_albedo             = "";
    std::string texture_normal             = "";
    std::string texture_metallic_roughness = "";

    /// Alpha-test mask for AlphaMode::Mask (CUTOUT), glTF/Unity convention: only the **alpha**
    /// channel is read, sampled at the mesh's UV and multiplied into albedo.a before the cutoff
    /// test (see gpu_alpha_cutoff()/has_alpha_mask()). Always linear, never sRGB -- it's a scalar
    /// coverage value, not color data. Ignored by SdfRenderer (SDFs have no UVs to sample with).
    std::string texture_alpha_mask         = "";

    /// Empty (the default) selects the stock surface shader for this material's
    /// alpha_mode -- exactly today's behaviour. Otherwise the name of a shader registered
    /// in a pipeline::SurfaceShaderRegistry (see gfxcoopa/pipeline/surface_shader.h and the
    /// layered-shaders plan's gfx/surface/*.glsl backbones): a scene author writing
    /// `shader: foliage` picks the same G-buffer/shadow/shadow-cube backbone their
    /// alpha_mode already selects, but with foliage's displacement/shading hooks instead of
    /// the identity default. An unregistered name is a scene-load error (see
    /// SurfaceShaderRegistry::require()), never a silent fallback to stock.
    std::string shader = "";

    /// Four author-defined floats reaching the surface shader's gfx_params (see
    /// gfx/surface/gbuffer_vs.glsl) -- e.g. foliage's wind strength/frequency/direction.xy,
    /// or water's wave amplitude/speed/direction.xy. Meaningless (and unread by every
    /// backbone) when shader is empty. Zero-initialized rather than left uninitialized so a
    /// material that sets `shader` but forgets a param gets 0.0, not garbage.
    glm::vec4 shader_params = {0.0f, 0.0f, 0.0f, 0.0f};

    // Populated by register_render_components()'s "MeshRenderer" parser once the corresponding
    // texture_* path above has been loaded via coopa::asset::AssetManager. All four handles are
    // consumed: engine::util::MaterialTextureCache binds each (or a neutral 1x1 fallback --
    // white for alpha_mask/albedo/metallic_roughness, flat-up for normal) into the material
    // descriptor set every textured pass (G-buffer, shadow, forward transparent, transparent
    // capture, probe capture) binds -- see has_albedo_map()/has_normal_map()/
    // has_metallic_roughness_map()/has_alpha_mask().
    coopa::asset::AssetHandle<coopa::gfx::engine::data::Texture> albedo_handle;
    coopa::asset::AssetHandle<coopa::gfx::engine::data::Texture> normal_handle;
    coopa::asset::AssetHandle<coopa::gfx::engine::data::Texture> metallic_roughness_handle;
    coopa::asset::AssetHandle<coopa::gfx::engine::data::Texture> alpha_mask_handle;

    /** @brief Returns true when this material must be drawn by the forward transparent pass. */
    bool is_blended() const { return alpha_mode == AlphaMode::Blend; }

    /** @brief Returns true when this material should be refracted by the forward MESH pass. */
    bool has_refraction() const { return is_blended() && refraction; }

    /**
     * @brief Returns true when this Mask material has a loaded alpha mask texture to sample.
     *
     * When false (Opaque/Blend, or a Mask material with no texture_alpha_mask), the G-buffer
     * and shadow passes bind a 1x1 white fallback instead, which collapses the shader-side test
     * back to today's constant-alpha behaviour.
     */
    bool has_alpha_mask() const { return alpha_mode == AlphaMode::Mask && alpha_mask_handle.is_loaded(); }

    /**
     * @brief Returns true when this material has a loaded albedo (base color) map to sample.
     *
     * Unlike has_alpha_mask(), this does NOT gate on alpha_mode -- an albedo map modulates
     * the base color for Opaque/Mask/Blend materials alike, not just Mask ones.
     */
    bool has_albedo_map() const { return albedo_handle.is_loaded(); }

    /** @brief Returns true when this material has a loaded tangent-space normal map to sample. */
    bool has_normal_map() const { return normal_handle.is_loaded(); }

    /** @brief Returns true when this material has a loaded metallic/roughness map to sample. */
    bool has_metallic_roughness_map() const { return metallic_roughness_handle.is_loaded(); }

    /**
     * @brief Returns the alpha cutoff as the GPU shader sees it.
     *
     * A value of 0.0 disables the discard entirely, which is what OPAQUE and BLEND
     * materials need since only MASK performs an alpha test in the G-buffer pass. When
     * has_alpha_mask() is true, the shader multiplies this against the sampled mask's alpha
     * rather than the constant material.albedo.a alone -- see gbuffer.frag/shadow_depth.frag.
     *
     * @return 0.0 for non-Mask materials, otherwise alpha_cutoff clamped to (0, 1].
     */
    float gpu_alpha_cutoff() const {
        return alpha_mode == AlphaMode::Mask ? glm::clamp(alpha_cutoff, 0.0001f, 1.0f) : 0.0f;
    }

    /** @brief Returns true when this material contributes any emissive radiance at all. */
    bool is_emissive() const {
        return emissive_strength > 0.0f && (emissive.r > 0.0f || emissive.g > 0.0f || emissive.b > 0.0f);
    }

    /**
     * @brief Returns the emissive term as the GPU shader sees it.
     *
     * Colour and strength collapse into one pre-multiplied vec4 for the same reason
     * gpu_alpha_cutoff() folds alpha_mode into its result: it keeps the shader side to a
     * single add, and it makes the batching predicates compare what actually reaches the
     * GPU, so {1,0,0} x 2.0 and {2,0,0} x 1.0 correctly merge into one batch. Negative
     * authored values are clamped away -- a negative emissive would subtract light from
     * the frame, which no pass expects.
     *
     * @return vec4(emissive * emissive_strength, 0.0); .w is reserved and currently unread.
     */
    glm::vec4 gpu_emissive() const {
        return glm::vec4(glm::max(emissive, glm::vec3(0.0f)) * glm::max(emissive_strength, 0.0f), 0.0f);
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
     * @brief Sets the GPU mesh handle.
     *
     * Called once AssetManager::load()/load_async() has been kicked off for
     * this renderer's mesh_path. The handle may still be Loading at this
     * point (async) -- see is_ready().
     *
     * @param mesh Refcounted handle to the GPU-resident Mesh asset.
     */
    void set_mesh(coopa::asset::AssetHandle<coopa::gfx::engine::data::Mesh> mesh) {
        mesh_ = std::move(mesh);
    }

    /**
     * @brief Returns the mesh handle. Dereference only after checking is_ready() (or the handle's own is_loaded()).
     */
    const coopa::asset::AssetHandle<coopa::gfx::engine::data::Mesh>& get_mesh() const { return mesh_; }

    /**
     * @brief Returns true if this renderer has a valid GPU mesh ready to draw.
     */
    bool is_ready() const { return mesh_.is_loaded(); }

    /// Slot 0's material -- the whole mesh for a mesh without material slots.
    PBRMaterial material;

    /**
     * Materials for the mesh's other slots (submeshes, see data::MeshPart): entry i is slot
     * i + 1. A slot with no entry here falls back to `material`. Filled from YAML
     * `materials:` -- a list (by slot index, entry 0 = slot 0) or a map keyed by the mesh's
     * slot names, which can only be resolved once the mesh has loaded (resolve_slot_names()).
     */
    std::vector<PBRMaterial> slot_materials;

    /** @brief The material for slot `slot`. */
    const PBRMaterial& material_for(uint32_t slot) const {
        if (slot == 0 || slot > slot_materials.size()) return material;
        return slot_materials[slot - 1];
    }
    /** @brief Writable slot material (grows the table; slot 0 is `material`). */
    PBRMaterial& material_for_mut(uint32_t slot) {
        if (slot == 0) return material;
        while (slot_materials.size() < slot) slot_materials.push_back(material);
        return slot_materials[slot - 1];
    }

    /// Materials keyed by slot NAME, waiting for the mesh to load (see resolve_slot_names()).
    std::vector<std::pair<std::string, PBRMaterial>> pending_named_materials;

    /**
     * @brief Applies pending name-keyed materials once the mesh (and so its slot names) is
     *        ready. Cheap no-op otherwise; the render pipeline calls it every frame.
     */
    void resolve_slot_names() {
        if (pending_named_materials.empty() || !is_ready()) return;
        const auto& mesh = *mesh_.get();
        for (auto& [name, mat] : pending_named_materials) {
            const int slot = mesh.slot_index(name);
            if (slot < 0) continue;   // unknown slot name: ignored, as an unknown key would be
            material_for_mut(static_cast<uint32_t>(slot)) = mat;
        }
        pending_named_materials.clear();
    }

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

    /// Multiplies this renderer's projected screen size before LOD selection: > 1 keeps
    /// detail longer (a hero prop), < 1 drops it sooner. The mesh's own LOD table (its
    /// `lods` block or `.lod.yaml` sidecar -- see data::Mesh::build_cpu) sets the levels.
    float lod_bias = 1.0f;

    /// False pins this renderer to LOD 0 and ignores the mesh's cull_screen_size.
    bool lods_enabled = true;

private:
    std::string mesh_path_; /**< Logical mesh path from YAML. */
    coopa::asset::AssetHandle<coopa::gfx::engine::data::Mesh> mesh_; /**< Refcounted GPU-resident mesh handle. */
};

} // namespace components
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_COMPONENTS_MESH_RENDERER_H
