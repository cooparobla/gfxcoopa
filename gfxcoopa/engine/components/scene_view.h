/**
 * @file scene_view.h
 * @brief Flattened, renderer-facing view over a coopa::scene::Scene.
 *
 * coopa::scene::Scene deliberately knows nothing about render component types
 * (see coopa/scene/scene.h) — it only exposes generic get_components<T>() /
 * find_first_component<T>() queries. SceneView is the adapter that restores
 * the convenience API blendy's PbrRenderPipeline and gfxcoopa's GI code used
 * to get directly from Scene (active_camera(), get_renderable_objects(), ...),
 * built entirely on top of those generic queries. This is what keeps
 * template<typename Scene> code in gi_baker.h / gi_system.h /
 * PbrRenderPipeline working unchanged in shape — pass a SceneView wherever
 * those templates expect a "Scene".
 */

#ifndef GFXCOOPA_ENGINE_COMPONENTS_SCENE_VIEW_H
#define GFXCOOPA_ENGINE_COMPONENTS_SCENE_VIEW_H

#include <coopa/scene/scene.h>
#include <coopa/scene/scene_object.h>
#include <coopa/scene/components/transform_component.h>

#include <gfxcoopa/engine/components/mesh_renderer.h>
#include <gfxcoopa/engine/components/camera_component.h>
#include <gfxcoopa/engine/components/directional_light.h>
#include <gfxcoopa/engine/components/point_light.h>
#include <gfxcoopa/engine/components/gi_probe_volume.h>
#include <gfxcoopa/engine/components/reflection_probe.h>

#include <vector>

namespace coopa {
namespace gfx {
namespace engine {
namespace components {

/**
 * @struct RenderableRef
 * @brief One renderable SceneObject, pre-resolved to its MeshRenderer/Transform.
 */
struct RenderableRef {
    coopa::scene::SceneObject*        object;    /**< Non-owning; the SceneObject carrying the renderer. */
    MeshRenderer*                     renderer;  /**< Non-owning; never null when this ref exists. */
    coopa::scene::TransformComponent* transform; /**< Non-owning; may be null if the object has none. */
};

/**
 * @class SceneView
 * @brief Renderer-facing flattening of a Scene: active camera/light plus flat
 *        component lists, refreshed with a single hierarchy walk.
 *
 * Usage:
 * @code
 * coopa::gfx::engine::components::SceneView view(scene_mgr.get_active_scene());
 * // once per frame, after Scene::update():
 * view.refresh();
 * rp.render(renderer, view);
 * @endcode
 */
class SceneView {
public:
    /**
     * @brief Wraps a Scene and performs the initial flattening pass.
     * @param scene The scene to view. Must outlive this SceneView.
     */
    explicit SceneView(coopa::scene::Scene& scene) : scene_(scene) {
        refresh();
    }

    /**
     * @brief Re-flattens the scene. Call once per frame after any hierarchy changes.
     *
     * A single DFS per component type, via Scene::get_components<T>() /
     * find_first_component<T>() — cheap relative to a render frame, and far
     * simpler than tracking incremental hierarchy edits.
     */
    void refresh() {
        renderables_.clear();
        for (auto* obj : scene_.find_objects_with<MeshRenderer>()) {
            renderables_.push_back(RenderableRef{ obj, obj->get_component<MeshRenderer>(), obj->get_transform() });
        }
        point_lights_       = scene_.get_components<PointLightComponent>();
        gi_probe_volumes_   = scene_.get_components<GiProbeVolumeComponent>();
        reflection_probes_  = scene_.get_components<ReflectionProbeComponent>();
        active_camera_      = scene_.find_first_component<CameraComponent>();
        active_light_       = scene_.find_first_component<DirectionalLightComponent>();
    }

    /** @brief Returns the first active CameraComponent found in the hierarchy, or nullptr. */
    CameraComponent* active_camera() const { return active_camera_; }

    /** @brief Returns the first active DirectionalLightComponent found in the hierarchy, or nullptr. */
    DirectionalLightComponent* active_light() const { return active_light_; }

    /** @brief Returns every active SceneObject with a MeshRenderer, pre-resolved to renderer/transform. */
    const std::vector<RenderableRef>& get_renderable_objects() const { return renderables_; }

    /** @brief Returns every active PointLightComponent in the hierarchy. */
    const std::vector<PointLightComponent*>& get_point_lights() const { return point_lights_; }

    /** @brief Returns every active GiProbeVolumeComponent in the hierarchy. */
    const std::vector<GiProbeVolumeComponent*>& get_gi_probe_volumes() const { return gi_probe_volumes_; }

    /** @brief Returns every active ReflectionProbeComponent in the hierarchy. */
    const std::vector<ReflectionProbeComponent*>& get_reflection_probes() const { return reflection_probes_; }

    /**
     * @brief Passthrough to the underlying Scene's root objects.
     *
     * Lets consumers that still walk the hierarchy themselves (e.g.
     * PbrRenderPipeline::render(), which needs its own traversal to split
     * renderables into opaque/transparent lists) use a SceneView wherever a
     * bare Scene was previously expected.
     */
    const std::vector<std::unique_ptr<coopa::scene::SceneObject>>& root_objects() const {
        return scene_.root_objects();
    }

    /** @brief Returns the underlying Scene. */
    coopa::scene::Scene& scene() const { return scene_; }

private:
    coopa::scene::Scene& scene_;

    std::vector<RenderableRef>              renderables_;
    std::vector<PointLightComponent*>       point_lights_;
    std::vector<GiProbeVolumeComponent*>    gi_probe_volumes_;
    std::vector<ReflectionProbeComponent*>  reflection_probes_;
    CameraComponent*           active_camera_ = nullptr;
    DirectionalLightComponent* active_light_  = nullptr;
};

} // namespace components
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_COMPONENTS_SCENE_VIEW_H
