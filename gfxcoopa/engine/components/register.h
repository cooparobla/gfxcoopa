/**
 * @file register.h
 * @brief Registers gfxcoopa's render components as coopa::scene::SceneLoader parsers.
 *
 * libcoopa's SceneLoader only understands hierarchy plus Transform/Animation —
 * every renderer-specific component (MeshRenderer, Camera, lights, GI/reflection
 * probes) is parsed here instead, keeping the scene system itself free of any
 * gfxcoopa/Vulkan dependency. Call register_render_components() once at startup,
 * before the first SceneLoader::load() that needs these components.
 *
 * MeshRenderer needs GPU handles (Device/Allocator/CommandPool) to upload mesh
 * data during parsing — those are captured by reference in the registered
 * lambdas, which is why this function takes them as parameters instead of a
 * parser receiving them per-call (SceneLoader::ComponentParser's signature has
 * no room for them, deliberately, so the scene system stays decoupled). This
 * means the referenced Device/Allocator/CommandPool must outlive every
 * subsequent SceneLoader::load() call; call
 * coopa::scene::SceneLoader::clear_component_parsers() before destroying them.
 *
 * Mesh and texture loading are routed through a coopa::asset::AssetManager
 * rather than the ad-hoc mesh_cache map this used to build inline — see
 * gfxcoopa/engine/loaders/{mesh_loader.h,texture_loader.h} for the
 * loaders registered against it. The caller owns the AssetManager (typically
 * for the app's whole lifetime) and must register those two loaders on it
 * before calling register_render_components(); this function only calls
 * assets.load()/load_async(), never registers loaders itself, so a caller
 * can substitute its own loader (e.g. a caml-backed one) if it wants to.
 */

#ifndef GFXCOOPA_ENGINE_COMPONENTS_REGISTER_H
#define GFXCOOPA_ENGINE_COMPONENTS_REGISTER_H

#include <coopa/scene/scene_loader.h>
#include <coopa/scene/scene_object.h>
#include <coopa/asset/asset_manager.h>
#include <fkYAML/node.hpp>

#include <gfxcoopa/core/device.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/engine/data/mesh.h>
#include <gfxcoopa/engine/data/texture.h>

#include <gfxcoopa/engine/components/mesh_renderer.h>
#include <gfxcoopa/engine/components/camera_component.h>
#include <gfxcoopa/engine/components/directional_light.h>
#include <gfxcoopa/engine/components/point_light.h>
#include <gfxcoopa/engine/components/environment_light.h>
#include <gfxcoopa/engine/components/gi_probe_volume.h>
#include <gfxcoopa/engine/components/reflection_probe.h>
#include <gfxcoopa/engine/components/fog_volume.h>
#include <gfxcoopa/engine/components/sdf_renderer.h>
#include <gfxcoopa/engine/components/sdf_shape.h>

#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace coopa {
namespace gfx {
namespace engine {
namespace components {

/**
 * @brief Loads and parses a YAML file directly via fkYAML — the mesh-file
 *        equivalent of SceneLoader's own default document loading, since mesh
 *        files are not routed through SceneLoader::set_document_loader().
 *
 * Retained for callers that still need a raw YAML read; mesh loading itself
 * no longer uses this directly (see MeshLoader::decode_typed()).
 */
inline fkyaml::node load_yaml_file_(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("[gfxcoopa] Failed to open file: " + path);
    }
    return fkyaml::node::deserialize(ifs);
}

/**
 * @brief Parses a `material:` YAML block into a PBRMaterial, shared by the
 * "MeshRenderer" and "SdfRenderer" parsers below so the two renderer types'
 * material surface can never drift apart (see SdfRenderer's own doc: it
 * reuses PBRMaterial verbatim for exactly this reason).
 */
inline void parse_pbr_material_(const fkyaml::node& mat_node, PBRMaterial& material,
                                coopa::asset::AssetManager& assets,
                                const coopa::scene::SceneLoader::ParseContext& ctx) {
    if (mat_node.contains("albedo")) {
        const auto& alb = mat_node.at("albedo");
        material.albedo = {
            alb.at("r").get_value<float>(),
            alb.at("g").get_value<float>(),
            alb.at("b").get_value<float>()
        };
    }
    if (mat_node.contains("metallic"))  material.metallic  = mat_node.at("metallic").get_value<float>();
    if (mat_node.contains("roughness")) material.roughness = mat_node.at("roughness").get_value<float>();
    if (mat_node.contains("ao"))        material.ao        = mat_node.at("ao").get_value<float>();

    if (mat_node.contains("emissive")) {
        const auto& em = mat_node.at("emissive");
        material.emissive = {
            em.at("r").get_value<float>(),
            em.at("g").get_value<float>(),
            em.at("b").get_value<float>()
        };
    }
    if (mat_node.contains("emissive_strength")) material.emissive_strength = mat_node.at("emissive_strength").get_value<float>();

    if (mat_node.contains("alpha"))        material.alpha        = mat_node.at("alpha").get_value<float>();
    if (mat_node.contains("alpha_cutoff")) material.alpha_cutoff = mat_node.at("alpha_cutoff").get_value<float>();
    if (mat_node.contains("alpha_mode")) {
        std::string mode = mat_node.at("alpha_mode").get_value<std::string>();
        if (mode == "BLEND") {
            material.alpha_mode = AlphaMode::Blend;
        } else if (mode == "MASK" || mode == "CLIP") {
            material.alpha_mode = AlphaMode::Mask;
        } else {
            material.alpha_mode = AlphaMode::Opaque;
        }
    }

    if (mat_node.contains("texture_albedo")) {
        material.texture_albedo = mat_node.at("texture_albedo").get_value<std::string>();
        material.albedo_handle = assets.load_async<data::Texture>(material.texture_albedo, ctx.scene_dir);
    }
    if (mat_node.contains("texture_normal")) {
        material.texture_normal = mat_node.at("texture_normal").get_value<std::string>();
        material.normal_handle = assets.load_async<data::Texture>(material.texture_normal, ctx.scene_dir);
    }
    if (mat_node.contains("texture_metallic_roughness")) {
        material.texture_metallic_roughness = mat_node.at("texture_metallic_roughness").get_value<std::string>();
        material.metallic_roughness_handle = assets.load_async<data::Texture>(material.texture_metallic_roughness, ctx.scene_dir);
    }
}

/// Parses a `{x:, y:, z:}` node into a glm::vec3, leaving components at
/// `fallback`'s when absent -- the same partial-override convention every
/// other vec3 field in this file already follows (DirectionalLight's
/// direction/color, FogVolume's extent/color, etc.).
inline glm::vec3 parse_vec3_(const fkyaml::node& node, const glm::vec3& fallback) {
    glm::vec3 v = fallback;
    if (node.contains("x")) v.x = node.at("x").get_value<float>();
    if (node.contains("y")) v.y = node.at("y").get_value<float>();
    if (node.contains("z")) v.z = node.at("z").get_value<float>();
    return v;
}

/**
 * @brief Registers parsers for every gfxcoopa render component with SceneLoader.
 *
 * @param device    Vulkan logical device, used for mesh/texture buffer creation.
 * @param allocator VMA allocator, used for mesh/texture memory allocation.
 * @param cmd_pool  Command pool for one-shot mesh/texture upload transfers.
 * @param assets    AssetManager with MeshLoader and TextureLoader already
 *                  registered (see gfxcoopa/engine/loaders/). Must outlive
 *                  every subsequent SceneLoader::load() call, same as
 *                  device/allocator/cmd_pool.
 */
inline void register_render_components(core::Device& device,
                                       memory::Allocator& allocator,
                                       command::CommandPool& cmd_pool,
                                       coopa::asset::AssetManager& assets) {
    using coopa::scene::SceneLoader;
    using coopa::scene::SceneObject;

    SceneLoader::register_component_parser("MeshRenderer",
        [&assets](
            const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext& ctx) {
            auto* mr = obj.add_component<MeshRenderer>();
            std::string mesh_path_key;
            if (node.contains("mesh_path")) {
                mesh_path_key = node.at("mesh_path").get_value<std::string>();
            }
            mr->set_mesh_path(mesh_path_key);

            if (node.contains("affects_reflection_probes"))
                mr->affects_reflection_probes = node.at("affects_reflection_probes").get_value<bool>();

            if (!mesh_path_key.empty()) {
                std::string mesh_virtual_path = "meshes/" + mesh_path_key + ".yaml";
                auto mesh_handle = assets.load<data::Mesh>(mesh_virtual_path, ctx.scene_dir);
                if (mesh_handle.is_failed()) {
                    std::cerr << "[register_render_components] Failed to load mesh '"
                              << mesh_virtual_path << "' (scene_dir=" << ctx.scene_dir
                              << "): " << mesh_handle.error() << std::endl;
                }
                mr->set_mesh(std::move(mesh_handle));
            }

            if (node.contains("material")) {
                parse_pbr_material_(node.at("material"), mr->material, assets, ctx);
            }
        });

    SceneLoader::register_component_parser("Camera",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* cam = obj.add_component<CameraComponent>();

            if (node.contains("projection")) {
                std::string p_str = node.at("projection").get_value<std::string>();
                cam->type = (p_str == "ORTHO" || p_str == "Orthographic" || p_str == "orthographic")
                            ? CameraType::Orthographic : CameraType::Perspective;
            } else if (node.contains("type")) {
                std::string type_str = node.at("type").get_value<std::string>();
                if (type_str != "Camera") {
                    cam->type = (type_str == "ORTHO" || type_str == "Orthographic" || type_str == "orthographic")
                                ? CameraType::Orthographic : CameraType::Perspective;
                }
            }

            if (node.contains("fov")) cam->fov = node.at("fov").get_value<float>();

            if (node.contains("orthographic_size")) {
                cam->orthographic_size = node.at("orthographic_size").get_value<float>();
            } else if (node.contains("ortho_size")) {
                cam->orthographic_size = node.at("ortho_size").get_value<float>();
            } else if (node.contains("ortho_scale")) {
                cam->set_ortho_scale(node.at("ortho_scale").get_value<float>());
            }

            if (node.contains("near_clip_plane")) {
                cam->clip_start = node.at("near_clip_plane").get_value<float>();
            } else if (node.contains("clip_start")) {
                cam->clip_start = node.at("clip_start").get_value<float>();
            }

            if (node.contains("far_clip_plane")) {
                cam->clip_end = node.at("far_clip_plane").get_value<float>();
            } else if (node.contains("clip_end")) {
                cam->clip_end = node.at("clip_end").get_value<float>();
            }

            if (node.contains("lens"))          cam->lens          = node.at("lens").get_value<float>();
            if (node.contains("sensor_width"))  cam->sensor_width  = node.at("sensor_width").get_value<float>();
            if (node.contains("sensor_height")) cam->sensor_height = node.at("sensor_height").get_value<float>();

            if (node.contains("main")) {
                cam->is_main = node.at("main").get_value<bool>();
            } else if (node.contains("is_main")) {
                cam->is_main = node.at("is_main").get_value<bool>();
            }
        });

    SceneLoader::register_component_parser("DirectionalLight",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* dl = obj.add_component<DirectionalLightComponent>();

            if (node.contains("direction")) {
                const auto& d = node.at("direction");
                float dx = d.contains("x") ? d.at("x").get_value<float>() : dl->direction.x;
                float dy = d.contains("y") ? d.at("y").get_value<float>() : dl->direction.y;
                float dz = d.contains("z") ? d.at("z").get_value<float>() : dl->direction.z;
                dl->direction = glm::normalize(glm::vec3(dx, dy, dz));
            }
            if (node.contains("color")) {
                const auto& c = node.at("color");
                dl->color.r = c.contains("r") ? c.at("r").get_value<float>() : dl->color.r;
                dl->color.g = c.contains("g") ? c.at("g").get_value<float>() : dl->color.g;
                dl->color.b = c.contains("b") ? c.at("b").get_value<float>() : dl->color.b;
            }
            if (node.contains("intensity")) {
                dl->intensity = node.at("intensity").get_value<float>();
            }
            if (node.contains("ambient")) {
                const auto& a = node.at("ambient");
                dl->ambient.r = a.contains("r") ? a.at("r").get_value<float>() : dl->ambient.r;
                dl->ambient.g = a.contains("g") ? a.at("g").get_value<float>() : dl->ambient.g;
                dl->ambient.b = a.contains("b") ? a.at("b").get_value<float>() : dl->ambient.b;
            }
            if (node.contains("cast_shadows")) {
                dl->cast_shadows = node.at("cast_shadows").get_value<bool>();
            }
        });

    SceneLoader::register_component_parser("PointLight",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* pl = obj.add_component<PointLightComponent>();

            if (node.contains("color")) {
                const auto& c = node.at("color");
                pl->color.r = c.contains("r") ? c.at("r").get_value<float>() : pl->color.r;
                pl->color.g = c.contains("g") ? c.at("g").get_value<float>() : pl->color.g;
                pl->color.b = c.contains("b") ? c.at("b").get_value<float>() : pl->color.b;
            }
            if (node.contains("intensity")) pl->intensity = node.at("intensity").get_value<float>();
            if (node.contains("range"))     pl->range     = node.at("range").get_value<float>();
            if (node.contains("cast_shadows")) pl->cast_shadows = node.at("cast_shadows").get_value<bool>();
            if (node.contains("attenuation_constant"))
                pl->attenuation_constant = node.at("attenuation_constant").get_value<float>();
            if (node.contains("attenuation_linear"))
                pl->attenuation_linear = node.at("attenuation_linear").get_value<float>();
            if (node.contains("attenuation_quadratic"))
                pl->attenuation_quadratic = node.at("attenuation_quadratic").get_value<float>();
        });

    SceneLoader::register_component_parser("GiProbeVolume",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* gv = obj.add_component<GiProbeVolumeComponent>();
            if (node.contains("origin")) {
                const auto& o = node.at("origin");
                gv->origin.x = o.contains("x") ? o.at("x").get_value<float>() : gv->origin.x;
                gv->origin.y = o.contains("y") ? o.at("y").get_value<float>() : gv->origin.y;
                gv->origin.z = o.contains("z") ? o.at("z").get_value<float>() : gv->origin.z;
            }
            if (node.contains("extent")) {
                const auto& e = node.at("extent");
                gv->extent.x = e.contains("x") ? e.at("x").get_value<float>() : gv->extent.x;
                gv->extent.y = e.contains("y") ? e.at("y").get_value<float>() : gv->extent.y;
                gv->extent.z = e.contains("z") ? e.at("z").get_value<float>() : gv->extent.z;
            }
            if (node.contains("grid_resolution")) {
                const auto& g = node.at("grid_resolution");
                gv->grid_resolution.x = g.contains("x") ? g.at("x").get_value<int>() : gv->grid_resolution.x;
                gv->grid_resolution.y = g.contains("y") ? g.at("y").get_value<int>() : gv->grid_resolution.y;
                gv->grid_resolution.z = g.contains("z") ? g.at("z").get_value<int>() : gv->grid_resolution.z;
            }
            if (node.contains("update_mode")) {
                std::string mode = node.at("update_mode").get_value<std::string>();
                gv->update_mode = (mode == "dynamic") ? GiUpdateMode::Dynamic : GiUpdateMode::Static;
            }
            if (node.contains("gi_intensity"))
                gv->gi_intensity = node.at("gi_intensity").get_value<float>();
        });

    SceneLoader::register_component_parser("FogVolume",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* fv = obj.add_component<FogVolumeComponent>();
            if (node.contains("shape")) {
                std::string shape = node.at("shape").get_value<std::string>();
                fv->shape = (shape == "sphere" || shape == "Sphere")
                            ? FogVolumeShape::Sphere : FogVolumeShape::Box;
            }
            if (node.contains("extent")) {
                const auto& e = node.at("extent");
                fv->extent.x = e.contains("x") ? e.at("x").get_value<float>() : fv->extent.x;
                fv->extent.y = e.contains("y") ? e.at("y").get_value<float>() : fv->extent.y;
                fv->extent.z = e.contains("z") ? e.at("z").get_value<float>() : fv->extent.z;
            }
            if (node.contains("color")) {
                const auto& c = node.at("color");
                fv->color.r = c.contains("r") ? c.at("r").get_value<float>() : fv->color.r;
                fv->color.g = c.contains("g") ? c.at("g").get_value<float>() : fv->color.g;
                fv->color.b = c.contains("b") ? c.at("b").get_value<float>() : fv->color.b;
            }
            if (node.contains("density")) fv->density = node.at("density").get_value<float>();
            if (node.contains("falloff")) fv->falloff = node.at("falloff").get_value<float>();
        });

    SceneLoader::register_component_parser("SdfRenderer",
        [&assets](
            const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext& ctx) {
            auto* sr = obj.add_component<SdfRenderer>();

            if (node.contains("bounds_center"))
                sr->bounds_center = parse_vec3_(node.at("bounds_center"), sr->bounds_center);
            if (node.contains("bounds_extent"))
                sr->bounds_extent = parse_vec3_(node.at("bounds_extent"), sr->bounds_extent);

            if (node.contains("cast_shadows"))
                sr->cast_shadows = node.at("cast_shadows").get_value<bool>();
            if (node.contains("affects_reflection_probes"))
                sr->affects_reflection_probes = node.at("affects_reflection_probes").get_value<bool>();
            if (node.contains("max_steps"))
                sr->max_steps = node.at("max_steps").get_value<int>();
            if (node.contains("surface_epsilon"))
                sr->surface_epsilon = node.at("surface_epsilon").get_value<float>();
            if (node.contains("normal_epsilon"))
                sr->normal_epsilon = node.at("normal_epsilon").get_value<float>();
            if (node.contains("smoothing"))
                sr->smoothing = node.at("smoothing").get_value<float>();

            if (node.contains("material")) {
                parse_pbr_material_(node.at("material"), sr->material, assets, ctx);
            }
        });

    SceneLoader::register_component_parser("SdfShape",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* shape = obj.add_component<SdfShape>();

            if (node.contains("shape")) {
                std::string s = node.at("shape").get_value<std::string>();
                if (s == "Box" || s == "box") {
                    shape->type = SdfShapeType::Box;
                } else if (s == "Plane" || s == "plane") {
                    shape->type = SdfShapeType::Plane;
                } else {
                    shape->type = SdfShapeType::Sphere;
                }
            }
            if (node.contains("params"))
                shape->params = parse_vec3_(node.at("params"), shape->params);
            if (node.contains("rounding"))
                shape->rounding = node.at("rounding").get_value<float>();
            if (node.contains("op")) {
                std::string o = node.at("op").get_value<std::string>();
                if (o == "Subtract" || o == "subtract") {
                    shape->op = SdfOperation::Subtract;
                } else if (o == "Intersect" || o == "intersect") {
                    shape->op = SdfOperation::Intersect;
                } else {
                    shape->op = SdfOperation::Union;
                }
            }
            if (node.contains("blend"))
                shape->blend = node.at("blend").get_value<float>();
        });

    SceneLoader::register_component_parser("ReflectionProbe",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* rp = obj.add_component<ReflectionProbeComponent>();
            if (node.contains("box_extent")) {
                const auto& e = node.at("box_extent");
                rp->box_extent.x = e.contains("x") ? e.at("x").get_value<float>() : rp->box_extent.x;
                rp->box_extent.y = e.contains("y") ? e.at("y").get_value<float>() : rp->box_extent.y;
                rp->box_extent.z = e.contains("z") ? e.at("z").get_value<float>() : rp->box_extent.z;
            }
            if (node.contains("resolution"))
                rp->resolution = static_cast<uint32_t>(node.at("resolution").get_value<int>());
            if (node.contains("blend_distance"))
                rp->blend_distance = node.at("blend_distance").get_value<float>();
            if (node.contains("importance"))
                rp->importance = node.at("importance").get_value<int>();
            if (node.contains("intensity"))
                rp->intensity = node.at("intensity").get_value<float>();
        });

    SceneLoader::register_component_parser("EnvironmentLight",
        [](const fkyaml::node& node, SceneObject& obj, const SceneLoader::ParseContext&) {
            auto* el = obj.add_component<EnvironmentLightComponent>();
            if (node.contains("sky_color")) {
                const auto& c = node.at("sky_color");
                el->sky_color.r = c.contains("r") ? c.at("r").get_value<float>() : el->sky_color.r;
                el->sky_color.g = c.contains("g") ? c.at("g").get_value<float>() : el->sky_color.g;
                el->sky_color.b = c.contains("b") ? c.at("b").get_value<float>() : el->sky_color.b;
            }
            if (node.contains("ground_color")) {
                const auto& c = node.at("ground_color");
                el->ground_color.r = c.contains("r") ? c.at("r").get_value<float>() : el->ground_color.r;
                el->ground_color.g = c.contains("g") ? c.at("g").get_value<float>() : el->ground_color.g;
                el->ground_color.b = c.contains("b") ? c.at("b").get_value<float>() : el->ground_color.b;
            }
            if (node.contains("sky_intensity"))
                el->sky_intensity = node.at("sky_intensity").get_value<float>();
            if (node.contains("hdri_path"))
                el->hdri_path = node.at("hdri_path").get_value<std::string>();
        });
}

} // namespace components
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // GFXCOOPA_ENGINE_COMPONENTS_REGISTER_H
