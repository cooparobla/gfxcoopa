# `gfxcoopa` Architecture & Submodules

`gfxcoopa` is a header-only C++20 Vulkan wrapper designed to bring OpenGL-like simplicity and modern C++ RAII ergonomics to Vulkan development.

All header files reside under `gfxcoopa/` and are organized into functional submodules within the `coopa::gfx` namespace.

---

## Submodule Architecture Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             GFXCOOPA LAYER ARCHITECTURE                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                               ┌───────────────────────┐
                               │   Application Code    │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │     presentation      │  (Window, Renderer)
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
        ┌───────────────────────┐┌───────────────────┐┌───────────────────────┐
        │        engine         ││     pipeline      ││        command        │
        │(Targets, Mesh, Lights)││(Shaders, Passes)  ││(CmdPool, CmdBuffer)   │
        └───────────┬───────────┘└─────────┬─────────┘└───────────┬───────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │        memory         │                     │         core          │
        │(Allocator, VMA, Image)│                     │(Instance, Device, Swap)│
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │         util          │  (Volk, Error, Debug)
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │      Vulkan API       │
                               └───────────────────────┘
```

---

## Submodule Directory Map

| Submodule | Namespace | Description |
|---|---|---|
| [`command/`](command/README.md) | `coopa::gfx::command` | Command buffers, command pools, and CPU/GPU sync primitives (Fences, Semaphores). |
| [`core/`](core/README.md) | `coopa::gfx::core` | Vulkan instance initialization, GLFW surface integration, physical/logical device management, and swapchain handling. |
| [`engine/`](engine/README.md) | `coopa::gfx::engine` | High-level rendering engine utilities: meshes, light/camera UBOs, offscreen render targets, shadow maps, BRDF LUTs, cubemaps, samplers, and GI probe volumes. |
| [`memory/`](memory/README.md) | `coopa::gfx::memory` | Vulkan Memory Allocator (VMA) RAII integration, GPU buffer management, and image allocation/transitions. |
| [`pipeline/`](pipeline/README.md) | `coopa::gfx::pipeline` | Shaders, render passes, descriptor pools/layouts/sets, and graphics pipeline state configuration. |
| [`presentation/`](presentation/README.md) | `coopa::gfx::presentation` | GLFW windowing wrapper and multi-buffered frame renderer orchestration (acquire, record, submit, present). |
| [`util/`](util/README.md) | `coopa::gfx::util` | Debug messengers, format selection helpers, Volk dynamic loader initialization, and error checking macros. |

---

## Core Principles

- **Header-Only C++20**: Include what you need. Zero compilation boilerplate.
- **RAII Lifecycle Management**: Vulkan objects automatically cleanup resources upon destruction.
- **Volk & VMA Powered**: Dynamic function loading via Volk and GPU memory management via Vulkan Memory Allocator.
- **Sensible Defaults**: Concise API surface with customizable default configurations.
