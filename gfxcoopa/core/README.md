# `gfxcoopa::core` Submodule

The `gfxcoopa::core` submodule encapsulates the core Vulkan setup components: instance creation, window surface integration, physical/logical device selection, and swapchain management.

---

## Core Subsystem Dependency Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            CORE SYSTEM DEPENDENCY HIERARCHY                              │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                               ┌───────────────────────┐
                               │      volkInitialize   │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │       Instance        │
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │        Window         │                     │    DebugMessenger     │
        └───────────┬───────────┘                     └───────────────────────┘
                    │ creates VkSurfaceKHR
                    ▼
        ┌───────────────────────┐
        │        Surface        │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │        Device         │  (Selects Physical GPU & creates VkDevice)
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │       Swapchain       │  (Manages VkSwapchainKHR & Swapchain Image Views)
        └───────────────────────┘
```

---

## Header Files

| File | Primary Class / Struct | Description |
|---|---|---|
| [`instance.h`](instance.h) | [`Instance`](instance.h) | RAII wrapper for `VkInstance`. Performs one-time Volk dynamic loader initialization, enables Vulkan validation layers in debug builds, and manages instance extensions. |
| [`surface.h`](surface.h) | [`Surface`](surface.h) | RAII wrapper around `VkSurfaceKHR`. Integrates GLFW window handles with Vulkan and provides surface capability/format support query helpers (`query_support()`). |
| [`device.h`](device.h) | [`Device`](device.h) | Physical GPU selection (preferring discrete GPUs), logical `VkDevice` creation, queue family indexing (graphics and present queues), queue retrieval, and `wait_idle()`. |
| [`swapchain.h`](swapchain.h) | [`Swapchain`](swapchain.h) | Swapchain management wrapper (`VkSwapchainKHR`). Handles image extent resolution, image format selection, present mode selection, swapchain image views, and dynamic recreation (`recreate()`) on window resize. |

---

## Usage Example

```cpp
#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/core/surface.h>
#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/presentation/window.h>

coopa::gfx::presentation::Window window("My Vulkan App", 1280, 720);
coopa::gfx::core::Instance       instance("app_name");
coopa::gfx::core::Surface        surface(instance, window.handle());
coopa::gfx::core::Device         device(instance, surface);
coopa::gfx::core::Swapchain      swapchain(device, surface, 1280, 720);

// Handle window resizing
if (window.was_resized()) {
    auto [width, height] = window.framebuffer_size();
    swapchain.recreate(width, height);
}
```
