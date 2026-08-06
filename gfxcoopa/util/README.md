# `gfxcoopa::util` Submodule

The `gfxcoopa::util` submodule provides foundational utilities including error checking macros, format selection helpers, Volk dynamic loader initialization, and Vulkan validation layer debug message handlers.

---

## Util Architecture & Error Flow Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            UTIL ARCHITECTURE & ERROR FLOW                                │
└──────────────────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────┐                     ┌───────────────────────┐
    │       volk_init       │                     │    debug_messenger    │
    ├───────────────────────┤                     ├───────────────────────┤
    │ - init_volk()         │                     │ - DebugMessenger      │
    │   (Loads Vulkan Symbols│                     │   (Validation Layer   │
    └───────────────────────┘                     │    Error Callback)    │
                                                  └───────────────────────┘

    ┌───────────────────────┐                     ┌───────────────────────┐
    │        format         │                     │         error         │
    ├───────────────────────┤                     ├───────────────────────┤
    │ - format_from_channels│                     │ - GFX_VK_CHECK(expr)  │
    │ - format_byte_size    │                     │ - vk_result_string()  │
    └───────────────────────┘                     └───────────────────────┘
```

---

## Header Files

| File | Primary Class / Struct / Function | Description |
|---|---|---|
| [`debug_messenger.h`](debug_messenger.h) | [`DebugMessenger`](debug_messenger.h) | RAII wrapper for `VkDebugUtilsMessengerEXT`. Configures Vulkan validation layer message callback to log warnings, errors, and performance details to `std::cerr`. |
| [`error.h`](error.h) | `GFX_VK_CHECK(expr)`, `vk_result_string()` | Macro for evaluating Vulkan `VkResult` expressions and throwing descriptive `std::runtime_error` exceptions with context on failure, along with a helper function translating `VkResult` enum values to strings. |
| [`format.h`](format.h) | `format_from_channels()`, `format_byte_size()`, `format_has_depth()`, `format_has_stencil()` | Utility functions mapping channel counts and sRGB flags to corresponding `VkFormat` enums, computing format stride in bytes, and checking depth/stencil properties. |
| [`volk_init.h`](volk_init.h) | `init_volk()` | One-time initialization guard function that safely invokes `volkInitialize()` to dynamically load entry points from `libvulkan.so`. |

---

## Usage Example

```cpp
#include <gfxcoopa/util/volk_init.h>
#include <gfxcoopa/util/error.h>
#include <gfxcoopa/util/format.h>
#include <gfxcoopa/util/debug_messenger.h>

// One-time volk initialization
coopa::gfx::util::init_volk();

// Error check macro
GFX_VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

// Format lookup helper (e.g. 4 channels, linear space)
VkFormat format = coopa::gfx::util::format_from_channels(4, false); // VK_FORMAT_R8G8B8A8_UNORM

// Setup debug messenger for validation layer logs
coopa::gfx::util::DebugMessenger messenger(instance);
```
