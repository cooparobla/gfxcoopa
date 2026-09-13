# Utility Submodule (`coopa::gfx::util`)

The `coopa::gfx::util` submodule provides debug validation messengers, format helpers, Volk dynamic loader initialization, and error checking macros.

---

## Utility Component Graph

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             UTILITY COMPONENT HIERARCHY                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────────────┐
                                │       volk_init       │  (volkInitialize)
                                └───────────┬───────────┘
                                            │
       ┌────────────────────────────────────┼────────────────────────────────────┐
       ▼                                    ▼                                    ▼
┌───────────────┐                  ┌─────────────────┐                  ┌────────────────┐
│DebugMessenger │                  │  GFX_VK_CHECK   │                  │ Format Helpers │
└───────────────┘                  └─────────────────┘                  └────────────────┘
```

---

## File Breakdown

### [debug_messenger.h](debug_messenger.h)
- **Role**: Vulkan validation layer debug messenger (`VkDebugUtilsMessengerEXT`).
- **Key Classes / Structs**: `DebugMessenger`.
- **Details**: Captures Vulkan API warning/error logs and routes formatted debug reports to stdout/stderr.

### [error.h](error.h)
- **Role**: Error checking macros and string translation utilities.
- **Key Classes / Structs**: `GFX_VK_CHECK()`, `vk_result_string()`.
- **Details**: Throws `std::runtime_error` with source filename and line number when Vulkan API functions return non-success codes.

### [format.h](format.h)
- **Role**: Vulkan format selection helpers.
- **Key Classes / Structs**: `find_supported_format()`, `find_depth_format()`.
- **Details**: Queries physical device format properties to select depth/stencil attachment formats (`VK_FORMAT_D32_SFLOAT`, `VK_FORMAT_D24_UNORM_S8_UINT`, etc.).

### [volk_init.h](volk_init.h)
- **Role**: Dynamic Vulkan loader initialization wrapper.
- **Key Classes / Structs**: `init_volk()`.
- **Details**: Calls `volkInitialize()` to dynamically load Vulkan entry points without linking against static Vulkan loader libraries.

---

## Usage Example

```cpp
#include <gfxcoopa/util/volk_init.h>
#include <gfxcoopa/util/error.h>
#include <gfxcoopa/util/format.h>

// 1. Initialize Volk dynamic loader
coopa::gfx::util::init_volk();

// 2. Check Vulkan API result macro
VkResult result = vkCreateInstance(&instance_info, nullptr, &instance);
GFX_VK_CHECK(result);

// 3. Find optimal depth format
VkFormat depth_format = coopa::gfx::util::find_depth_format(physical_device);
```
