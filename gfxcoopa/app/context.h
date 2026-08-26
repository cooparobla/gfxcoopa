/**
 * @file context.h
 * @brief Owns the full Vulkan/windowing bring-up (Window -> Instance ->
 * Surface -> Device -> Allocator -> Swapchain -> CommandPool -> RenderPass
 * -> Renderer) plus timing and the frame loop, so a consumer application
 * never has to construct and order these nine objects by hand.
 *
 * Every pre-seal consumer either hand-rolled this exact 9-object sequence
 * inline (blendy's test.cpp, pixengine's demo.cpp, uicoopa's
 * test_window.cpp) or wrapped it in its own bespoke engine class
 * (toyengine's core::Engine). Context is that sequence, written once here,
 * with the two known bring-up bugs already fixed:
 *
 *  - presentation::Renderer::handle_resize() never called
 *    core::Swapchain::recreate() on its own (a caller-must-remember-to
 *    latent bug) -- Context installs the full correct sequence via
 *    Renderer::set_resize_handler() instead.
 *  - ONESHOT/MAX_FRAMES headless-testing env vars, previously reimplemented
 *    per-consumer (toyengine's core::Engine has them; blendy's test.cpp
 *    reimplements them inline; pixengine's demo.cpp has neither), are
 *    handled once by ContextConfig::from_env().
 *
 * Consumers that need their own asset/scene/config layer (as toyengine's
 * Engine does) are expected to COMPOSE a Context as their first member
 * (constructed first, destroyed last) rather than reimplement bring-up.
 */

#ifndef COOPA_GFX_APP_CONTEXT_H
#define COOPA_GFX_APP_CONTEXT_H

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <gfxcoopa/core/instance.h>
#include <gfxcoopa/core/surface.h>
#include <gfxcoopa/core/device.h>
#include <gfxcoopa/core/swapchain.h>
#include <gfxcoopa/memory/allocator.h>
#include <gfxcoopa/command/command_pool.h>
#include <gfxcoopa/command/command_buffer.h>
#include <gfxcoopa/pipeline/render_pass.h>
#include <gfxcoopa/presentation/window.h>
#include <gfxcoopa/presentation/renderer.h>
#include <gfxcoopa/types/enums.h>
#include <gfxcoopa/types/format.h>
#include <gfxcoopa/types/clear.h>
#include <gfxcoopa/detail/vk_convert.h>
#include <coopa/util/time.h>

namespace coopa {
namespace gfx {
namespace app {

/**
 * @struct ContextConfig
 * @brief Everything Context needs to bring up a window + device + swapchain.
 */
struct ContextConfig {
    std::string title      = "gfxcoopa";
    uint32_t    width      = 1280;
    uint32_t    height     = 720;
    bool        resizable  = true;
    bool        vsync      = true;
    bool        validation = true;
    Format      depth_format = Format::D32_Sfloat;

    bool        headless_oneshot = false; ///< True if ONESHOT was set via from_env().
    uint32_t    max_frames = 0;           ///< 0 = unlimited; run() stops after this many frames otherwise.

    /**
     * @brief Overlays the ONESHOT/MAX_FRAMES environment variables (the
     * convention every headless test target in this workspace already
     * uses) onto `base`.
     *
     * ONESHOT (if set, any value) forces max_frames to 1 and sets
     * headless_oneshot, taking priority over MAX_FRAMES if both are set --
     * matching toyengine::core::Engine's pre-existing precedence exactly.
     *
     * @param base Config to start from (e.g. one with title/size already set).
     * @return `base` with the env overlay applied.
     */
    static ContextConfig from_env(ContextConfig base) {
        if (const char* mf = std::getenv("MAX_FRAMES")) {
            base.max_frames = static_cast<uint32_t>(std::atoll(mf));
        }
        if (std::getenv("ONESHOT")) {
            base.max_frames = 1;
            base.headless_oneshot = true;
        }
        return base;
    }

    /// @brief Overload of from_env() defaulting `base` to ContextConfig{}.
    /// Split from the overload above (rather than a default argument)
    /// because a default argument of `ContextConfig{}` inside
    /// ContextConfig's own member function hits a genuine C++ aggregate-
    /// initialization-in-a-default-argument limitation (verified: GCC 13
    /// rejects it with "could not convert '{}' ... to 'ContextConfig'").
    static ContextConfig from_env() { return from_env(ContextConfig{}); }
};

/**
 * @struct FrameCallbacks
 * @brief The callbacks Context::frame()/run() invoke each frame, mirroring
 * presentation::Renderer::draw_frame()'s callback shape in sealed types.
 */
struct FrameCallbacks {
    /// @brief Records draw commands inside the swapchain render pass. Required.
    std::function<void(command::CommandBuffer&)> record;
    /// @brief Optional: records an earlier, separate render pass into the
    /// same command buffer before `record` runs -- see
    /// presentation::Renderer::draw_frame()'s pre_pass_fn docs.
    std::function<void(command::CommandBuffer&)> pre_pass;
    /// @brief Optional: invoked after a swapchain resize/recreate, with the new extent.
    std::function<void(Extent2D)> on_resize;
    /// @brief Clear color for the swapchain render pass.
    ClearColor clear{0.0f, 0.0f, 0.0f, 1.0f};
};

/**
 * @class Context
 * @brief Owns Window/Instance/Surface/Device/Allocator/Swapchain/
 * CommandPool/RenderPass/Renderer, plus frame timing and the main loop.
 *
 * Usage:
 * @code
 * gfx::app::Context ctx(gfx::app::ContextConfig::from_env({.title = "My App"}));
 * ctx.run([&](float dt) { my_scene.update(dt); },
 *         gfx::app::FrameCallbacks{ .record = [&](command::CommandBuffer& cmd) {
 *             my_pipeline.render(cmd, my_scene);
 *         }});
 * @endcode
 * or, driving the loop by hand:
 * @code
 * while (!ctx.should_close()) {
 *     ctx.poll();
 *     ctx.frame([&](command::CommandBuffer& cmd) { ... });
 * }
 * @endcode
 */
class Context {
public:
    /// @brief Brings up the full window + device + swapchain stack.
    explicit Context(const ContextConfig& config = {})
        : config_(config)
    {
        window_   = std::make_unique<presentation::Window>(config.title, config.width, config.height, config.resizable);
        instance_ = std::make_unique<core::Instance>(config.title, config.validation);
        surface_  = std::make_unique<core::Surface>(*instance_, *window_);
        device_   = std::make_unique<core::Device>(*instance_, *surface_);
        allocator_= std::make_unique<memory::Allocator>(*instance_, *device_);

        auto [fb_w, fb_h] = window_->framebuffer_size();
        swapchain_ = std::make_unique<core::Swapchain>(*device_, *surface_, fb_w, fb_h, config.vsync);

        cmd_pool_ = std::make_unique<command::CommandPool>(*device_, device_->graphics_family());

        // Renderer::create_framebuffers() only ever binds ONE color view per
        // swapchain framebuffer -- it has no depth image/view of its own to
        // bind as a second attachment. A RenderPass built with a real depth
        // format therefore does NOT match the framebuffers Renderer builds
        // against it (verified: triggers VUID-VkFramebufferCreateInfo-
        // attachmentCount-00876). Every existing caller of Renderer already
        // works around this by passing VK_FORMAT_UNDEFINED explicitly (see
        // gfxcoopa's own test.cpp fixture setup) despite RenderPass's raw
        // constructor defaulting depth_format to VK_FORMAT_D32_SFLOAT --
        // that default is only correct for a render pass NOT paired with
        // Renderer. config.depth_format is intentionally NOT used here; it
        // is exposed via depth_format() for a consumer building its OWN
        // separate depth-supporting render pass/target (e.g. a G-buffer),
        // which is unrelated to the swapchain present pass Context owns.
        render_pass_ = std::make_unique<pipeline::RenderPass>(*device_, swapchain_->image_format(),
                                                               VK_FORMAT_UNDEFINED);
        renderer_    = std::make_unique<presentation::Renderer>(*device_, *swapchain_, *render_pass_, *cmd_pool_);

        install_resize_handler();
    }

    /// @brief Waits for the device to go idle, then tears down in reverse
    /// bring-up order automatically via the unique_ptr members' destructors.
    ~Context() {
        if (device_) device_->wait_idle();
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // --- Owned object access ---
    presentation::Window&   window()       { return *window_; }
    core::Device&           device()       { return *device_; }
    core::Swapchain&        swapchain()    { return *swapchain_; }
    memory::Allocator&      allocator()    { return *allocator_; }
    command::CommandPool&   command_pool() { return *cmd_pool_; }
    pipeline::RenderPass&   render_pass()  { return *render_pass_; }
    presentation::Renderer& renderer()     { return *renderer_; }

    // --- Derived state ---
    Extent2D extent() const       { return detail::from_vk(swapchain_->extent()); }
    /// @brief The swapchain/present render pass's color format.
    Format   color_format() const { return detail::from_vk(swapchain_->image_format()); }
    /// @brief The depth format from ContextConfig, for a consumer's OWN
    /// depth-supporting render pass/target (e.g. a G-buffer) -- NOT used by
    /// Context's own swapchain render pass, which is always depth-less; see
    /// the constructor's comment on why.
    Format   depth_format() const { return config_.depth_format; }
    uint32_t frames_in_flight() const { return presentation::MAX_FRAMES_IN_FLIGHT; }
    uint32_t current_frame() const    { return renderer_->current_frame(); }
    /// @brief The configured frame limit (0 = unlimited), from ContextConfig::max_frames
    /// -- for a consumer driving its own loop (rather than run()) that still wants
    /// to honor ONESHOT/MAX_FRAMES without re-parsing the env vars itself.
    uint32_t max_frames() const { return config_.max_frames; }

    // --- Loop driving ---
    bool should_close() const { return window_->should_close(); }

    /// @brief Clears per-frame input state, polls OS events, and advances
    /// the frame timer. Call once per iteration before frame().
    void poll() {
        window_->new_frame();
        window_->poll_events();
        time_.update();
    }

    float    delta_time() const  { return time_.delta_time(); }
    double   elapsed() const     { return time_.elapsed(); }
    uint64_t frame_index() const { return time_.frame_count(); }

    /**
     * @brief Acquires, records, submits, and presents one frame.
     * @param cb Callbacks for this frame (see FrameCallbacks).
     * @return False if the frame was skipped (e.g. mid-resize); true otherwise.
     */
    bool frame(const FrameCallbacks& cb) {
        pending_on_resize_ = cb.on_resize;
        VkClearColorValue vk_clear = detail::to_vk(cb.clear);
        return renderer_->draw_frame(cb.record, vk_clear, nullptr, cb.pre_pass);
    }

    /// @brief Convenience overload taking just a record callback.
    bool frame(std::function<void(command::CommandBuffer&)> record) {
        FrameCallbacks cb;
        cb.record = std::move(record);
        return frame(cb);
    }

    /// @brief Records and submits a one-time command buffer outside the
    /// per-frame loop (e.g. a BRDF LUT bake, a GI probe capture at load
    /// time), blocking until the GPU finishes. See command::CommandPool::submit_once().
    void submit_once(std::function<void(command::CommandBuffer&)> record) {
        cmd_pool_->submit_once(std::move(record));
    }

    /// @brief Blocks until all GPU work has completed.
    void wait_idle() { device_->wait_idle(); }

    /**
     * @brief Runs the full poll -> update -> frame loop until the window
     * closes or (in headless test mode) config's max_frames is reached.
     * @param update Called once per iteration with the frame's delta time,
     *   before frame(). May be empty.
     * @param cb     Callbacks passed to frame() every iteration.
     */
    void run(std::function<void(float dt)> update, const FrameCallbacks& cb) {
        uint32_t frames_done = 0;
        while (!should_close()) {
            poll();
            if (update) update(delta_time());
            frame(cb);
            ++frames_done;
            if (config_.max_frames != 0 && frames_done >= config_.max_frames) break;
        }
    }

private:
    /**
     * @brief Installs the FULL correct resize sequence on the Renderer:
     * wait_idle -> poll framebuffer size until non-zero (handles minimize)
     * -> Swapchain::recreate() -> Renderer::recreate_framebuffers() -> the
     * caller's FrameCallbacks::on_resize. See presentation::Renderer::
     * set_resize_handler()'s docs for why the internal default path alone
     * is insufficient (it never calls Swapchain::recreate()).
     */
    void install_resize_handler() {
        renderer_->set_resize_handler([this]() {
            device_->wait_idle();
            auto [w, h] = window_->framebuffer_size();
            while ((w == 0 || h == 0) && !window_->should_close()) {
                window_->wait_events();
                std::tie(w, h) = window_->framebuffer_size();
            }
            swapchain_->recreate(w, h);
            renderer_->recreate_framebuffers();
            window_->reset_resized();
            if (pending_on_resize_) pending_on_resize_(Extent2D{w, h});
        });
    }

    ContextConfig config_;

    // Declaration order is bring-up/teardown order.
    std::unique_ptr<presentation::Window>   window_;
    std::unique_ptr<core::Instance>         instance_;
    std::unique_ptr<core::Surface>          surface_;
    std::unique_ptr<core::Device>           device_;
    std::unique_ptr<memory::Allocator>      allocator_;
    std::unique_ptr<core::Swapchain>        swapchain_;
    std::unique_ptr<command::CommandPool>   cmd_pool_;
    std::unique_ptr<pipeline::RenderPass>   render_pass_;
    std::unique_ptr<presentation::Renderer> renderer_;

    coopa::util::Time              time_;
    std::function<void(Extent2D)>  pending_on_resize_; ///< Set by frame(); read by install_resize_handler()'s lambda.
};

} // namespace app
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_APP_CONTEXT_H
