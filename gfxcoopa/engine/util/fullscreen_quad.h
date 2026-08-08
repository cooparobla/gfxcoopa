/**
 * @file fullscreen_quad.h
 * @brief Utility for recording a fullscreen triangle draw (no vertex buffer needed).
 *
 * Uses the standard fullscreen triangle trick: 3 hard-coded clip-space positions
 * in the vertex shader, driven by gl_VertexIndex. No vertex buffer binding required.
 * Used by the upscale pass and all post-processing stages.
 */

#ifndef GFXCOOPA_ENGINE_UTIL_FULLSCREEN_QUAD_H
#define GFXCOOPA_ENGINE_UTIL_FULLSCREEN_QUAD_H

#include <gfxcoopa/command/command_buffer.h>

namespace coopa {
namespace gfx {
namespace engine {
namespace util {

/**
 * @class FullscreenQuad
 * @brief Stateless helper that records a fullscreen triangle draw.
 *
 * The matching vertex shader (upscale.vert, edge_detect.vert, etc.) must
 * derive positions from gl_VertexIndex to cover the full NDC space.
 *
 * Usage:
 * @code
 * FullscreenQuad fsq;
 * fsq.draw(cmd);  // records vkCmdDraw(3)
 * @endcode
 */
class FullscreenQuad {
public:
    /**
     * @brief Records a draw call for 3 vertices (one large triangle covering the screen).
     * @param cmd The command buffer to record into.
     */
    void draw(command::CommandBuffer& cmd) const {
        cmd.draw(3);
    }
};

} // namespace util
} // namespace engine
} // namespace gfx
} // namespace coopa

#endif // COOPA_GFX_ENGINE_FULLSCREEN_QUAD_H
