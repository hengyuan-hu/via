"""MCP server for controlling the browser-based robot interface.

* each tool owns its `description`, `input_schema`, and implementation
* `list_tools()` renders MCP tool definitions from the objects
* `call_tool()` dispatches through a small handler map

Start `record_sim.py` first; this server attaches to the Chromium instance that
record_sim launches with a CDP endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import logging
import os
import pathlib
import time
from dataclasses import dataclass

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from playwright.async_api import Browser, Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

# Self-contained logging: this server is launched as a bare script
# (`.venv/bin/python via/mcp_server.py`) without the repo root on PYTHONPATH,
# so it can't import via.utils. The format mirrors via.utils.LOG_FORMAT. Logs go to
# stderr, which `claude -p` folds into the per-seed claude log (and away from the
# stdout the MCP stdio protocol owns).
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Server("sphinx-robot")

# Ports for record_sim's stack, derived from the SPHINX_BASE_PORT env var
# (default 8100) so this server can attach to a non-default stack -- set the same
# env var for record_sim and this MCP server to drive a second instance. Must
# match record_sim.py: BASE_PORT UI HTTP, +1 broadcast WS + JSON, +3 Chromium CDP.
_BASE_PORT = int(os.environ.get("SPHINX_BASE_PORT", "8100"))
CDP_URL = f"http://localhost:{_BASE_PORT + 3}"
SIM_HTTP_BASE = f"http://localhost:{_BASE_PORT + 1}"
UI_URL = f"http://localhost:{_BASE_PORT}"

# Short pauses (seconds) that let the browser repaint before a screenshot.
FRAME_DELAY_S = 0.1  # One repaint frame: crosshair overlay, zoom, camera pose.
SETTLE_DELAY_S = 0.15  # Let a drag / mesh swap / camera reset settle.
# The gripper open/closed swap loads an OBJ asynchronously, so the new state is not
# visible right after the keypress. Poll for the flip instead of reading back once.
TOGGLE_POLL_DELAY_S = 0.1  # Interval between state checks while the swap commits.
TOGGLE_POLL_ATTEMPTS = 20  # Up to ~2 s total for the OBJ fetch + parse to land.

# Upper bound (ms) to wait for a waypoint execution to report completion (the
# browser's __waypointDoneCount tick) before snapshotting anyway.
EXEC_TIMEOUT_MS = 30_000


async def _fetch_sim_json(path: str) -> dict | None:
    """Best-effort GET of a JSON sidecar from record_sim's broadcast server."""
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{SIM_HTTP_BASE}/{path}")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


CLICK_SETTLE_DELAY_S = 0.2  # Clicks trigger more UI work; wait a bit longer.
WHEEL_TICK_DELAY_S = 0.04  # Space out wheel ticks so the browser handles each.
KEY_PRESS_DELAY_S = 0.06  # Seconds between successive keypresses so the UI handles each.
# Distance (UI units), not a delay: how far one f/b keypress moves the gripper along
# its approach axis. Mirrors the UI's hardcoded 0.06 step; same value as the delay above
# is coincidental.
GRIPPER_KEY_STEP_UI = 0.06
GRIPPER_KEY_STEP_M = GRIPPER_KEY_STEP_UI / 10.0

# Shared coordinate-frame reference, prepended to the description of every tool
# that sets frames_preamble = True. Single source of truth for the robot frame
# and the approach/opening orientation convention the spatial tools report in;
# the 1 unit = 0.1 m scale matches ui_to_robot().
FRAMES_PREAMBLE = (
    "Frame (shared by all spatial tools):\n"
    "  robot - metres; the canonical frame every pose, delta, and camera target "
    "reports in and that every tool input expects.\n"
    "  orientation - the gripper's pose is reported as two robot-frame unit "
    "vectors: 'approach' (the direction the fingers reach) and 'opening' (the "
    "fingertip-to-fingertip axis the jaws open/close along).\n"
    "Robot Z position carries a fixed height offset, so a canonical top-down "
    "gripper reads z~1.0, not 0; reason about position deltas.\n\n"
)


@dataclass
class ToolContext:
    name: str
    browser: Browser
    page: Page
    ui_z_offset: float = 0.8
    # Absolute path of the current demo's screenshots/ folder, published by
    # record_sim via /env.json. None until the first successful refresh.
    screenshots_dir: str | None = None

    async def refresh_env_info(self) -> None:
        """Refresh UI/robot-frame metadata from record_sim if available."""
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"{SIM_HTTP_BASE}/env.json")
                resp.raise_for_status()
                info = resp.json()
            self.ui_z_offset = float(info.get("ui_z_offset", self.ui_z_offset))
            screenshots_dir = info.get("screenshots_dir")
            if screenshots_dir:
                self.screenshots_dir = str(screenshots_dir)
        except Exception:
            pass

    async def _screenshot_jpeg(self) -> bytes:
        """Take a 1400x900 CSS-pixel JPEG screenshot of the browser."""
        return await self.page.screenshot(type="jpeg", quality=85, scale="css")

    def _image_content(self, img: bytes) -> types.ImageContent:
        return types.ImageContent(
            type="image",
            data=base64.b64encode(img).decode(),
            mimeType="image/jpeg",
        )

    def _save_screenshot(self, img: bytes) -> pathlib.Path:
        """Save screenshot bytes into the current demo's screenshots/ folder
        (published by record_sim via /env.json), with a newest-first filename
        prefix. Falls back to ./data/screenshots when record_sim has not reported
        a location yet."""
        out_dir = pathlib.Path(self.screenshots_dir or "data/screenshots")
        out_dir.mkdir(parents=True, exist_ok=True)

        now = time.time_ns()
        reverse_ns = 9_999_999_999_999_999_999 - now
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return_path = out_dir / f"{reverse_ns:019d}_{self.name}_{stamp}.jpg"
        return_path.write_bytes(img)
        return return_path

    async def snap(self, text: str | None = None) -> list[types.ContentBlock]:
        """Capture, save, and return a screenshot as MCP content."""
        img = await self._screenshot_jpeg()
        path = self._save_screenshot(img)
        # Report the path relative to the demo folder, not absolute: the absolute
        # path would hand an agent with any filesystem/shell access (e.g. the codex
        # harness, whose shell can't be disabled) the location of the live episode
        # dir and, transitively, the repo's ground-truth files — an asymmetric
        # cheating vector the eval's tool restrictions are meant to close. Humans
        # reading transcripts still get a usable pointer.
        rel = pathlib.Path(os.path.relpath(path, pathlib.Path(path).parent.parent))
        message = f"{text}\n  saved screenshot: {rel}" if text else f"Saved screenshot to {rel}"
        return [
            types.TextContent(type="text", text=message),
            self._image_content(img),
        ]

    async def show_cursor(self, x: int, y: int) -> None:
        """Render or move the persistent crosshair overlay shown in screenshots.

        The overlay is a symmetric reticle (crosshair arms with a small center
        dot) centered exactly on (x, y). Being symmetric, it has no directional
        body to bias visual judgment: the target point is its geometric center,
        and the gap between the arms leaves the target pixel visible.
        """
        await self.page.evaluate(
            """([x, y]) => {
                const oldDrag = document.getElementById('mcp-drag-overlay');
                if (oldDrag) oldDrag.remove();

                let cursor = document.getElementById('mcp-visible-cursor');
                if (!cursor) {
                    cursor = document.createElement('div');
                    cursor.id = 'mcp-visible-cursor';
                    cursor.innerHTML = `
                        <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
                          <g stroke-linecap="round">
                            <path d="M14 3 V10 M14 18 V25 M3 14 H10 M18 14 H25"
                                  stroke="black" stroke-width="4" fill="none"/>
                            <path d="M14 3 V10 M14 18 V25 M3 14 H10 M18 14 H25"
                                  stroke="white" stroke-width="2" fill="none"/>
                            <circle cx="14" cy="14" r="1.6"
                                    fill="white" stroke="black" stroke-width="1"/>
                          </g>
                        </svg>`;
                    Object.assign(cursor.style, {
                        position: 'fixed',
                        width: '28px',
                        height: '28px',
                        pointerEvents: 'none',
                        zIndex: '2147483647',
                        filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.85))',
                    });
                    document.body.appendChild(cursor);
                }
                cursor.style.left = (x - 14) + 'px';
                cursor.style.top = (y - 14) + 'px';
            }""",
            [x, y],
        )

    async def show_drag_feedback(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Show a start marker, end crosshair, and arrow path for drag screenshots."""
        await self.show_cursor(x2, y2)
        await self.page.evaluate(
            """([x1, y1, x2, y2]) => {
                const old = document.getElementById('mcp-drag-overlay');
                if (old) old.remove();

                const ns = 'http://www.w3.org/2000/svg';
                const svg = document.createElementNS(ns, 'svg');
                svg.id = 'mcp-drag-overlay';
                svg.setAttribute('width', String(window.innerWidth));
                svg.setAttribute('height', String(window.innerHeight));
                svg.setAttribute('viewBox', `0 0 ${window.innerWidth} ${window.innerHeight}`);
                Object.assign(svg.style, {
                    position: 'fixed',
                    left: '0',
                    top: '0',
                    width: '100vw',
                    height: '100vh',
                    pointerEvents: 'none',
                    zIndex: '2147483646',
                    transition: 'opacity 0.8s ease-out',
                    opacity: '1',
                });

                const defs = document.createElementNS(ns, 'defs');
                const makeMarker = (id, color, size) => {
                    const marker = document.createElementNS(ns, 'marker');
                    marker.id = id;
                    marker.setAttribute('markerWidth', String(size));
                    marker.setAttribute('markerHeight', String(size));
                    marker.setAttribute('refX', String(size - 1));
                    marker.setAttribute('refY', String(size / 2));
                    marker.setAttribute('orient', 'auto');
                    marker.setAttribute('markerUnits', 'strokeWidth');
                    const tip = document.createElementNS(ns, 'path');
                    tip.setAttribute('d', `M 0 0 L ${size} ${size / 2} L 0 ${size} z`);
                    tip.setAttribute('fill', color);
                    marker.appendChild(tip);
                    defs.appendChild(marker);
                };
                makeMarker('mcp-drag-arrow', '#fde047', 7);
                svg.appendChild(defs);

                const makeLine = (color, width, markerId, opacity) => {
                    const line = document.createElementNS(ns, 'line');
                    line.setAttribute('x1', String(x1));
                    line.setAttribute('y1', String(y1));
                    line.setAttribute('x2', String(x2));
                    line.setAttribute('y2', String(y2));
                    line.setAttribute('stroke', color);
                    line.setAttribute('stroke-width', String(width));
                    line.setAttribute('stroke-linecap', 'round');
                    line.setAttribute('opacity', String(opacity));
                    line.setAttribute('marker-end', `url(#${markerId})`);
                    svg.appendChild(line);
                };
                makeLine('#fde047', 4, 'mcp-drag-arrow', 0.96);

                const makeCircle = (cx, cy, r, fill, stroke, width) => {
                    const circle = document.createElementNS(ns, 'circle');
                    circle.setAttribute('cx', String(cx));
                    circle.setAttribute('cy', String(cy));
                    circle.setAttribute('r', String(r));
                    circle.setAttribute('fill', fill);
                    circle.setAttribute('stroke', stroke);
                    circle.setAttribute('stroke-width', String(width));
                    svg.appendChild(circle);
                };
                makeCircle(x1, y1, 10, 'rgba(0,0,0,0.86)', 'rgba(0,0,0,0.86)', 1);
                makeCircle(x1, y1, 6, '#22c55e', 'white', 2);
                makeCircle(x2, y2, 10, 'rgba(0,0,0,0.86)', 'rgba(0,0,0,0.86)', 1);
                makeCircle(x2, y2, 6, '#38bdf8', 'white', 2);

                document.body.appendChild(svg);
                setTimeout(() => {
                    svg.style.opacity = '0';
                    setTimeout(() => svg.remove(), 900);
                }, 5000);
            }""",
            [x1, y1, x2, y2],
        )

    async def perform_drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        button: str = "left",
        steps: int = 25,
        held_key: str | None = None,
    ) -> None:
        """Press at (x1, y1), drag to (x2, y2) with the given mouse button while
        optionally holding ``held_key``, release, then draw drag feedback and let
        the view settle."""
        if held_key:
            await self.page.keyboard.down(held_key)
        try:
            await self.page.mouse.move(x1, y1)
            await self.page.mouse.down(button=button)  # type: ignore
            await self.page.mouse.move(x2, y2, steps=steps)
            await self.page.mouse.up(button=button)  # type: ignore
        finally:
            if held_key:
                await self.page.keyboard.up(held_key)
        await self.show_drag_feedback(x1, y1, x2, y2)
        await asyncio.sleep(SETTLE_DELAY_S)

    async def uv_to_xy(self, u: float, v: float) -> tuple[int, int, int, int]:
        """Convert normalized screenshot coordinates to browser viewport pixels."""
        size = await self.page.evaluate(
            """() => ({
            width: window.innerWidth,
            height: window.innerHeight,
        })"""
        )
        width = int(size["width"])
        height = int(size["height"])
        x = min(max(round(float(u) * width), 0), max(width - 1, 0))
        y = min(max(round(float(v) * height), 0), max(height - 1, 0))
        return x, y, width, height

    async def contact_point_text(self, x: int, y: int) -> str:
        """Describe the salient point a click at browser pixel (x, y) would select.

        Read-only preview: runs the same pick the real teleport-click uses
        (window.previewContactPointUI) without moving cubeMesh or the gripper.
        """
        preview = await self.page.evaluate(
            "([x, y]) => window.previewContactPointUI"
            " ? window.previewContactPointUI(x, y) : null",
            [x, y],
        )
        if not preview:
            return "  contact point: unavailable"
        surface = preview.get("surface")
        point = preview.get("point")
        if point:
            robot = self.ui_to_robot(point["x"], point["y"], point["z"])
            where = "" if surface == "pointcloud" else f" via {surface} feed"
            return (
                f"  contact point if clicked here{where} (robot, m): "
                f"x={robot['x']:.4f}  y={robot['y']:.4f}  z={robot['z']:.4f}"
            )
        if surface == "gripper":
            return "  contact point: none - over the blue target gripper (a click here drags, not places)"
        if surface == "other":
            return "  contact point: none - not a placement surface (point-cloud canvas or a camera feed)"
        # previewContactPointUI requires a genuine cloud hit (no snap-to-nearest
        # fallback), so a null point on the canvas/feeds means (u, v) is over empty
        # space with no cloud point under the cursor.
        return "  contact point: none - (u, v) is not over a point in the cloud"

    def ui_to_robot(self, ui_x: float, ui_y: float, ui_z: float) -> dict[str, float]:
        """Convert UI-frame Three.js coordinates to robot-frame metres."""
        return {
            "x": round(ui_x / 10.0, 4),
            "y": round(-ui_z / 10.0, 4),
            "z": round(ui_y / 10.0 + self.ui_z_offset, 4),
        }

    def robot_to_ui(self, x: float, y: float, z: float) -> dict[str, float]:
        """Inverse of ui_to_robot: robot-frame metres -> UI Three.js units.

        Unrounded so camera_set_pose round-trips a camera_get_pose target.
        """
        return {
            "x": x * 10.0,
            "y": (z - self.ui_z_offset) * 10.0,
            "z": -y * 10.0,
        }

    async def camera_pose(self) -> dict | None:
        """Return camera pose in the azimuth/elevation/distance convention."""
        return await self.page.evaluate(
            """() => {
            if (typeof camera === 'undefined' || !camera) return null;
            const target = (typeof cameraTarget !== 'undefined' && cameraTarget)
                ? cameraTarget
                : (window.cameraTarget || { x: 0, y: 0, z: 0 });
            const pos = camera.position;
            const dx = pos.x - target.x;
            const dy = pos.y - target.y;
            const dz = pos.z - target.z;
            const distance = Math.hypot(dx, dy, dz);
            const azimuth = distance > 1e-9
                ? (Math.atan2(dx, dz) * 180 / Math.PI + 360) % 360
                : 0;
            const elevation = distance > 1e-9
                ? Math.asin(dy / distance) * 180 / Math.PI
                : 0;
            return {
                target: { x: target.x, y: target.y, z: target.z },
                distance,
                azimuth,
                elevation,
            };
        }"""
        )

    def format_camera_pose(self, pose: dict | None, heading: str = "Camera pose") -> str:
        if pose is None:
            return "Camera pose unavailable: browser camera object not found."

        def fmt(value: float) -> str:
            return f"{float(value):.4f}"

        target = pose["target"]
        robot_target = self.ui_to_robot(target["x"], target["y"], target["z"])
        return (
            f"{heading}\n"
            f"  azimuth:   {fmt(pose['azimuth'])} deg (horizontal orbit angle)\n"
            f"  elevation: {fmt(pose['elevation'])} deg (0=horizontal, 90=top-down)\n"
            f"  distance:  {fmt(pose['distance'] / 10.0)} m (camera to target)\n"
            f"  target (robot, m): x={fmt(robot_target['x'])}  "
            f"y={fmt(robot_target['y'])}  z={fmt(robot_target['z'])}"
        )

    async def camera_pose_text(self, heading: str = "Camera pose") -> str:
        return self.format_camera_pose(await self.camera_pose(), heading)

    async def gripper_pose(self) -> dict | None:
        """Return the blue target gripper pose at the fingertip center."""
        pose = await self.page.evaluate(
            """() => {
            if (typeof selectedObject === 'undefined' || !selectedObject) return null;
            const pos = selectedObject.position;
            const q = new THREE.Quaternion().setFromEuler(selectedObject.rotation);
            // Report orientation as two robot-frame unit vectors instead of Euler
            // angles: a single local rotation couples across all three Euler
            // components off the top-down pose, and the YZX decomposition is
            // multivalued, so Euler reads (and their deltas) mislead. approach =
            // local -Y (where the fingers reach); opening = local +Z (the
            // fingertip-to-fingertip jaw axis). A Three.js direction (x,y,z) maps
            // to a robot-frame direction (x,-z,y), matching ui_to_robot.
            const toRobotDir = (v) => ({ x: v.x, y: -v.z, z: v.y });
            const approach = new THREE.Vector3(0, -1, 0).applyQuaternion(q);
            const opening = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
            let url = null;
            if (
                typeof controlPoints !== 'undefined' &&
                typeof meshUrls !== 'undefined' &&
                Array.isArray(controlPoints)
            ) {
                const idx = controlPoints.indexOf(selectedObject);
                if (idx >= 0) url = meshUrls[idx] || null;
            }
            return {
                ui_position: { x: pos.x, y: pos.y, z: pos.z },
                robot_approach: toRobotDir(approach),
                robot_opening: toRobotDir(opening),
                gripper_open: url === null ? null : !url.includes('_closed'),
            };
        }"""
        )
        if pose is None:
            return None
        pose["robot_position"] = self.ui_to_robot(
            pose["ui_position"]["x"],
            pose["ui_position"]["y"],
            pose["ui_position"]["z"],
        )
        return pose

    def format_gripper_pose(self, pose: dict | None) -> str:
        if pose is None:
            return "Gripper pose unavailable: blue target gripper object not found."

        def fmt(value: float) -> str:
            return f"{float(value):.4f}"

        def fmt_dir(value: float) -> str:
            return f"{float(value):+.3f}"

        robot_pos = pose["robot_position"]
        approach = pose["robot_approach"]
        opening = pose["robot_opening"]
        state = pose.get("gripper_open")
        if state is None:
            state_text = "unknown"
        else:
            state_text = "open" if state else "closed"
        return (
            "Current Target Gripper Pose:\n"
            "Position is the midpoint between the fingertips of the blue target gripper. "
            "Orientation is given as two robot-frame unit vectors, approach and opening.\n"
            f"  position (robot, m):    x={fmt(robot_pos['x'])}  "
            f"y={fmt(robot_pos['y'])}  z={fmt(robot_pos['z'])}\n"
            f"  approach (robot, unit): x={fmt_dir(approach['x'])}  "
            f"y={fmt_dir(approach['y'])}  z={fmt_dir(approach['z'])}  "
            "(direction the fingers reach; the grasp-approach axis)\n"
            f"  opening (robot, unit):  x={fmt_dir(opening['x'])}  "
            f"y={fmt_dir(opening['y'])}  z={fmt_dir(opening['z'])}  "
            "(fingertip-to-fingertip axis the jaws open/close along)\n"
            f"  gripper: {state_text}"
        )

    async def gripper_pose_text(self) -> str:
        return self.format_gripper_pose(await self.gripper_pose())

    async def is_translation_mode(self) -> bool:
        active = await self.page.evaluate(
            """() => {
            const translation = typeof isTranslationMode === 'undefined' || !!isTranslationMode;
            const rotating = typeof transformControl !== 'undefined' && !!transformControl;
            return translation && !rotating;
        }"""
        )
        return bool(active)

    async def ensure_translation_mode(self) -> bool:
        """Return True if the gripper is movable, recovering silently if not.

        The agent is intentionally unaware of UI modes, so if a lingering
        rotation mode would block a move, force-exit it with Escape (which
        discards any pending rotation) and re-check. Only a genuinely
        unrecoverable state returns False, letting the caller fail with
        mode-free wording.
        """
        if await self.is_translation_mode():
            return True
        await self.page.keyboard.press("Escape")
        await asyncio.sleep(SETTLE_DELAY_S)
        return await self.is_translation_mode()


class ToolBase:
    name: str
    description: str
    input_schema: dict
    frames_preamble: bool = False  # prepend FRAMES_PREAMBLE for spatial tools

    def as_mcp_tool(self) -> types.Tool:
        description = self.description
        if self.frames_preamble:
            description = FRAMES_PREAMBLE + description
        return types.Tool(
            name=self.name,
            description=description,
            inputSchema=self.input_schema,
        )

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        raise NotImplementedError

    async def _start_hits_gripper(self, ctx: ToolContext, x: int, y: int) -> bool:
        hit = await ctx.page.evaluate(
            """([x, y]) => {
            if (typeof renderer === 'undefined' || !renderer) return false;
            if (typeof camera === 'undefined' || !camera) return false;
            if (typeof selectedObject === 'undefined' || !selectedObject) return false;

            const rect = renderer.domElement.getBoundingClientRect();
            if (
                x < rect.left || x > rect.right ||
                y < rect.top || y > rect.bottom ||
                rect.width <= 0 || rect.height <= 0
            ) {
                return false;
            }

            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(new THREE.Vector2(
                ((x - rect.left) / rect.width) * 2 - 1,
                -((y - rect.top) / rect.height) * 2 + 1
            ), camera);
            selectedObject.updateMatrixWorld(true);
            return raycaster.intersectObject(selectedObject, false).length > 0;
        }""",
            [x, y],
        )
        return bool(hit)


class ScreenshotTool(ToolBase):
    name = "screenshot"
    description = (
        "Capture a screenshot of the robot interface.\n\n"
        "The window shows:\n"
        "  - Left sidebar: two camera feeds. Both are rendered from the real "
        "scene and they refresh only after execute_waypoint moves the real "
        "gripper.\n"
        "      - 3rd-person view: a fixed camera looking at the table and gripper. "
        "Best for analyzing the full scene layout (objects relative to each other "
        "and the gripper).\n"
        "      - Wrist view: a camera on the gripper looking down between the fingers. "
        "Best for the close-up gripper-object relationship (whether the gripper is "
        "well positioned for a grasp or place).\n"
        "  - Right main view: a 3-D point cloud reconstruction of the scene, with "
        "robot-frame axes (X red, Y green, Z blue) and the blue target gripper mesh you "
        "command. Best for 3D / depth understanding that the flat camera images "
        "can't give.\n\n"
        "Call this first, or whenever you need a fresh view without taking an action."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        return await ctx.snap()


class HoverTool(ToolBase):
    name = "hover"
    description = (
        "Move the mouse to normalized screenshot coordinates (u, v) and show "
        "a visible crosshair overlay in the returned screenshot.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Reading the overlay:\n"
        "  The exact (u, v) point is the crosshair's center dot.\n\n"
        "Use hover to inspect a spot before acting on it. It takes no action and "
        "gives two readouts:\n"
        "  - Visual: the crosshair (a persistent DOM overlay, not the OS cursor) "
        "marks exactly where (u, v) lands, confirming it's on the intended UI "
        "element or point-cloud region.\n"
        "  - Contact point (robot-frame x/y/z, m), computed read-only. On the "
        "point-cloud canvas or a camera feed it reports the cloud point under the "
        "cursor (the ray needn't be pixel-exact - it hits within a small radius). "
        "Two uses:\n"
        "      1. Probe the robot-frame coordinate of any point of interest.\n"
        "      2. Preview a teleport - when it lands on a point it is exactly the "
        "salient point a gripper_teleport_via_click at this same (u, v) would select. "
        "Note this is the clicked point itself, NOT the resulting gripper pose (which "
        "lands ~0.07 m back along the approach axis).\n"
        "    It reads 'none' when (u, v) is not over a cloud point (empty space), "
        "over the blue target gripper, or over UI chrome rather than the point-cloud "
        "canvas or a camera feed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
        },
        "required": ["u", "v"],
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        u = float(arguments["u"])
        v = float(arguments["v"])
        x, y, width, height = await ctx.uv_to_xy(u, v)
        await ctx.page.mouse.move(x, y)
        await ctx.show_cursor(x, y)
        await asyncio.sleep(FRAME_DELAY_S)
        return await ctx.snap(
            f"Hovered at u={u:.4f}, v={v:.4f} "
            f"(browser pixel x={x}, y={y}, viewport={width}x{height})\n"
            f"{await ctx.contact_point_text(x, y)}"
        )


# Gripper tools
class GripperTeleportViaClickTool(ToolBase):
    name = "gripper_teleport_via_click"
    frames_preamble = True
    description = (
        "Fast-move the blue target gripper near a clicked salient point. "
        "Returns a screenshot. Best for quickly setting a subtask's starting "
        "position; refine afterward with gripper_drag/move/rotate/toggle.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Where to click:\n"
        "  - Left-panel 2-D camera views: best when the target is clearly "
        "visible in either camera feed.\n"
        "  - Right-panel 3-D point-cloud view: best when locating the target "
        "takes some 3D reasoning about depth.\n"
        "  - Click the object's center, not its edge, so the gripper lands "
        "centered over it for a reliable grasp or place.\n\n"
        "What happens:\n"
        "  The gripper lands ~0.07 m back from the clicked point along its "
        "current approach axis (opposite the reach direction), not on the point "
        "itself - so a short forward move/descent then reaches it. E.g. if the "
        "gripper currently points straight down, this puts it ~0.07 m directly "
        "above the click. Translation only - the gripper's orientation is "
        "preserved.\n"
        "  The readout reports the clicked point as the 'selected salient point' "
        "and the resulting gripper pose separately; check both, plus the "
        "screenshot. Clicking the blue target gripper itself does nothing here - that "
        "is reserved for gripper_drag."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
        },
        "required": ["u", "v"],
    }

    async def _surface_at(self, ctx: ToolContext, x: int, y: int) -> dict:
        return await ctx.page.evaluate(
            """([x, y]) => {
            const el = document.elementFromPoint(x, y);
            const closest = (selector) =>
                el && typeof el.closest === 'function' ? el.closest(selector) : null;

            if (closest('#btn-record')) {
                return { kind: 'action', label: 'Execute Waypoint button', tool: 'execute_waypoint' };
            }
            if (closest('#btn-end')) {
                return { kind: 'action', label: 'End Episode button', tool: 'end_episode' };
            }

            const cam = closest('#cam-agentview, #cam-wrist');
            if (cam) {
                return {
                    kind: 'gripper_surface',
                    label: cam.id === 'cam-agentview' ? 'agentview camera feed' : 'wrist camera feed',
                };
            }

            if (typeof renderer !== 'undefined' && renderer && el === renderer.domElement) {
                if (
                    typeof camera !== 'undefined' && camera &&
                    typeof selectedObject !== 'undefined' && selectedObject
                ) {
                    const rect = renderer.domElement.getBoundingClientRect();
                    const raycaster = new THREE.Raycaster();
                    raycaster.setFromCamera(new THREE.Vector2(
                        ((x - rect.left) / rect.width) * 2 - 1,
                        -((y - rect.top) / rect.height) * 2 + 1
                    ), camera);
                    selectedObject.updateMatrixWorld(true);
                    if (raycaster.intersectObject(selectedObject, false).length > 0) {
                        return {
                            kind: 'unsupported',
                            label: 'blue target gripper mesh; use gripper_drag'
                        };
                    }
                }
                return { kind: 'gripper_surface', label: '3-D point-cloud canvas', hold: 's' };
            }

            let label = 'unknown UI surface';
            if (el) {
                label = el.id ? `#${el.id}` : el.tagName.toLowerCase();
            }
            return { kind: 'unsupported', label };
        }""",
            [x, y],
        )

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        u = float(arguments["u"])
        v = float(arguments["v"])
        x, y, width, height = await ctx.uv_to_xy(u, v)
        surface = await self._surface_at(ctx, x, y)

        await ctx.show_cursor(x, y)
        if surface["kind"] == "action":
            await asyncio.sleep(FRAME_DELAY_S)
            return await ctx.snap(
                f"gripper_teleport_via_click did not run: u={u:.4f}, v={v:.4f} lands on the "
                f"{surface['label']}. Use the {surface['tool']} tool instead.\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"{await ctx.gripper_pose_text()}"
            )

        if surface["kind"] != "gripper_surface":
            await asyncio.sleep(FRAME_DELAY_S)
            return await ctx.snap(
                f"gripper_teleport_via_click did not run: u={u:.4f}, v={v:.4f} lands on "
                f"{surface['label']}, not a gripper placement surface.\n"
                "  Click the 3-D point-cloud canvas or a camera feed, or use "
                "execute_waypoint/end_episode for episode actions.\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"{await ctx.gripper_pose_text()}"
            )

        # Placing a salient point on the 3-D canvas now requires holding "s" so
        # stray clicks meant to orbit don't teleport the gripper. _surface_at
        # tags that surface with hold='s'; the camera-feed handlers don't gate
        # on "s", so only hold it when the surface asks for it.
        hold_key = surface.get("hold")
        if hold_key:
            await ctx.page.keyboard.down(hold_key)
        try:
            await ctx.page.mouse.click(x, y, button="left")
        finally:
            if hold_key:
                await ctx.page.keyboard.up(hold_key)
        await asyncio.sleep(CLICK_SETTLE_DELAY_S)

        target = await ctx.page.evaluate(
            """() => {
            if (typeof cubeMesh === 'undefined' || !cubeMesh) return null;
            if (!cubeMesh.visible) return null;
            return { x: cubeMesh.position.x, y: cubeMesh.position.y, z: cubeMesh.position.z };
        }"""
        )

        if target:
            robot = ctx.ui_to_robot(target["x"], target["y"], target["z"])
            text = (
                f"Teleported blue target gripper from {surface['label']} click "
                f"at u={u:.4f}, v={v:.4f}\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"  selected salient point (robot, m): "
                f"x={robot['x']:.4f}  y={robot['y']:.4f}  z={robot['z']:.4f}\n"
                f"{await ctx.gripper_pose_text()}"
            )
        else:
            text = (
                f"Clicked {surface['label']} at u={u:.4f}, v={v:.4f}, "
                "but no 3-D gripper target is available.\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"{await ctx.gripper_pose_text()}"
            )

        return await ctx.snap(text)


class GripperDragTool(ToolBase):
    name = "gripper_drag"
    frames_preamble = True
    description = (
        "Drag the blue target gripper between normalized screenshot coords "
        "(u across 0->1 left->right, v down 0->1 top->bottom). Returns a screenshot.\n\n"
        "The blue target gripper is the target pose you command; this tool moves it. "
        "The gray point-cloud gripper is actual sensor data and is not a draggable mesh.\n\n"
        "Start point: u1/v1 must ray-hit the solid blue target gripper mesh - a finger, wrist, "
        "or body, not the empty gap between the fingers or the gray gripper. A miss is "
        "refused with a no-op.\n\n"
        "Motion model: free drag moves the gripper with the cursor in the view "
        "plane, ~0.0006 m per CSS pixel of mouse movement, before constraints.\n\n"
        "Movement modes - free is the default (no lock); x/y/z lock motion to one "
        "robot-frame axis (the fixed robot X/Y/Z, not the gripper-attached gizmo "
        "axes that gripper_rotate spins about) and are exact at any camera angle, "
        "so trust the pose readout, not the picture:\n"
        "  free - unconstrained: follows the cursor in the camera view plane (camera-dependent)\n"
        "  x    - lock to robot X only\n"
        "  y    - lock to robot Y only\n"
        "  z    - lock to robot Z / vertical\n\n"
        "Tip: hover first to confirm u1/v1 lands on the gripper, then use small single-axis "
        "drags. The screenshot shows start/end markers, end crosshair, the arrow path, and the "
        "updated pose."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u1": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Start fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v1": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Start fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "u2": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "End fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v2": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "End fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "constraint": {
                "type": "string",
                "enum": ["free", "x", "y", "z"],
                "description": "Movement constraint. Default: free.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Mouse move interpolation steps. Default: 25.",
            },
        },
        "required": ["u1", "v1", "u2", "v2"],
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        u1 = float(arguments["u1"])
        v1 = float(arguments["v1"])
        u2 = float(arguments["u2"])
        v2 = float(arguments["v2"])
        x1, y1, width, height = await ctx.uv_to_xy(u1, v1)
        x2, y2, _, _ = await ctx.uv_to_xy(u2, v2)
        steps = min(max(int(arguments.get("steps", 25)), 1), 100)

        constraint = str(arguments.get("constraint", "free")).lower()
        if constraint not in {"free", "x", "y", "z"}:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Invalid gripper_drag constraint {constraint!r}; "
                        "expected 'free', 'x', 'y', or 'z'."
                    ),
                )
            ]

        if not await ctx.ensure_translation_mode():
            return await ctx.snap(
                "gripper_drag did not run: couldn't move the gripper right now; try again."
            )

        if not await self._start_hits_gripper(ctx, x1, y1):
            await ctx.show_cursor(x1, y1)
            await asyncio.sleep(FRAME_DELAY_S)
            return await ctx.snap(
                f"gripper_drag did not run: u1={u1:.4f}, v1={v1:.4f} "
                "does not ray-hit solid geometry on the blue target gripper mesh.\n"
                f"  browser pixel: x={x1}, y={y1}, viewport={width}x{height}\n"
                "  aim at a solid blue jaw block, wrist, or body; avoid the hollow "
                "slot between jaws and the inactive gray gripper.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        held_key = None if constraint == "free" else constraint
        await ctx.perform_drag(x1, y1, x2, y2, steps=steps, held_key=held_key)

        return await ctx.snap(
            f"Gripper dragged u={u1:.4f}, v={v1:.4f} -> u={u2:.4f}, v={v2:.4f} "
            f"[constraint={constraint}]\n"
            f"  browser pixels: ({x1}, {y1}) -> ({x2}, {y2}), "
            f"viewport={width}x{height}, steps={steps}\n"
            f"{await ctx.gripper_pose_text()}"
        )


class GripperAdvanceOrRetreatTool(ToolBase):
    name = "gripper_advance_or_retreat"
    frames_preamble = True
    description = (
        "Move the blue target gripper along its approach axis with f/b. "
        "Returns a screenshot.\n\n"
        "  direction='f' - step along the approach axis (toward where "
        "the fingers point)\n"
        "  direction='b' - step opposite the approach axis (retreat / "
        "back out)\n"
        "  In robot frame the direction follows the current orientation (see "
        "Frame above): top-down, 'f' is -Z (down) and 'b' is +Z (lift / back "
        "out); on a side-grasp the same steps are horizontal.\n\n"
        "If the current orientation makes the f/b direction ambiguous, call "
        "gripper_get_pose first to see the approach axis drawn in the "
        "screenshot.\n\n"
        "Scale:\n"
        f"  One step is about {GRIPPER_KEY_STEP_M:.3f} m along the approach axis. "
        "Use 1-3 steps for small corrections and larger counts for coarse "
        "approach/retreat.\n\n"
        "The returned text reports the actual pose delta after f/b keypresses."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["f", "b"],
                "description": "f moves forward; b moves backward.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of f/b keypresses. Default: 1.",
            },
        },
        "required": ["direction"],
    }

    KEY_LABELS = {"f": "forward", "b": "backward"}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        raw_direction = arguments.get("direction")
        direction = "" if raw_direction is None else str(raw_direction).strip().lower()
        if direction not in self.KEY_LABELS:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Invalid gripper_advance_or_retreat direction {direction!r}; "
                        "expected 'f' or 'b'."
                    ),
                )
            ]
        key = direction

        pose = await ctx.gripper_pose()
        if pose is None:
            return await ctx.snap(
                "gripper_advance_or_retreat did not run: no blue target gripper is selected.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        if not await ctx.ensure_translation_mode():
            return await ctx.snap(
                "gripper_advance_or_retreat did not run: couldn't move the gripper right now; try again."
            )

        steps = min(max(int(arguments.get("steps", 1)), 1), 10)

        for _ in range(steps):
            await ctx.page.keyboard.press(key)
            await asyncio.sleep(KEY_PRESS_DELAY_S)
        await asyncio.sleep(SETTLE_DELAY_S)

        new_pose = await ctx.gripper_pose()
        if new_pose is None:
            return await ctx.snap(
                f"gripper_advance_or_retreat pressed '{key}' {steps} time(s), but the active "
                "gripper pose is unavailable after the move."
            )

        before_robot = pose["robot_position"]
        after_robot = new_pose["robot_position"]
        delta_robot = {axis: after_robot[axis] - before_robot[axis] for axis in ("x", "y", "z")}
        label = self.KEY_LABELS[key]
        expected_distance = steps * GRIPPER_KEY_STEP_M

        def fmt(value: float) -> str:
            return f"{float(value):.4f}"

        return await ctx.snap(
            f"Gripper moved {label} with '{key}' x{steps} "
            f"(nominal approach-axis distance {expected_distance:.4f} m).\n"
            f"  delta (robot, m): x={fmt(delta_robot['x'])}  "
            f"y={fmt(delta_robot['y'])}  z={fmt(delta_robot['z'])}\n"
            f"{ctx.format_gripper_pose(new_pose)}"
        )


MAX_TRANSLATE_STEP_M = 0.1


class GripperTranslateTool(ToolBase):
    name = "gripper_translate"
    frames_preamble = True
    description = (
        "Translate the blue target gripper by a metric offset along the robot "
        "frame axes, in metres. Returns a screenshot. You specify the offset "
        "directly - no start point, mesh hit, or mouse drag needed - so it is the "
        "exact, camera-independent way to shift the gripper along the world axes. "
        "This is the keyed equivalent of gripper_drag's x/y/z axis locks without "
        "the cursor imprecision; the orientation is unchanged.\n\n"
        "Axes are the fixed robot X/Y/Z (the colored arrows), not the "
        "gripper-attached gizmo axes that gripper_rotate spins about, so the "
        "offsets are exact at any camera angle - trust the pose readout:\n"
        "  d_x - along robot X\n"
        "  d_y - along robot Y\n"
        "  d_z - along robot Z (vertical: +z lifts, -z lowers)\n\n"
        "Set one offset for a pure single-axis move, or several to move "
        "diagonally; omitted offsets default to 0 and at least one must be "
        f"nonzero. Each offset must be within +/-{MAX_TRANSLATE_STEP_M} m to keep "
        "moves gradual; larger requests are rejected, so call repeatedly to "
        "translate further. The returned text reports "
        "the actual pose delta after the move."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "d_x": {
                "type": "number",
                "minimum": -MAX_TRANSLATE_STEP_M,
                "maximum": MAX_TRANSLATE_STEP_M,
                "description": (
                    "Offset along robot X, metres. Must be within "
                    f"+/-{MAX_TRANSLATE_STEP_M}. Default: 0."
                ),
            },
            "d_y": {
                "type": "number",
                "minimum": -MAX_TRANSLATE_STEP_M,
                "maximum": MAX_TRANSLATE_STEP_M,
                "description": (
                    "Offset along robot Y, metres. Must be within "
                    f"+/-{MAX_TRANSLATE_STEP_M}. Default: 0."
                ),
            },
            "d_z": {
                "type": "number",
                "minimum": -MAX_TRANSLATE_STEP_M,
                "maximum": MAX_TRANSLATE_STEP_M,
                "description": (
                    "Offset along robot Z (vertical); +z lifts, -z lowers, metres. "
                    f"Must be within +/-{MAX_TRANSLATE_STEP_M}. Default: 0."
                ),
            },
        },
        "required": [],
    }

    @staticmethod
    def _coerce(value: object) -> float:
        return 0.0 if value is None else float(value)

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        d_x = self._coerce(arguments.get("d_x"))
        d_y = self._coerce(arguments.get("d_y"))
        d_z = self._coerce(arguments.get("d_z"))

        # Per-axis magnitude (+/-MAX_TRANSLATE_STEP_M) is enforced by input_schema,
        # but guard defensively in case a client skips validation.
        for label, value in (("d_x", d_x), ("d_y", d_y), ("d_z", d_z)):
            if abs(value) > MAX_TRANSLATE_STEP_M + 1e-9:
                return [
                    types.TextContent(
                        type="text",
                        text=(
                            f"gripper_translate did not run: invalid gripper_translate "
                            f"{label}={value:.4f}; each offset must be within "
                            f"+/-{MAX_TRANSLATE_STEP_M} m."
                        ),
                    )
                ]

        if d_x == 0.0 and d_y == 0.0 and d_z == 0.0:
            return await ctx.snap(
                "gripper_translate did not run: all offsets are zero; pass a "
                "nonzero d_x, d_y, or d_z.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        pose = await ctx.gripper_pose()
        if pose is None:
            return await ctx.snap(
                "gripper_translate did not run: no blue target gripper is selected.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        if not await ctx.ensure_translation_mode():
            return await ctx.snap(
                "gripper_translate did not run: couldn't move the gripper right now; try again."
            )

        # Inputs are robot-frame metres; the JS scene works in UI units (1 UI = 0.1 m).
        # Invert ui_to_robot (rx=ui_x/10, ry=-ui_z/10, rz=ui_y/10+offset) on the deltas:
        #   d_ui_x = d_x*10, d_ui_y = d_z*10, d_ui_z = -d_y*10. Move selectedObject's
        # position (the same object the axis-locked drag nudges) and resync the path.
        ok = await ctx.page.evaluate(
            """([dux, duy, duz]) => {
            if (typeof selectedObject === 'undefined' || !selectedObject) return null;
            selectedObject.position.x += dux;
            selectedObject.position.y += duy;
            selectedObject.position.z += duz;
            if (typeof updateCurvePath === 'function') updateCurvePath();
            return true;
        }""",
            [d_x * 10.0, d_z * 10.0, -d_y * 10.0],
        )
        if ok is None:
            return await ctx.snap(
                "gripper_translate did not run: the blue target gripper object was not found."
            )
        await asyncio.sleep(SETTLE_DELAY_S)

        new_pose = await ctx.gripper_pose()
        if new_pose is None:
            return await ctx.snap(
                "gripper_translate moved the gripper, but the active pose is unavailable after the move."
            )

        before_robot = pose["robot_position"]
        after_robot = new_pose["robot_position"]
        delta_robot = {axis: after_robot[axis] - before_robot[axis] for axis in ("x", "y", "z")}

        def fmt(value: float) -> str:
            return f"{float(value):.4f}"

        return await ctx.snap(
            f"Gripper translated by (robot, m) d_x={d_x:.4f}, d_y={d_y:.4f}, d_z={d_z:.4f}.\n"
            f"  delta (robot, m): x={fmt(delta_robot['x'])}  "
            f"y={fmt(delta_robot['y'])}  z={fmt(delta_robot['z'])}\n"
            f"{ctx.format_gripper_pose(new_pose)}"
        )


class GripperRotateTool(ToolBase):
    name = "gripper_rotate"
    frames_preamble = True
    description = (
        "Rotate the blue target gripper about one gizmo axis (x, y, or z) by a "
        "signed angle in degrees. Returns a screenshot.\n\n"
        "IMPORTANT: call `gripper_show_rotation_gizmo` before EVERY rotation and "
        "pick the axis and direction from the gizmo you see - do not guess. The "
        "x/y/z are the gripper's own gizmo axes, not the world/robot X/Y/Z (the "
        "colored arrows); they line up only at the top-down pose and move with the "
        "gripper after any tilt, so the same letter points a different way once it "
        "is tilted.\n\n"
        "The gripper has three mutually perpendicular gizmo axes: the approach axis "
        "(where the fingers reach), the opening axis (the finger line), and a "
        "third axis perpendicular to both. Each rotation spins about one, leaving "
        "it fixed and re-aiming the other two (the arrow on each ring shows the + "
        "direction):\n"
        "* x - about the third axis; both the approach and opening axes re-aim.\n"
        "* y - about the opening axis; opening stays fixed, approach re-aims.\n"
        "* z - about the approach axis; approach stays fixed, opening re-aims "
        "(twist in place).\n\n"
        "The returned text reports the resulting approach and opening axes; the "
        "rotation is in place (position unchanged). "
        "If you are unsure about the axis or direction, rotate in small amount and observe "
        "its outcome."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "axis": {
                "type": "string",
                "enum": ["x", "y", "z"],
                "description": (
                    "Gizmo rotation axis to spin about: x, y, or z. Choose it from "
                    "gripper_show_rotation_gizmo."
                ),
            },
            "angle": {
                "type": "number",
                "minimum": -90,
                "maximum": 90,
                "description": (
                    "Signed rotation angle in degrees, within "
                    "[-90, 90]. Prefer small steps (e.g. 10-45 degrees) so you can "
                    "observe the intermediate result and adjust before overshooting."
                ),
            },
        },
        "required": ["axis", "angle"],
    }

    VALID_AXES = ("x", "y", "z")

    def _angle_to_keys(self, angle: float) -> list[str]:
        """Render an angle in degrees as the keypresses the UI parses, mapping
        the sign and decimal point to their Playwright key names."""
        text = f"{angle:.2f}".rstrip("0").rstrip(".")
        if text in ("", "-", "-0"):
            text = "0"
        keys: list[str] = []
        for ch in text:
            if ch == "-":
                keys.append("Minus")
            elif ch == ".":
                keys.append("Period")
            else:
                keys.append(ch)
        return keys

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        raw_axis = arguments.get("axis")
        axis = "" if raw_axis is None else str(raw_axis).strip().lower()
        if axis not in self.VALID_AXES:
            return [
                types.TextContent(
                    type="text",
                    text=(f"Invalid gripper_rotate axis {axis!r}; " "expected 'x', 'y', or 'z'."),
                )
            ]

        raw_angle = arguments.get("angle")
        try:
            angle = float(raw_angle)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Invalid gripper_rotate angle {raw_angle!r}; "
                        "expected a number of degrees."
                    ),
                )
            ]
        angle = min(max(angle, -90.0), 90.0)

        pose = await ctx.gripper_pose()
        if pose is None:
            return await ctx.snap(
                "gripper_rotate did not run: no blue target gripper is selected.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        # UI rotation sequence: enter rotation mode, pick the axis, type the
        # signed angle, then Enter. The UI's keydown handler applies the rotation
        # and exits to translation mode on Enter whenever the angle input is
        # non-empty, and _angle_to_keys always emits at least "0", so Enter
        # should leave rotation mode. The Escape fallback below makes that an
        # invariant rather than a trusted assumption.
        sequence = ["r", axis, *self._angle_to_keys(angle), "Enter"]
        for key in sequence:
            await ctx.page.keyboard.press(key)
            await asyncio.sleep(KEY_PRESS_DELAY_S)
        await asyncio.sleep(SETTLE_DELAY_S)

        # Defensive: if the Enter above ever fails to auto-exit rotation mode,
        # force-exit with Escape so rotation-mode state can't leak into the next
        # tool's is_translation_mode() guard. Escape discards any pending
        # rotation rather than applying it, so the pose-change check below still
        # surfaces a no-op to the caller.
        if not await ctx.is_translation_mode():
            await ctx.page.keyboard.press("Escape")
            await asyncio.sleep(SETTLE_DELAY_S)
            if not await ctx.is_translation_mode():
                return await ctx.snap(
                    "gripper_rotate could not complete: the gripper didn't "
                    "settle into a movable state after the rotation; try again.\n"
                    f"{await ctx.gripper_pose_text()}"
                )

        new_pose = await ctx.gripper_pose()
        if new_pose is None:
            return await ctx.snap(
                "gripper_rotate ran, but the blue target gripper pose is "
                "unavailable after the rotation."
            )

        # Detect a no-op from the orientation vectors and position rather than
        # Euler angles (Euler couples and is multivalued off the top-down pose).
        def vec_changed(a: dict, b: dict) -> bool:
            return any(abs(a[ax] - b[ax]) > 1e-6 for ax in ("x", "y", "z"))

        changed = (
            vec_changed(pose["robot_approach"], new_pose["robot_approach"])
            or vec_changed(pose["robot_opening"], new_pose["robot_opening"])
            or vec_changed(pose["robot_position"], new_pose["robot_position"])
        )

        def fmt(value: float) -> str:
            return f"{float(value):.4f}"

        if abs(angle) > 1e-9 and not changed:
            return await ctx.snap(
                "gripper_rotate ran, but the gripper pose did not "
                "change. Check that the blue target gripper is selected.\n"
                f"{ctx.format_gripper_pose(new_pose)}"
            )

        return await ctx.snap(
            f"Rotated {fmt(angle)} deg about gizmo '{axis}' axis (right-handed). "
            "Rotation is in place (position unchanged).\n"
            f"{ctx.format_gripper_pose(new_pose)}"
        )


class GripperToggleTool(ToolBase):
    name = "gripper_toggle"
    description = (
        "Open or close the blue target gripper. Returns a screenshot.\n\n"
        "Wraps the UI 'g' key, which swaps the gripper mesh between open and "
        "closed. Pass state to drive a known result without checking the current "
        "state first:\n"
        "  state='toggle' - flip the current state (default)\n"
        "  state='open'   - ensure open  (no-op if already open)\n"
        "  state='closed' - ensure closed (no-op if already closed)\n\n"
        "The returned message and screenshot report the resulting open/closed "
        "state, read back after the swap."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["toggle", "open", "closed"],
                "description": "Target state. Default: toggle.",
            },
        },
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        state = str(arguments.get("state", "toggle")).lower()
        if state not in {"toggle", "open", "closed"}:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Invalid gripper_toggle state {state!r}; "
                        "expected 'toggle', 'open', or 'closed'."
                    ),
                )
            ]

        pose = await ctx.gripper_pose()
        if pose is None or pose.get("gripper_open") is None:
            return await ctx.snap(
                "gripper_toggle did not run: no blue target gripper is selected, or its "
                "open/closed state is unknown.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        is_open = bool(pose["gripper_open"])
        want_open = (not is_open) if state == "toggle" else (state == "open")
        if want_open == is_open:
            return await ctx.snap(
                f"Gripper already {'open' if is_open else 'closed'}; no change.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        await ctx.page.keyboard.press("g")

        # The UI swaps the gripper mesh asynchronously (it fetches + parses an OBJ),
        # so the new open/closed state is not visible immediately. Poll until it
        # flips instead of reading back once and racing the load.
        new_open = is_open
        for _ in range(TOGGLE_POLL_ATTEMPTS):
            await asyncio.sleep(TOGGLE_POLL_DELAY_S)
            new_pose = await ctx.gripper_pose()
            new_open = None if new_pose is None else new_pose.get("gripper_open")
            if new_open is None or bool(new_open) != is_open:
                break

        if new_open is not None and bool(new_open) == is_open:
            return await ctx.snap(
                f"gripper_toggle pressed 'g' but the state did not change (still "
                f"{'open' if is_open else 'closed'}).\n"
                f"{await ctx.gripper_pose_text()}"
            )

        result = "unknown" if new_open is None else ("open" if new_open else "closed")
        verb = "toggled" if state == "toggle" else "set"
        return await ctx.snap(f"Gripper {verb} -> {result}.\n" f"{await ctx.gripper_pose_text()}")


class ResetTargetGripperTool(ToolBase):
    name = "gripper_reset"
    frames_preamble = True
    description = (
        "Reset the blue target gripper back onto the real gripper's current pose - "
        "position, orientation, and open/closed state. Returns a screenshot.\n\n"
        "Use this to undo experimental edits: freely try gripper_drag, "
        "gripper_advance_or_retreat, gripper_rotate, or gripper_toggle to plan a "
        "move, see how it looks, then reset to the real pose before committing the "
        "move you actually want. This only moves the blue target gripper - it does "
        "not move the real robot and records no waypoint."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    # Maps the UI snap function's failure status to a human-readable reason.
    # ('no_selection' is already caught by the gripper_pose() guard below, so it
    # can't reach here; an unexpected status falls through to the generic message.)
    _STATUS_REASONS = {
        "no_real_pose": "the real gripper pose has not been received yet",
        "no_fn": "this UI build does not support reset (snapSelectedToRealGripper missing)",
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        if await ctx.gripper_pose() is None:
            return await ctx.snap(
                "gripper_reset did not run: no blue target gripper is selected.\n"
                f"{await ctx.gripper_pose_text()}"
            )

        # No translation-mode guard: snapSelectedToRealGripper exits rotation mode
        # itself if a gizmo is open, so reset works in any mode (and doubles as an
        # escape hatch out of a stuck rotation).
        result = await ctx.page.evaluate(
            "() => (typeof snapSelectedToRealGripper === 'function') "
            "? snapSelectedToRealGripper() : { status: 'no_fn' }"
        )
        status = (result or {}).get("status")
        if status != "ok":
            reason = self._STATUS_REASONS.get(status, f"unexpected status {status!r}")
            return await ctx.snap(
                f"gripper_reset did not run: {reason}.\n" f"{await ctx.gripper_pose_text()}"
            )

        # Position/orientation update synchronously. The open/closed swap loads an
        # OBJ asynchronously (like gripper_toggle), so when the state flips, poll
        # for the mesh to land before snapping; otherwise just let the move settle.
        if result.get("swapped"):
            want_open = bool(result.get("wantOpen"))
            for _ in range(TOGGLE_POLL_ATTEMPTS):
                await asyncio.sleep(TOGGLE_POLL_DELAY_S)
                new_pose = await ctx.gripper_pose()
                if new_pose is not None and bool(new_pose.get("gripper_open")) == want_open:
                    break
        else:
            await asyncio.sleep(SETTLE_DELAY_S)

        return await ctx.snap(
            "Reset the blue target gripper to the real gripper's current pose "
            "(position, orientation, open/closed).\n"
            f"{await ctx.gripper_pose_text()}"
        )


class GetGripperPoseTool(ToolBase):
    name = "gripper_get_pose"
    frames_preamble = True
    description = (
        "Return the current blue target gripper pose and visually show its "
        "approach axis in the screenshot. Besides giving a readout of the "
        "exact current gripper pose, the drawn approach axis makes this the "
        "best orientation check before using gripper_advance_or_retreat: it "
        "shows which way f/b will step. Returns a screenshot.\n\n"
        "Reports the center of the fingertips of the blue target gripper in "
        "robot metres, its orientation as approach and opening robot-frame unit "
        "vectors, and open/closed state. "
        "The visual axis guide is shown without moving the gripper."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        pose = await ctx.gripper_pose()
        if pose is None:
            return await ctx.snap(await ctx.gripper_pose_text())

        await ctx.page.keyboard.down("a")
        await asyncio.sleep(FRAME_DELAY_S)
        try:
            return await ctx.snap(
                "Showing current gripper approach axis visually; use this "
                "to choose gripper_advance_or_retreat f/b direction and interpret gripper_drag. "
                "No movement.\n"
                f"{ctx.format_gripper_pose(pose)}"
            )
        finally:
            await ctx.page.keyboard.up("a")


class ShowGripperRotationGizmoTool(ToolBase):
    name = "gripper_show_rotation_gizmo"
    frames_preamble = True
    description = (
        "Show the gripper's rotation gizmo (the three rotation rings) together "
        "with its approach axis in the screenshot. Always call this first to "
        "choose the gripper_rotate axis: it gives an intuitive visual "
        "understanding of the rotation axes (the x/y/z gizmo rings). The arrow on "
        "each rotation ring points to the + direction of that rotation. Returns a "
        "screenshot.\n\n"
        "Reports the center of the fingertips of the blue target gripper in "
        "robot metres, its orientation as approach and opening robot-frame unit "
        "vectors, and open/closed state."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        pose = await ctx.gripper_pose()
        if pose is None:
            return await ctx.snap(await ctx.gripper_pose_text())

        await ctx.page.keyboard.down("q")
        await asyncio.sleep(FRAME_DELAY_S)
        try:
            return await ctx.snap(
                "Showing current gripper rotation gizmo and approach axis "
                "visually; use this to choose gripper_rotate axis and direction. "
                "No movement.\n"
                f"{ctx.format_gripper_pose(pose)}"
            )
        finally:
            await ctx.page.keyboard.up("q")


# Camera tools
_DRAG_UV_SCHEMA = {
    "type": "object",
    "properties": {
        "u1": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Start fraction across the screenshot, 0.0 left to 1.0 right.",
        },
        "v1": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Start fraction down the screenshot, 0.0 top to 1.0 bottom.",
        },
        "u2": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "End fraction across the screenshot, 0.0 left to 1.0 right.",
        },
        "v2": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "End fraction down the screenshot, 0.0 top to 1.0 bottom.",
        },
        "steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Mouse move interpolation steps. Default: 25.",
        },
    },
    "required": ["u1", "v1", "u2", "v2"],
}


class CameraOrbitViaDragTool(ToolBase):
    name = "camera_orbit_via_drag"
    description = (
        "Orbit the camera around the scene by dragging between normalized "
        "screenshot coordinates. Returns a screenshot.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Drag open space in the 3-D view: open space means the canvas away from "
        "the blue target gripper mesh; a drag starting on that blue target gripper is "
        "refused with a no-op (it would move the gripper instead of orbiting, the "
        "same hit-test gripper_drag uses), so start on open space to orbit or use "
        "gripper_drag. Horizontal drag rotates around the scene; drag down -> "
        "toward top-down, drag up -> toward a side/horizon view. Sensitivity: "
        "d_azimuth ~= -0.286*W*du deg, d_elevation ~= 0.286*H*dv deg, where W/H are "
        "viewport pixels. Tip: side/horizon views can make azimuth look symmetric, "
        "so use a raised/top-down view to check alignment.\n\n"
        "Use mostly horizontal or mostly vertical drags for predictable orbit "
        "adjustments; diagonal drags couple azimuth and elevation. To shift the "
        "camera target instead of orbiting, use camera_pan_via_drag. The returned "
        "screenshot includes a start marker, end marker, crosshair overlay at the "
        "end point, an arrow path, and the updated camera pose."
    )
    input_schema = _DRAG_UV_SCHEMA

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        u1 = float(arguments["u1"])
        v1 = float(arguments["v1"])
        u2 = float(arguments["u2"])
        v2 = float(arguments["v2"])
        x1, y1, width, height = await ctx.uv_to_xy(u1, v1)
        x2, y2, _, _ = await ctx.uv_to_xy(u2, v2)
        steps = min(max(int(arguments.get("steps", 25)), 1), 100)

        if await self._start_hits_gripper(ctx, x1, y1):
            await ctx.show_cursor(x1, y1)
            await asyncio.sleep(FRAME_DELAY_S)
            return await ctx.snap(
                f"camera_orbit_via_drag did not run: u1={u1:.4f}, v1={v1:.4f} "
                "ray-hits the blue target gripper mesh, so the drag would move the "
                "gripper instead of orbiting the camera.\n"
                f"  browser pixel: x={x1}, y={y1}, viewport={width}x{height}\n"
                "  start on open space (away from the blue target gripper) to orbit, or "
                "use gripper_drag to move the gripper.\n"
                f"{await ctx.camera_pose_text()}"
            )

        await ctx.perform_drag(x1, y1, x2, y2, button="left", steps=steps)

        return await ctx.snap(
            f"Orbited u={u1:.4f}, v={v1:.4f} -> u={u2:.4f}, v={v2:.4f}\n"
            f"  browser pixels: ({x1}, {y1}) -> ({x2}, {y2}), "
            f"viewport={width}x{height}, steps={steps}\n"
            f"{await ctx.camera_pose_text()}"
        )


MAX_ORBIT_STEP_DEG = 45.0
# Elevation 0 is the exact horizon (allowed). 90 (exact top-down) is degenerate
# (azimuth collapses to 0) and not useful, so keep a small margin off the top.
# The camera can reach the horizon but never exactly 90.
MIN_ELEVATION_DEG = 0.0
MAX_ELEVATION_DEG = 85.0


class CameraOrbitViaKeyTool(ToolBase):
    name = "camera_orbit_via_key"
    description = (
        "Orbit the camera around the scene target by an angular delta, in degrees. "
        "Returns a screenshot. You specify the two orbit angles directly - no start "
        "point or open-space click needed - so it is the simplest way to reframe.\n\n"
        "Both deltas are RELATIVE to the current pose (read camera_get_pose for the "
        "absolute angles). The distance and look-at target are unchanged - this only "
        "rotates the viewpoint.\n\n"
        "Directions:\n"
        "  +d_azimuth swings the camera counterclockwise seen from above while "
        "-d_azimuth swings it clockwise. At the default view, azimuth=90 with "
        "Y=0 looking towards -X. +d_azimuth will swing the camera to +Y and "
        "gradually move towards looking at -Y at azimuth=180.\n"
        "  +d_elevation raises the camera toward a top-down view (elevation -> 90); "
        "-d_elevation lowers it toward a front or side horizontal view "
        "(elevation -> 0).\n\n"
        f"Each delta must be within +/-{MAX_ORBIT_STEP_DEG:.0f} deg to keep camera "
        "moves gradual; larger requests are rejected, so call repeatedly to orbit "
        f"further. Elevation is clamped to [{MIN_ELEVATION_DEG:.0f}, "
        f"{MAX_ELEVATION_DEG:.0f}] (the horizon is allowed; kept just off exact "
        "top-down, which is degenerate); azimuth wraps at 360. Returns the updated "
        "camera pose."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "d_azimuth": {
                "type": "number",
                "minimum": -MAX_ORBIT_STEP_DEG,
                "maximum": MAX_ORBIT_STEP_DEG,
                "description": (
                    "Change in horizontal orbit angle, degrees. + orbits the camera "
                    f"left, - orbits right. Must be within +/-{MAX_ORBIT_STEP_DEG:.0f}."
                ),
            },
            "d_elevation": {
                "type": "number",
                "minimum": -MAX_ORBIT_STEP_DEG,
                "maximum": MAX_ORBIT_STEP_DEG,
                "description": (
                    "Change in vertical angle, degrees. + raises the camera toward "
                    "top-down (90), - lowers it toward the horizon (0). Must be "
                    f"within +/-{MAX_ORBIT_STEP_DEG:.0f}."
                ),
            },
        },
        "required": ["d_azimuth", "d_elevation"],
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        # Per-step magnitude (+/-MAX_ORBIT_STEP_DEG) is enforced by input_schema,
        # which rejects out-of-range requests before we get here.
        d_az = float(arguments["d_azimuth"])
        d_el = float(arguments["d_elevation"])
        clamp_notes = []

        pose = await ctx.camera_pose()
        if pose is None:
            return [
                types.TextContent(
                    type="text",
                    text="Camera pose unavailable: browser camera object not found.",
                )
            ]

        # azimuth wraps (periodic); elevation clamps to the documented range.
        new_azimuth = (float(pose["azimuth"]) + d_az) % 360.0
        new_elevation = min(
            max(float(pose["elevation"]) + d_el, MIN_ELEVATION_DEG), MAX_ELEVATION_DEG
        )
        if new_elevation != float(pose["elevation"]) + d_el:
            clamp_notes.append(
                f"elevation clamped to {new_elevation:.4f} "
                f"(valid range [{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}])"
            )

        ok = await ctx.page.evaluate(
            """([az_deg, el_deg]) => {
            if (typeof camera === 'undefined' || !camera) return null;
            const target = (typeof cameraTarget !== 'undefined' && cameraTarget)
                ? cameraTarget
                : (window.cameraTarget || new THREE.Vector3(0, 0, 0));
            const offset = new THREE.Vector3().subVectors(camera.position, target);
            const radius = offset.length();
            const theta = az_deg * Math.PI / 180;
            const phi = (90 - el_deg) * Math.PI / 180;
            offset.setFromSphericalCoords(radius, phi, theta);
            camera.position.copy(target).add(offset);
            camera.lookAt(target);
            return true;
        }""",
            [new_azimuth, new_elevation],
        )

        if ok is None:
            return [
                types.TextContent(
                    type="text",
                    text="Camera pose unavailable: browser camera object not found.",
                )
            ]

        await asyncio.sleep(FRAME_DELAY_S)
        lines = [f"Orbited camera by d_azimuth={d_az:.4f}, d_elevation={d_el:.4f} deg"]
        lines.extend(f"  note: {note}" for note in clamp_notes)
        lines.append(await ctx.camera_pose_text())
        return await ctx.snap("\n".join(lines))


class CameraPanViaDragTool(ToolBase):
    name = "camera_pan_via_drag"
    description = (
        "Pan the camera (shift its look-at target) by dragging between normalized "
        "screenshot coordinates. Returns a screenshot.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "ZOOM-INDEPENDENT - the same drag shifts the robot-frame target by the "
        "same metre amount at any zoom/distance (the 0.001 m/px factors are "
        "constant). Let du=u2-u1, dv=v2-v1, az=current azimuth from "
        "camera_get_pose, and W/H=viewport pixels; then "
        "d_target ~= (-0.001*W*du*cos(az), -0.001*W*du*sin(az), 0.001*H*dv) m "
        "(robot). For a desired robot shift (dx,dy,dz): dv=dz/(0.001*H), "
        "du=-(dx*cos(az)+dy*sin(az))/(0.001*W). Horizontal pan only moves along "
        "camera-right, so orbit first with camera_orbit_via_drag if needed.\n\n"
        "The returned screenshot includes a start marker, end marker, crosshair "
        "overlay at the end point, an arrow path, and the updated camera pose."
    )
    input_schema = _DRAG_UV_SCHEMA

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        u1 = float(arguments["u1"])
        v1 = float(arguments["v1"])
        u2 = float(arguments["u2"])
        v2 = float(arguments["v2"])
        x1, y1, width, height = await ctx.uv_to_xy(u1, v1)
        x2, y2, _, _ = await ctx.uv_to_xy(u2, v2)
        steps = min(max(int(arguments.get("steps", 25)), 1), 100)

        await ctx.perform_drag(x1, y1, x2, y2, button="right", steps=steps)

        return await ctx.snap(
            f"Panned u={u1:.4f}, v={v1:.4f} -> u={u2:.4f}, v={v2:.4f}\n"
            f"  browser pixels: ({x1}, {y1}) -> ({x2}, {y2}), "
            f"viewport={width}x{height}, steps={steps}\n"
            f"{await ctx.camera_pose_text()}"
        )


MAX_PAN_STEP_M = 0.2


class CameraPanViaKeyTool(ToolBase):
    name = "camera_pan_via_key"
    description = (
        "Pan the camera (shift its look-at point) by a metric offset, in robot-frame "
        "metres. Returns a screenshot. You specify the shift directly - no start "
        "point or drag needed - so it is the simplest way to re-center the view "
        "without rotating.\n\n"
        "The viewing angle (azimuth/elevation) and distance are unchanged - the "
        "camera and its target slide together by the same amount, so only the "
        "framing moves.\n\n"
        "Directions (relative to the current view):\n"
        "  +d_right pans right: the look-at point slides toward screen-right, so you "
        "see more of what was off the right edge (scene content appears to slide "
        "left). -d_right pans left.\n"
        "  +d_up pans up along world-Z (robot +z): you see more above (scene content "
        "appears to slide down). -d_up pans down.\n\n"
        f"Each offset must be within +/-{MAX_PAN_STEP_M} m to keep moves gradual; "
        "larger requests are rejected, so call repeatedly to pan further. Returns "
        "the updated camera pose, whose target shows the new framing center."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "d_right": {
                "type": "number",
                "minimum": -MAX_PAN_STEP_M,
                "maximum": MAX_PAN_STEP_M,
                "description": (
                    "Horizontal pan along camera-right, robot metres. + pans right "
                    "(reveals scene to the right), - pans left. Must be within "
                    f"+/-{MAX_PAN_STEP_M}."
                ),
            },
            "d_up": {
                "type": "number",
                "minimum": -MAX_PAN_STEP_M,
                "maximum": MAX_PAN_STEP_M,
                "description": (
                    "Vertical pan along world-Z (robot +z), metres. + pans up "
                    "(reveals scene above), - pans down. Must be within "
                    f"+/-{MAX_PAN_STEP_M}."
                ),
            },
        },
        "required": ["d_right", "d_up"],
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        # Per-step magnitude (+/-MAX_PAN_STEP_M) is enforced by input_schema,
        # which rejects out-of-range requests before we get here.
        d_right = float(arguments["d_right"])
        d_up = float(arguments["d_up"])

        # Inputs are robot-frame metres; the JS camera works in UI units (1 UI = 0.1 m).
        # Pan in THREE world space: shift camera and target along the camera's own
        # horizontal right vector (screen-right) and world-up (robot +z = UI +y).
        ok = await ctx.page.evaluate(
            """([d_right_ui, d_up_ui]) => {
            if (typeof camera === 'undefined' || !camera) return null;
            const target = (typeof cameraTarget !== 'undefined' && cameraTarget)
                ? cameraTarget
                : (window.cameraTarget || null);
            if (!target) return null;
            camera.updateMatrixWorld();
            const e = camera.matrixWorld.elements;
            const right = new THREE.Vector3(e[0], e[1], e[2]);
            right.y = 0;
            if (right.lengthSq() < 1e-9) { right.set(1, 0, 0); }
            right.normalize();
            const delta = new THREE.Vector3();
            delta.addScaledVector(right, d_right_ui);
            delta.y += d_up_ui;
            camera.position.add(delta);
            target.x += delta.x;
            target.y += delta.y;
            target.z += delta.z;
            if (typeof camera.lookAt === 'function') camera.lookAt(target);
            return true;
        }""",
            [d_right * 10.0, d_up * 10.0],
        )

        if ok is None:
            return [
                types.TextContent(
                    type="text",
                    text="Camera pan unavailable: browser camera object not found.",
                )
            ]

        await asyncio.sleep(FRAME_DELAY_S)
        lines = [f"Panned camera by d_right={d_right:.4f}, d_up={d_up:.4f} m"]
        lines.append(await ctx.camera_pose_text())
        return await ctx.snap("\n".join(lines))


class CameraZoomTool(ToolBase):
    name = "camera_zoom"
    description = (
        "Zoom the camera toward normalized screenshot coordinates. Returns a screenshot.\n\n"
        "The browser zooms toward the point under the mouse. u/v should be "
        "inside the point-cloud view.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Positive steps zoom in; negative steps zoom out. Internally this sends "
        "scroll-wheel events to the browser. Each step changes the "
        "camera-to-target distance by a constant ~0.03 m, regardless of current "
        "zoom level or direction (so steps ~= |delta_distance| / 0.03). "
        "Off-center zoom centers also shift composition toward that "
        "point; use this deliberately to reframe while zooming. Returns the "
        "updated camera pose."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Zoom center fraction across the screenshot.",
            },
            "v": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Zoom center fraction down the screenshot.",
            },
            "steps": {
                "type": "integer",
                "minimum": -20,
                "maximum": 20,
                "description": "Scroll steps. Positive zooms in; negative zooms out. Default: 1.",
            },
        },
        "required": ["u", "v"],
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        u = float(arguments["u"])
        v = float(arguments["v"])
        steps = min(max(int(arguments.get("steps", 1)), -20), 20)
        x, y, width, height = await ctx.uv_to_xy(u, v)

        await ctx.page.mouse.move(x, y)
        await ctx.show_cursor(x, y)
        direction = -1 if steps > 0 else 1
        for _ in range(abs(steps)):
            await ctx.page.mouse.wheel(delta_x=0, delta_y=direction * 100)
            await asyncio.sleep(WHEEL_TICK_DELAY_S)
        await asyncio.sleep(FRAME_DELAY_S)

        if steps == 0:
            action = "no zoom"
        else:
            action = "zoom in" if steps > 0 else "zoom out"
        return await ctx.snap(
            f"Camera zoom {steps} steps ({action}); zoom center u={u:.4f}, v={v:.4f}\n"
            f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
            f"{await ctx.camera_pose_text()}"
        )


class CameraResetTool(ToolBase):
    name = "camera_reset"
    description = (
        "Restore the initial camera view. Returns a screenshot and the updated "
        "camera pose.\n\n"
        "Strategy: the default view is framed so most operations are viable from "
        "it, so prefer working there and return to it with this tool after "
        "exploring. Reach for camera_orbit_via_key / camera_pan_via_key only "
        "when the default view is insufficient - e.g. to resolve an occlusion, or "
        "to judge the gripper-to-object relationship (depth, alignment, clearance) "
        "from another angle - then reset here to re-establish a known frame."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        await ctx.page.keyboard.press("h")
        await asyncio.sleep(SETTLE_DELAY_S)
        return await ctx.snap(f"Camera reset to initial view\n{await ctx.camera_pose_text()}")


class GetCameraPoseTool(ToolBase):
    name = "camera_get_pose"
    description = (
        "Return the current camera pose without taking an action.\n\n"
        "The camera orbits a target point, reported in the robot frame (metres) "
        "like every other pose: target (robot, m) is the world point the camera "
        "looks at, and distance is the camera-to-target separation in metres. "
        "azimuth/elevation are the viewing direction onto that target: azimuth is "
        "the horizontal orbit angle in degrees; elevation is 0=horizontal (level "
        f"with the target) and 90=top-down, but orbiting keeps it within "
        f"[{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}] so it never reaches the "
        "degenerate top-down pole (where azimuth would collapse to 0). The target is "
        "reported because pan and "
        "off-center zoom shift the scene center. Use these values to reason about "
        "the current framing before adjusting it with other camera tools."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        return [types.TextContent(type="text", text=await ctx.camera_pose_text())]


class SetCameraPoseTool(ToolBase):
    name = "camera_set_pose"
    description = (
        "Set the camera pose. Returns a screenshot.\n\n"
        "Same robot-frame convention as camera_get_pose: azimuth is the horizontal "
        "orbit angle in degrees; elevation is 0=horizontal and 90=top-down (at "
        "exactly 90 azimuth has no effect and reads back as 0); distance "
        "is the camera-to-target separation in metres (omit to keep the current "
        "zoom). target_x/y/z is the world point to look at, in robot-frame metres "
        "(e.g. pass a gripper or object position straight from gripper_get_pose to "
        "frame it). By default the current target is preserved; pass all three of "
        "target_x, target_y, target_z together to move it. Returns the updated pose."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "azimuth": {
                "type": "number",
                "minimum": 0,
                "maximum": 360,
                "description": "Horizontal angle in degrees (0-360).",
            },
            "elevation": {
                "type": "number",
                "minimum": 0,
                "maximum": 90,
                "description": "Vertical angle in degrees (0=horizontal, 90=top-down).",
            },
            "distance": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 10,
                "description": "Camera-to-target distance in metres (positive). Omit to keep current.",
            },
            "target_x": {
                "type": "number",
                "description": "Optional look-at target x, robot-frame metres.",
            },
            "target_y": {
                "type": "number",
                "description": "Optional look-at target y, robot-frame metres.",
            },
            "target_z": {
                "type": "number",
                "description": "Optional look-at target z, robot-frame metres.",
            },
        },
        "required": ["azimuth", "elevation"],
    }

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        # azimuth wraps (periodic); elevation/distance clamp to the documented range.
        azimuth = float(arguments["azimuth"]) % 360.0
        elevation = min(max(float(arguments["elevation"]), 0.0), 90.0)
        distance = arguments.get("distance")
        # Inputs are robot-frame metres; the JS camera works in UI units (1 UI = 0.1 m).
        distance = None if distance is None else min(max(float(distance), 0.05), 10.0) * 10.0
        target_keys = ("target_x", "target_y", "target_z")
        target_values = [arguments.get(key) for key in target_keys]
        has_target = any(value is not None for value in target_values)
        if has_target and not all(value is not None for value in target_values):
            return [
                types.TextContent(
                    type="text",
                    text="camera_set_pose expects target_x, target_y, and target_z together.",
                )
            ]
        if not has_target:
            target = None
        else:
            robot_xyz = [float(value) for value in target_values if value is not None]
            ui_target = ctx.robot_to_ui(robot_xyz[0], robot_xyz[1], robot_xyz[2])
            target = [ui_target["x"], ui_target["y"], ui_target["z"]]

        ok = await ctx.page.evaluate(
            """([az_deg, el_deg, dist, next_target]) => {
            if (typeof camera === 'undefined' || !camera) return null;
            const target = (typeof cameraTarget !== 'undefined' && cameraTarget)
                ? cameraTarget
                : (window.cameraTarget || new THREE.Vector3(0, 0, 0));
            if (next_target !== null) {
                if (typeof target.set === 'function') {
                    target.set(next_target[0], next_target[1], next_target[2]);
                } else {
                    target.x = next_target[0];
                    target.y = next_target[1];
                    target.z = next_target[2];
                }
            }
            const offset = new THREE.Vector3().subVectors(camera.position, target);
            const radius = dist !== null ? dist : offset.length();
            const theta = az_deg * Math.PI / 180;
            const phi = (90 - el_deg) * Math.PI / 180;
            offset.setFromSphericalCoords(radius, phi, theta);
            camera.position.copy(target).add(offset);
            camera.lookAt(target);
            return true;
        }""",
            [azimuth, elevation, distance, target],
        )

        if ok is None:
            return [
                types.TextContent(
                    type="text",
                    text="Camera pose unavailable: browser camera object not found.",
                )
            ]

        await asyncio.sleep(FRAME_DELAY_S)
        return await ctx.snap(await ctx.camera_pose_text("Camera pose set"))


class ExecuteTool(ToolBase):
    name = "execute_waypoint"
    description = (
        "Execute the current gripper target and record one waypoint. Returns a screenshot.\n\n"
        "This asks the robot controller to move the real gripper to the current "
        "target pose: position, orientation, and final open/closed gripper state. "
        "Use this after meaningful target-setting operations such as "
        "gripper_teleport_via_click, gripper_drag, gripper_advance_or_retreat, gripper_rotate, "
        "or gripper_toggle. Execute a gripper_toggle on its own — do not combine it "
        "with a pose change in the same waypoint, since moving and opening/closing "
        "the gripper at once is unreliable.\n\n"
        "Execution uses the final target as the next base state for later operations. "
        "The controller interpolates between the current pose and target pose; it "
        "does not replay the exact UI trajectory used to place the target. For "
        "example, if you toggled the gripper twice before executing, it only tries "
        "to match the final open/closed state, without intermediate toggles.\n\n"
        "If this waypoint achieves the task goal, the result reports TASK SUCCEEDED "
        "and that the connection is about to close. When you see that, stop: the "
        "episode is done and no further tools should be called."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        # Snapshot exactly when execution finishes rather than after a fixed
        # sleep: the browser ticks __waypointDoneCount on the update_ui frame
        # record_sim sends once move_to() returns. Latch the count, click, then
        # wait for it to advance.
        prev = await ctx.page.evaluate("() => window.__waypointDoneCount || 0")
        await ctx.page.locator("#btn-record").click()

        note = ""
        try:
            await ctx.page.wait_for_function(
                "prev => (window.__waypointDoneCount || 0) > prev",
                arg=prev,
                timeout=EXEC_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError:
            note = (
                f"\n(Warning: no completion signal within "
                f"{EXEC_TIMEOUT_MS // 1000}s; screenshot may be mid-execution.)"
            )

        await asyncio.sleep(FRAME_DELAY_S)  # let the final frame repaint

        # record_sim refreshes the success flag right after applying the
        # waypoint and before the update_ui frame we just waited on, so by now
        # /success.json reflects this waypoint's outcome.
        info = await _fetch_sim_json("success.json")
        if info and info.get("success"):
            note += (
                "\n\n✅ TASK SUCCEEDED. The episode goal is satisfied and the UI "
                "is closing automatically. This is the end of the task: stop here "
                "and do not call any further tools — they will fail against the "
                "closed connection."
            )

        return await ctx.snap(
            "Executed current target and recorded waypoint.\n"
            f"{await ctx.gripper_pose_text()}{note}"
        )


class EndEpisodeTool(ToolBase):
    name = "end_episode"
    description = (
        "Save/end the current episode and reset or advance the task. Returns a screenshot.\n\n"
        "Call this only after the final waypoint for the episode has been executed."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(
        self,
        ctx: ToolContext,
        arguments: dict,
    ) -> list[types.ContentBlock]:
        # Single task: /success.json's live flag is the whole verdict, so read it
        # (the multi-task /eval_summary.json aggregate is degenerate here). Fetch
        # before clicking #btn-end so the flag still reflects this episode.
        info = await _fetch_sim_json("success.json")
        if info is None:
            text = "Episode ended and saved. Success unknown: /success.json unavailable."
        else:
            verdict = "SUCCESS" if info.get("success") else "FAILURE"
            task = info.get("task")
            suffix = f" ({task})" if task else ""
            text = f"Episode ended and saved. Task verdict: {verdict}{suffix}."

        # Snap BEFORE clicking #btn-end: ending the episode tears down / resets the
        # UI, so the post-click frame is meaningless. This is the only tool that
        # screenshots before its action rather than after. Capture the final
        # meaningful frame, then click to end.
        result = await ctx.snap(text)
        await ctx.page.locator("#btn-end").click()
        return result


TOOL_HANDLERS: dict[str, ToolBase] = {
    "screenshot": ScreenshotTool(),
    "hover": HoverTool(),
    "gripper_teleport_via_click": GripperTeleportViaClickTool(),
    "gripper_drag": GripperDragTool(),
    "gripper_translate": GripperTranslateTool(),
    "gripper_advance_or_retreat": GripperAdvanceOrRetreatTool(),
    "gripper_rotate": GripperRotateTool(),
    "gripper_toggle": GripperToggleTool(),
    "gripper_reset": ResetTargetGripperTool(),
    "gripper_get_pose": GetGripperPoseTool(),
    "gripper_show_rotation_gizmo": ShowGripperRotationGizmoTool(),
    # "camera_orbit_via_drag": CameraOrbitViaDragTool(),
    # "camera_pan_via_drag": CameraPanViaDragTool(),
    "camera_orbit_via_key": CameraOrbitViaKeyTool(),
    "camera_pan_via_key": CameraPanViaKeyTool(),
    "camera_zoom": CameraZoomTool(),
    "camera_reset": CameraResetTool(),
    "camera_get_pose": GetCameraPoseTool(),
    # "camera_set_pose": SetCameraPoseTool(),
    "execute_waypoint": ExecuteTool(),
    "end_episode": EndEpisodeTool(),
}

_ctx: ToolContext | None = None


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        TOOL_HANDLERS["screenshot"].as_mcp_tool(),
        TOOL_HANDLERS["hover"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_teleport_via_click"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_drag"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_translate"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_toggle"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_reset"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_advance_or_retreat"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_rotate"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_get_pose"].as_mcp_tool(),
        TOOL_HANDLERS["gripper_show_rotation_gizmo"].as_mcp_tool(),
        # TOOL_HANDLERS["camera_orbit_via_drag"].as_mcp_tool(),
        # TOOL_HANDLERS["camera_pan_via_drag"].as_mcp_tool(),
        TOOL_HANDLERS["camera_orbit_via_key"].as_mcp_tool(),
        TOOL_HANDLERS["camera_pan_via_key"].as_mcp_tool(),
        TOOL_HANDLERS["camera_zoom"].as_mcp_tool(),
        TOOL_HANDLERS["camera_reset"].as_mcp_tool(),
        TOOL_HANDLERS["camera_get_pose"].as_mcp_tool(),
        # TOOL_HANDLERS["camera_set_pose"].as_mcp_tool(),
        TOOL_HANDLERS["execute_waypoint"].as_mcp_tool(),
        TOOL_HANDLERS["end_episode"].as_mcp_tool(),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.ContentBlock]:
    if _ctx is None:
        return [
            types.TextContent(
                type="text",
                text="Browser not ready - is record_sim.py running?",
            )
        ]

    tool = TOOL_HANDLERS.get(name)
    if tool is None:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    ctx = ToolContext(
        name=name,
        browser=_ctx.browser,
        page=_ctx.page,
        ui_z_offset=_ctx.ui_z_offset,
        screenshots_dir=_ctx.screenshots_dir,
    )
    await ctx.refresh_env_info()
    _ctx.ui_z_offset = ctx.ui_z_offset
    _ctx.screenshots_dir = ctx.screenshots_dir
    return await tool(ctx, arguments or {})


async def _attach_to_record_sim_page(pw) -> tuple[Browser, Page]:
    logger.info(f"Connecting to record_sim browser at {CDP_URL} ...")
    browser = None
    for _ in range(int(10 / 0.5)):
        try:
            browser = await pw.chromium.connect_over_cdp(CDP_URL)
            break
        except Exception:
            await asyncio.sleep(0.5)

    if browser is None:
        raise RuntimeError(
            f"could not connect to record_sim's browser at {CDP_URL}. " "Start record_sim first."
        )

    page = None
    for browser_context in browser.contexts:
        for existing in browser_context.pages:
            if existing.url.startswith(UI_URL):
                page = existing
                break
        if page is not None:
            break

    if page is None:
        browser_context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await browser_context.new_page()
        await page.goto(UI_URL)
        await page.wait_for_timeout(3000)

    # CDP attachment otherwise reports the title-bar-shrunk content area.
    await page.set_viewport_size({"width": 1400, "height": 900})
    return browser, page


async def main() -> None:
    global _ctx
    async with async_playwright() as pw:
        browser, page = await _attach_to_record_sim_page(pw)
        _ctx = ToolContext(name="server", browser=browser, page=page)
        logger.info(f"Attached to record_sim browser (page url={page.url}).")

        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
