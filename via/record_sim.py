from __future__ import annotations
import glob
import json
import logging
import os
import shutil
import signal
import subprocess
import socket
import sys
import tempfile
import time
import urllib.request
import argparse
import asyncio
import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
import msgpack
import pyrallis
import numpy as np
import cv2

from via import utils
from via import serve
from via.episode_recorder import EpisodeRecorder, ActMode
from via.build_video_index import build_video_index
from via.sim_env import SimEnv, SimEnvConfig

logger = logging.getLogger(__name__)

# Consecutive ports for this session's stack, from SPHINX_BASE_PORT (default 8100) so
# multiple stacks can run at once (e.g. SPHINX_BASE_PORT=8200). mcp_server.py and run_eval.py
# read the same env var; init_webcontent rewrites the template's 8100/8101/8102 literals to
# match. record_sim owns the Chromium the UI runs in (CDP on +3); the MCP server attaches
# over CDP to drive that same page.
#   BASE_PORT + 0  UI static HTTP server (serve.http_server)
#   BASE_PORT + 1  broadcast WebSocket + JSON endpoints (/env.json, /success.json, /cam/*)
#   BASE_PORT + 2  UI command listener WebSocket
#   BASE_PORT + 3  Chromium CDP (remote debugging)
BASE_PORT = int(os.environ.get("SPHINX_BASE_PORT", "8100"))
HTTP_PORT = BASE_PORT
BROADCAST_PORT = BASE_PORT + 1
UI_LISTEN_PORT = BASE_PORT + 2
CDP_PORT = BASE_PORT + 3
STACK_PORTS = (HTTP_PORT, BROADCAST_PORT, UI_LISTEN_PORT, CDP_PORT)

UI_CONTROL_PANEL_REFERENCE = """
Browser UI control panel reference
----------------------------------
Primary actions:
  Execute Waypoint - execute current target and record a waypoint
  End Episode     - save the episode and reset

Camera:
  Left-drag empty point-cloud view - orbit
  Right-drag or middle-drag        - pan
  Scroll                           - zoom
  h                                - restore initial camera view

Position:
  Click point cloud - place gripper
  Click camera feed - place gripper from image ray
  Drag gripper      - free move
  x + drag          - constrain to robot X
  y + drag          - constrain to robot Y
  z + drag          - constrain to robot Z / vertical
  p + drag          - constrain to horizontal plane
  a                 - show gripper approach axis
  f / b             - forward/back along gripper approach axis
  c                 - snap forward to last click with small overshoot

Orientation:
  r                 - enter rotation mode
  x/y/z angle Enter - rotate by degrees
  esc               - exit rotation mode

Gripper:
  g                 - toggle gripper open/closed
"""


def _build_env(task: str, on_screen_render: bool, controller_mode: str):
    """Build the unified SimEnv for a `--task` string. Returns (env, cfg_class, ui_z_offset).

    SimEnv owns task dispatch + per-task config (robomimic presets lift/square/stack, LIBERO
    keys libero_spatial/<id>). ui_z_offset is the tabletop height (0.8 robomimic / 0.9 LIBERO)
    the UI uses to map its z=0 table to/from world z. record_sim_state=True makes demos
    replayable. on_screen_render opens the robomimic mujoco viewer (ignored by LIBERO); the
    offscreen renderer feeding the point-cloud/camera UI is always on, so headless runs pass False.
    """
    env = SimEnv(
        task,
        on_screen_render=on_screen_render,
        verbose=True,
        record_sim_state=True,
        controller_mode=controller_mode,
    )
    return env, SimEnvConfig, env.pc_config.z_offset


def _task_slug(task: str) -> str:
    """Filesystem-safe per-task subfolder name. `libero_spatial/1` → `libero_spatial_1`."""
    return task.replace("/", "_")


def _next_demo_idx(data_folder: str) -> int:
    """Next demoNNNNN index in `data_folder`: one past the highest existing demoNNNNN/, or 0."""
    next_idx = 0
    for entry in glob.glob(os.path.join(data_folder, "demo[0-9]*")):
        if not os.path.isdir(entry):
            continue
        try:
            next_idx = max(next_idx, int(os.path.basename(entry)[len("demo") :]) + 1)
        except ValueError:
            continue
    return next_idx


# Path to Playwright's downloaded Chromium, cached: the sync_playwright context only reads
# the path, never launches. record_sim launches this binary DIRECTLY for the UI browser (see
# _start_session_browser), keeping Playwright's node driver off the high-bandwidth point-cloud
# WebSocket path -- where it accumulates traffic into multi-GB swap over a session.
_CHROMIUM_PATH: str | None = None


def _chromium_executable() -> str:
    global _CHROMIUM_PATH
    if _CHROMIUM_PATH is None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            _CHROMIUM_PATH = p.chromium.executable_path
    return _CHROMIUM_PATH


class InteractiveBot:
    def __init__(
        self,
        task,
        num_point,
        stream_freq,
        data_root,
        seed,
        demo_folder: None | str,
        on_screen_render: bool,
        controller_mode: str,
        record_ui: bool = False,
    ):
        assert isinstance(task, str) and task, "must specify a single task string"

        self.num_point = num_point
        self.stream_freq = stream_freq
        self._data_root = data_root
        # Which waypoint controller move_to uses: "p" or "pi".
        self._controller_mode = controller_mode
        # Open the robomimic mujoco viewer window? Off for headless/parallel runs
        # (run_eval). LIBERO ignores it; observations/UI are unaffected either way.
        self._on_screen_render = on_screen_render
        # Optional exact demo folder (the equivalent of one demoNNNNN/). When set
        # it is used verbatim instead of an auto-numbered subfolder of
        # <data_root>/<task_slug>/; see _build.
        self._demo_folder_override = demo_folder
        self.task = task
        self.seed = seed

        # record_sim owns the Chromium the UI runs in: one visible browser with a CDP endpoint
        # for the MCP server to attach/drive. record_ui also records a session-wide webm
        # (browser-side, so it captures the red click marker etc.), finalized at teardown as
        # ui_recording.webm. Off by default; the browser is always launched. Handles below.
        self._record_ui = record_ui
        self._pw = None  # sync_playwright handle
        self._browser = None  # owned Chromium
        self._ui_context = None  # recording context
        self._ui_page = None  # the page at :8100
        # Default path: Chromium launched DIRECTLY as a subprocess (no Playwright), keeping the
        # leaky node driver off the streaming hot path. The Playwright handles above are used
        # only when record_ui is set (needed for browser-side video). Kept for teardown.
        self._chrome_proc: subprocess.Popen | None = None
        self._chrome_profile: str | None = None
        # Budget is sim steps (LIBERO horizon = cfg.max_len); one_episode auto-ends with
        # timed_out=True at the cap. Wall-clock start is for the verdict log only.
        self._task_start_time: float = 0.0
        self._timed_out: bool = False
        # Set by the SIGTERM handler (installed in one_episode) when an external
        # supervisor -- run_eval's terminate_group on a wall-clock timeout, or a
        # Ctrl-C -- asks us to stop. The episode loop watches it so it can break
        # cleanly and finalize the mp4 (moov atom) + segments + verdict instead of
        # being killed mid-stream, which would leave an unplayable video behind.
        self._stop_requested: bool = False

        # Shared success flag (0/1) updated by the main process from
        # `env._check_success()` and read by the broadcast child process to
        # serve /success.json. Created here so it's inherited by the fork.
        self._success_flag = mp.Value("i", 0)

        # Manager-backed state visible to the broadcast child so /env.json and
        # /success.json (which the MCP server polls) reflect the run. dict()
        # values must be plain JSON types.
        self._mgr = mp.Manager()
        self._shared: dict = self._mgr.dict()

        # Build the env. _build_env may take a few seconds for LIBERO (BDDL load
        # + offscreen renderer).
        self._build()

    def _build(self):
        """Build env + recorder for self.task; set self.{data_folder, demo_folder, env,
        recorder, _ui_z_offset} and the shared env-info dict for the broadcast child.

        One recorder run = one demo, all of it in self.demo_folder (recorder, video finalizer,
        verdict sidecar, MCP screenshots); only env_cfg.yaml lives one level up in data_folder,
        shared across demos. Default folder is auto-numbered demoNNNNN/ under <root>/<task_slug>/;
        --demo_folder names the exact folder instead, with env_cfg.yaml in its parent.
        """
        if self._demo_folder_override:
            self.demo_folder = os.path.abspath(self._demo_folder_override)
            self.data_folder = os.path.dirname(self.demo_folder)
            # Recover a demoNNNNN index from the folder name when it follows that
            # convention; otherwise it's just an id for the verdict/log.
            base = os.path.basename(self.demo_folder)
            tail = base[len("demo") :]
            self.demo_idx = int(tail) if base.startswith("demo") and tail.isdigit() else 0
            self.video_index_root = os.path.dirname(self.data_folder)
        else:
            self.data_folder = os.path.join(self._data_root, _task_slug(self.task))
            self.demo_idx = _next_demo_idx(self.data_folder)
            self.demo_folder = os.path.join(self.data_folder, f"demo{self.demo_idx:05d}")
            self.video_index_root = self._data_root

        os.makedirs(self.data_folder, exist_ok=True)
        self.screenshots_dir = os.path.join(self.demo_folder, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        logger.info(utils.wrap_ruler("save location"))
        logger.info(f"[save] demo {self.demo_idx:05d} -> {os.path.abspath(self.demo_folder)}/")
        logger.info(f"[save] env_cfg.yaml stays at {os.path.abspath(self.data_folder)}/")

        env, cfg_class, ui_z_offset = _build_env(
            self.task, self._on_screen_render, self._controller_mode
        )
        assert env.cfg.record_sim_state

        self.env = env
        self._cfg_class = cfg_class
        self._ui_z_offset = ui_z_offset

        # Recorder writes its camera mp4 straight into this demo's folder as
        # cameras.mp4 (video_name="cameras"). Each episode owns its folder, so no
        # auto-numbered naming or post-hoc rename is needed.
        self.recorder = EpisodeRecorder(self.demo_folder, video_name="cameras")
        self._dump_or_check_env_cfg()

        # Publish task info to the broadcast child (single task).
        self._shared["task"] = str(self.task)
        self._shared["language"] = str(getattr(self.env, "language_instruction", "") or "")
        self._shared["ui_z_offset"] = float(self._ui_z_offset)
        self._shared["sim_steps_budget"] = int(self.env.cfg.max_len)
        self._shared["sim_steps_used"] = 0
        # Absolute paths exposed to the MCP server: data_folder is the task
        # folder; demo_folder is this run's demoNNNNN/; screenshots_dir is where the
        # MCP server writes its screenshots for this demo.
        self._shared["data_folder"] = os.path.abspath(self.data_folder)
        self._shared["demo_folder"] = os.path.abspath(self.demo_folder)
        self._shared["screenshots_dir"] = os.path.abspath(self.screenshots_dir)
        self._success_flag.value = 0
        self._timed_out = False

    # ── UI screen recording ───────────────────────────────────────────────────

    def _start_session_browser(self):
        """Bring up the single Chromium the UI runs in, owned by record_sim.

        Default (record_ui off): launch Chromium DIRECTLY as a subprocess with a CDP endpoint,
        no Playwright. record_sim streams a high-bandwidth point-cloud WebSocket into the page
        every sim step; a Playwright-owned page instruments that traffic and its node driver
        balloons into multi-GB swap. Raw launch keeps Playwright out -- the MCP server still
        attaches over CDP (connect_over_cdp doesn't instrument the page that way).

        record_ui on: fall back to Playwright (the only way to get browser-side video),
        accepting the memory cost.

        Call AFTER the child servers are forked so Playwright's driver thread isn't inherited.
        Best-effort: failure leaves the sim running without an attachable/recorded browser.
        """
        # Wait for the forked webserver to actually serve HTTP_PORT before pointing a
        # browser at it: the navigation is one-shot and won't retry, so launching too
        # early leaves the window stuck on ERR_CONNECTION_REFUSED (see _wait_http_ready).
        if not self._wait_http_ready(15.0):
            logger.warning(
                f"[session_browser] UI server on :{HTTP_PORT} not ready after 15s; "
                "browser may show ERR_CONNECTION_REFUSED"
            )
        if self._record_ui:
            self._start_session_browser_playwright()
            return
        try:
            # No start_new_session: Chromium stays in record_sim's process group, so a hard
            # kill of the group (run_eval/run_learn terminate_group) cascades to the browser
            # tree. Graceful teardown uses terminate() (_stop_raw_chromium); killpg here would
            # suicide record_sim itself.
            self._chrome_profile = tempfile.mkdtemp(prefix="sphinx_chrome_")
            url = f"http://localhost:{HTTP_PORT}"
            args = [
                _chromium_executable(),
                f"--remote-debugging-port={CDP_PORT}",
                f"--user-data-dir={self._chrome_profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-extensions",
                "--disable-sync",
                "--disable-default-apps",
                # 1400x900 content at 1x scale. Window is +32px tall so the macOS
                # title bar leaves a 1400x900 content area (the MCP also pins this viewport).
                "--force-device-scale-factor=1",
                "--window-size=1400,932",
                # Suppress the "Chrome for Testing is only for automated testing"
                # infobar that a visible window would show.
                "--test-type",
                "--disable-infobars",
                # Keep the renderer live when the window is unfocused/occluded: parallel eval
                # runs many windows, only one focused; else the render loop / WS pump throttle
                # and the MCP's screenshots go stale. (Playwright sets these; a raw launch opts in.)
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=CalculateNativeWinOcclusion",
                # App mode: a chromeless window so content ~= window size (closest
                # to Playwright's chromeless viewport). Carries the URL itself.
                f"--app={url}",
            ]
            # The +32px in --window-size is the macOS title bar; other platforms have
            # different decorations, so the content area may not be exactly 1400x900.
            if sys.platform != "darwin":
                logger.warning(
                    f"--window-size=1400,932 adds 32px for the macOS title bar; on "
                    f"{sys.platform} window decorations differ, so double-check the "
                    f"screenshot is 1400x900 manually."
                )
            self._chrome_proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if self._wait_cdp_ready(15.0):
                logger.info(
                    f"[session_browser] raw Chromium ready on CDP :{CDP_PORT} (visible window)"
                )
            else:
                logger.warning(
                    f"[session_browser] CDP :{CDP_PORT} not up 15s after launch (visible window)"
                )
        except Exception as e:
            logger.warning(f"could not start raw Chromium: {e}")
            self._chrome_proc = None

    def _wait_cdp_ready(self, timeout: float) -> bool:
        """Poll the Chromium CDP endpoint until it answers, i.e. the browser is up."""
        deadline = time.time() + timeout
        url = f"http://localhost:{CDP_PORT}/json/version"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=0.5):
                    return True
            except Exception:
                time.sleep(0.25)
        return False

    def _wait_http_ready(self, timeout: float) -> bool:
        """Poll the UI static server until it answers, i.e. the page is fetchable.

        init_webcontent forks the webserver (mp.Process) but doesn't wait for it to
        bind; the browser then navigates to http://localhost:HTTP_PORT ONCE and never
        retries, so launching it before that child binds leaves the window stuck on
        ERR_CONNECTION_REFUSED. Gate the launch on this to close the startup race."""
        deadline = time.time() + timeout
        url = f"http://localhost:{HTTP_PORT}/"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=0.5):
                    return True
            except Exception:
                time.sleep(0.1)
        return False

    def _start_session_browser_playwright(self):
        """Playwright launch path, used only when record_ui is set: it's the only way
        to get browser-side video recording. Because the recorded surface IS the page
        that gets clicked, local-only UI state (the red click marker, gripper drag,
        ...) appears in the webm. The CDP endpoint is exposed too so the MCP server
        still attaches the same way. See _start_session_browser for why the default
        path avoids Playwright entirely."""
        from playwright.sync_api import sync_playwright

        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=False,
                args=[f"--remote-debugging-port={CDP_PORT}"],
            )
            ctx_kwargs = {
                "viewport": {"width": 1400, "height": 900},
                "device_scale_factor": 1,
            }
            if self._record_ui:
                rec_dir = os.path.join(self.demo_folder, ".ui_rec_tmp")
                os.makedirs(rec_dir, exist_ok=True)
                ctx_kwargs["record_video_dir"] = rec_dir
                ctx_kwargs["record_video_size"] = {"width": 1400, "height": 900}
            self._ui_context = self._browser.new_context(**ctx_kwargs)
            self._ui_page = self._ui_context.new_page()
            self._ui_page.goto(f"http://localhost:{HTTP_PORT}")
            self._ui_page.wait_for_timeout(3000)  # WebGL + WebSocket handshake
            state = "recording" if self._record_ui else "no recording"
            logger.info(f"[session_browser] ready on CDP :{CDP_PORT} ({state})")
        except Exception as e:
            logger.warning(f"could not start session browser: {e}")
            self._pw = self._browser = self._ui_context = self._ui_page = None

    def _stop_raw_chromium(self):
        """Terminate the directly-launched Chromium (default path) and drop its temp profile.
        terminate() (SIGTERM), NOT killpg -- Chromium shares record_sim's process group and
        reaps its own renderer/GPU helpers on exit. A hard group kill is the safety net if this
        never runs. Idempotent."""
        proc = self._chrome_proc
        self._chrome_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            except Exception as e:
                logger.warning(f"raw chromium cleanup: {e}")
        if self._chrome_profile:
            shutil.rmtree(self._chrome_profile, ignore_errors=True)
            self._chrome_profile = None

    def _stop_session_browser(self):
        """Close the owned browser. Raw default: terminate + drop temp profile
        (_stop_raw_chromium). record_ui path: finalize the webm next to the demo's mp4 as
        ui_recording.webm for build_video_index. Idempotent / best-effort."""
        if self._chrome_proc is not None or self._chrome_profile is not None:
            self._stop_raw_chromium()
            return
        if self._ui_context is None and self._browser is None and self._pw is None:
            return
        src_path = None
        try:
            if self._record_ui and self._ui_page is not None:
                try:
                    src_path = self._ui_page.video.path() if self._ui_page.video else None
                except Exception:
                    src_path = None
            if self._ui_context is not None:
                self._ui_context.close()  # finalizes the webm on disk
            if self._browser is not None:
                self._browser.close()
        except Exception as e:
            logger.warning(f"session browser cleanup: {e}")
        finally:
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass
            self._pw = self._browser = self._ui_context = self._ui_page = None

        if not self._record_ui:
            return

        # Resolve the produced webm (prefer video.path(); else glob the rec dir).
        rec_dir = os.path.join(self.demo_folder, ".ui_rec_tmp")
        if not (src_path and os.path.exists(src_path)):
            cands = sorted(
                glob.glob(os.path.join(rec_dir, "*.webm")), key=lambda p: os.path.getmtime(p)
            )
            src_path = cands[-1] if cands else None
        if src_path and os.path.exists(src_path):
            # Finalize into this demo's folder as ui_recording.webm.
            dst = os.path.join(self.demo_folder, "ui_recording.webm")
            try:
                os.replace(src_path, dst)
                logger.info(f"[session_browser] wrote {dst}")
            except Exception as e:
                logger.warning(f"could not finalize session webm: {e}")
        else:
            logger.warning("no session webm produced")
        # Drop the now-empty Playwright temp recording dir.
        shutil.rmtree(rec_dir, ignore_errors=True)

    def _refresh_success_flag(self) -> bool:
        # SimEnv.success is refreshed each apply_action() for both robomimic and
        # LIBERO; mirror it into the shared flag the broadcast child serves.
        ok = self.env.success
        self._success_flag.value = 1 if ok else 0
        return ok

    def _record_verdict(self):
        """Record the demo's verdict after `one_episode` returns (End Episode).

        Writes the per-demo verdict.json sidecar.
        Read off disk by run_eval.py and build_video_index.py
        """
        success = self._refresh_success_flag()
        elapsed = time.time() - self._task_start_time if self._task_start_time else 0.0
        sim_steps_used = int(getattr(self.env, "num_step", 0))
        sim_steps_budget = int(self.env.cfg.max_len)
        verdict = {
            "task": self.task,
            "language": str(getattr(self.env, "language_instruction", "") or ""),
            "success": success,
            "timed_out": bool(self._timed_out),
            "sim_steps_used": sim_steps_used,
            "sim_steps_budget": sim_steps_budget,
            "elapsed_seconds": round(elapsed, 1),
            "demo_idx": self.demo_idx,
            "demo_folder": os.path.abspath(self.demo_folder),
            "data_folder": os.path.abspath(self.data_folder),
            # Files that make up this demo (see the save layout in _build). The UI
            # recording is finalized into demo_folder at teardown (_stop_session_browser).
            "cameras_mp4": os.path.join(self.demo_folder, "cameras.mp4"),
            "episode_npz": os.path.join(self.demo_folder, "episode.npz"),
            "ui_webm": os.path.join(self.demo_folder, "ui_recording.webm"),
            "env_cfg": os.path.join(self.data_folder, "env_cfg.yaml"),
            "screenshots_dir": os.path.abspath(self.screenshots_dir),
        }
        # The verdict sidecar lives inside this demo's folder.
        sidecar_path = os.path.join(self.demo_folder, "verdict.json")
        try:
            with open(sidecar_path, "w") as f:
                # default= coerces stray numpy scalars (e.g. int64) so a single
                # non-native value can't drop the whole sidecar.
                json.dump(
                    verdict,
                    f,
                    indent=2,
                    default=lambda o: o.item() if hasattr(o, "item") else str(o),
                )
        except Exception as e:
            logger.warning(f"could not write {sidecar_path}: {e}")

        marker = "✓" if success else ("⏱" if self._timed_out else "✗")
        step_tag = f"{sim_steps_used}/{sim_steps_budget} steps"
        suffix = (
            f" (horizon hit, {step_tag}, {elapsed:.0f}s wall)"
            if self._timed_out
            else f" ({step_tag}, {elapsed:.0f}s wall)"
        )
        logger.info(f"[{marker}] {self.task} — success={success}{suffix}")

    def _dump_or_check_env_cfg(self):
        cfg_path = os.path.join(self.data_folder, "env_cfg.yaml")
        if not os.path.exists(cfg_path):
            logger.info(f"saving env cfg to {cfg_path}")
            pyrallis.dump(self.env.cfg, open(cfg_path, "w"))  # type: ignore
        else:
            pass
            # assert utils.check_cfg(
            #     self._cfg_class, cfg_path, self.env.cfg
            # ), f"Error: {self.data_folder} contains a different config than the current one"

    def reset(self):
        np.random.seed(self.seed)
        # LIBERO ignores np.random for its initial layout: SimEnv.reset() applies
        # a saved init state chosen by cfg.init_state_index, so without this the
        # seed wouldn't change a LIBERO start (every seed would reuse index 0).
        # Drive that index from the seed; SimEnv.reset() wraps it modulo the
        # number of saved states. Robomimic has no init states, so this is a
        # no-op there and the np.random.seed above governs its layout.
        self.env.cfg.init_state_index = self.seed
        self.env.reset(render=True)
        self._success_flag.value = 0

    def transform_robotframe_to_uiframe(self, waypoints):
        waypoints = np.array(waypoints) + np.array([0.0, 0.0, -self._ui_z_offset])
        transf = R.from_euler("x", -90, degrees=True)
        waypoints_ui = transf.apply(waypoints)
        rescale_amt = 10
        waypoints_ui *= rescale_amt
        return waypoints_ui

    def transform_uiframe_to_robotframe(self, waypoints):
        waypoints_rob = waypoints.copy()
        waypoints_rob /= 10.0
        transf = R.from_euler("x", 90, degrees=True)
        waypoints_rob = transf.apply(waypoints_rob)
        waypoints_rob += np.array([0.0, 0.0, self._ui_z_offset])
        return waypoints_rob

    def prepare_point_cloud(self, obs):
        points, colors = self.env.get_point_cloud(obs, crop_table=bool(self.env.cfg.crop_table))

        if len(points) > self.num_point:
            # Evenly spaced indices spanning the whole cloud (points are in
            # raster order, so plain stride+truncate would systematically drop a
            # spatial band). Negligible cost vs the gather/serialize below.
            idxs = np.linspace(0, len(points) - 1, self.num_point, dtype=int)
            points = points[idxs]
            colors = colors[idxs]

        points_ui = self.transform_robotframe_to_uiframe(points).ravel().astype(np.float32)
        colors_out = colors.ravel().astype(np.float32)
        # Return raw bytes: ~100x faster to serialize than Python float lists
        return points_ui.tobytes(), colors_out.tobytes()

    # Camera K/E must be sampled in the same process that steps the sim so
    # the wrist camera's extrinsic reflects the current EE pose. Producers
    # call this and attach the result to the queue message; the broadcast
    # loop (which runs in a separate process with a stale env fork) just
    # forwards what's in the message instead of calling self.env.get_camera_*.
    _CAM_LABEL_TO_NAMES = [
        ("agentview_image", "agentview", "agentview"),
        ("robot0_eye_in_hand_image", "wrist", "robot0_eye_in_hand"),
    ]

    def collect_cam_info(self, obs: dict) -> dict:
        info = {}
        for obs_key, label, cam_name in self._CAM_LABEL_TO_NAMES:
            if obs_key not in obs:
                continue

            K = self.env.get_camera_intrinsics(cam_name)
            E = self.env.get_camera_extrinsics(cam_name)
            info[label] = {
                "K": K.ravel().tolist(),
                "E": E.ravel().tolist(),
                "img_size": self.env.cfg.image_size,
            }
        return info

    def calculate_fingertip_offset(self, ee_euler: np.ndarray) -> np.ndarray:
        home_fingertip_offset = np.array([0, 0, -0.0])
        ee_euler_adjustment = ee_euler.copy() - np.array([-np.pi, 0, 0])
        fingertip_offset = (
            R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_fingertip_offset
        )
        return fingertip_offset

    def init_webcontent(self, obs: dict):
        ee_pos = obs["ee_pos"]
        ee_euler = obs["ee_euler"]
        fingertip_pos = ee_pos + self.calculate_fingertip_offset(ee_euler)
        fingertip_pos_ui = self.transform_robotframe_to_uiframe(
            fingertip_pos.reshape(1, 3)
        ).squeeze()
        ee_euler_ui = np.array([ee_euler[0] + np.pi, ee_euler[1], ee_euler[2]])

        fingertip_pos_code = "new THREE.Vector3(%.2f, %.2f, %.2f)" % (
            fingertip_pos_ui[0],
            fingertip_pos_ui[1],
            fingertip_pos_ui[2],
        )
        ee_euler_code = "new THREE.Euler(%.2f, %.2f, %.2f)" % (
            ee_euler_ui[0],
            ee_euler_ui[1],
            ee_euler_ui[2],
        )

        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "UI", "template_index.html"
        )
        with open(template_path) as f:
            html_content = f.read()

        html_content = html_content % (
            1,
            self.num_point,
            fingertip_pos_code,
            ee_euler_code,
            "'franka'",  # gripper name
            f"{self._ui_z_offset:.4f}",  # UI_Z_OFFSET (matches tabletop per env)
        )

        # The template hardcodes the default stack ports (8100/8101/8102); rewrite
        # them to this run's actual ports so the served page talks to the right
        # servers when SPHINX_BASE_PORT shifts the stack. No-op at the default base.
        html_content = (
            html_content.replace("localhost:8100", f"localhost:{HTTP_PORT}")
            .replace("localhost:8101", f"localhost:{BROADCAST_PORT}")
            .replace("localhost:8102", f"localhost:{UI_LISTEN_PORT}")
        )

        # The generated page lives in this run's demo folder (archived with the
        # rest of the demo). serve.http_server serves that folder on :8100 and
        # falls back to interactive_utils/ for the shared franka .obj meshes.
        index_path = os.path.join(self.demo_folder, "index.html")
        with open(index_path, "w") as f:
            f.write(html_content)

        # Start Server
        # this starts the localhost which we visit on browser
        # hold it in self to prevent destruction
        self.webserver_proc = mp.Process(
            target=serve.http_server, args=(self.demo_folder, HTTP_PORT)
        )
        self.webserver_proc.start()

    def init_ui_listen_process(self):
        ui_queue = mp.Queue(maxsize=1)

        async def listen_ui(websocket):
            async for message in websocket:
                message = json.loads(message)
                if not len(message):
                    continue

                if not ui_queue.empty():
                    logger.warning(
                        "the ui_queue is not empty, dropping new UI command. "
                        "This should not happen"
                    )
                    continue

                data = message[-1]  # Retrieve the last waypoint in the UI
                click_ui_pos = [
                    data["click"]["x"],
                    data["click"]["y"],
                    data["click"]["z"],
                ]
                fingertip_ui_pos = [
                    data["position"]["x"],
                    data["position"]["y"],
                    data["position"]["z"],
                ]
                rotation = [
                    data["orientation"]["x"],
                    data["orientation"]["y"],
                    data["orientation"]["z"],
                ]

                info = {
                    "click_ui_pos": click_ui_pos,
                    "fingertip_ui_pos": fingertip_ui_pos,
                    "rotation": rotation,
                    "gripper_open": float(
                        data.get("url") == f"http://localhost:{HTTP_PORT}/franka.obj"
                    ),
                    "done": data["done"],
                }
                logger.info(f"[ui_cmd] gripper_open: {info['gripper_open']}, done: {info['done']}")
                # block=True should take no extra time as the queue should be empty
                ui_queue.put(info, block=True)

        async def listen_ui_main():
            async with websockets.serve(listen_ui, "localhost", UI_LISTEN_PORT):
                await asyncio.Future()

        # Build the coroutine inside the child (fork) so the parent never holds
        # an un-awaited coroutine object (avoids "coroutine was never awaited").
        self.listen_process = mp.Process(target=lambda: asyncio.run(listen_ui_main()))
        self.listen_process.start()

        return ui_queue

    def init_ui_update_process(self):
        ui_update_queue = mp.Queue(maxsize=1)
        # Shared state read by the broadcast child below. `_shared` carries the
        # task / language / ui_z_offset served over /env.json and /success.json.
        success_flag = self._success_flag
        shared = self._shared

        async def send_data_to_web_main():
            connected = set()
            pcl_cache = [None, None]  # [points_bytes, colors_bytes]
            # Latest sim-side JPEG per camera, served over HTTP so external
            # consumers (e.g. the MCP server's point_cam → Gemini path) can
            # read the unrescaled, single-pass-encoded image directly from
            # the sim instead of screenshotting the rendered <img>.
            latest_cam_jpegs: dict[str, bytes] = {}

            async def on_connect(websocket):
                connected.add(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    connected.discard(websocket)

            def _json_response(payload: dict) -> Response:
                body = json.dumps(payload).encode()
                return Response(
                    200,
                    "OK",
                    Headers(
                        [
                            ("Content-Type", "application/json"),
                            ("Content-Length", str(len(body))),
                            ("Cache-Control", "no-store"),
                        ]
                    ),
                    body,
                )

            # Cache the last successful read from `shared` so that requests
            # arriving after the Manager process has been torn down (graceful
            # exit) still get a sensible payload instead of crashing the
            # websocket handshake with BrokenPipeError.
            shared_cache: dict = {}

            def safe_shared_get(key, default):
                try:
                    val = shared.get(key, default)
                except (BrokenPipeError, EOFError, ConnectionResetError, FileNotFoundError):
                    return shared_cache.get(key, default)
                shared_cache[key] = val
                return val

            async def process_request(connection, request):
                path = request.path
                if path == "/env.json":
                    return _json_response(
                        {
                            "task": str(safe_shared_get("task", "")),
                            "language": str(safe_shared_get("language", "")),
                            "ui_z_offset": float(safe_shared_get("ui_z_offset", 0.8)),
                            "task_idx": 0,
                            "n_tasks": 1,
                            "data_folder": str(safe_shared_get("data_folder", "")),
                            "demo_folder": str(safe_shared_get("demo_folder", "")),
                            "screenshots_dir": str(safe_shared_get("screenshots_dir", "")),
                            "sim_steps_budget": int(safe_shared_get("sim_steps_budget", 0)),
                            "sim_steps_used": int(safe_shared_get("sim_steps_used", 0)),
                        }
                    )
                if path == "/success.json":
                    return _json_response(
                        {
                            "success": bool(success_flag.value),
                            "task": str(safe_shared_get("task", "")),
                            "language": str(safe_shared_get("language", "")),
                        }
                    )
                if path.startswith("/cam/") and path.endswith(".jpg"):
                    name = path[len("/cam/") : -len(".jpg")]
                    body = latest_cam_jpegs.get(name)
                    if body is None:
                        return Response(
                            503,
                            "Service Unavailable",
                            Headers([("Content-Length", "0")]),
                            b"",
                        )
                    return Response(
                        200,
                        "OK",
                        Headers(
                            [
                                ("Content-Type", "image/jpeg"),
                                ("Content-Length", str(len(body))),
                                ("Cache-Control", "no-store"),
                            ]
                        ),
                        body,
                    )
                return None  # fall through to the normal WebSocket handshake

            async def broadcast_loop():
                loop = asyncio.get_event_loop()
                while True:
                    to_send = await loop.run_in_executor(None, ui_update_queue.get)
                    obs = to_send["obs"]
                    update_ui = to_send["update_ui"]
                    skip_pcl = to_send.get("skip_pcl", False)

                    if skip_pcl and pcl_cache[0] is not None:
                        points_bytes, colors_bytes = pcl_cache
                    else:
                        points_bytes, colors_bytes = self.prepare_point_cloud(obs)
                        pcl_cache[0], pcl_cache[1] = points_bytes, colors_bytes

                    ee_pos = obs["ee_pos"]
                    ee_euler = obs["ee_euler"]
                    gripper_open = int(obs["gripper_open"].item() > 0.9)

                    fingertip_pos = ee_pos + self.calculate_fingertip_offset(ee_euler)
                    fingertip_pos_ui = (
                        self.transform_robotframe_to_uiframe(fingertip_pos.reshape(1, 3))
                        .squeeze()
                        .tolist()
                    )
                    ee_euler_ui = [ee_euler[0] + np.pi, ee_euler[1], ee_euler[2]]

                    cam_images = {}
                    # cam_info (K/E) is computed by the main process alongside
                    # the obs that produced these images, so the wrist
                    # camera's extrinsic matches the EE pose at obs time.
                    # We forward it as-is — calling self.env.get_camera_* here
                    # would read this child process's stale env fork.
                    cam_info = to_send.get("cam_info", {})
                    for obs_key, label, _ in self._CAM_LABEL_TO_NAMES:
                        if obs_key in obs:
                            _, buf = cv2.imencode(
                                ".jpg", obs[obs_key][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75]
                            )
                            jpeg_bytes = buf.tobytes()
                            cam_images[label] = jpeg_bytes
                            latest_cam_jpegs[label] = jpeg_bytes

                    data = {
                        "positions": points_bytes,
                        "colors": colors_bytes,
                        "fingertip_pos_ui": fingertip_pos_ui,
                        "ee_euler_ui": ee_euler_ui,
                        "gripper_action": [1 - gripper_open],
                        "update_ui": update_ui,
                        "cam_images": cam_images,
                        "cam_info": cam_info,
                    }

                    msg = msgpack.packb(data)
                    dead = set()
                    for ws in list(connected):
                        try:
                            await ws.send(msg)
                        except Exception:
                            dead.add(ws)
                    connected.difference_update(dead)

            asyncio.create_task(broadcast_loop())
            async with websockets.serve(
                on_connect, "localhost", BROADCAST_PORT, process_request=process_request
            ):
                await asyncio.Future()

        # Build the coroutine inside the child (fork) so the parent never holds
        # an un-awaited coroutine object (avoids "coroutine was never awaited").
        self.send_process = mp.Process(target=lambda: asyncio.run(send_data_to_web_main()))
        self.send_process.start()
        return ui_update_queue

    def teardown(self):
        """Stop the child processes started for the live UI so the program can
        exit. webserver_proc (serve_forever), listen_process / send_process
        (await asyncio.Future()) and the Manager all run forever and are
        non-daemon, so without an explicit terminate() the multiprocessing
        atexit join blocks the interpreter from ever exiting. Best-effort and
        idempotent."""
        # Close the owned UI browser, finalizing the session webm.
        self._stop_session_browser()
        for name in ("send_process", "listen_process", "webserver_proc"):
            proc = getattr(self, name, None)
            if proc is None:
                continue
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"could not stop {name}: {e}")
        mgr = getattr(self, "_mgr", None)
        if mgr is not None:
            try:
                mgr.shutdown()
            except Exception:
                pass

    def apply_waypoint_mode(self, ui_cmd: dict, recorder: EpisodeRecorder, send_queue: mp.Queue):
        click_pos = np.array(ui_cmd["click_ui_pos"]).astype(np.float64)
        click_pos = self.transform_uiframe_to_robotframe(click_pos.reshape(1, 3)).squeeze()

        fingertip_pos_cmd = np.array(ui_cmd["fingertip_ui_pos"])
        ee_euler_cmd = np.array(
            [
                ui_cmd["rotation"][0] - np.pi,
                -ui_cmd["rotation"][2],
                ui_cmd["rotation"][1],
            ]
        )
        ee_pos_cmd = self.transform_uiframe_to_robotframe(fingertip_pos_cmd.reshape(1, 3)).squeeze()
        ee_pos_cmd -= self.calculate_fingertip_offset(ee_euler_cmd)
        gripper_open_cmd = ui_cmd["gripper_open"]

        obs = self.env.observe()
        action = np.concatenate([ee_pos_cmd, ee_euler_cmd, [gripper_open_cmd]], dtype=np.float32)
        recorder.record(ActMode.Waypoint, obs, action, click_pos=click_pos)

        def stream_fn(obs):
            try:
                send_queue.put_nowait(
                    {
                        "obs": obs,
                        "update_ui": False,
                        "skip_pcl": True,
                        "cam_info": self.collect_cam_info(obs),
                    }
                )
            except Exception:
                pass

        self.env.move_to(
            ee_pos_cmd,
            ee_euler_cmd,
            gripper_open_cmd,
            recorder=recorder,
            render=True,
            stream_fn=stream_fn,
        )

    def record_demo(self):
        logger.info(utils.wrap_ruler(f"record session — task: {self.task}"))
        self.reset()
        obs = self.env.observe()
        self.init_webcontent(obs)
        ui_queue = self.init_ui_listen_process()
        send_queue = self.init_ui_update_process()

        # Own the UI browser for the whole session (after the servers above are
        # forked, so Playwright's driver thread isn't inherited). The MCP server
        # attaches to this browser over CDP; recording is session-wide.
        self._start_session_browser()

        logger.info("reset done, sending first frame")
        send_queue.put(
            {
                "obs": obs,
                "update_ui": True,
                "cam_info": self.collect_cam_info(obs),
            }
        )

        logger.info(f"episode start! task: {self.task}")
        self.one_episode(ui_queue, send_queue, self.recorder)

        # Single episode: record the verdict and exit. The finally in main()
        # tears down the servers + finalizes the webm. On a SIGTERM stop we skip the
        # verdict on purpose -- one_episode already finalized the video, and run_eval
        # (which knows whether this was a real wall-clock timeout or a resumable
        # interrupt) owns the verdict for that case; see one_episode's stop branch.
        if self._stop_requested:
            logger.info("stopped by signal — video finalized; leaving verdict to supervisor")
        else:
            self._record_verdict()
            logger.info(
                "demo complete — finalizing video, stopping UI servers, rebuilding video.html"
            )

    def one_episode(self, ui_queue: mp.Queue, send_queue: mp.Queue, recorder: EpisodeRecorder):
        stopwatch = utils.Stopwatch()

        # Catch SIGTERM (run_eval's terminate_group sends it, with a ~20s grace
        # before SIGKILL) so a wall-clock-timeout kill breaks the loop cleanly and
        # still finalizes the video, rather than dropping an unplayable mp4. Installed
        # HERE, after every mp.Process child (Manager, webserver, listen/send) has been
        # forked, so the handler isn't inherited by those children -- they keep the
        # default disposition and die promptly on the same group SIGTERM.
        def _on_sigterm(signum, frame):
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _on_sigterm)

        # Wall-clock start, for the verdict log only.
        self._task_start_time = time.time()
        budget = int(self.env.cfg.max_len)
        self._shared["sim_steps_budget"] = budget
        self._shared["sim_steps_used"] = 0

        while True:
            # An external stop (SIGTERM) -- typically run_eval killing the seed after
            # the agent blew the wall-clock cap, or a Ctrl-C. Break so end_episode(
            # save=True) below finalizes the mp4 (moov atom) + segments instead of the
            # process being killed mid-write. We deliberately do NOT write a verdict on
            # this path (see record_demo): run_eval is the authority on whether the stop
            # was a gradable timeout or a resumable interrupt, so writing one here would
            # mislabel a Ctrl-C'd seed as a timeout failure and skip it on rerun.
            if self._stop_requested:
                logger.info(f"[stop] task {self.task} received SIGTERM — finalizing recording")
                break

            # Publish current sim-step usage so MCP / clients see it live.
            used = int(getattr(self.env, "num_step", 0))
            self._shared["sim_steps_used"] = used

            # Per-task sim-step budget (matches LIBERO benchmark horizon).
            # `self.env.terminal` flips True at the same boundary; either
            # condition force-ends the demo.
            if used >= budget or self.env.terminal:
                if used >= budget:
                    logger.info(
                        f"[budget] task {self.task} hit sim-step horizon ({used}/{budget}) — end"
                    )
                    self._timed_out = True
                else:
                    logger.info(
                        f"[terminal] task {self.task} terminated with success: {self.env.success}"
                    )
                self._refresh_success_flag()
                break

            # waypoint mode
            if not ui_queue.empty():
                ui_cmd = ui_queue.get()
                if ui_cmd["done"]:
                    self._refresh_success_flag()
                    break

                self.apply_waypoint_mode(ui_cmd, recorder, send_queue)
                self._refresh_success_flag()
                # we can afford to call observe again here in waypoint mode
                fresh_obs = self.env.observe()
                send_queue.put(
                    {
                        "obs": fresh_obs,
                        "update_ui": True,
                        "cam_info": self.collect_cam_info(fresh_obs),
                    }
                )
                continue

            with utils.FreqGuard(self.stream_freq), stopwatch.time("stream"):
                obs = self.env.observe()
                send_queue.put(
                    {
                        "obs": obs,
                        "update_ui": False,
                        "cam_info": self.collect_cam_info(obs),
                    }
                )

        recorder.end_episode(save=True)


def _prime_fork_dns():
    """Work around a macOS fork() segfault in the forked server children.

    The UI servers run as forked processes (mp start method 'fork', required so
    each child inherits the live sim env, which isn't picklable for 'spawn'). On
    macOS a forked child that calls getaddrinfo — which both
    HTTPServer.server_bind (via socket.getfqdn) and websockets.serve("localhost")
    do — segfaults inside libsystem_info: the lazy one-time init of its
    address-sorting / os_log machinery cannot run safely after fork(), so the
    child dies with SIGSEGV ("Python quit unexpectedly") and the port never binds.

    Running that one-time init HERE, in the parent before any fork, lets the
    children inherit the completed state and skip the crashing path. No-op off mac.
    """
    if sys.platform != "darwin":
        return
    try:
        socket.getaddrinfo("localhost", None, proto=socket.IPPROTO_TCP)
        socket.getfqdn("")  # the HTTPServer.server_bind path
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sim demo recorder. --task accepts robomimic presets "
            "(lift, square, stack) or a LIBERO key (libero_spatial/<id>). "
            "Use --list-tasks to enumerate."
        ),
    )
    parser.add_argument("--num_point", type=int, default=100_000)
    parser.add_argument("--stream_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1, help="Seed for env.reset()'s initial state.")
    parser.add_argument("--task", type=str, default="square", help="lift, stack, libero_goal/0...")
    parser.add_argument("--data_root", type=str, default="data/dev1", help="<root>/<task>/demoNNN")
    parser.add_argument(
        "--demo_folder", type=str, default=None, help="overrides data_root's auto-numbered demoNNN"
    )
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--controller_mode", type=str, default="pi", choices=["p", "pi"])
    parser.add_argument("--record_ui", action="store_true", help="Record the session UI to webm")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    utils.setup_logging()

    if args.list_tasks:
        from via.sim_env import list_supported_tasks

        list_supported_tasks()
        return

    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    mp.set_start_method("fork")  # compatibility on mac
    _prime_fork_dns()  # macOS: stop forked UI servers segfaulting in getaddrinfo

    # Free every port this stack binds so a previous crashed/killed run can't
    # leave any squatting (a clean exit cleans up its own children, but a hard
    # kill can leak a bound port). Scoped to this stack's ports, so a second
    # instance on a different SPHINX_BASE_PORT won't disturb the first.
    for port in STACK_PORTS:
        utils.kill_process_on_port(port)

    robot = InteractiveBot(
        args.task,
        args.num_point,
        args.stream_freq,
        args.data_root,
        args.seed,
        demo_folder=args.demo_folder,
        on_screen_render=bool(args.render),
        controller_mode=args.controller_mode,
        record_ui=args.record_ui,
    )

    try:
        robot.record_demo()
    finally:
        # record_demo's child processes (web server + the two websocket
        # processes + Manager) run forever and are non-daemon; without this
        # the multiprocessing atexit join hangs the interpreter on exit.
        robot.teardown()
        # Best-effort refresh of <data_root>/videos.html. Guarded so a failure
        # here can't mask whatever exception drove us into this finally.
        try:
            index_path = build_video_index(robot.video_index_root)
            logger.info(f"[build_video_index] wrote {index_path}")
        except Exception as e:
            logger.warning(f"could not update video index: {e}")


if __name__ == "__main__":
    main()
