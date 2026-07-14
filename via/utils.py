"""Shared utilities for the simulation and evaluation stack.

This module contains point-cloud, process, timing, and logging helpers used by
the active `via` package:

  deproject            <- interactive_scripts/vision_utils/pc_utils.py  (sim_env.py)
  wrap_ruler           <- common_utils/helper.py                        (record_sim.py)
  kill_process_on_port <- common_utils/helper.py                        (record_sim.py)
  Stopwatch            <- common_utils/stopwatch.py                     (record_sim.py)
  FreqGuard            <- common_utils/freq_guard.py                    (record_sim.py)

  setup_logging        <- shared logging config for every entry point
"""

import logging
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import tabulate

logger = logging.getLogger(__name__)


# --- logging ---------------------------------------------------------------

# Format:  [HH:MM:SS LEVEL file:line] message
#   - time to the second; level and call-site file:line in one [ ] group.
# mcp_server.py duplicates this string inline (it's a bare script in its own
# venv and can't import via); keep the two in sync.
LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Third-party loggers that chatter at INFO; pinned to WARNING so they don't drown
# out the via logs once the root level is INFO. Extend as needed.
_QUIET_LOGGERS = ("asyncio", "websockets", "httpx", "httpcore", "matplotlib", "PIL")


def setup_logging(level: int = logging.INFO) -> None:
    """Install a single stderr handler with our format on the root logger.

    Every entry point (run_eval, run_learn, record_sim, serve, sim_env, ...)
    calls this once at startup; library modules just do
    `logger = logging.getLogger(__name__)` and inherit this config.

    Output goes to stderr so run_eval/run_learn keep capturing a child's logs:
    they redirect a child's stdout+stderr into a logs/ file
    (Popen(..., stdout=f, stderr=subprocess.STDOUT)), so stderr lands there too.

    Idempotent: re-running replaces the handler rather than stacking another.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# --- from interactive_scripts/vision_utils/pc_utils.py ---------------------


def deproject(depth_image, K, tf=np.eye(4), base_units=-3):
    # Convert depth image to meters
    depth_image_m = depth_image * (10**base_units)

    h, w = depth_image.shape
    i, j = np.indices((h, w))

    # Create homogeneous coordinates for pixels
    pixels_homog = np.stack([j.ravel(), i.ravel(), np.ones_like(i).ravel()], axis=0)

    # Compute the 3D points in the camera frame
    depth_arr = depth_image_m.ravel()
    points_3d = np.linalg.inv(K) @ pixels_homog * depth_arr

    # Transform the points to the target frame
    points_3d_homog = np.vstack([points_3d, np.ones(points_3d.shape[1])])
    points_3d_transf = (tf @ points_3d_homog).T[:, :3]

    return points_3d_transf


# --- from common_utils/helper.py -------------------------------------------


def kill_process_on_port(port):
    try:
        # Find process using the port
        result = subprocess.check_output(["lsof", "-i", f":{port}"])
        lines = result.splitlines()
        for line in lines[1:]:  # Skip the header line
            columns = line.split()
            pid = int(columns[1])
            logger.info("[*] Found existing server process")
            logger.info(f"[*] Killing process with PID: {pid} on port {port}")
            os.kill(pid, signal.SIGKILL)
    except subprocess.CalledProcessError:
        logger.info("[*] Starting server")
    except Exception as e:
        logger.warning(f"[*] Server error occurred: {e}")


def wrap_ruler(text: str, max_len=40):
    text_len = len(text)
    if text_len > max_len:
        return text

    left_len = (max_len - text_len) // 2
    right_len = max_len - text_len - left_len
    return ("=" * left_len) + text + ("=" * right_len)


class FreqGuard:
    def __init__(self, control_hz, slack_time=0.001):
        self.control_hz = control_hz
        self.slack_time = slack_time

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t_curr = time.time()
        t_end = self.t_start + 1 / self.control_hz
        t_wait = t_end - t_curr
        if t_wait > 0:
            t_sleep = t_wait - self.slack_time
            if t_sleep > 0:
                time.sleep(t_sleep)
            while time.time() < t_end:
                pass


# --- from common_utils/stopwatch.py ----------------------------------------


class Stopwatch:
    """stop watch in MS"""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.time()

        self.init_time = time.time()
        self.records_for_freq = {}

    @property
    def total_time(self):
        return time.time() - self.init_time

    @property
    def elapsed_time_since_reset(self):
        return time.time() - self.reset_time

    def count(self, key):
        return len(self.times[key])

    def reset(self):
        self.times = defaultdict(list)
        self.reset_time = time.time()

    def record_for_freq(self, key):
        if key not in self.records_for_freq:
            self.records_for_freq[key] = {"time": time.time(), "count": 0}

        delta_time = time.time() - self.records_for_freq[key]["time"]
        if delta_time > 1:
            freq = self.records_for_freq[key]["count"] / delta_time
            logger.info(f"Freq of {key}: duration: {delta_time:.2f}, freq: {freq:.2f}")
            self.records_for_freq[key] = {"time": time.time(), "count": 0}

        self.records_for_freq[key]["count"] += 1

    @contextmanager
    def time(self, key):
        t = time.time()
        yield

        self.times[key].append(1000 * (time.time() - t))  # record in ms

    def summary(self, reset=True):
        headers = ["name", "num", "t/call (ms)", "%"]
        total = 0
        times = {}
        for k, v in self.times.items():
            if len(v) == 0:
                continue
            sum_t = np.sum(v)
            mean_t = sum_t / len(v)
            times[k] = (len(v), sum_t, mean_t)
            total += sum_t

        rows = []
        for k, (num, sum_t, mean_t) in times.items():
            rows.append([k, f"{num:.1f}", f"{mean_t:.1f}", f"{100 * sum_t / total:.1f}"])

        rows.append(["total(s)", 1, f"{total/1000:.1f}", f"{100 * total / total:.1f}"])
        table = tabulate.tabulate(rows, headers=headers, tablefmt="orgtbl")
        logger.info("Timer Info:\n%s", table)

        if reset:
            self.reset()
