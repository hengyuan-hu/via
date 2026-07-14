"""Shared, harness-agnostic infrastructure for the eval/learn drivers.

Home for the things every harness and driver needs and that must NOT depend on
run_eval (so the harness modules can import them without a cycle): repo paths,
the logging helpers, the generic subprocess terminate(), and the registry of
live agent processes that main()'s Ctrl-C path tears down.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The MCP config handed to `claude --mcp-config` (and translated to codex `-c`
# overrides); also the single source of truth for which server(s) to allowlist
# (see claude_harness.allowed_tools()).
MCP_CONFIG_PATH = os.path.join(REPO_ROOT, ".mcp.json")

# The robot guide the agent reads before operating (build_prompt points it here;
# the codex harness ships the same content as AGENTS.md). Snapshotted into each
# run's task folder so results stay interpretable against the exact guide in
# force (see run_eval.backup_guide_file).
CLAUDE_MD_PATH = os.path.join(REPO_ROOT, "CLAUDE.md")


def log(msg: str) -> None:
    """Info-level log for the drivers. stacklevel=2 makes file:line point at the
    caller (run_eval / run_learn / a harness) rather than at this wrapper. logging
    is itself thread-safe, so concurrent seeds no longer need an explicit lock."""
    logger.info(msg, stacklevel=2)


def warn(msg: str) -> None:
    """Warning-level companion to log() (was the old `[warn] ...` prints)."""
    logger.warning(msg, stacklevel=2)


def get_json_file(path: str):
    """Parse a JSON file, returning None on ANY failure (missing, unreadable,
    malformed). The single JSON-file-read policy for the drivers and harnesses:
    callers that want a fallback write `get_json_file(p) or {}`, which also
    coerces falsy-but-valid JSON (null, [], "") to the fallback."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# Agent (claude/codex) procs currently running, so a Ctrl-C in main() can tear
# them all down promptly instead of waiting for each worker thread to unwind to
# its own finally. A proc is added right after launch and dropped once it's gone.
# Guarded because seeds run in a thread pool. These procs share the driver's group
# (killpg would suicide the driver), so main() reaps them with plain terminate();
# sims lead their own session and get terminate_group in run_eval.
_active_agents: set[subprocess.Popen] = set()
_agents_lock = threading.Lock()


def register_agent(proc: subprocess.Popen) -> None:
    with _agents_lock:
        _active_agents.add(proc)


def unregister_agent(proc: subprocess.Popen) -> None:
    with _agents_lock:
        _active_agents.discard(proc)


def active_agents() -> list[subprocess.Popen]:
    """Snapshot of the live agent procs, for main()'s teardown."""
    with _agents_lock:
        return list(_active_agents)
