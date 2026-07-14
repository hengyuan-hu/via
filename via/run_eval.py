"""Outer-loop eval driver: run ONE task across N seeds, all in parallel.

Each seed is an isolated stack on its own SPHINX_BASE_PORT (= exp.base_port +
i*stride), so seeds run at once without interfering. Per seed this driver:
  1. launches a fresh `record_sim.py` for that stack and waits for it to come up,
  2. launches a fresh headless `claude -p` whose MCP server attaches to that
     seed's browser (via SPHINX_BASE_PORT) and drives the gripper to completion,
  3. waits for record_sim to save the episode (verdict.json + video), then
     scrapes the verdict.

Seeds run in a thread pool, up to --max-parallel at once (default: all). Pass
several experiments and they share ONE --max-parallel budget across all their
seeds (ports are allocated globally so concurrent experiments never collide).
Reruns are incremental: a seed that already saved a verdict.json is reused, so a
rerun only fills in the seeds that didn't finish. --print-results skips running
and just reports the verdicts on disk.

Experiments are dataclass configs in EXPERIMENTS. Run from the repo root:
    python -m via.run_eval                                   # list experiments
    python -m via.run_eval stack-min-opus
    python -m via.run_eval stack-min-opus lg8-min-opus -j 4  # both, 4 seeds at a time
    python -m via.run_eval stack-min-opus --print-results    # just show verdicts
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

from via.agent_common import (
    CLAUDE_MD_PATH,
    MCP_CONFIG_PATH,
    REPO_ROOT,
    active_agents,
    get_json_file,
    log,
    terminate,
    warn,
)

# Importing the harness modules self-registers them (see harness.get_harness), so
# get_harness(exp.harness) resolves "claude"/"codex". Adding a new backend is one
# new module implementing Harness + self-registering; import it here to load it.
from via import claude_harness  # noqa: F401  (import for self-registration)
from via import codex_harness  # noqa: F401  (import for self-registration)
from via.harness import SeedContext, build_prompt, get_harness
from via.utils import setup_logging

# Shared defaults for simulation streams and startup timing.
NUM_POINT = 100_000
STREAM_FREQ = 1
SETTLE = 3.0  # extra seconds after readiness before launching the agent
CODEX_MODEL = "gpt-5.5"  # codex-harness model for the codex groups

# Task goals, written once and referenced by the experiments below.
LANG = {
    "stack": "Pick up the red cube and place it onto the green cube.",
    "lg0": "Open the middle (2nd from the top) drawer of the cabinet.",
    "lg7": "Turn on the stove.",
    "lg8": "Put the bowl on the plate.",
    "t_block": (
        "Build a structure exactly two blocks tall: the blue block on the "
        "bottom, and both green blocks resting directly on top of it, side by "
        "side at the same height. Neither green block may sit on the other or "
        "on the table."
    ),
    "rainbow": "Rearrange the blocks so that it forms a rainbow. Be creative.",
}


def task_slug(task: str) -> str:
    """Filesystem-safe per-task folder name. `libero_spatial/0` -> `libero_spatial_0`."""
    return task.replace("/", "_")


@dataclass(frozen=True)
class Experiment:
    """One eval: a single task run across N seeds, all in parallel.

    Seed slot i (i = 0..n_seeds-1; the actual seed is seed_start + i) gets its
    own port stack at base_port + i*port_stride. record_sim binds 4 consecutive
    ports per stack and the stride leaves head-room, so the seeds never collide
    and run concurrently. Each seed's episode is saved into its own folder
    <data_root>/<task_slug>/seed<seed>/ via record_sim's --demo_folder.
    """

    task: str
    language: str  # natural-language goal handed to the agent.
    # Optional path (relative to the repo root, or absolute) to a per-task
    # instructions.md. Claude receives it via --append-system-prompt-file (the same
    # channel the harness uses for CLAUDE.md); codex receives it inside the
    # auto-loaded AGENTS.md -- either way the agent gets it even though file reads
    # are locked down.
    instruct_file: str | None = None
    n_seeds: int = 2
    seed_start: int = 1
    data_root: str = "data/run_eval/dev"
    # Sub-dir name under data_root (default: the task slug). Set this to keep many
    # runs of the same task apart under a shared data_root.
    run_name: str | None = None
    base_port: int = 8100
    port_stride: int = 100
    harness: str = "claude"  # agent harness, "claude" or "codex"
    model: str = "opus"
    reasoning_effort: str = "xhigh"
    episode_timeout: float = 3600.0  # wall-clock cap on one agent episode
    ready_timeout: float = 180.0  # wait for a seed's sim UI + browser to come up
    grace: float = 60.0  # wait for record_sim to save after the agent exits
    force_rerun: bool = False  # rerun every seed even if it already saved a verdict

    def __post_init__(self) -> None:
        # Validate against the live harness registry (raises KeyError naming the
        # registered keys), so adding a harness never requires touching this check.
        harness = get_harness(self.harness)
        if harness.supported_efforts and self.reasoning_effort not in harness.supported_efforts:
            raise ValueError(
                f"reasoning_effort {self.reasoning_effort!r} not supported by the "
                f"{self.harness!r} harness; valid: {sorted(harness.supported_efforts)}"
            )
        # Enforce at construction (i.e. import time, when EXPERIMENTS is built) that
        # a named instruct_file actually exists, so a typo'd path fails loudly here
        # instead of letting claude launch without the per-task instructions.
        if self.instruct_file:
            path = os.path.join(REPO_ROOT, self.instruct_file)
            assert os.path.isfile(path), f"Experiment.instruct_file not found: {path}"


def task_folder(exp: Experiment) -> str:
    """Sub-dir under data_root that holds this run's episodes. Defaults to the task
    slug; exp.run_name overrides it so many runs of the SAME task can sit side by
    side under one data_root (e.g. debug/run1, debug/run2) without clobbering."""
    return exp.run_name or task_slug(exp.task)


def seed_demo_folder(exp: Experiment, seed: int) -> str:
    """Where this seed's episode is saved (handed to record_sim's --demo_folder).
    Single source of truth so the run path and the finished-seed check agree."""
    return os.path.join(REPO_ROOT, exp.data_root, task_folder(exp), f"seed{seed}")


def stack_ports(base_port: int) -> tuple[int, ...]:
    """The 4 consecutive ports record_sim binds for one stack (see record_sim.py)."""
    return tuple(base_port + i for i in range(4))


def kill_ports(ports) -> None:
    """Best-effort free of ONE stack's ports. record_sim cleans up its own
    children on exit, but a crashed/killed run can leave a port bound; this keeps
    the next run from failing to bind. Scoped to the given ports so parallel
    seeds never kill each other. No-op without `lsof`."""
    for port in ports:
        try:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
        except FileNotFoundError:
            return  # no lsof; rely on record_sim's own cleanup
        for pid in out.stdout.split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


def get_json(url: str, timeout: float = 1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def cdp_ready(base_port: int) -> bool:
    return get_json(f"http://localhost:{base_port + 3}/json/version", timeout=0.5) is not None


def wait_for_sim_ready(proc: subprocess.Popen, base_port: int, ready_timeout: float, settle: float):
    """Poll until this seed's record_sim is fully up: /env.json serving a
    demo_folder AND the Chromium CDP endpoint accepting connections (so claude's MCP
    server can attach to a real page rather than racing the browser launch).

    Returns the /env.json dict, or None if the process died or the timeout hit.
    """
    env_url = f"http://localhost:{base_port + 1}/env.json"
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return None  # record_sim exited before becoming ready
        info = get_json(env_url)
        if info and info.get("demo_folder") and cdp_ready(base_port):
            time.sleep(settle)  # let the page finish its WebGL/WS handshake
            return info
        time.sleep(0.5)
    return None


_logged_prompts: set[tuple[str, str | None]] = set()
_prompt_log_lock = threading.Lock()


def log_prompt_once(prompt: str, instruct_file: str | None) -> None:
    """Print each distinct (prompt, instructions) combination a single time.
    Seeds of one experiment share a prompt, so logging per seed is just noise —
    but prompts DO differ across experiments (language/instruct_file) and across
    harnesses (guide filename), so a mixed run logs one banner per variant rather
    than misattributing the first variant to every agent."""
    key = (prompt, instruct_file)
    with _prompt_log_lock:
        if key in _logged_prompts:
            return
        _logged_prompts.add(key)
    instructions = "None" if instruct_file is None else open(instruct_file).read()
    log(
        "\n=== agent prompt (same for every seed of this experiment) ===\n"
        f"{prompt}\n"
        "===========================================\n"
        "================ instructions =============\n"
        f"{instructions}\n"
        "==========================================="
    )


# Sims currently running, so a Ctrl-C in main() can tear them all down promptly
# instead of waiting for each worker thread to unwind to its own finally. A sim is
# added right after launch and dropped once it's gone. Guarded because seeds run in a
# thread pool. Sims get terminate_group (own session); the agent procs are tracked
# separately in agent_common (they share our group, so plain terminate()).
_active_sims: set[subprocess.Popen] = set()
_active_procs_lock = threading.Lock()

# Set on Ctrl-C so a worker that hasn't launched the agent yet bails out instead of
# spinning up a fresh sim/agent that nothing would then reap. Worker threads never
# receive the KeyboardInterrupt themselves (Python delivers it only to the main
# thread), so this flag is how the interrupt reaches them.
_shutdown = threading.Event()


def terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM-then-SIGKILL the whole process GROUP of `proc`.

    `proc` is a record_sim launched with start_new_session=True, so it leads its
    own process group and its mp-forked children (the web + two websocket servers
    and the mp.Manager) inherit that group. Signalling the group reaches all of
    them; a plain proc.terminate()/.kill() hits only the parent PID and orphans
    the children -- they block forever in serve_forever / asyncio.Future, get
    reparented to launchd, and pile up. Idempotent and best-effort.

    Only ever call this on a sim (own session). The claude/reflect procs share the
    orchestrator's group, so group-killing them would take the driver down too --
    use plain terminate() for those."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    # 20s SIGTERM grace: worst case record_sim finishes the in-flight waypoint
    # (<=5s) then finalizes mp4 + npz + webm before exiting; clean exits return
    # from proc.wait immediately so the wide window costs nothing.
    for sig, wait_s in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return  # group already gone
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue  # escalate to SIGKILL


def read_verdict(demo_folder: str | None):
    if not demo_folder:
        return None
    return get_json_file(os.path.join(demo_folder, "verdict.json"))


def finished_result(seed: int, base_port: int, demo_folder: str) -> dict | None:
    """Build a result dict from an already-saved verdict.json, or None if this
    seed hasn't finished yet (no verdict). Used both to skip reruns of finished
    seeds and to report results in --print-results mode."""
    verdict = read_verdict(demo_folder)
    if verdict is None:
        return None
    # A timed-out episode is graded as a failure regardless of the raw success
    # flag: the run hit the sim's step budget instead of finishing on its own terms.
    timed_out = bool(verdict.get("timed_out"))
    return {
        "seed": seed,
        "base_port": base_port,
        "demo_folder": demo_folder,
        "status": "ok",
        "success": bool(verdict.get("success")) and not timed_out,
        "timed_out": timed_out,
        "sim_steps_used": verdict.get("sim_steps_used"),
        "elapsed_seconds": verdict.get("elapsed_seconds"),
    }


def write_timeout_verdict(demo_folder: str, exp: Experiment) -> None:
    """Persist a synthetic verdict for a seed whose agent blew the wall-clock
    cap (episode_timeout). In that case the agent is killed before it can call
    end_episode, so record_sim never reaches _record_verdict — its one_episode loop
    just keeps streaming (waypoint mode doesn't advance sim steps on its own, so it
    never hits the horizon either) and writes no verdict.json. Without a verdict on
    disk, the next run can't tell this seed apart from one that never started and
    would re-drive it (see finished_result, which keys the rerun-skip off
    verdict.json existing). Writing this marker — graded a timeout failure, the same
    as a sim-step-horizon timeout — makes the seed count against the success rate AND
    be skipped on rerun. `wall_clock_timeout` distinguishes it from record_sim's own
    horizon-hit verdict; the fields finished_result reads are kept compatible."""
    verdict = {
        "success": False,
        "timed_out": True,
        "wall_clock_timeout": True,
        "elapsed_seconds": round(exp.episode_timeout, 1),
        "sim_steps_used": None,  # unknown: record_sim was terminated mid-stream
    }
    path = os.path.join(demo_folder, "verdict.json")
    try:
        os.makedirs(demo_folder, exist_ok=True)
        with open(path, "w") as f:
            json.dump(verdict, f, indent=2)
    except Exception as e:
        warn(f"could not write timeout verdict {path}: {e}")


def run_seed(exp: Experiment, seed: int, base_port: int, logs_dir: str) -> dict:
    """Run one seed end to end on its own port stack. Returns a result dict.

    A seed that already saved a verdict.json is treated as done: we return its
    existing result (tagged "reused") without relaunching, so re-running an
    experiment only fills in the seeds that didn't finish."""
    tag = f"seed{seed}"
    demo_folder = seed_demo_folder(exp, seed)

    # A queued seed that the pool starts in the instant around Ctrl-C: bail before
    # launching anything (main() has already torn down / will not wait for us).
    if _shutdown.is_set():
        return {
            "seed": seed,
            "base_port": base_port,
            "demo_folder": demo_folder,
            "status": "skipped",
            "success": None,
        }

    if not exp.force_rerun:
        existing = finished_result(seed, base_port, demo_folder)
        if existing is not None:
            existing["reused"] = True
            log(f"[{tag}] already finished (success={existing['success']}); skipping rerun")
            return existing

    ports = stack_ports(base_port)
    kill_ports(ports)  # only THIS seed's ports — parallel seeds must not kill each other

    sim_log = os.path.join(logs_dir, f"{tag}_port{base_port}_sim.log")
    agent_log = os.path.join(logs_dir, f"{tag}_port{base_port}_{exp.harness}.log")

    # This seed's isolated stack. SPHINX_BASE_PORT drives record_sim's ports and,
    # inherited through `claude -p`, the MCP server that attaches to this browser.
    # SPHINX_BASE_PORT is all the children need from us: the sim picks up
    # PYTHONPATH + OPENBLAS_NUM_THREADS by sourcing set_env.sh (below), and the
    # claude/MCP child imports no repo modules so it needs no PYTHONPATH.
    env = dict(os.environ)
    env["SPHINX_BASE_PORT"] = str(base_port)

    # Source set_env.sh so the sim runs in the activated venv (hence bare
    # `python`) and inherits its OPENBLAS_NUM_THREADS cap (single source of truth
    # -- the point-cloud deproject() must not fan thin matmuls across every core).
    # `exec "$@"` replaces the shell with python in-place so the PID we get back
    # is the sim itself (signals/terminate reach it directly); the args ride
    # through as positional params, no re-quoting. Only the sim needs this -- the
    # claude/MCP child doesn't use numpy.
    sim_cmd = [
        "bash",
        "-c",
        f"source {shlex.quote(os.path.join(REPO_ROOT, 'set_env.sh'))} && exec \"$@\"",
        "_",  # $0 placeholder; real argv follows
        "python",
        "-m",
        "via.record_sim",
        "--task",
        exp.task,
        "--demo_folder",
        demo_folder,
        "--seed",
        str(seed),
        "--num_point",
        str(NUM_POINT),
        "--stream_freq",
        str(STREAM_FREQ),
        # Headless: eval runs are parallel/offscreen — no robomimic viewer window.
        "--render",
        "0",
    ]

    result: dict = {"seed": seed, "base_port": base_port, "demo_folder": demo_folder}
    with open(sim_log, "w") as slog:
        # start_new_session: the sim leads its own process group so terminate_group
        # can take down its forked children + Manager in one signal (no orphans).
        sim = subprocess.Popen(
            sim_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=slog,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with _active_procs_lock:
            _active_sims.add(sim)
        try:
            info = wait_for_sim_ready(sim, base_port, exp.ready_timeout, SETTLE)
            if info is None:
                terminate_group(sim)
                result["status"] = "sim_failed_to_start"
                result["success"] = None
                return result

            # Trust the demo_folder record_sim reports over the one we passed in.
            demo_folder = info.get("demo_folder") or demo_folder
            result["demo_folder"] = demo_folder
            log(
                f"[{tag}] sim ready on :{base_port}  task={info.get('task')!r}  demo_folder={demo_folder}"
            )

            harness = get_harness(exp.harness)
            instruct_file = (
                os.path.join(REPO_ROOT, exp.instruct_file) if exp.instruct_file else None
            )

            # wait_for_sim_ready above can block up to ready_timeout; if Ctrl-C
            # landed in that window, bail now rather than launch an agent that
            # nothing would reap (the sim is torn down by the finally below).
            if _shutdown.is_set():
                result["status"] = "skipped"
                result["success"] = None
                return result

            # Same prompt template for every harness; only the guide filename differs
            # (Claude: CLAUDE.md; Codex: the same content as AGENTS.md).
            prompt = build_prompt(exp.language, guide=harness.guide_filename)
            log_prompt_once(prompt, instruct_file)
            log(f"[{tag}] launching {harness.key} ({exp.model}) ...")
            t0 = time.time()
            # The harness owns everything agent-specific: writing the watch/resume
            # scripts, running the agent, and archiving conversation.txt.
            outcome = harness.drive(
                SeedContext(
                    tag=tag,
                    prompt=prompt,
                    model=exp.model,
                    reasoning_effort=exp.reasoning_effort,
                    log_path=agent_log,
                    timeout=exp.episode_timeout,
                    env=env,
                    demo_folder=demo_folder,
                    task=exp.task,
                    seed=seed,
                    base_port=base_port,
                    instruct_file=instruct_file,
                )
            )
            result["session_id"] = outcome.session_id
            result["agent_status"] = outcome.status
            log(
                f"[{tag}] {harness.key} finished ({outcome.status}, {time.time() - t0:.0f}s); "
                "waiting for record_sim to save ..."
            )

            # success / end_episode make record_sim exit on its own; an abandoned
            # episode would stream forever, so terminate after the grace window.
            try:
                sim.wait(timeout=exp.grace)
            except subprocess.TimeoutExpired:
                log(f"[{tag}] record_sim still running after grace; terminating (no verdict).")
        finally:
            terminate_group(sim)  # idempotent; clean-exit, timeout, and interrupt paths
            with _active_procs_lock:
                _active_sims.discard(sim)

    # (conversation.txt is archived by the harness's drive(), right after the agent
    # exits — see ClaudeHarness/CodexHarness.)
    finished = finished_result(seed, base_port, demo_folder)
    if finished is not None:
        result.update(finished)  # keeps session_id / agent_status already on result
        return result

    # No verdict was saved. If claude blew the wall-clock cap, record_sim never got
    # an end_episode — grade that as a timeout failure (success=False), not an
    # ungraded result, so it counts against the success rate. Persist a synthetic
    # verdict so the seed is also skipped on rerun (record_sim wrote none). Any other
    # no-verdict case (e.g. the sim crashed) stays ungraded and reruns.
    if result.get("agent_status") == "timeout":
        write_timeout_verdict(demo_folder, exp)
        result["status"] = "timeout"
        result["success"] = False
        result["timed_out"] = True
    else:
        result["status"] = "no_verdict"
        result["success"] = None
    return result


def print_one(res: dict, tag: str = "") -> None:
    s = res.get("success")
    mark = "✓" if s else ("·" if s is None else "✗")
    detail = ""
    if res.get("sim_steps_used") is not None:
        detail = f"  steps={res['sim_steps_used']}  {res.get('elapsed_seconds')}s"
    flags = ""
    if res.get("timed_out"):
        flags += " [timed_out]"
    if res.get("reused"):
        flags += " [reused]"
    if res.get("status") not in ("ok", None):
        flags += f" [{res.get('status')}]"
    # tag names the experiment so seeds stay attributable when runs interleave.
    label = f"{tag}  " if tag else ""
    log(f"  [{mark}] {label}seed {res.get('seed')}{detail}{flags}")


def summarize_and_write(
    results, exp: Experiment, name: str, out_dir: str, interrupted: bool, write: bool = True
) -> None:
    n = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    n_graded = sum(1 for r in results if r.get("success") is not None)
    rate = (n_success / n_graded) if n_graded else 0.0

    out_path = None
    if write:  # --print-results just reports; only real runs persist a new file
        summary = {
            "experiment": name,
            "task": exp.task,
            "n_seeds": exp.n_seeds,
            "seed_start": exp.seed_start,
            "harness": exp.harness,
            "model": exp.model,
            "reasoning_effort": exp.reasoning_effort,
            "n_success": n_success,
            "n_graded": n_graded,
            "success_rate": rate,
            "interrupted": interrupted,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "results": results,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"eval_results_{ts}.json")
        try:
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            warn(f"could not write {out_path}: {e}")
            out_path = "(unwritten)"

    log("\n=== summary ===")
    log(f"  experiment: {name}  (task: {exp.task})")
    ungraded = n - n_graded
    ungraded_note = f"  ({ungraded} ungraded)" if ungraded else ""
    log(f"  success: {n_success}/{n_graded} graded = {rate * 100:.1f}%{ungraded_note}")
    if interrupted:
        log("  (interrupted before all seeds finished)")
    if out_path is not None:
        log(f"  results: {out_path}")


def backup_guide_file(out_dir: str, harness_key: str) -> None:
    """Snapshot the agent guide that drove this run into the task folder.

    Claude reads CLAUDE.md directly; Codex gets the same content through a
    generated AGENTS.md, so Codex runs are labeled backup_AGENTS.md even though the
    source content still comes from CLAUDE.md. The label should match what the
    harness told the agent to read.

    Only-if-missing: an incremental rerun must not clobber the snapshot of what
    drove the already-finished seeds (a CLAUDE.md edit between runs would otherwise
    overwrite it)."""
    guide_name = get_harness(harness_key).guide_filename
    dst = os.path.join(out_dir, f"backup_{guide_name}")
    if os.path.exists(dst):
        return  # preserve the original snapshot across reruns
    try:
        shutil.copy(CLAUDE_MD_PATH, dst)
        log(f"  backed up {guide_name} guide -> {dst}")
    except Exception as e:
        warn(f"could not back up {guide_name} guide to {dst}: {e}")


def backup_instruct_file(out_dir: str, exp: Experiment) -> None:
    """Snapshot the per-task instruct_file (if any) into the task folder, mirroring
    backup_guide_file so a run's results stay tied to the exact instructions seen.
    Only-if-missing for the same reason: a rerun preserves the original snapshot."""
    if not exp.instruct_file:
        return
    src = os.path.join(REPO_ROOT, exp.instruct_file)
    dst = os.path.join(out_dir, f"backup_{os.path.basename(exp.instruct_file)}")
    if os.path.exists(dst):
        return  # preserve the original snapshot across reruns
    try:
        shutil.copy(src, dst)
        log(f"  backed up instruct file -> {dst}")
    except Exception as e:
        warn(f"could not back up instruct file {src} -> {dst}: {e}")


# Experiments. Run with: python -m via.run_eval <name>
EXPERIMENTS: dict[str, Experiment] = {
    ####### dev: quick smoke tests (few seeds, always rerun)
    "stack-codex-dev": Experiment(
        task="stack",
        language=LANG["stack"],
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=3,
        data_root="data/run_eval/dev",
        run_name="codex",
        force_rerun=True,
    ),
    "stack-opus-dev": Experiment(
        task="stack",
        language=LANG["stack"],
        model="opus",
        n_seeds=3,
        data_root="data/run_eval/dev",
        run_name="opus",
        force_rerun=True,
    ),
    ####### opus (claude harness): min prompts
    "stack-min-opus": Experiment(
        task="stack",
        language=LANG["stack"],
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/min_opus",
    ),
    "lg0-min-opus": Experiment(
        task="libero_goal/0",
        language=LANG["lg0"],
        instruct_file="prompts/open_drawer_min.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/min_opus",
    ),
    "lg7-min-opus": Experiment(
        task="libero_goal/7",
        language=LANG["lg7"],
        instruct_file="prompts/turn_stove_min.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/min_opus",
    ),
    "lg8-min-opus": Experiment(
        task="libero_goal/8",
        language=LANG["lg8"],
        instruct_file="prompts/bowl_plate_min.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/min_opus",
    ),
    "t-block-min-opus": Experiment(
        task="t_block",
        language=LANG["t_block"],
        instruct_file="prompts/t_block_min.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/min_opus",
    ),
    "rainbow-min-opus": Experiment(
        task="rainbow",
        language=LANG["rainbow"],
        instruct_file="prompts/rainbow_min.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/min_opus",
        episode_timeout=7200,
    ),
    ####### opus (claude harness): detailed prompts
    "lg0-opus": Experiment(
        task="libero_goal/0",
        language=LANG["lg0"],
        instruct_file="prompts/open_drawer.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/detailed_opus",
    ),
    "lg7-opus": Experiment(
        task="libero_goal/7",
        language=LANG["lg7"],
        instruct_file="prompts/turn_stove.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/detailed_opus",
    ),
    "lg8-opus": Experiment(
        task="libero_goal/8",
        language=LANG["lg8"],
        instruct_file="prompts/bowl_plate.md",
        model="opus",
        n_seeds=10,
        data_root="data/run_eval/detailed_opus",
    ),
    ####### fable (claude harness): min prompts
    "stack-min-fable": Experiment(
        task="stack",
        language=LANG["stack"],
        model="fable",
        n_seeds=10,
        data_root="data/run_eval/min_fable",
    ),
    "lg0-min-fable": Experiment(
        task="libero_goal/0",
        language=LANG["lg0"],
        instruct_file="prompts/open_drawer_min.md",
        model="fable",
        n_seeds=10,
        data_root="data/run_eval/min_fable",
    ),
    "lg7-min-fable": Experiment(
        task="libero_goal/7",
        language=LANG["lg7"],
        instruct_file="prompts/turn_stove_min.md",
        model="fable",
        n_seeds=10,
        data_root="data/run_eval/min_fable",
    ),
    "lg8-min-fable": Experiment(
        task="libero_goal/8",
        language=LANG["lg8"],
        instruct_file="prompts/bowl_plate_min.md",
        model="fable",
        n_seeds=10,
        data_root="data/run_eval/min_fable",
    ),
    "t-block-min-fable": Experiment(
        task="t_block",
        language=LANG["t_block"],
        instruct_file="prompts/t_block_min.md",
        model="fable",
        n_seeds=10,
        data_root="data/run_eval/min_fable",
    ),
    "rainbow-min-fable": Experiment(
        task="rainbow",
        language=LANG["rainbow"],
        instruct_file="prompts/rainbow_min.md",
        model="fable",
        n_seeds=10,
        data_root="data/run_eval/min_fable",
        episode_timeout=7200,
    ),
    ####### codex harness: min prompts
    "stack-min-codex": Experiment(
        task="stack",
        language=LANG["stack"],
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/min_codex",
    ),
    "lg0-min-codex": Experiment(
        task="libero_goal/0",
        language=LANG["lg0"],
        instruct_file="prompts/open_drawer_min.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/min_codex",
    ),
    "lg7-min-codex": Experiment(
        task="libero_goal/7",
        language=LANG["lg7"],
        instruct_file="prompts/turn_stove_min.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/min_codex",
    ),
    "lg8-min-codex": Experiment(
        task="libero_goal/8",
        language=LANG["lg8"],
        instruct_file="prompts/bowl_plate_min.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/min_codex",
    ),
    "t-block-min-codex": Experiment(
        task="t_block",
        language=LANG["t_block"],
        instruct_file="prompts/t_block_min.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/min_codex",
    ),
    "rainbow-min-codex": Experiment(
        task="rainbow",
        language=LANG["rainbow"],
        instruct_file="prompts/rainbow_min.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/min_codex",
        episode_timeout=7200,
    ),
    ####### codex harness: detailed prompts
    "lg0-codex": Experiment(
        task="libero_goal/0",
        language=LANG["lg0"],
        instruct_file="prompts/open_drawer.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/detailed_codex",
    ),
    "lg7-codex": Experiment(
        task="libero_goal/7",
        language=LANG["lg7"],
        instruct_file="prompts/turn_stove.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/detailed_codex",
    ),
    "lg8-codex": Experiment(
        task="libero_goal/8",
        language=LANG["lg8"],
        instruct_file="prompts/bowl_plate.md",
        harness="codex",
        model=CODEX_MODEL,
        n_seeds=10,
        data_root="data/run_eval/detailed_codex",
    ),
}


def list_experiments() -> None:
    log("available experiments:")
    for name, exp in EXPERIMENTS.items():
        log(
            f"  {name:22s} harness={exp.harness:6s} task={exp.task!r}  "
            f"n_seeds={exp.n_seeds}  base_port={exp.base_port}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more experiments, sharing a single -j budget across them.",
    )
    parser.add_argument("experiment", nargs="*", help="experiment name(s) (omit to list).")
    parser.add_argument("-j", "--max-parallel", type=int, default=1)
    parser.add_argument("--print-results", action="store_true", help="no run; print summary.")
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help=(
            "override the starting port for THIS invocation (default: the experiment's "
            "base_port, usually 8100). Use to run a second run_eval in parallel on a "
            "non-colliding range -- give it enough headroom above the other run's "
            "highest seed (base_port + (n_seeds-1)*port_stride + 3)."
        ),
    )
    args = parser.parse_args()
    setup_logging()

    if not args.experiment:
        list_experiments()
        return
    # Dedupe but keep order; a repeated name would re-run the same seed folders.
    names = list(dict.fromkeys(args.experiment))
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        warn(f"unknown experiment(s): {', '.join(map(repr, unknown))}")
        list_experiments()
        sys.exit(2)
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

    # Plan each experiment's seeds and ports. Ports are allocated GLOBALLY across
    # all selected experiments (a running counter, not each exp's base_port) so
    # experiments that run at once never bind the same stack. A lone experiment
    # retains its configured base port and stride.
    port_stride = max(EXPERIMENTS[n].port_stride for n in names)
    # --base-port shifts this whole invocation's stacks so a second, parallel
    # run_eval can avoid the ports the first one cycles through.
    next_port = (
        args.base_port
        if args.base_port is not None
        else min(EXPERIMENTS[n].base_port for n in names)
    )
    plan = []  # (name, exp, out_dir, seeds, ports)
    for name in names:
        exp = EXPERIMENTS[name]
        out_dir = os.path.join(REPO_ROOT, exp.data_root, task_folder(exp))
        seeds = [exp.seed_start + i for i in range(exp.n_seeds)]
        ports = [next_port + i * port_stride for i in range(exp.n_seeds)]
        if ports:
            next_port = ports[-1] + port_stride
        plan.append((name, exp, out_dir, seeds, ports))

    # Guard against two selected experiments writing the same seed folder (same
    # task + data_root): concurrent writes there would corrupt each other's episode.
    seen: dict[str, str] = {}
    for name, exp, _out_dir, seeds, _ports in plan:
        for seed in seeds:
            folder = seed_demo_folder(exp, seed)
            if folder in seen:
                parser.error(
                    f"experiments {seen[folder]!r} and {name!r} both write {folder}; "
                    "give them different data_root or task."
                )
            seen[folder] = name

    if args.print_results:
        # Read-only: report whatever verdicts exist on disk, run nothing.
        for name, exp, out_dir, seeds, ports in plan:
            log(f"=== results: experiment={name}  task={exp.task} ===")
            results = []
            for seed, port in zip(seeds, ports):
                demo_folder = seed_demo_folder(exp, seed)
                res = finished_result(seed, port, demo_folder) or {
                    "seed": seed,
                    "base_port": port,
                    "demo_folder": demo_folder,
                    "status": "no_verdict",
                    "success": None,
                }
                results.append(res)
                print_one(res, name if len(plan) > 1 else "")
            summarize_and_write(results, exp, name, out_dir, interrupted=False, write=False)
        return

    # Per-exp setup (log dir + snapshots), and flatten to per-seed work items that
    # all feed ONE pool, so --max-parallel bounds total concurrency across experiments.
    work = []  # (name, exp, seed, port, logs_dir)
    for name, exp, out_dir, seeds, ports in plan:
        logs_dir = os.path.join(out_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        backup_guide_file(out_dir, exp.harness)
        backup_instruct_file(out_dir, exp)
        for seed, port in zip(seeds, ports):
            work.append((name, exp, seed, port, logs_dir))

    total_seeds = len(work)
    # Cap concurrency; the rest queue in the pool and start as workers free up.
    max_parallel = min(args.max_parallel or total_seeds, total_seeds)
    lines = [
        f"=== eval: {len(plan)} experiment(s), {total_seeds} seeds total, "
        f"{max_parallel} at a time ===",
    ]
    for name, exp, _out_dir, seeds, ports in plan:
        lines.append(
            f"  {name}  harness={exp.harness}  task={exp.task!r}  model={exp.model}  "
            f"seeds->ports={dict(zip(seeds, ports))}"
        )
    log("\n".join(lines))

    results_by_name: dict[str, list[dict]] = {name: [] for name, *_ in plan}
    all_ports = [port for _name, _exp, _seed, port, _logs in work]
    interrupted = False
    pool = ThreadPoolExecutor(max_workers=max_parallel)
    futs = {
        pool.submit(run_seed, exp, seed, port, logs_dir): (name, seed)
        for name, exp, seed, port, logs_dir in work
    }
    try:
        for fut in as_completed(futs):
            name, seed = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"seed": seed, "status": "error", "success": None, "error": repr(e)}
            results_by_name[name].append(res)
            print_one(res, name if len(plan) > 1 else "")
    except KeyboardInterrupt:
        interrupted = True
        log("\n[interrupted] killing all stacks ...")
    finally:
        # Stop the bleeding first: flag shutdown so any worker about to launch a
        # seed bails, and cancel_futures so seeds still QUEUED in the pool never
        # start. Without cancel_futures, pool.shutdown(wait=False) leaves queued
        # work in place and the workers (which the interpreter's atexit join blocks
        # on) keep draining it -- launching fresh sims/claude that never got the
        # terminal's Ctrl-C and so would run to episode_timeout. This is THE reason
        # a Ctrl-C used to leave sessions alive: not orphaned children, but the pool
        # marching on through the remaining seeds.
        _shutdown.set()
        pool.shutdown(wait=False, cancel_futures=True)
        # Tear down everything still in flight. Sims die by process group (parent +
        # forked servers + Manager + Chromium share it -- no orphans). Agent procs
        # (claude/codex) get a plain terminate: they share OUR group, so killpg would
        # suicide the driver, and they do NOT exit just because their sim/MCP died, so
        # we can't rely on that or on the terminal signal (absent under nohup/background).
        # kill_ports is belt-and-suspenders for anything (e.g. a browser helper) that escaped.
        with _active_procs_lock:
            sims = list(_active_sims)
        for sim in sims:
            terminate_group(sim)
        for proc in active_agents():
            terminate(proc)
        for port in all_ports:
            kill_ports(stack_ports(port))
        for name, exp, out_dir, _seeds, _ports in plan:
            results = results_by_name[name]
            results.sort(key=lambda r: r.get("seed", 0))
            summarize_and_write(results, exp, name, out_dir, interrupted)


if __name__ == "__main__":
    # python -m via.run_eval stack-min-codex -j 3
    main()
