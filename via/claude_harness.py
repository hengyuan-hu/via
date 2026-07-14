"""Claude Code harness for the eval driver — the built-in `claude -p` path.

Holds the Claude-specific primitives (run_claude + the debug-script writers +
the MCP tool allowlist) and the ClaudeHarness that wires them into the uniform
Harness interface. run_learn imports run_claude / write_*_script / allowed_tools
directly from here.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import uuid

from via.agent_common import (
    MCP_CONFIG_PATH,
    REPO_ROOT,
    get_json_file,
    log,
    register_agent,
    terminate,
    unregister_agent,
    warn,
)
from via.harness import DriveOutcome, Harness, SeedContext, register_harness

# Built-in tools denied to the eval agent so it can ONLY perceive/act through the
# sphinx2 MCP robot tools. Without this, the agent can read ground-truth task
# state off disk (e.g. the .bddl object poses, sim_env.py, this file) and shortcut
# the perception the eval is meant to measure. CLAUDE.md is unaffected — the
# harness injects it into context directly, not via the Read tool. Task is denied
# too, else a spawned subagent would regain filesystem access. Only real tool names
# are listed: an unknown name makes Claude warn "matches no known tool".
DISALLOWED_TOOLS = [
    "Bash",
    "BashOutput",
    "KillShell",
    "KillBash",
    "Read",
    "Edit",
    "Write",
    "NotebookEdit",
    "Glob",
    "Grep",
    "Task",
    "WebFetch",
    "WebSearch",
]


def allowed_tools(config_path: str) -> list[str]:
    """Allowlist Claude to the MCP robot tools without enumerating them.

    A server-level grant ("mcp__<server>") permits every tool that server
    exposes, so this auto-follows any tool added to or removed from
    mcp_server.py — no list to keep in sync. The server name(s) are read from
    the same .mcp.json passed to --mcp-config, so a rename there stays in sync
    too.
    """
    cfg = get_json_file(config_path) or {}
    names = list((cfg.get("mcpServers") or {}).keys())
    if not names:
        names = ["sphinx2"]  # fall back to the known server name
    return [f"mcp__{name}" for name in names]


def run_claude(
    prompt: str,
    model: str,
    log_path: str,
    timeout: float,
    tools: list[str],
    env: dict,
    session_id: str,
    reasoning_effort: str,
    instruct_file: str | None = None,
) -> str:
    """Drive one seed with a fresh headless Claude. Returns 'ok' or 'timeout'.
    `env` carries this seed's SPHINX_BASE_PORT, inherited by the sphinx2 MCP
    server so it attaches to this seed's browser. Stdout+stderr (incl. the MCP
    server's) go to log_path. `session_id` pins the conversation's id up front so
    it can be resumed later for debugging (see write_resume_script).
    `instruct_file`, when given, is an absolute path to per-task instructions
    appended to the system prompt (see Experiment.instruct_file). `reasoning_effort`
    pins the thinking level via --effort (the analog of codex's
    model_reasoning_effort), never inheriting the CLI default."""
    cmd = [
        "claude",
        "-p",
        prompt,
        "--session-id",
        session_id,
        "--mcp-config",
        MCP_CONFIG_PATH,
        "--strict-mcp-config",
        "--model",
        model,
        "--effort",
        reasoning_effort,
    ]
    # Inject per-task instructions into the system prompt (like CLAUDE.md). Not a
    # tool, so it's unaffected by --disallowedTools below. Must come before the
    # variadic tool flags so --allowedTools can stay last.
    if instruct_file:
        cmd += ["--append-system-prompt-file", instruct_file]
    # Both --disallowedTools and --allowedTools are variadic; the first is
    # terminated by the next flag token, so disallowed must precede allowed and
    # --allowedTools stays last.
    cmd += ["--disallowedTools", *DISALLOWED_TOOLS]
    cmd += ["--allowedTools", *tools]

    # Match the codex harness's MCP timeout budget (startup_timeout_sec=90,
    # tool_timeout_sec=180 in codex_mcp_overrides) so neither arm of the eval
    # fails its sphinx2 attach purely because the machine is loaded. Claude Code
    # reads these in milliseconds.
    env = {**env, "MCP_TIMEOUT": str(90_000), "MCP_TOOL_TIMEOUT": str(180_000)}

    with open(log_path, "w") as log_f:
        # stdin=DEVNULL for parity with run_codex: `claude -p` appends piped
        # non-tty stdin to the prompt, so an inherited stdin under nohup/another
        # driver could contaminate the episode (or block); the robot eval never
        # feeds stdin.
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        # Track so main()'s Ctrl-C path can terminate it directly. claude shares the
        # driver's process group, so an interactive Ctrl-C reaches it via the terminal
        # too -- but a non-foreground run (nohup/background/under another driver) gets
        # no group signal, and claude does NOT exit just because its sim/MCP died, so
        # the explicit terminate in main() is what actually reaps it there.
        register_agent(proc)
        try:
            proc.wait(timeout=timeout)
            return "ok"
        except subprocess.TimeoutExpired:
            terminate(proc)
            return "timeout"
        except KeyboardInterrupt:
            terminate(proc)
            raise
        finally:
            unregister_agent(proc)


def write_resume_script(
    demo_folder: str,
    session_id: str,
    base_port: int,
    model: str,
    task: str,
    seed: int,
    instruct_file: str | None = None,
) -> str:
    """Drop <demo_folder>/resume_claude.sh: the command to resume this episode's
    Claude Code session for debugging. The id is pinned via --session-id at launch (so
    it's known up front, before the run even finishes). Claude stores sessions
    per-project keyed by cwd, so the script cd's to the repo root first.

    The active command resumes WITHOUT the MCP server, which always works for
    inspecting/continuing the transcript; a commented recipe shows how to relaunch
    the sim on this port and resume WITH the robot tools attached.

    `instruct_file`, when given, is the per-task instructions file that drove the
    episode. It's re-appended to the resumed session's system prompt via
    --append-system-prompt-file: the system prompt isn't stored in the saved
    transcript (only messages are), so a plain --resume rebuilds it from flags and
    would otherwise drop the task notes the driving agent actually saw."""
    q = shlex.quote
    # The driving agent got these via --append-system-prompt-file; replay them so the
    # resumed session sees the same task notes (CLAUDE.md is auto-loaded, this isn't).
    append = f" --append-system-prompt-file {q(instruct_file)}" if instruct_file else ""
    script = f"""#!/usr/bin/env bash
# Resume the Claude Code session that drove this episode (for debugging).
# Run from anywhere; it cd's to the repo root (sessions are stored per-project).
cd {q(REPO_ROOT)} || exit 1

# Inspect / continue the conversation (no robot tools; always works):
claude --resume {session_id} --model {q(model)}{append} "$@"

# To actually re-drive the robot, first start a sim on this port in another shell
# (set_env.sh activates the venv + sets PYTHONPATH and OPENBLAS_NUM_THREADS):
#   source set_env.sh && SPHINX_BASE_PORT={base_port} \\
#     python -m via.record_sim --task {q(task)} \\
#     --demo_folder {q(demo_folder)} --seed {seed} --render 0
# then resume WITH the MCP robot tools attached (same tool restriction as the
# eval, so a re-drive can't read ground-truth off disk either):
#   SPHINX_BASE_PORT={base_port} claude --resume {session_id}{append} \\
#     --mcp-config {q(MCP_CONFIG_PATH)} --strict-mcp-config --model {q(model)} \\
#     --disallowedTools {' '.join(DISALLOWED_TOOLS)}
"""
    path = os.path.join(demo_folder, "resume_claude.sh")
    try:
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)
    except Exception as e:
        warn(f"could not write {path}: {e}")
    return path


def write_inspect_script(demo_folder: str, session_id: str) -> str:
    """Drop <demo_folder>/inspect_claude.sh: a read-only view of this seed's
    Claude Code session, streaming the agent's messages and tool calls. The id is
    pinned via --session-id at launch, so this can be written (and run) before the
    episode finishes. Session ids are unique UUIDs, so it globs across projects
    rather than re-deriving Claude's per-project transcript dir from cwd.

    Two modes: no args follows the transcript live (Ctrl-C to stop); `dump` prints
    the whole conversation once and exits — used to archive it when the seed ends."""
    script = f"""#!/usr/bin/env bash
# Inspect the Claude Code session driving this episode (read-only).
#   (no args)  follow the transcript live; Ctrl-C to stop watching.
#   dump       print the whole conversation once and exit (used to archive it).
SID={session_id}
MODE="${{1:-follow}}"

find_transcript() {{ ls "$HOME"/.claude/projects/*/"$SID".jsonl 2>/dev/null | head -1; }}

if [ "$MODE" = dump ]; then
  F=$(find_transcript)
  [ -n "$F" ] || {{ echo "no transcript for session $SID" >&2; exit 1; }}
  FOLLOW=""  # read once and exit
else
  # Wait for the transcript to appear (claude creates it at launch), then follow it.
  echo "waiting for session $SID transcript ..."
  while :; do
    F=$(find_transcript)
    [ -n "$F" ] && break
    sleep 0.5
  done
  echo "tailing $F"
  FOLLOW="-f"  # stream live until Ctrl-C
fi

if command -v jq >/dev/null 2>&1; then
  tail -n +1 $FOLLOW "$F" \\
    | jq -rc 'select(.type=="assistant")
              | (.timestamp[0:19] + "Z" | fromdateiso8601 | strflocaltime("%H:%M:%S")) as $t
              | .message.content[]
              | if .type=="text"      then "[\\($t)] [say]  " + .text
                elif .type=="tool_use" then "[\\($t)] [tool] " + .name + "  " + (.input|tostring)
                else empty end'
else
  echo "(install jq for readable output; showing raw transcript)"
  tail -n +1 $FOLLOW "$F"
fi
"""
    path = os.path.join(demo_folder, "inspect_claude.sh")
    try:
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)
    except Exception as e:
        warn(f"could not write {path}: {e}")
    return path


class ClaudeHarness(Harness):
    key = "claude"
    guide_filename = "CLAUDE.md"
    # `--effort` levels for the adaptive-reasoning models we run (Opus 4.8 / Fable 5).
    # `max` is Claude-only (unbounded; no codex analog); `minimal` is codex-only.
    supported_efforts = frozenset({"low", "medium", "high", "xhigh", "max"})

    def drive(self, ctx: SeedContext) -> DriveOutcome:
        # Pin the session id up front so the resume/inspect scripts (written before
        # launch) survive a crash and can be used to watch the run live.
        session_id = str(uuid.uuid4())
        resume_path = write_resume_script(
            ctx.demo_folder,
            session_id,
            ctx.base_port,
            ctx.model,
            ctx.task,
            ctx.seed,
            ctx.instruct_file,
        )
        inspect_path = write_inspect_script(ctx.demo_folder, session_id)
        log(f"[{ctx.tag}] watch this session live:           bash {inspect_path}")
        log(f"[{ctx.tag}] resume this session for debugging: bash {resume_path}")

        status = run_claude(
            ctx.prompt,
            ctx.model,
            ctx.log_path,
            ctx.timeout,
            allowed_tools(MCP_CONFIG_PATH),
            ctx.env,
            session_id,
            ctx.reasoning_effort,
            ctx.instruct_file,
        )

        # Archive the full conversation next to the episode (the same view
        # inspect_claude.sh streams live, dumped once) so a finished run stays
        # interpretable without Claude's per-project transcript dir. Best-effort.
        convo_path = os.path.join(ctx.demo_folder, "conversation.txt")
        try:
            with open(convo_path, "w") as cf:
                subprocess.run(
                    ["bash", inspect_path, "dump"], stdout=cf, stderr=subprocess.STDOUT, timeout=60
                )
            log(f"[{ctx.tag}] saved conversation -> {convo_path}")
        except Exception as e:
            warn(f"[{ctx.tag}] could not save conversation: {e}")

        return DriveOutcome(status, session_id)


register_harness(ClaudeHarness())  # self-register on import (see harness.get_harness)
