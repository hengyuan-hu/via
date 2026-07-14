"""OpenAI Codex CLI harness for the eval driver — the Codex counterpart to
the built-in `claude -p` path (see claude_harness.py).

Kept as close as possible to the Claude harness so Claude-vs-Codex results stay
comparable. What is held IDENTICAL:
  * the task prompt (harness.build_prompt, only the guide filename differs);
  * the robot guide (CLAUDE.md) and the per-task instruct_file — delivered to
    Codex as an auto-loaded AGENTS.md instead of Claude's CLAUDE.md + system
    prompt, the same content through the analogous channel;
  * the robot MCP server, translated from the SAME .mcp.json Claude gets via
    --mcp-config (single source of truth for both harnesses);
  * verdict scraping and everything downstream in run_eval (harness-agnostic).

Differences forced by Codex's design, each mitigated to preserve the eval's
intent:
  * Codex's shell tool CANNOT be disabled (unlike Claude's DISALLOWED_TOOLS,
    which strips Read/Bash/etc. so the agent can't read ground-truth off disk).
    We approximate that isolation by (a) running Codex from an empty per-seed
    TEMP dir outside the repo/data tree, so its shell/apply_patch see no repo
    source and no episode artifacts; (b) disabling the two built-ins that could
    otherwise pull data in — `tools.view_image` and `tools.web_search` (Claude
    has no view_image and is denied WebFetch/WebSearch, so this matches); and
    (c) stripping SPHINX_BASE_PORT from the codex process env (the MCP server
    child gets it via the -c env table instead), so the shell can't trivially
    derive record_sim's localhost endpoints (/env.json, /success.json), which
    serve unauthenticated ground truth. Residual risk, accepted: a determined
    agent could still port-scan localhost or infer paths from tool-result text;
    closing that fully would need auth on record_sim's endpoints.
  * Codex can't pre-pin a session id (Claude uses --session-id), so it's parsed
    from the `--json` `thread.started` event after launch; the resume script is
    written then rather than up front.
  * Perception is the inline image the MCP `screenshot` tool returns; Codex
    0.142.5 renders inline MCP image content natively (validated), so view_image
    is unnecessary and safe to disable.
  * Codex exec auto-CANCELS MCP tool calls under the default approval flow, so we
    pass --dangerously-bypass-approvals-and-sandbox (the run is externally
    orchestrated, like Claude's, which is likewise unsandboxed and merely
    tool-restricted).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from via.agent_common import (
    CLAUDE_MD_PATH,
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


def _toml_str(s: str) -> str:
    """A TOML basic string for `s`. JSON string syntax is a compatible subset for
    the filesystem paths / port strings we emit (both quote with " and escape \\)."""
    return json.dumps(s)


def _looks_like_path(arg: str) -> bool:
    """Heuristic: a CLI arg that names a repo-relative file we should absolute-ify
    (so it still resolves once Codex runs from an isolated -C dir)."""
    return "/" in arg or arg.endswith(".py")


def codex_mcp_overrides(base_port: int) -> list[str]:
    """Translate run_eval's .mcp.json into Codex `-c mcp_servers.*` overrides.

    Same server(s), made absolute (Codex runs from an isolated cwd, so relative
    command/args wouldn't resolve) and given this seed's SPHINX_BASE_PORT in the
    server env — exactly what Claude's MCP server inherits from the seed env.
    Degrades to no flags on a missing/malformed .mcp.json (same policy as
    allowed_tools) rather than raising — this is also called post-run by
    write_codex_resume_script, where an exception would discard a completed
    episode's outcome."""
    cfg = get_json_file(MCP_CONFIG_PATH) or {}
    servers = cfg.get("mcpServers") or {}
    flags: list[str] = []
    for name, spec in servers.items():
        cmd = spec.get("command", "")
        cmd_abs = cmd if os.path.isabs(cmd) else os.path.join(REPO_ROOT, cmd)
        args_abs = []
        for a in spec.get("args") or []:
            args_abs.append(
                a if (os.path.isabs(a) or not _looks_like_path(a)) else os.path.join(REPO_ROOT, a)
            )
        env_items = dict(spec.get("env") or {})
        env_items["SPHINX_BASE_PORT"] = str(base_port)
        args_toml = "[" + ",".join(_toml_str(a) for a in args_abs) + "]"
        env_toml = "{" + ",".join(f"{k}={_toml_str(v)}" for k, v in env_items.items()) + "}"
        # startup/tool timeouts bumped: the MCP server attaches to a live Chromium
        # over CDP and the first screenshot round-trips through the browser, well
        # past Codex's 10s/60s defaults on a loaded machine.
        table = (
            f"mcp_servers.{name}={{command={_toml_str(cmd_abs)},"
            f"args={args_toml},env={env_toml},"
            f"startup_timeout_sec=90,tool_timeout_sec=180}}"
        )
        flags += ["-c", table]
    return flags


def write_codex_agents_md(demo_folder: str, instruct_file: str | None) -> str:
    """Create this seed's isolated Codex working dir and drop an AGENTS.md that
    Codex auto-loads (validated). AGENTS.md = CLAUDE.md + the per-task instruct_file
    (under an explicit task-instructions heading, since Claude receives that file
    through the higher-salience system-prompt channel and codex has no equivalent
    flag) — the same two documents, so the agent starts from identical guidance.

    The working dir is a fresh TEMP dir outside the repo/data tree: codex's shell
    can't be disabled, and a cwd inside demo_folder would hand it the live episode
    artifacts (screenshots/, env_cfg.yaml, verdict.json) via `ls ..` — ground truth
    Claude's DISALLOWED_TOOLS deny. A copy of AGENTS.md is also snapshotted into
    demo_folder so the run stays interpretable (the codex analog of
    backup_guide_file) and so the resume script's `-C {demo_folder}` auto-loads the
    exact guidance the episode saw. Returns the working-dir path (Codex's -C root)."""
    cwd = tempfile.mkdtemp(prefix="sphinx_codex_cwd_")
    parts = []
    try:
        with open(CLAUDE_MD_PATH) as f:
            parts.append(f.read())
    except OSError as e:
        warn(f"[codex] could not read CLAUDE.md for AGENTS.md: {e}")
    if instruct_file:
        try:
            with open(instruct_file) as f:
                parts.append("# Task-specific instructions\n\n" + f.read())
        except OSError as e:
            warn(f"[codex] could not read instruct_file for AGENTS.md: {e}")
    content = "\n\n".join(parts)
    with open(os.path.join(cwd, "AGENTS.md"), "w") as f:
        f.write(content)
    try:  # snapshot next to the episode (see docstring); best-effort
        os.makedirs(demo_folder, exist_ok=True)
        with open(os.path.join(demo_folder, "AGENTS.md"), "w") as f:
            f.write(content)
    except OSError as e:
        warn(f"[codex] could not snapshot AGENTS.md into {demo_folder}: {e}")
    return cwd


def session_id_from_log(log_path: str | Path) -> str | None:
    """Pull the Codex session (thread) id out of the --json event stream. Codex
    can't pre-pin it the way Claude accepts --session-id, so we read it back from
    the `thread.started` event to key the resume script and the usage record."""
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "thread.started":
                    return ev.get("thread_id") or (ev.get("item") or {}).get("thread_id")
    except FileNotFoundError:
        return None
    return None


def run_codex(
    prompt: str,
    model: str,
    reasoning_effort: str,
    log_path: str,
    timeout: float,
    env: dict,
    demo_folder: str,
    base_port: int,
    instruct_file: str | None,
) -> tuple[str, str | None]:
    """Drive one seed with a fresh headless `codex exec`. Returns (status, session_id)
    where status is 'ok' or 'timeout' (mirrors run_claude). The `--json` event stream
    is written to log_path (the transcript inspect/dump read); MCP-server stderr goes
    to log_path + '.stderr' so it can't corrupt the JSONL. The proc is tracked via the
    shared agent registry so a Ctrl-C tears it down like a claude proc.

    SPHINX_BASE_PORT is stripped from the codex process env: only the MCP server
    needs it, and it gets it via the -c env table (codex_mcp_overrides). Left in
    place, codex's un-disableable shell could derive record_sim's localhost
    endpoints (/success.json ground truth) from it — a channel Claude (Bash denied)
    doesn't have."""
    env = {k: v for k, v in env.items() if k != "SPHINX_BASE_PORT"}
    cwd = write_codex_agents_md(demo_folder, instruct_file)
    cmd = ["codex", "exec"]
    cmd += codex_mcp_overrides(base_port)
    # Match Claude's tool restriction as closely as Codex allows: no web reach and
    # no reading images off disk (perception is the inline MCP screenshot). The shell
    # tool can't be disabled; the isolated -C dir stands in for that.
    cmd += ["-c", "tools.web_search=false"]
    cmd += ["-c", "tools.view_image=false"]
    cmd += ["-c", f"model_reasoning_effort={_toml_str(reasoning_effort)}"]
    cmd += [
        "-m",
        model,
        # exec auto-cancels MCP tool calls under the default approval flow; bypass so
        # the robot tools actually run. The run is externally orchestrated (as Claude's
        # is), so dropping Codex's sandbox does not widen the trust boundary.
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",  # the isolated -C dir is not a git repo
        "-C",
        cwd,
        "--json",
        prompt,
    ]

    # No in-process retry on a transient provider failure (e.g. "at capacity"): codex
    # exits 0 with a turn.failed event and no end_episode, so record_sim writes no
    # verdict and the seed lands in `no_verdict` -- which run_eval re-drives on the next
    # invocation (its skip keys on verdict.json existing). That rerun path is both the
    # safety net and safer than retrying here: it restarts on a FRESH sim, whereas an
    # in-process retry would resume a new conversation on an already-manipulated scene
    # if the failure hit mid-episode. Observed rate ~1/152 seeds, and that one recovered
    # via rerun anyway.
    status = "ok"
    try:
        with open(log_path, "w") as log_f, open(log_path + ".stderr", "w") as err_f:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,  # else exec reads stdin and appends a <stdin> block
            )
            register_agent(proc)
            try:
                proc.wait(timeout=timeout)
                status = "ok"
            except subprocess.TimeoutExpired:
                terminate(proc)
                status = "timeout"
            except KeyboardInterrupt:
                terminate(proc)
                raise
            finally:
                unregister_agent(proc)
    finally:
        # Drop this seed's isolated -C temp dir on every exit (success, timeout,
        # Ctrl-C): it's disposable once codex has exited -- AGENTS.md is snapshotted
        # into demo_folder and the resume script uses `-C demo_folder`, not this.
        shutil.rmtree(cwd, ignore_errors=True)

    return status, session_id_from_log(log_path)


# --- Codex usage archiving: token counts and API-equivalent cost per episode. ---
# write_usage_record drops a codex_usage.json next to each episode when it ends
# (tools/codex_usage.py backfills the same records over finished runs). Token counts
# come from the per-request ~/.codex/sessions transcript when available (exact, and
# the only source that can price long-context requests correctly), else from the run
# log's cumulative `turn.completed` total. Costs are API-equivalent only (MODEL_RATES,
# standard tier); they exclude tier/credit/tax adjustments and never double-count
# reasoning tokens (already inside output_tokens).

PRICING_AS_OF = "2026-07-12"
LONG_CONTEXT_THRESHOLD = 272_000
MODEL_RATES: dict[str, dict[str, float]] = {
    "gpt-5.5": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
    "gpt-5.6-sol": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
}


def _canonical_model(model: str) -> str | None:
    for name in MODEL_RATES:
        if model == name or model.startswith(name + "-"):
            return name
    return None


def find_session_transcript(session_id: str | None) -> Path | None:
    if not session_id:
        return None
    root = Path.home() / ".codex" / "sessions"
    return next(root.rglob(f"*-{session_id}.jsonl"), None)


def _usage_from_completed_turn(log_path: Path) -> dict[str, int] | None:
    usage = None
    try:
        with log_path.open("rb") as stream:
            for raw in stream:
                if b'"turn.completed"' not in raw:
                    continue
                event = json.loads(raw)
                if event.get("type") == "turn.completed" and event.get("usage"):
                    usage = event["usage"]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return usage


def _session_usage(
    session_path: Path, rates: dict[str, float] | None
) -> tuple[dict[str, int] | None, list[dict[str, Any]], dict[str, float] | None, int,]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    saw_usage = False
    requests: list[dict[str, Any]] = []
    context_compactions = 0
    cost_breakdown = (
        {
            "uncached_input": 0.0,
            "cached_input": 0.0,
            "output": 0.0,
            "long_context_surcharge": 0.0,
            "total": 0.0,
        }
        if rates
        else None
    )
    try:
        with session_path.open("rb") as stream:
            for raw in stream:
                if b'"token_count"' not in raw or b'"last_token_usage"' not in raw:
                    continue
                event = json.loads(raw)
                payload = event.get("payload") or {}
                if event.get("type") != "event_msg" or payload.get("type") != "token_count":
                    continue
                usage = (payload.get("info") or {}).get("last_token_usage") or {}
                if not usage:
                    continue
                if not any(int(usage.get(key, 0) or 0) for key in totals):
                    context_compactions += int(bool(usage.get("total_tokens")))
                    continue
                saw_usage = True
                for key in totals:
                    totals[key] += int(usage.get(key, 0) or 0)
                request = {key: int(usage.get(key, 0) or 0) for key in totals}
                request["uncached_input_tokens"] = (
                    request["input_tokens"] - request["cached_input_tokens"]
                )
                request["total_tokens"] = request["input_tokens"] + request["output_tokens"]
                if rates:
                    request_cost = _request_cost(request, rates)
                    request["long_context_pricing"] = request_cost.pop("long_context")
                    request["api_cost_estimate_usd"] = request_cost["total"]
                    assert cost_breakdown is not None
                    for key in cost_breakdown:
                        cost_breakdown[key] += request_cost[key]
                requests.append(request)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None, [], None, 0
    if cost_breakdown:
        cost_breakdown = {key: round(value, 8) for key, value in cost_breakdown.items()}
    return (totals if saw_usage else None), requests, cost_breakdown, context_compactions


def _request_cost(
    usage: dict[str, int], rates: dict[str, float], long_context_pricing: bool = True
) -> dict[str, Any]:
    # `long_context_pricing` gates OpenAI's >LONG_CONTEXT_THRESHOLD surcharge, which
    # is a PER-REQUEST property. Callers must pass False for a cumulative
    # (turn.completed) total: its input sum spans many requests and almost always
    # clears the threshold, so applying the surcharge there overcharges (a single
    # request may never have been long-context). Only the per-request session path
    # can price it, so it keeps the default True.
    input_tokens = usage.get("input_tokens", 0)
    cached_tokens = usage.get("cached_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    uncached_cost = (input_tokens - cached_tokens) * rates["input"] / 1_000_000
    cached_cost = cached_tokens * rates["cached_input"] / 1_000_000
    output_cost = output_tokens * rates["output"] / 1_000_000
    long_context = long_context_pricing and input_tokens > LONG_CONTEXT_THRESHOLD
    surcharge = uncached_cost + cached_cost + output_cost * 0.5 if long_context else 0.0
    return {
        "uncached_input": round(uncached_cost, 8),
        "cached_input": round(cached_cost, 8),
        "output": round(output_cost, 8),
        "long_context_surcharge": round(surcharge, 8),
        "total": round(uncached_cost + cached_cost + output_cost + surcharge, 8),
        "long_context": long_context,
    }


def build_usage_record(
    log_path: str | Path,
    model: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    log_path = Path(log_path)
    session_id = session_id or session_id_from_log(log_path)
    canonical_model = _canonical_model(model)
    rates = MODEL_RATES.get(canonical_model or "")
    session_path = find_session_transcript(session_id)
    usage = None
    requests: list[dict[str, Any]] | None = None
    cost_breakdown = None
    context_compactions = None
    source = None

    if session_path:
        usage, requests, cost_breakdown, context_compactions = _session_usage(session_path, rates)
        if usage:
            source = "codex_session_transcript"
    if usage is None:
        usage = _usage_from_completed_turn(log_path)
        if usage:
            source = "turn.completed"
            # Cumulative total: no per-request split, so the long-context surcharge
            # can't be attributed and is omitted (see _request_cost).
            cost_breakdown = (
                _request_cost(usage, rates, long_context_pricing=False) if rates else None
            )
            if cost_breakdown:
                cost_breakdown.pop("long_context")

    notes = [
        "API-equivalent estimate; not a historical Codex subscription charge.",
        "Reasoning output tokens are included in output_tokens and are not charged twice.",
    ]
    if source == "turn.completed":
        notes.append(
            "Long-context surcharge omitted: it is per-request and cannot be "
            "reconstructed from a cumulative total, so cost may be underestimated "
            "for runs with long-context requests."
        )
    notes.append(
        "Excludes service-tier, negotiated-price, credit, tax, and unreported "
        "cache-write differences."
    )
    return {
        "schema_version": 2,
        "provider": "openai",
        "model": model,
        "session_id": session_id,
        "token_source": source,
        "token_usage": usage,
        "request_usage": requests,
        "context_compactions": context_compactions,
        "api_cost_estimate": {
            "currency": "USD",
            "amount": cost_breakdown.get("total") if cost_breakdown else None,
            "cost_breakdown_usd": cost_breakdown,
            "pricing_as_of": PRICING_AS_OF,
            "service_tier": "standard",
            "rates_per_million_tokens": rates,
            "long_context_threshold_input_tokens": LONG_CONTEXT_THRESHOLD,
            "long_context_requests": (
                sum(bool(request.get("long_context_pricing")) for request in requests)
                if requests is not None
                else None
            ),
            "notes": notes,
        },
    }


def write_usage_record(
    demo_folder: str | Path,
    log_path: str | Path,
    model: str,
    session_id: str | None = None,
) -> Path:
    path = Path(demo_folder) / "codex_usage.json"
    record = build_usage_record(log_path, model, session_id)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def write_codex_inspect_script(demo_folder: str, log_path: str, model: str) -> str:
    """Drop <demo_folder>/inspect_codex.sh: a read-only view of this seed's Codex
    run, streaming agent messages and tool calls. Codex's transcript is the --json
    stream we already tee to log_path, so (unlike the Claude inspect script) this
    just follows that file — no per-project transcript dir to locate.

    No args follows live (Ctrl-C to stop); `dump` prints once and exits; `usage`
    prints the token counters reported by Codex; `cost` converts those counters
    using the standard API prices for the run's model, baked in at generation
    time from MODEL_RATES (the single price table)."""
    q = shlex.quote
    rates = MODEL_RATES.get(_canonical_model(model) or "")
    if rates:
        rates_sh = (
            f"INPUT_RATE={rates['input']:g}; CACHED_RATE={rates['cached_input']:g}; "
            f"OUTPUT_RATE={rates['output']:g}"
        )
    else:
        rates_sh = 'echo "no API price table configured for model $MODEL" >&2; exit 1'
    script = f"""#!/usr/bin/env bash
# Inspect the Codex session driving this episode (read-only).
#   (no args)  follow the run live; Ctrl-C to stop watching.
#   dump       print the whole conversation once and exit (used to archive it).
#   usage      print the token usage reported by Codex and exit.
#   cost       estimate the equivalent standard API cost and exit.
LOG={q(log_path)}
USAGE_FILE={q(os.path.join(demo_folder, "codex_usage.json"))}
MODEL={q(model)}
MODE="${{1:-follow}}"

if [ "$MODE" = usage ] || [ "$MODE" = cost ]; then
  [ -f "$LOG" ] || {{ echo "no codex log at $LOG" >&2; exit 1; }}
  if command -v jq >/dev/null 2>&1; then
    if [ -f "$USAGE_FILE" ]; then
      if ! jq -e '.token_usage != null' "$USAGE_FILE" >/dev/null; then
        echo "no recoverable token usage archived in $USAGE_FILE" >&2
        exit 1
      fi
      if [ "$MODE" = usage ]; then
        jq -r '"source: " + (.token_source // "unknown")
               + "\ninput_tokens: " + ((.token_usage.input_tokens // 0) | tostring)
               + "\ncached_input_tokens: " + ((.token_usage.cached_input_tokens // 0) | tostring)
               + "\noutput_tokens: " + ((.token_usage.output_tokens // 0) | tostring)
               + "\nreasoning_output_tokens: " + ((.token_usage.reasoning_output_tokens // 0) | tostring)' \
          "$USAGE_FILE"
      else
        jq -r '"model: " + .model
               + "\npricing_as_of: " + .api_cost_estimate.pricing_as_of
               + "\napi_equivalent_cost_usd: $" + (.api_cost_estimate.amount | tostring)
               + "\nThis is an API-equivalent estimate, not a Codex subscription charge."' \
          "$USAGE_FILE"
      fi
      exit
    fi
    SID=$(jq -r 'select(.type=="thread.started") | .thread_id' "$LOG" | head -1)
    if [ -n "$SID" ]; then
      SESSION=$(find "$HOME/.codex/sessions" -type f -name "*-$SID.jsonl" -print -quit 2>/dev/null)
    fi
    USAGE=$(jq -c 'select(.type=="turn.completed" and .usage != null) | .usage' "$LOG" | tail -1)
    SOURCE="$LOG"
    if [ -z "$USAGE" ]; then
      if [ -n "${{SESSION:-}}" ]; then
        USAGE=$(jq -c 'select(.type=="event_msg"
                              and .payload.type=="token_count"
                              and .payload.info.total_token_usage != null)
                       | .payload.info.total_token_usage' "$SESSION" | tail -1)
        SOURCE="$SESSION"
      fi
    fi
    [ -n "$USAGE" ] || {{
      echo "no usage found in the run log or Codex session transcript" >&2
      exit 1
    }}
    if [ "$MODE" = usage ]; then
      jq -nr --argjson usage "$USAGE" --arg source "$SOURCE" '
        "source: \\($source)\n"
        + "input_tokens: \\($usage.input_tokens // 0)\n"
        + "cached_input_tokens: \\($usage.cached_input_tokens // 0)\n"
        + "output_tokens: \\($usage.output_tokens // 0)\n"
        + "reasoning_output_tokens: \\($usage.reasoning_output_tokens // 0)"'
    else
      {rates_sh}
      if [ -n "${{SESSION:-}}" ]; then
        COST=$(jq -nr --argjson input_rate "$INPUT_RATE" \
          --argjson cached_rate "$CACHED_RATE" --argjson output_rate "$OUTPUT_RATE" '
          reduce inputs as $event (0;
            if ($event.type=="event_msg"
                and $event.payload.type=="token_count"
                and $event.payload.info.last_token_usage != null)
            then $event.payload.info.last_token_usage as $u
              | (if ($u.input_tokens // 0) > {LONG_CONTEXT_THRESHOLD} then 2 else 1 end) as $input_multiplier
              | (if $input_multiplier == 2 then 1.5 else 1 end) as $output_multiplier
              | . + (((($u.input_tokens // 0) - ($u.cached_input_tokens // 0)) * $input_rate
                      + ($u.cached_input_tokens // 0) * $cached_rate) * $input_multiplier
                     + ($u.output_tokens // 0) * $output_rate * $output_multiplier) / 1000000
            else . end)' "$SESSION")
      else
        COST=$(jq -nr --argjson usage "$USAGE" --argjson input_rate "$INPUT_RATE" \
          --argjson cached_rate "$CACHED_RATE" --argjson output_rate "$OUTPUT_RATE" '
          ($usage.input_tokens // 0) as $input
          | ($usage.cached_input_tokens // 0) as $cached
          | ($usage.output_tokens // 0) as $output
          | (($input - $cached) * $input_rate
             + $cached * $cached_rate
             + $output * $output_rate) / 1000000')
      fi
      printf 'model: %s\n' "$MODEL"
      echo "pricing_as_of: {PRICING_AS_OF}"
      printf 'standard_api_rates_per_1m: input=$%s, cached=$%s, output=$%s\n' \
        "$INPUT_RATE" "$CACHED_RATE" "$OUTPUT_RATE"
      printf 'api_equivalent_cost_usd: $%.6f\n' "$COST"
      echo "This is an API-equivalent estimate, not a Codex subscription charge."
    fi
  else
    echo "install jq to extract usage from $LOG" >&2
    exit 1
  fi
  exit
elif [ "$MODE" = dump ]; then
  [ -f "$LOG" ] || {{ echo "no codex log at $LOG" >&2; exit 1; }}
  FOLLOW=""
else
  echo "waiting for codex log $LOG ..."
  while [ ! -f "$LOG" ]; do sleep 0.5; done
  echo "tailing $LOG"
  FOLLOW="-f"
fi

if command -v jq >/dev/null 2>&1; then
  tail -n +1 $FOLLOW "$LOG" \\
    | jq -rc 'select(.type=="item.completed") | .item
              | if .type=="agent_message"  then "[say]  " + .text
                elif .type=="reasoning"     then "[think] " + (.text // "")
                elif .type=="mcp_tool_call" then "[tool] " + .server + "/" + .tool
                     + "  " + (.arguments|tostring) + "  -> " + (.status // "")
                elif .type=="command_execution" then "[shell] " + (.command // "")
                     + "  -> " + (.status // "") + " (exit " + ((.exit_code // "?")|tostring) + ")"
                else empty end'
else
  echo "(install jq for readable output; showing raw JSONL)"
  tail -n +1 $FOLLOW "$LOG"
fi
"""
    path = os.path.join(demo_folder, "inspect_codex.sh")
    try:
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)
    except OSError as e:
        warn(f"could not write {path}: {e}")
    return path


def write_codex_resume_script(
    demo_folder: str,
    session_id: str | None,
    base_port: int,
    model: str,
    task: str,
    seed: int,
) -> str:
    """Drop <demo_folder>/resume_codex.sh: the command to resume this episode's Codex
    session for debugging. Mirrors write_resume_script for Claude — an always-works
    interactive continue line (`codex resume`, the analog of `claude --resume`; the
    headless `codex exec resume` would block reading a prompt from stdin), plus a
    commented recipe to relaunch the sim on this port/seed and resume WITH the robot
    tools attached. Both resume forms use `-C {demo_folder}`, whose snapshotted
    AGENTS.md (see write_codex_agents_md) auto-loads the exact guidance the episode
    saw. Written after the run, once the session id is known (Codex can't pre-pin
    it)."""
    q = shlex.quote
    if not session_id:
        body = "# No Codex session id was captured (run may have failed before start).\n"
    else:
        # Each -c flag is shell-quoted: the TOML inline tables contain braces,
        # commas, and double quotes that bash would otherwise brace-expand and
        # quote-strip into garbage.
        mcp_flags = " ".join(q(f) for f in codex_mcp_overrides(base_port))
        body = f"""# Inspect / continue the conversation interactively (no robot tools; always works):
codex resume {q(session_id)} --model {q(model)} -C {q(demo_folder)} "$@"

# To actually re-drive the robot, first start a sim on this port in another shell
# (set_env.sh activates the venv + sets PYTHONPATH and OPENBLAS_NUM_THREADS):
#   source set_env.sh && SPHINX_BASE_PORT={base_port} \\
#     python -m via.record_sim --task {q(task)} \\
#     --demo_folder {q(demo_folder)} --seed {seed} --render 0
# then resume WITH the MCP robot tools attached (same tool restrictions as the
# eval; -C is the episode folder — its AGENTS.md auto-loads, but unlike the eval's
# isolated temp cwd this exposes the episode artifacts, which is fine for debugging):
#   codex resume {q(session_id)} {mcp_flags} \\
#     -c tools.web_search=false -c tools.view_image=false --model {q(model)} \\
#     --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \\
#     -C {q(demo_folder)} "continue the task"
"""
    script = f"""#!/usr/bin/env bash
# Resume the Codex session that drove this episode (for debugging).
# Run from anywhere; it cd's to the repo root.
cd {q(REPO_ROOT)} || exit 1

{body}"""
    path = os.path.join(demo_folder, "resume_codex.sh")
    try:
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)
    except OSError as e:
        warn(f"could not write {path}: {e}")
    return path


def dump_codex_conversation(log_path: str, out_path: str) -> None:
    """Parse the captured --json event stream into a readable transcript next to the
    episode (the same view inspect_codex.sh streams live). Best-effort; mirrors the
    conversation.txt Claude gets from `inspect_claude.sh dump`.

    Renders `command_execution` items too: Codex's shell can't be disabled, so a run
    may include shell commands (unlike Claude, whose Bash is denied). Surfacing them
    here — rather than only the MCP tool calls — keeps conversation.txt an honest
    record for auditing (the codex_audit validator scans the raw log for the same)."""
    lines: list[str] = []
    try:
        with open(log_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "item.completed":
                    continue
                item = ev.get("item") or {}
                t = item.get("type")
                if t == "agent_message":
                    lines.append("[say]  " + (item.get("text") or ""))
                elif t == "reasoning":
                    lines.append("[think] " + (item.get("text") or ""))
                elif t == "mcp_tool_call":
                    lines.append(
                        f"[tool] {item.get('server')}/{item.get('tool')}  "
                        f"{json.dumps(item.get('arguments'))}  -> {item.get('status')}"
                    )
                elif t == "command_execution":
                    lines.append(
                        f"[shell] {item.get('command')}  "
                        f"-> {item.get('status')} (exit {item.get('exit_code')})"
                    )
    except FileNotFoundError:
        lines = ["(no codex log captured)"]
    try:
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        warn(f"could not write {out_path}: {e}")


class CodexHarness(Harness):
    key = "codex"
    guide_filename = "AGENTS.md"
    # `model_reasoning_effort` levels codex exec accepts. `minimal` is codex-only;
    # `max` is Claude-only. `xhigh` is the shared top level for a fair A/B.
    supported_efforts = frozenset({"minimal", "low", "medium", "high", "xhigh"})

    def drive(self, ctx: SeedContext) -> DriveOutcome:
        # The inspect script just follows the --json log (written below), so it can be
        # dropped now; the resume script needs the session id, so it waits for the run.
        inspect_path = write_codex_inspect_script(ctx.demo_folder, ctx.log_path, ctx.model)
        log(f"[{ctx.tag}] watch this session live:           bash {inspect_path}")

        status, session_id = run_codex(
            ctx.prompt,
            ctx.model,
            ctx.reasoning_effort,
            ctx.log_path,
            ctx.timeout,
            ctx.env,
            ctx.demo_folder,
            ctx.base_port,
            ctx.instruct_file,
        )

        try:
            usage_path = write_usage_record(ctx.demo_folder, ctx.log_path, ctx.model, session_id)
            log(f"[{ctx.tag}] saved Codex usage -> {usage_path}")
        except OSError as e:
            warn(f"could not archive Codex usage for {ctx.tag}: {e}")

        resume_path = write_codex_resume_script(
            ctx.demo_folder, session_id, ctx.base_port, ctx.model, ctx.task, ctx.seed
        )
        log(f"[{ctx.tag}] resume this session for debugging: bash {resume_path}")

        convo_path = os.path.join(ctx.demo_folder, "conversation.txt")
        dump_codex_conversation(ctx.log_path, convo_path)
        log(f"[{ctx.tag}] saved conversation -> {convo_path}")

        return DriveOutcome(status, session_id)


register_harness(CodexHarness())  # self-register on import (see harness.get_harness)
