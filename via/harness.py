"""The agent-harness abstraction the eval driver dispatches through.

A Harness knows how to drive ONE seed's agent to completion and leave behind the
debug artifacts (watch/resume scripts, conversation.txt). run_eval's run_seed is
harness-agnostic: it brings up the sim, builds a SeedContext, and calls
`get_harness(exp.harness).drive(ctx)`. Adding a new agent (e.g. a Gemini CLI) is
one new module implementing Harness + one register_harness() call — no edits to
run_seed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


def build_prompt(language: str, guide: str = "CLAUDE.md") -> str:
    # `guide` names the auto-loaded instructions file so the prompt is otherwise
    # identical across harnesses: Claude reads CLAUDE.md, Codex reads the same
    # content delivered as AGENTS.md (see codex_harness.write_codex_agents_md).
    task_line = f"Task: {language}\n\n" if language else ""
    return (
        "Control the simulated robot arm to complete the following task.\n\n"
        f"{task_line}"
        f"Make sure you read {guide} before operating. It contains useful introduction and tips for the UI.\n"
        "Once you think you have completed the task, call end_episode to save the episode and end the server.\n"
        "Act fully autonomously; do not ask questions."
    )


@dataclass
class SeedContext:
    """Everything a harness needs to drive one seed. Built by run_eval.run_seed
    once the sim is up; harness-agnostic (no claude/codex specifics)."""

    tag: str  # e.g. "seed3", for log lines
    prompt: str  # already built with this harness's guide filename
    model: str  # interpreted per harness ("opus"/"fable" | "gpt-5.5")
    reasoning_effort: str  # always set; codex --> model_reasoning_effort, claude --> --effort
    log_path: str  # per-seed agent log / transcript
    timeout: float  # episode wall-clock cap
    env: dict  # carries SPHINX_BASE_PORT for this seed's stack
    demo_folder: str  # where the episode + debug artifacts are saved
    task: str
    seed: int
    base_port: int
    instruct_file: str | None  # absolute path to per-task instructions, or None


@dataclass
class DriveOutcome:
    """What run_seed needs back from a harness run."""

    status: str  # "ok" | "timeout"
    session_id: str | None


class Harness(ABC):
    """One agent backend. Subclasses set `key` and `guide_filename` and implement
    drive(). drive() owns ALL agent-specific work for a seed: writing the debug
    scripts, logging the watch/resume hints, running the agent to completion, and
    archiving conversation.txt."""

    key: str = ""  # registry key; matches Experiment.harness
    guide_filename: str = "CLAUDE.md"  # the auto-loaded guide the prompt references
    # Reasoning-effort levels this harness's CLI accepts (SeedContext.reasoning_effort
    # is validated against this at Experiment construction). Empty = no validation.
    supported_efforts: frozenset[str] = frozenset()

    @abstractmethod
    def drive(self, ctx: SeedContext) -> DriveOutcome:
        ...


_REGISTRY: dict[str, Harness] = {}


def register_harness(h: Harness) -> None:
    _REGISTRY[h.key] = h


def get_harness(key: str) -> Harness:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown harness {key!r}; registered: {sorted(_REGISTRY)}")
