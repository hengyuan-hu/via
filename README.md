# VIA: Visual Interface Agent for Robot Control

VIA lets a general-purpose coding agent control a simulated robot through a browser-based
3D interface and MCP tools.

## Setup

We use uv to manage packages. Install [uv](https://docs.astral.sh/uv/) if you haven't.

### Clone and install dependencies

```shell
git clone https://github.com/hengyuan-hu/via.git
cd via
git submodule update --init --recursive           # LIBERO

uv venv --python 3.10 .venv
uv sync
.venv/bin/python -m playwright install chromium    # browser for the MCP server (one-time)
```

`uv sync` installs everything in `uv.lock`, including the subset of LIBERO's runtime deps
declared in `pyproject.toml` (`bddl`, `gym`, `future`, `easydict`, `cloudpickle`).

`torch`/`torchvision` come from PyTorch's own indexes (`[tool.uv.sources]`), so one lock works
on both platforms — CUDA 12.1 wheels on Linux, CPU/MPS on macOS. On a newer NVIDIA driver,
bump the `pytorch-cu121` index in `pyproject.toml` to `cu124`.

### LIBERO

LIBERO is vendored as a git submodule at `third_party/LIBERO` and put on `PYTHONPATH`;
`set_env.sh` handles this. On first import it writes `~/.libero/config.yaml` via an interactive prompt.
Seed it once:

```shell
source set_env.sh
echo N | python -c "import libero.libero"
```

## Usage

`source set_env.sh` once per shell (activates `.venv`, sets `PYTHONPATH` for the repo root and
LIBERO).

List the available simulation tasks or start an interactive recording:

```shell
python -m via.record_sim --list-tasks
python -m via.record_sim --task stack
```

List the configured evaluations or run one across its configured seeds:

```shell
python -m via.run_eval
python -m via.run_eval t-block-min-opus
```

Evaluation runs also require the CLI for the configured harness (`claude` or `codex`) to be
installed and authenticated. The MCP server (`via/mcp_server.py`) is launched automatically
from the same `.venv` via `.mcp.json`; you do not run it by hand.

## Repo layout notes

- `via/` — the active code: the flat `via.*` package (`record_sim`, `sim_env`, `run_eval`, `mcp_server`, and their helpers), plus `via/UI/` for the browser assets.
- `prompts/` — per-task instruction files referenced by the evaluations in `run_eval`.
- `pyproject.toml` / `uv.lock` — the single 3.10 environment (sim stack + MCP server).
- `set_env.sh` — activates `.venv` and sets `PYTHONPATH` (repo root + LIBERO submodule).
- `third_party/LIBERO` — LIBERO submodule (on `PYTHONPATH`, not pip-installed).
