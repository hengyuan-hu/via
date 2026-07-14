"""Standalone builder for the demo review page.

Scans a data folder for per-demo verdict sidecars
(`<folder>/<task_slug>/demoNNNNN.json`, written by record_sim) and writes a
static `<folder>/videos.html` that lists every demo with its verdict and embeds
the camera mp4 and (if present) the UI webm. Decoupled from record_sim so it can
be run on any data folder after recording:

    python -m via.build_video_index data/dev1
"""

from __future__ import annotations

import argparse
import glob
import html as _html
import json
import logging
import os
import re

from via.utils import setup_logging

logger = logging.getLogger(__name__)


def _natkey(s: str) -> list:
    """Natural sort key so seed2 sorts before seed10 (not lexicographically)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def scan_verdicts(folder: str) -> list[dict]:
    """Load every per-demo verdict sidecar under <folder>, natural-sorted by path.

    Current layout (record_sim): <folder>/<task_slug>/<demo_folder>/verdict.json.
    Legacy layout: <folder>/<task_slug>/demoNNNNN.json. Both are picked up. Each
    sidecar is the dict record_sim writes per demo; entries missing a task or a
    camera video path are skipped (this drops the eval_results_*.json aggregate).
    The resolved sidecar path is attached as `_json_path` for the html links."""
    out: list[dict] = []
    if not os.path.isdir(folder):
        return out
    paths = glob.glob(os.path.join(folder, "*", "*", "verdict.json")) + glob.glob(
        os.path.join(folder, "*", "demo*.json")
    )
    seen: set[str] = set()
    for js in sorted(paths, key=_natkey):
        if js in seen:
            continue
        seen.add(js)
        try:
            with open(js) as f:
                v = json.load(f)
        except Exception:
            continue
        if isinstance(v, dict) and "task" in v and (v.get("cameras_mp4") or v.get("mp4")):
            v["_json_path"] = js
            out.append(v)
    return out


def _ui_webm_for(v: dict) -> str | None:
    """Resolve a demo's UI recording. Prefer an explicit `ui_webm` field; else
    fall back to a `.webm` sibling of the camera mp4 (demoNNNNN.webm)."""
    cand = v.get("ui_webm")
    if cand and os.path.exists(cand):
        return cand
    mp4 = v.get("cameras_mp4") or v.get("mp4", "")
    if mp4:
        sibling = mp4[:-4] + ".webm" if mp4.endswith(".mp4") else mp4 + ".webm"
        if os.path.exists(sibling):
            return sibling
    return None


def build_video_index(folder: str) -> str:
    """Scan <folder> and (re)write <folder>/videos.html. Returns the path."""
    index_path = os.path.join(folder, "videos.html")
    os.makedirs(folder, exist_ok=True)
    verdicts = scan_verdicts(folder)
    n_done = len(verdicts)
    n_success = sum(1 for v in verdicts if v.get("success"))
    rate_pct = (100.0 * n_success / n_done) if n_done else 0.0

    rows: list[str] = []
    for i, v in enumerate(verdicts, 1):
        if v.get("success"):
            badge_css, badge_text = "ok", "SUCCESS"
        elif v.get("timed_out"):
            badge_css, badge_text = "timeout", "TIMEOUT"
        else:
            badge_css, badge_text = "fail", "FAIL"
        rel_mp4 = os.path.relpath(v.get("cameras_mp4") or v.get("mp4", ""), folder)
        rel_json = os.path.relpath(v["_json_path"], folder)

        ui_webm = _ui_webm_for(v)
        if ui_webm:
            rel_ui = os.path.relpath(ui_webm, folder)
            ui_cell = (
                f'<video class="ui-video" controls autoplay muted loop preload="auto" '
                f'src="{_html.escape(rel_ui)}"></video>'
            )
        else:
            ui_cell = '<span class="muted">(no UI video)</span>'
        cam_cell = (
            f'<video class="cam-video" controls autoplay muted loop preload="auto" '
            f'src="{_html.escape(rel_mp4)}"></video>'
        )

        rows.append(
            f"""
        <tr>
          <td class="num">{i}</td>
          <td class="task">
            <div class="tid">{_html.escape(v['task'])}</div>
            <div class="lang">{_html.escape(v.get('language', ''))}</div>
            <div class="meta"><a href="{_html.escape(rel_json)}">verdict json</a></div>
          </td>
          <td class="verdict"><span class="badge {badge_css}">{badge_text}</span></td>
          <td class="steps">
            {v.get('sim_steps_used', '?')}/{v.get('sim_steps_budget', '?')}
            <div class="meta">{v.get('elapsed_seconds', '?')}s wall</div>
          </td>
          <td class="vid ui">{ui_cell}</td>
          <td class="vid cam">{cam_cell}</td>
        </tr>"""
        )

    title = _html.escape(os.path.basename(os.path.abspath(folder)))
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Demo videos — {title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background:#ffffff; color:#111827; margin:0; padding:24px; }}
  h1 {{ margin:0 0 6px 0; font-size:22px; }}
  .summary {{ color:#4b5563; margin-bottom:18px; }}
  .rate {{ font-weight:600; }}
  .controls {{ display:flex; align-items:center; gap:10px; margin:12px 0 18px 0;
               padding:10px 14px; background:#f3f4f6; border:1px solid #e5e7eb;
               border-radius:6px; font-size:13px; color:#374151; }}
  .controls label {{ font-weight:600; }}
  .controls select {{ font-size:13px; padding:3px 6px; border-radius:4px;
                      border:1px solid #d1d5db; background:#ffffff; color:#111827; }}
  table.demos {{ border-collapse:separate; border-spacing:0; width:100%; }}
  table.demos thead th {{ position:sticky; top:0; background:#ffffff; color:#4b5563;
                          font-weight:500; font-size:12px; text-align:left; padding:8px 10px;
                          border-bottom:1px solid #e5e7eb; text-transform:uppercase;
                          letter-spacing:0.06em; }}
  table.demos td {{ background:#ffffff; border-bottom:1px solid #e5e7eb; padding:10px;
                    vertical-align:top; }}
  td.num {{ width:40px; color:#9ca3af; font-variant-numeric:tabular-nums; }}
  td.task {{ width:240px; }}
  td.task .tid {{ font-weight:600; }}
  td.task .lang {{ color:#374151; font-size:13px; margin-top:2px; }}
  td.verdict {{ width:100px; }}
  td.steps {{ width:100px; color:#374151; font-variant-numeric:tabular-nums; }}
  td.vid.ui  {{ width:640px; }}
  td.vid.cam {{ width:420px; }}
  td.vid video {{ width:100%; border-radius:5px; background:#000; }}
  .meta {{ color:#9ca3af; font-size:11px; margin-top:3px; }}
  .meta a {{ color:#6b7280; }}
  .muted {{ color:#9ca3af; }}
  .badge {{ font-size:11px; padding:2px 8px; border-radius:999px; font-weight:600;
            letter-spacing:0.04em; }}
  .badge.ok       {{ background:#d1fae5; color:#065f46; }}
  .badge.fail     {{ background:#fee2e2; color:#991b1b; }}
  .badge.timeout  {{ background:#fef3c7; color:#92400e; }}
</style>
</head>
<body>
  <h1>Demo videos — {title}</h1>
  <p class="summary">
    <span class="rate">{n_success}/{n_done} success ({rate_pct:.1f}%)</span>
  </p>
  <div class="controls">
    <label for="ui-speed">UI video speed:</label>
    <select id="ui-speed">
      <option value="0.5">0.5×</option>
      <option value="1">1×</option>
      <option value="1.5">1.5×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
      <option value="8">8×</option>
      <option value="10">10×</option>
      <option value="16" selected>16×</option>
    </select>
  </div>
  <table class="demos">
    <thead>
      <tr><th>#</th><th>task</th><th>verdict</th><th>sim steps</th>
          <th>UI</th><th>cameras</th></tr>
    </thead>
    <tbody>
{''.join(rows)}
    </tbody>
  </table>
<script>
  (function() {{
    var sel = document.getElementById('ui-speed');
    if (!sel) return;
    function applyRate() {{
      var r = parseFloat(sel.value);
      document.querySelectorAll('video.ui-video').forEach(function(v) {{
        v.playbackRate = r;
      }});
    }}
    sel.addEventListener('change', applyRate);
    document.querySelectorAll('video.ui-video').forEach(function(v) {{
      v.addEventListener('loadedmetadata', function() {{
        v.playbackRate = parseFloat(sel.value);
      }});
    }});
  }})();
</script>
</body>
</html>"""
    with open(index_path, "w") as f:
        f.write(html_doc)
    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Build <folder>/videos.html from the demo verdict sidecars "
        "under a record_sim data folder."
    )
    parser.add_argument("folder", help="Data folder root (e.g. data/dev1).")
    args = parser.parse_args()
    setup_logging()
    path = build_video_index(args.folder)
    n = len(scan_verdicts(args.folder))
    logger.info(f"[build_video_index] wrote {path} ({n} demo(s))")


if __name__ == "__main__":
    main()
