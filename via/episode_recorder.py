import json
import logging
import os
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ActMode(Enum):
    Waypoint = 0
    Dense = 1
    Terminate = 2
    Interpolate = 3


class EpisodeRecorder:
    """Records a single episode into one folder: streams tiled camera views to
    <episode_dir>/<video_name>.mp4.

    With record_npz=True it additionally retains the full per-step record (obs,
    action, mode, waypoint index, click, reward) and writes it to
    <episode_dir>/<video_name>.npz on end_episode(save=True). It defaults off
    because a session that only needs the mp4 would otherwise accumulate the full
    obs (~1.4 MB/step) in memory for nothing.

    The recorder owns only this one folder; the caller decides where it sits
    (e.g. the demoNNNNN/ layout in record_sim).
    """

    def __init__(self, episode_dir, video_name, record_npz=False, vis_dim=(320, 240)):
        os.makedirs(episode_dir, exist_ok=True)
        self.episode_dir = episode_dir
        self.video_name = video_name
        self.record_npz = record_npz
        self.vis_dim = vis_dim
        self._reset()

    def _reset(self):
        self.episode = []
        self.waypoint_idx = -1
        # Frames stream straight to the mp4 as they arrive (writer opened lazily
        # on the first frame, once the dims are known), so a long session never
        # buffers thousands of frames in memory.
        self._writer = None
        self._writer_path = None
        # mp4 frames written under each waypoint_idx, so the video can be sliced
        # back into per-waypoint rollout segments (see _write_segments). Just
        # counts -- no obs retained -- so it costs nothing. Frames before the
        # first waypoint (waypoint_idx < 0) are counted separately.
        self._wp_frame_counts: list[int] = []
        self._pre_wp_frames = 0

    def record(
        self,
        mode: ActMode,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        click_pos: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
    ):
        if mode == ActMode.Waypoint:
            self.waypoint_idx += 1

        # Retain the full step only when an npz is wanted; otherwise just stream
        # the frame and keep nothing.
        if self.record_npz:
            data = {
                "action": action,
                "mode": mode,
                "waypoint_idx": -1 if mode == ActMode.Dense else self.waypoint_idx,
                "click": click_pos,
                "obs": obs,
            }
            if reward is not None:
                data["reward"] = reward
            self.episode.append(data)

        views = [
            cv2.resize(v, self.vis_dim)
            for k, v in obs.items()
            if ("image" in k or "wrist" in k) and v.ndim == 3
        ]
        if views:
            self._write_frame(self._tile_views(views), mode)
            self._count_frame()

    def _count_frame(self):
        """Tally one written mp4 frame against the current waypoint_idx."""
        wi = self.waypoint_idx
        if wi < 0:
            self._pre_wp_frames += 1
            return
        while len(self._wp_frame_counts) <= wi:
            self._wp_frame_counts.append(0)
        self._wp_frame_counts[wi] += 1

    def _write_segments(self):
        """Dump <video_name>_segments.json: the [start, end) mp4 frame range of
        each waypoint's rollout, so a viewer can play the video per waypoint."""
        segments = []
        start = self._pre_wp_frames
        for i, count in enumerate(self._wp_frame_counts):
            segments.append({"waypoint": i, "start": start, "end": start + count})
            start += count
        path = os.path.join(self.episode_dir, f"{self.video_name}_segments.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "fps": 30,
                    "pre_waypoint_frames": self._pre_wp_frames,
                    "total_frames": start,
                    "segments": segments,
                },
                f,
                indent=2,
            )
        logger.info(f"saved {path}")

    @staticmethod
    def _tile_views(views):
        """Compose per-camera views into one frame: a 2x2 grid for exactly 4
        cameras, else a single horizontal row. All views share vis_dim, so the
        stacks line up cleanly."""
        if len(views) == 4:
            top = np.hstack(views[:2])
            bottom = np.hstack(views[2:])
            return np.vstack([top, bottom])
        return np.hstack(views)

    def _write_frame(self, vis, mode: ActMode):
        """Encode one tiled frame to the mp4. Dense steps get a green top bar.
        obs frames are RGB; the writer wants BGR, hence the channel swap."""
        if mode == ActMode.Dense:
            vis[:10, :, :] = (0, 255, 0)
        if self._writer is None:
            H, W = vis.shape[:2]
            self._writer_path = os.path.join(self.episode_dir, f"{self.video_name}.mp4")
            self._writer = cv2.VideoWriter(
                self._writer_path, cv2.VideoWriter_fourcc(*"avc1"), 30, (W, H)
            )
        self._writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def end_episode(self, save):
        # Frames were streamed as they arrived; just finalize the writer. A
        # writer exists only if at least one frame was recorded.
        if self._writer is not None:
            self._writer.release()
            if save:
                logger.info(f"saved {self._writer_path}")
                # Per-waypoint frame ranges for the mp4 just written.
                self._write_segments()
                # Loaded back via np.load(path, allow_pickle=True)["arr_0"].
                if self.record_npz and self.episode:
                    npz_path = os.path.join(self.episode_dir, f"{self.video_name}.npz")
                    np.savez_compressed(npz_path, self.episode)
                    logger.info(f"saved {npz_path}")
            else:
                # Discard: drop the partially written mp4.
                try:
                    os.remove(self._writer_path)
                except OSError:
                    pass
                logger.info("Episode discarded")
        else:
            logger.info("Episode discarded")

        self._reset()
