"""A scatter task built on the Stack backbone: 7 rainbow blocks on the table.

Reuses everything the `Stack` env provides (table arena, Panda mount, placement
sampler, reset plumbing) and only swaps the two stacking cubes for seven equally
sized rainbow blocks scattered randomly across the tabletop. Reward and success
detection are intentionally neutralized -- this task is for free-form
manipulation / data collection, not a scored objective.

Importing this module registers `RainbowScatter` with robosuite (the EnvMeta
metaclass auto-registers every env subclass by class name), so
`robosuite.make(env_name="RainbowScatter")` works once it is imported. sim_env.py
imports it and exposes it under the task key "rainbow".
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.transform_utils import convert_quat


# Block half-extents: a 4cm cube.
BLOCK_HALF_SIZE = 0.02

# Blocks scatter within +/- this in x and y (table frame). This square is carved
# into a GRID_N x GRID_N lattice of equal cells; each reset drops one block into
# each of a random subset of cells.
SCATTER_HALF_RANGE = 0.20
GRID_N = 3  # GRID_N**2 = 9 candidate cells, 7 of which get a block each reset

# The seven rainbow colors as RGBA, in spectral order.
RAINBOW = [
    ("red", [1.0, 0.0, 0.0, 1.0]),
    ("orange", [1.0, 0.65, 0.0, 1.0]),
    ("yellow", [1.0, 1.0, 0.0, 1.0]),
    ("green", [0.0, 1.0, 0.0, 1.0]),
    ("blue", [0.0, 0.0, 1.0, 1.0]),
    ("indigo", [0.2, 0.0, 0.6, 1.0]),
    ("violet", [0.8, 0.35, 1.0, 1.0]),
]


class RainbowScatter(Stack):
    """Seven rainbow blocks scattered randomly on the Stack tabletop.

    Inherits the Stack constructor and reset logic verbatim; only the model
    (objects), references, observables, reward and success are overridden.
    """

    def _load_model(self):
        # Skip Stack._load_model (it builds the two stacking cubes); reproduce the
        # shared arena/robot setup, then add the seven rainbow blocks instead.
        super(Stack, self)._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # seven equally sized rainbow blocks (4cm cubes)
        size = [BLOCK_HALF_SIZE, BLOCK_HALF_SIZE, BLOCK_HALF_SIZE]
        self.blocks = [
            BoxObject(
                name=f"block_{color_name}",
                size_min=size,
                size_max=size,
                rgba=rgba,
            )
            for color_name, rgba in RAINBOW
        ]

        # Scatter the blocks one-per-cell over a random subset of a grid (see
        # _make_placement_initializer). Rebuilt every reset so the cell choice
        # re-randomizes; this initial build keeps _load_model self-contained.
        self.placement_initializer = self._make_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.blocks,
        )

    def _make_placement_initializer(self):
        """Place the blocks at the centers of a GRID_N x GRID_N grid.

        Split [-SCATTER_HALF_RANGE, SCATTER_HALF_RANGE]^2 into GRID_N**2 equal
        cells, pick len(self.blocks) of them at random (7 of 9), and drop one block
        at the center of each chosen cell (still a random yaw). Returned as a
        composite sampler so every reset can rebuild it with a fresh cell choice.
        """
        edges = np.linspace(-SCATTER_HALF_RANGE, SCATTER_HALF_RANGE, GRID_N + 1)
        centers = (edges[:-1] + edges[1:]) / 2  # cell-center coordinate per axis
        cells = [(centers[i], centers[j]) for i in range(GRID_N) for j in range(GRID_N)]
        chosen = np.random.choice(len(cells), size=len(self.blocks), replace=False)

        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for block, cell_idx in zip(self.blocks, chosen):
            cx, cy = cells[cell_idx]
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{block.name}_sampler",
                    mujoco_objects=block,
                    x_range=[cx, cx],
                    y_range=[cy, cy],
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )
            )
        return sampler

    def _reset_internal(self):
        # Re-pick which cells are occupied before Stack samples + places the blocks.
        self.placement_initializer = self._make_placement_initializer()
        super()._reset_internal()

    def _setup_references(self):
        # Skip Stack._setup_references (it looks up cubeA/cubeB); set up the
        # shared references then one body id per rainbow block.
        super(Stack, self)._setup_references()
        self.block_body_ids = [
            self.sim.model.body_name2id(block.root_body) for block in self.blocks
        ]

    def _setup_observables(self):
        # Skip Stack._setup_observables (cubeA/cubeB sensors); add a pos/quat
        # observable per rainbow block on top of the shared robot observables.
        observables = super(Stack, self)._setup_observables()

        if self.use_object_obs:
            modality = "object"

            def _make_sensors(idx, name):
                @sensor(modality=modality)
                def block_pos(obs_cache, body_id=self.block_body_ids[idx]):
                    return np.array(self.sim.data.body_xpos[body_id])

                block_pos.__name__ = f"{name}_pos"

                @sensor(modality=modality)
                def block_quat(obs_cache, body_id=self.block_body_ids[idx]):
                    return convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

                block_quat.__name__ = f"{name}_quat"
                return [block_pos, block_quat]

            for idx, block in enumerate(self.blocks):
                for s in _make_sensors(idx, block.name):
                    observables[s.__name__] = Observable(
                        name=s.__name__,
                        sensor=s,
                        sampling_rate=self.control_freq,
                    )

        return observables

    # Reward and success are intentionally disabled for this free-form task.
    def reward(self, action=None):
        return 0.0

    def staged_rewards(self):
        return 0.0, 0.0, 0.0

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        # Skip Stack.visualize (it colors the gripper site by distance to cubeA).
        super(Stack, self).visualize(vis_settings=vis_settings)


def sample_arrangements_figure(
    seeds=range(1, 11),
    camera="agentview",
    out="data/figs/rainbow_arrangements.png",
):
    """Tile one initial arrangement per seed into a single 3x4 figure.

    Reuses one env, reseeding np.random before each reset so each panel is the
    reproducible layout for that seed (see how the grid cells and locations are
    sampled in _make_placement_initializer). 10 seeds fill 10 of the 12 cells.
    """
    import os

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")  # save-only, no GUI backend needed
    import matplotlib.pyplot as plt

    # Lazy import avoids a circular import: sim_env imports this module at load.
    from via.sim_env import SimEnv

    seeds = list(seeds)
    env = SimEnv("rainbow", on_screen_render=False, verbose=False)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, seed in zip(axes, seeds):
        np.random.seed(seed)
        env.reset()
        ax.imshow(env.observe_camera()[f"{camera}_image"])
        ax.set_title(f"seed {seed}", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    logger.info(f"saved {out} ({camera}) for seeds {seeds[0]}..{seeds[-1]}")
    return out


if __name__ == "__main__":
    from via.utils import setup_logging

    setup_logging()
    sample_arrangements_figure()
