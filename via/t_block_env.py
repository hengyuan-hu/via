"""T-block task (two green blocks on a red base), built on the Stack backbone.

Three 5cm cubes start in a left-right line on the Stack tabletop -- a red base in
the middle with one green block on each side (random per-side gaps). The goal is
to place BOTH green blocks on top of the red
block at the SAME level -- side by side, each resting on the red top face -- and
NOT stacked green-on-green-on-red. Because each green cube is as wide as the red
top face, the two greens necessarily overhang opposite edges and lean together,
which is what makes the arrangement delicate.

Reuses everything Stack provides (table arena, Panda mount, reset plumbing) and
only swaps the two stacking cubes for one red + two green blocks, plus a reward /
success detector for the "both green on red, same level" goal.

Importing this module registers `TBlock` with robosuite (the EnvMeta
metaclass auto-registers every env subclass by class name), so
`robosuite.make(env_name="TBlock")` works once it is imported. sim_env.py
imports it and exposes it under the task key "t_block".
"""

import numpy as np

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


# Block half-extents: a 5cm cube (all three blocks share this size).
BLOCK_HALF_SIZE = 0.025
BLOCK_SIZE = 2 * BLOCK_HALF_SIZE  # 0.05, the full edge length

# Initial layout: the three cubes in a left-right line -- red in the middle, one
# green on each side. GAP_MIN/GAP_MAX bound the empty space (m) between the faces
# of adjacent cubes, sampled independently per side each reset (center-to-center
# spacing is BLOCK_SIZE + gap). The whole line is also shifted by +/- LINE_JITTER
# in x and y each reset so the absolute positions vary by seed.
GAP_MIN = 0.05
GAP_MAX = 0.08
LINE_JITTER = 0.03

# Goal geometry. A green cube resting on the red top face has its center one full
# block height above the red center; both greens must sit at that level.
ON_RED_DZ = BLOCK_SIZE  # 0.05: expected green_z - red_z when resting on red
DZ_TOL = 0.012  # vertical slack on ON_RED_DZ (allows a little tilt)
XY_TOL = 0.045  # horizontal: green center within this of red center (footprints overlap)
SAME_LEVEL_TOL = 0.015  # the two greens must agree in z to this much (else it's a stack)

# Min Panda jaw width (m) for the gripper to count as "released". The fingers
# open to ~0.078; a clamped 5cm block holds them at ~0.05, so 0.06 sits cleanly
# between -- success cannot be claimed while still pinching/holding a green block.
GRIPPER_OPEN_MIN = 0.06

# Max block speeds for the arrangement to count as "settled" (at rest): linear
# (m/s) and angular (rad/s). The two-blocks-leaning goal is delicate, so success
# only registers once the blocks have actually stopped -- never mid-topple.
LIN_VEL_MAX = 0.01
ANG_VEL_MAX = 0.10

# The goal condition must hold for this many consecutive env steps before
# success fires -- a single frame can be a fluke (e.g. a zero-velocity instant
# mid-wobble slipping past the at-rest thresholds).
SUCCESS_HOLD_STEPS = 3

# Colors.
RED = [0.1, 0.3, 1.0, 1.0]  # bright blue base -- contrasts with both red and green
GREEN = [0.0, 0.4, 0.0, 1.0]  # dark green


class TBlock(Stack):
    """One red + two green 5cm cubes; goal is both greens on red at the same level.

    Inherits the Stack constructor and reset logic; the model (objects),
    references, observables, reward and success are overridden.
    """

    _success_streak = 0  # consecutive env steps _success_now() has held

    def _load_model(self):
        # Skip Stack._load_model (it builds the two stacking cubes); reproduce the
        # shared arena/robot setup, then add the red + two green blocks instead.
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

        # one red base + two identical green blocks, all 5cm cubes
        size = [BLOCK_HALF_SIZE, BLOCK_HALF_SIZE, BLOCK_HALF_SIZE]
        self.red_block = BoxObject(name="block_red", size_min=size, size_max=size, rgba=RED)
        self.green_blocks = [
            BoxObject(name=f"block_green{i}", size_min=size, size_max=size, rgba=GREEN)
            for i in (1, 2)
        ]
        self.blocks = [self.red_block, *self.green_blocks]

        # Lay the blocks out in a line (see _make_placement_initializer); rebuilt
        # every reset (see _reset_internal) so the gaps/offset re-randomize.
        self.placement_initializer = self._make_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.blocks,
        )

    def _make_placement_initializer(self):
        """Lay the three cubes out in a line: green, red, green (red in the middle).

        The line runs along the table y-axis (left-right in the default view).
        Each green sits one cube-width plus a random GAP_MIN..GAP_MAX gap from the
        red, sampled independently per side, and the whole line is shifted by a
        small random offset so absolute positions vary by seed. Every cube is
        axis-aligned (no yaw). Returned as a composite sampler (one fixed cell per
        cube) so every reset rebuilds it with fresh gaps/offset.
        """
        # small random shift of the whole line so seeds are not identical
        dx = np.random.uniform(-LINE_JITTER, LINE_JITTER)
        dy = np.random.uniform(-LINE_JITTER, LINE_JITTER)
        # center-to-center offset to each green: one cube width + a random face gap
        left_off = BLOCK_SIZE + np.random.uniform(GAP_MIN, GAP_MAX)
        right_off = BLOCK_SIZE + np.random.uniform(GAP_MIN, GAP_MAX)
        # red in the middle, one green on each side along y
        layout = {
            self.red_block.name: (dx, dy),
            self.green_blocks[0].name: (dx, dy + left_off),  # left
            self.green_blocks[1].name: (dx, dy - right_off),  # right
        }

        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for block in self.blocks:
            cx, cy = layout[block.name]
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{block.name}_sampler",
                    mujoco_objects=block,
                    x_range=[cx, cx],
                    y_range=[cy, cy],
                    rotation=0.0,  # axis-aligned: no yaw
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )
            )
        return sampler

    def _reset_internal(self):
        # Re-randomize the line layout (gaps + offset) before Stack places blocks.
        self.placement_initializer = self._make_placement_initializer()
        self._success_streak = 0
        super()._reset_internal()

    def _post_action(self, action):
        # advance the success dwell once per env step, before reward/done
        self._success_streak = self._success_streak + 1 if self._success_now() else 0
        return super()._post_action(action)

    def _setup_references(self):
        # Skip Stack._setup_references (it looks up cubeA/cubeB); set up the
        # shared references then one body id per block.
        super(Stack, self)._setup_references()
        self.red_body_id = self.sim.model.body_name2id(self.red_block.root_body)
        self.green_body_ids = [self.sim.model.body_name2id(b.root_body) for b in self.green_blocks]

    def _setup_observables(self):
        # Skip Stack._setup_observables (cubeA/cubeB sensors); add a pos/quat
        # observable per block on top of the shared robot observables.
        observables = super(Stack, self)._setup_observables()

        if self.use_object_obs:
            modality = "object"
            body_ids = [self.red_body_id, *self.green_body_ids]

            def _make_sensors(name, body_id):
                @sensor(modality=modality)
                def block_pos(obs_cache, body_id=body_id):
                    return np.array(self.sim.data.body_xpos[body_id])

                block_pos.__name__ = f"{name}_pos"

                @sensor(modality=modality)
                def block_quat(obs_cache, body_id=body_id):
                    return convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

                block_quat.__name__ = f"{name}_quat"
                return [block_pos, block_quat]

            for block, body_id in zip(self.blocks, body_ids):
                for s in _make_sensors(block.name, body_id):
                    observables[s.__name__] = Observable(
                        name=s.__name__,
                        sensor=s,
                        sampling_rate=self.control_freq,
                    )

        return observables

    # ------------------------------------------------------------------ reward

    def _green_on_red(self, green_pos, red_pos):
        """True if a green block is resting on the red top face (one level up).

        Requires the green center to be ~one block height above the red center
        (rules out sitting on the table, dz~=0, and double-stacking, dz~=0.08) and
        horizontally over the red footprint (so it is on the red, not beside it).
        """
        dz = green_pos[2] - red_pos[2]
        height_ok = abs(dz - ON_RED_DZ) < DZ_TOL
        horiz = np.linalg.norm(np.array(green_pos[:2]) - np.array(red_pos[:2]))
        return height_ok and horiz < XY_TOL

    def _robot_touching_blocks(self):
        """True while any robot or gripper geom touches any block.

        A two-finger _check_grasp misses load-bearing cheats: a single open
        fingertip, the finger side, the palm, or the arm can brace an otherwise
        collapsing arrangement. Any robot-block contact voids "hands off".
        """
        robot = self.robots[0]
        geoms = robot.gripper.contact_geoms + robot.robot_model.contact_geoms
        return any(self.check_contact(geoms, b) for b in self.blocks)

    def _gripper_jaw_width(self):
        """Current Panda jaw width (m): the span between the two finger joints."""
        idx = self.robots[0]._ref_gripper_joint_pos_indexes
        qpos = self.sim.data.qpos
        return abs(qpos[idx[0]] - qpos[idx[1]])

    def _at_rest(self):
        """True if every block has settled (linear & angular speed near zero).

        Reads each block's free-joint velocity (6-vector: linear then angular);
        keeps success from firing while the leaning arrangement is still in
        motion and about to collapse.
        """
        for obj in self.blocks:
            qvel = self.sim.data.get_joint_qvel(obj.joints[0])
            if np.linalg.norm(qvel[:3]) > LIN_VEL_MAX:
                return False
            if np.linalg.norm(qvel[3:]) > ANG_VEL_MAX:
                return False
        return True

    def staged_rewards(self):
        """(r_reach, r_place): light shaping toward placing both greens on red.

        - r_reach in [0, 0.25]: closeness of the gripper to the nearest green not
          yet on the red (0.25 once both are placed).
        - r_place in {0, 0.5, 1.0}: 0.5 per green correctly resting on the red.
        """
        red_pos = self.sim.data.body_xpos[self.red_body_id]
        gripper_site = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        green_pos = [self.sim.data.body_xpos[bid] for bid in self.green_body_ids]

        placed = [self._green_on_red(gp, red_pos) for gp in green_pos]

        unplaced_dists = [
            np.linalg.norm(gripper_site - gp) for gp, ok in zip(green_pos, placed) if not ok
        ]
        r_reach = (1 - np.tanh(10.0 * min(unplaced_dists))) * 0.25 if unplaced_dists else 0.25
        r_place = 0.5 * sum(placed)
        return r_reach, r_place

    def reward(self, action=None):
        """Sparse 2.0 on success; optional [0, ~1.25] shaping otherwise.

        Normalized/scaled by reward_scale / 2.0, matching Stack, so the max score
        equals reward_scale.
        """
        if self._check_success():
            reward = 2.0
        elif self.reward_shaping:
            r_reach, r_place = self.staged_rewards()
            reward = r_reach + r_place
        else:
            reward = 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0
        return reward

    def _success_now(self):
        """Both green blocks resting on the red block at the same level, hands off.

        True iff each green is on the red top face (height + footprint overlap),
        the two greens agree in height (so it is a side-by-side arrangement, not a
        green-on-green stack), the robot has let go (no robot-block contact AND
        jaws open past GRIPPER_OPEN_MIN), and the whole arrangement has settled
        (all blocks at rest). The contact veto stops any load-bearing cheat --
        grasping, pinning, or bracing with an open finger or the arm; the at-rest
        check stops success registering on a momentary pose while the leaning
        blocks are still toppling.
        """
        red_pos = self.sim.data.body_xpos[self.red_body_id]
        green_pos = [self.sim.data.body_xpos[bid] for bid in self.green_body_ids]

        both_on_red = all(self._green_on_red(gp, red_pos) for gp in green_pos)
        same_level = abs(green_pos[0][2] - green_pos[1][2]) < SAME_LEVEL_TOL
        hands_off = (
            not self._robot_touching_blocks() and self._gripper_jaw_width() > GRIPPER_OPEN_MIN
        )
        return both_on_red and same_level and hands_off and self._at_rest()

    def _check_success(self):
        """_success_now() sustained for the last SUCCESS_HOLD_STEPS env steps.

        The streak advances once per step in _post_action, so a state staged
        between steps (sim-state hacks) or a one-frame fluke never counts.
        """
        return self._success_streak >= SUCCESS_HOLD_STEPS

    def visualize(self, vis_settings):
        # Skip Stack.visualize (it colors the gripper site by distance to cubeA).
        super(Stack, self).visualize(vis_settings=vis_settings)
