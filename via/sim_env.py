import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import robosuite
from robosuite.utils import camera_utils
from robosuite.utils.mjcf_utils import new_element, array_to_string
from robosuite.utils.observables import Observable, sensor
from via.utils import deproject, setup_logging
from via.episode_recorder import EpisodeRecorder, ActMode
from via.robot_utils import (
    Proprio,
    WaypointReach,
    WaypointReachPI,
    WaypointReachConfig,
)

# Importing registers the RainbowScatter / TBlock envs with robosuite
# (see rainbow_env.py / t_block_env.py).
from via import rainbow_env  # noqa: F401
from via import t_block_env  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class PointCloudConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    # z lower bound to use instead of z_min when cropping out the table
    z_min_crop_table: float
    # Robot-frame z of the tabletop; the UI offsets by this to map its z=0 table
    # to/from world z.
    z_offset: float


RobomimicPC = PointCloudConfig(
    x_min=-0.3,
    x_max=0.3,
    y_min=-0.3,
    y_max=0.3,
    z_min=0.75,
    z_max=1.1,
    z_min_crop_table=0.81,
    z_offset=0.8,
)


# LIBERO's mounted Panda table sits higher than the robomimic tabletop. Tuned
# against libero_spatial/0 (eef at z≈1.17 at reset, tabletop ≈ 0.90). z_offset
# matches the per-task UI z-offset record_sim used for LIBERO (0.9). Every LIBERO
# suite (spatial, goal, object, 10, ...) shares the same Panda-mount table arena
# and agentview pose, so this one config is used for all of them.
LiberoPC = PointCloudConfig(
    x_min=-0.53,
    x_max=0.4,
    y_min=-0.4,
    y_max=0.4,
    z_min=0.85,
    z_max=1.30,
    z_min_crop_table=0.90,
    z_offset=0.9,
)


@dataclass
class CameraSpec:
    """A fixed world-frame camera to inject into the robosuite model.

    pos is the world-frame (x, y, z); quat is the MuJoCo (w, x, y, z) orientation
    -- same convention as the arena cameras in robosuite's table_arena.xml. The
    name must be unique and must not contain "eye_in_hand" unless the view should
    be excluded from the merged point cloud (get_point_cloud filters on that).
    """

    name: str
    pos: list[float]
    quat: list[float]


@dataclass
class SimEnvConfig:
    name: str
    max_len: int
    cameras: list[str]
    idle_step: int = 10
    image_size: int = 224
    robots: list[str] = field(default_factory=lambda: ["Panda"])
    waypoint_max_step: int = 50
    waypoint_reach: WaypointReachConfig = field(default_factory=WaypointReachConfig)
    crop_table: int = 1
    # When set, observe() includes the flattened MuJoCo sim state under "sim_state"
    # so demos can be deterministically replayed / reset (used by record_sim).
    record_sim_state: int = 0
    # LIBERO only: index into the task's ~50 saved init states applied at reset
    # for repeatable starts (-1 skips and uses the BDDL default). Ignored by
    # robomimic tasks.
    init_state_index: int = 0


# Robomimic presets supported by _create_robomimic_env (single source of truth
# for the dispatch error message and the --list output).
ROBOMIMIC_TASKS = ("lift", "square", "stack", "rainbow", "t_block")


# Extra fixed views to add: agentview orbited by these azimuths (deg, CCW seen
# from above). -x sits on the viewer's left, +x on the right.
_DEFAULT_ORBIT_ANGLES = [("leftview", -45.0), ("rightview", 45.0)]


def _orbit_agentview(
    ref_pos: np.ndarray,
    ref_quat_wxyz: np.ndarray,
    table_z: float,
    named_angles: list[tuple[str, float]],
) -> list[CameraSpec]:
    """Cameras identical to a reference view but orbited about the vertical axis
    through its look-at point: same radius, height and look-down angle, azimuth
    shifted by each angle (degrees, CCW seen from above). The reference is angle 0.

    ref_pos / ref_quat_wxyz are the reference camera's world-frame pose -- read
    live from the model so the orbit matches whatever arena is loaded (robomimic
    and LIBERO place agentview differently). table_z is the tabletop height the
    view stays centered on.

    Preview/tune with: python envs/sim_env.py --task square --test cams
    """
    rot = R.from_quat([*ref_quat_wxyz[1:], ref_quat_wxyz[0]])  # wxyz -> xyzw
    # the reference looks along its local -z; the point where that ray crosses the
    # tabletop is the orbit center, so every orbited view stays aimed at it.
    fwd = rot.apply([0.0, 0.0, -1.0])
    lookat = ref_pos + (table_z - ref_pos[2]) / fwd[2] * fwd

    specs = []
    for name, deg in named_angles:
        rz = R.from_euler("z", deg, degrees=True)
        pos = lookat + rz.apply(ref_pos - lookat)
        x, y, z, w = (rz * rot).as_quat()
        specs.append(CameraSpec(name=name, pos=pos.tolist(), quat=[w, x, y, z]))
    return specs


def _is_libero_task(task: str) -> bool:
    """True iff task looks like '<libero_suite>/<id>', e.g. 'libero_spatial/0'."""
    if "/" not in task:
        return False
    return task.split("/", 1)[0].startswith("libero_")


def _resolve_task_key(task_key: str):
    """Resolve '<suite>/<task_id>' to (benchmark, task_id, task, bddl_path)."""
    # Imported lazily so robomimic-only use never imports LIBERO.
    from libero.libero import benchmark, get_libero_path

    suite_name, idx_str = task_key.split("/", 1)
    try:
        task_id = int(idx_str)
    except ValueError as e:
        raise ValueError(f"task id must be an int, got {idx_str!r} (key {task_key!r})") from e

    bm_dict = benchmark.get_benchmark_dict()
    if suite_name not in bm_dict:
        raise ValueError(f"unknown LIBERO suite {suite_name!r}; available: {list(bm_dict)}")
    bm = bm_dict[suite_name]()
    if not (0 <= task_id < bm.n_tasks):
        raise ValueError(f"task_id {task_id} out of range for {suite_name} (n_tasks={bm.n_tasks})")
    task = bm.get_task(task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    if not os.path.exists(bddl_path):
        raise FileNotFoundError(f"BDDL file missing: {bddl_path}")
    return bm, task_id, task, bddl_path


class SimEnv:
    def __init__(
        self,
        task: str,
        on_screen_render=False,
        verbose=False,
        record_sim_state=False,
        controller_mode="p",
    ):
        self.task = task
        self.verbose = verbose
        # Which controller move_to uses to reach a waypoint: the P-only
        # WaypointReach ("p", default) or its PI variant ("pi") for tighter
        # convergence.
        assert controller_mode in ("p", "pi"), f"invalid controller_mode: {controller_mode}"
        self.waypoint_reach_cls = WaypointReachPI if controller_mode == "pi" else WaypointReach
        # LIBERO tasks set these; robomimic tasks leave them at the defaults.
        self.language_instruction = ""
        self._init_states = None
        self._task_id = -1
        # _create_env sets self.cfg, self.env, self.env_wrapper, self.ctrl_config,
        # self.pc_config and self.has_renderer (plus the LIBERO fields above).
        self._create_env(task, on_screen_render)
        # opt-in: have observe() emit the flattened sim state for replayable demos.
        self.cfg.record_sim_state = int(record_sim_state)
        assert self.ctrl_config["control_delta"]
        self.action_dim: int = self.env.action_dim

        # LIBERO tasks carry a language instruction; robomimic tasks have none.
        logger.info(f"[SimEnv] task {task!r} language instruction: {self.language_instruction!r}")

        if verbose:
            logger.info(f"ctrl_config:\n{pprint.pformat(self.ctrl_config)}")
            logger.info(f"control_freq: {self.env.control_freq}")
            logger.info(f"action dim: {self.action_dim}")

        # bookkeeping
        self.obs = {}
        self.curr_gripper_open = 1
        # max jaw width (m), measured in reset() with the gripper fully open;
        # used to normalize the raw width in observe_proprio()
        self.gripper_max_width = 0.0
        self.reward = -1
        self.terminal = False
        self.success = False
        self.num_step = 0

    def _create_env(self, task: str, on_screen_render: bool):
        """Dispatch to the right env builder based on the task name.

        Both builders set self.cfg, self.env, self.env_wrapper, self.ctrl_config,
        self.pc_config and self.has_renderer. Robomimic tasks ("lift", "square",
        "stack") build a robosuite env directly; LIBERO tasks
        ("<suite>/<id>", e.g. "libero_spatial/0" or "libero_goal/2") load a BDDL
        through OffScreenRenderEnv.
        """
        if _is_libero_task(task):
            self._create_libero_env(task)
        else:
            self._create_robomimic_env(task, on_screen_render)

    def _create_robomimic_env(self, task: str, on_screen_render: bool):
        """Create a robosuite env for a robomimic preset ("lift", "square", "stack")."""
        self.ctrl_config = robosuite.load_controller_config(default_controller="OSC_POSE")

        if task == "lift":
            cfg = SimEnvConfig(
                name="Lift",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=1000,
                crop_table=0,
            )
        elif task == "square":
            cfg = SimEnvConfig(
                name="NutAssemblySquare",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=1000,
                crop_table=1,
            )
        elif task == "stack":
            cfg = SimEnvConfig(
                name="Stack",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=1000,
                crop_table=0,
            )
        elif task == "rainbow":
            # 7 rainbow blocks scattered on the table; same backbone as Stack.
            cfg = SimEnvConfig(
                name="RainbowScatter",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=5000,
                crop_table=0,
            )
        elif task == "t_block":
            # 1 red + 2 blue blocks; goal is both blues on red at the same level.
            cfg = SimEnvConfig(
                name="TBlock",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=5000,
                crop_table=0,
            )
        else:
            raise ValueError(
                f"unsupported task {task!r}; use one of: {', '.join(ROBOMIMIC_TASKS)}, "
                f"or a LIBERO key like libero_spatial/0"
            )

        env = robosuite.make(
            env_name=cfg.name,
            robots=cfg.robots,
            controller_configs=self.ctrl_config,
            has_renderer=on_screen_render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=False,
            camera_names=cfg.cameras,
            camera_heights=cfg.image_size,
            camera_widths=cfg.image_size,
            camera_depths=True,
            # +200 so simulator does not timeout, we control timeout in this class
            horizon=cfg.max_len + cfg.idle_step + 200,
            render_camera="agentview",
        )
        self.cfg = cfg
        self.env = env
        # For robomimic, step()/reset() and the underlying sim are the same env.
        self.env_wrapper = env
        self.pc_config = RobomimicPC
        self.has_renderer = on_screen_render

        # Inject any extra fixed cameras before the first reset() renders them.
        self._add_custom_cameras()

    def _add_custom_cameras(self):
        """Inject the default extra cameras into the robosuite model and register them.

        robosuite only renders a camera that exists as a <camera> element in the
        MuJoCo model, and camera_names is consumed by make() before any custom
        camera is in the arena -- so these can't go through make(). Instead we:

          1. register an xml_processor that appends each <camera> to <worldbody>;
             robosuite re-applies it inside _initialize_sim() on every hard reset,
          2. extend the parallel camera-config lists that _setup_observables()
             zips over so a {name}_image / {name}_depth observable is created,
          3. append the name to cfg.cameras so observe_camera()/get_point_cloud()
             pick the view up (they just iterate cfg.cameras).

        The next reset() rebuilds the model with the cameras present.
        """
        # Read agentview's live world-frame pose so the orbit is correct for
        # whatever arena is loaded; works for both robomimic and LIBERO.
        sim = self.env.sim
        cid = sim.model.camera_name2id("agentview")
        specs = _orbit_agentview(
            np.array(sim.model.cam_pos[cid]),
            np.array(sim.model.cam_quat[cid]),
            self.pc_config.z_offset,
            _DEFAULT_ORBIT_ANGLES,
        )

        def _processor(xml: str) -> str:
            root = ET.fromstring(xml)
            worldbody = root.find("worldbody")
            assert worldbody is not None, "model xml has no <worldbody>"
            for spec in specs:
                if worldbody.find(f"camera[@name='{spec.name}']") is None:
                    worldbody.append(
                        new_element(
                            tag="camera",
                            name=spec.name,
                            pos=array_to_string(spec.pos),
                            quat=array_to_string(spec.quat),
                        )
                    )
            return ET.tostring(root, encoding="utf8").decode("utf8")

        self.env.set_xml_processor(_processor)

        size = self.cfg.image_size
        for spec in specs:
            if spec.name in self.env.camera_names:
                continue
            self.env.camera_names.append(spec.name)
            self.env.camera_widths.append(size)
            self.env.camera_heights.append(size)
            self.env.camera_depths.append(True)
            self.env.camera_segmentations.append(None)
            self.cfg.cameras.append(spec.name)

            # reset() only *modifies* observables it already knows about (it never
            # adds new keys), and an observable's real camera sensor renders on
            # construction -- which would fail now because the camera isn't in the
            # sim yet. So seed self._observables with no-op placeholders; the next
            # reset() injects the camera and swaps in the real rendering sensors.
            for suffix, shape in (("image", (size, size, 3)), ("depth", (size, size, 1))):

                @sensor(modality="image")
                def _placeholder(obs_cache, _shape=shape):
                    return np.zeros(_shape)

                self.env.add_observable(
                    Observable(
                        name=f"{spec.name}_{suffix}",
                        sensor=_placeholder,
                        sampling_rate=self.env.control_freq,
                    )
                )

    def _create_libero_env(self, task: str):
        """Create a LIBERO env for a "<suite>/<id>" task key.

        Any LIBERO suite is supported (libero_spatial, libero_goal, libero_10,
        ...); _resolve_task_key validates the suite/id against the benchmark dict
        and they all share the same Panda-mount table arena and PC config.
        OffScreenRenderEnv wraps the robosuite env: step()/reset() go through the
        wrapper (self.env_wrapper) while .sim, _check_success() and
        _get_observations() live on the underlying env (self.env).
        """
        # Imported lazily so robomimic-only use never imports LIBERO.
        from libero.libero.envs import OffScreenRenderEnv

        bm, task_id, task_obj, bddl_path = _resolve_task_key(task)
        self.language_instruction = task_obj.language
        self._task_id = task_id

        cfg = SimEnvConfig(
            name=task,
            cameras=["agentview", "robot0_eye_in_hand"],
            max_len=2000,
            crop_table=0,
            init_state_index=0,  # actual seed/init state is taken care of later
        )

        env_wrapper = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            robots=cfg.robots,
            camera_heights=cfg.image_size,
            camera_widths=cfg.image_size,
            camera_depths=True,
            camera_names=cfg.cameras,
            # +200: headroom so the env's own horizon is never hit (the real
            # budget is enforced in apply_action); else robosuite raises and
            # crashes mid-waypoint before a verdict is written.
            horizon=cfg.max_len + cfg.idle_step + 200,
        )

        self.cfg = cfg
        self.env_wrapper = env_wrapper
        self.env = env_wrapper.env
        # LIBERO hard-codes the OSC_POSE controller; read it back for move_to().
        self.ctrl_config = self.env.robots[0].controller_config
        self.pc_config = LiberoPC
        # OffScreenRenderEnv has no on-screen viewer.
        self.has_renderer = False

        # Inject the extra fixed cameras before the first reset() renders them.
        # OffScreenRenderEnv.reset() goes through the same robosuite hard reset,
        # so the xml_processor + observable plumbing works identically here.
        self._add_custom_cameras()

        # Each task ships ~50 saved init states; reset() applies one
        # deterministically by index for repeatable starts.
        try:
            self._init_states = bm.get_task_init_states(task_id)
        except Exception as e:
            if self.verbose:
                logger.warning(f"[libero] no init states for {task}: {e}")
            self._init_states = None

    def reset(self, render: bool = False):
        self.obs = self.env_wrapper.reset()

        # LIBERO: apply a deterministic saved init state for a repeatable start.
        if self._init_states is not None and self.cfg.init_state_index >= 0:
            idx = self.cfg.init_state_index % len(self._init_states)
            self.env_wrapper.set_init_state(self._init_states[idx])
            self.obs = self.env._get_observations(force_update=True)

        # provisional max width so observe_proprio() works during the idle
        # steps; the gripper starts open so this is already close to the max
        self.gripper_max_width = self._gripper_width()

        # run some idle steps with the gripper commanded open so that
        # everything is static
        for _ in range(self.cfg.idle_step):
            self.apply_action(np.zeros(3), np.zeros(3), 1)
            if render and self.has_renderer:
                self.env.render()

        # the gripper is now fully open and settled; record its width as the
        # max width used to normalize gripper_open in observe_proprio()
        self.gripper_max_width = self._gripper_width()

        # orient the gripper straight down (top-down) without moving it.
        # canonical down pose is ee_euler = [-pi, 0, 0]
        self.curr_gripper_open = 1
        proprio = self.observe_proprio()
        self.move_to(proprio.eef_pos, np.array([-np.pi, 0.0, 0.0]), 1, render=render)

        if self.verbose:
            proprio = self.observe_proprio()
            pos = proprio.eef_pos
            rpy = np.degrees(proprio.eef_euler)
            logger.info(f"[reset]: gripper open: {self.observe()['gripper_open']}")
            logger.info(f"[reset]: ee_pos: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m")
            logger.info(f"[reset]: ee_rpy: [{rpy[0]:+6.1f}, {rpy[1]:+6.1f}, {rpy[2]:+6.1f}] deg")

        self.curr_gripper_open = 1
        self.reward = -1
        self.terminal = False
        self.success = False
        self.num_step = 0

    def reset_to_sim_state(self, sim_state: np.ndarray):
        self.reset()
        self.env.sim.set_state_from_flattened(sim_state)
        self.env.sim.forward()
        self.obs = self.env._get_observations(force_update=True)

    def get_camera_intrinsics(self, camera: str) -> np.ndarray:
        matrix = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim, camera, self.cfg.image_size, self.cfg.image_size
        )
        if "eye_in_hand" in camera:
            # observe_camera() rotates the wrist feed 180, as if the physical
            # camera were mounted rotated 180 about its optical axis. Match that
            # in K by reflecting the principal point (focal length and skew are
            # unchanged); the rotation itself lives in the extrinsics below.
            n = self.cfg.image_size
            matrix = matrix.copy()
            matrix[0, 2] = n - matrix[0, 2]
            matrix[1, 2] = n - matrix[1, 2]
        return matrix

    def get_camera_extrinsics(self, camera: str) -> np.ndarray:
        matrix = camera_utils.get_camera_extrinsic_matrix(self.env.sim, camera)
        if "eye_in_hand" in camera:
            # Match the 180 wrist rotation (see observe_camera / get_camera_
            # intrinsics): rotate the camera frame 180 about its optical (z) axis,
            # i.e. flip the world-frame camera x and y axes. The optical axis and
            # the camera position (column 3) are unchanged.
            matrix = matrix.copy()
            matrix[:3, 0] *= -1
            matrix[:3, 1] *= -1
        return matrix

    def observe_camera(self, channel_first=False) -> dict[str, np.ndarray]:
        obs = {}

        for name in self.cfg.cameras:
            for key in [f"{name}_image", f"{name}_depth"]:
                image: np.ndarray = self.obs[key]
                image = image[::-1]  # flip because the default images are up-side-down
                # The eye-in-hand (wrist) camera is mounted rotated; rotate its
                # feed 180 so it reads right-way-up. The wrist view is excluded
                # from the point cloud (get_point_cloud), so geometry is
                # unaffected; the click->ray mapping in UI/template_index.html
                # (pickPointCloudFromCamUV) inverts wrist u/v to match.
                if "eye_in_hand" in name:
                    image = image[::-1, ::-1]  # rotate 180

                if channel_first:
                    image = image.transpose(2, 0, 1)
                obs[key] = image

        return obs

    def _gripper_width(self) -> float:
        # raw jaw width (m) measured between the two finger joints
        return self.obs["robot0_gripper_qpos"][0] - self.obs["robot0_gripper_qpos"][1]

    def observe_proprio(self) -> Proprio:
        return Proprio(
            eef_pos=self.obs["robot0_eef_pos"],
            eef_quat=self.obs["robot0_eef_quat"],
            # normalized jaw width: 0.0 closed -> 1.0 fully open
            gripper_open=self._gripper_width() / self.gripper_max_width,
        )

    def observe(self) -> dict[str, np.ndarray]:
        obs = self.observe_camera()

        proprio = self.observe_proprio()
        obs["ee_pos"] = proprio.eef_pos
        obs["ee_euler"] = proprio.eef_euler
        obs["ee_quat"] = proprio.eef_quat
        obs["gripper_open"] = proprio.gripper_open_np
        obs["proprio"] = proprio.eef_pos_euler

        if self.cfg.record_sim_state:
            obs["sim_state"] = self.env.sim.get_state().flatten()

        return obs

    def apply_action(
        self, ee_pos: np.ndarray, ee_euler: np.ndarray, gripper_open: float, is_delta=True
    ):
        assert is_delta, "positional not implemented yet"
        # gripper_open = 1 (open) -> gripper_action: -1
        # gripper_open = 0 (closed) -> gripper_action: 1
        gripper_action = 1 - 2 * gripper_open
        action = np.concatenate([ee_pos, ee_euler, [gripper_action]]).astype(np.float32)
        self.obs, self.reward, self.terminal, _ = self.env_wrapper.step(action)
        self.num_step += 1
        self.success = bool(self.env._check_success())

        # robomimic only flips `terminal` at the horizon, so also end on success
        # to match LIBERO, which terminates itself.
        if not _is_libero_task(self.task) and self.success:
            self.terminal = True

        # enforce the max_len budget here; LIBERO's terminal only flips on success.
        if self.num_step >= self.cfg.max_len:
            self.terminal = True

        if self.verbose:
            proprio = self.observe_proprio()
            pos = proprio.eef_pos
            rpy = np.degrees(proprio.eef_euler)
            logger.info(
                f"[apply_action] env: {self.num_step}/{self.cfg.max_len} step, reward: {self.reward}"
            )
            logger.info(f"[apply_action] ee_pos: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m")
            logger.info(
                f"[apply_action] ee_rpy: [{rpy[0]:+6.1f}, {rpy[1]:+6.1f}, {rpy[2]:+6.1f}] deg"
            )
            logger.info(f"[apply_action] gripper width: {proprio.gripper_open:.3f}")

    def move_to(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        recorder: Optional[EpisodeRecorder] = None,
        render: bool = False,
        stream_fn=None,
    ):
        self.update_pose(target_pos, target_euler, recorder, render, stream_fn)
        if self.terminal:
            return

        # gripper
        assert gripper_open == 0 or gripper_open == 1
        if self.curr_gripper_open != gripper_open:
            self.update_gripper(gripper_open, recorder, render, stream_fn=stream_fn)
            self.curr_gripper_open = gripper_open

    def update_pose(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        recorder: Optional[EpisodeRecorder],
        render: bool = False,
        stream_fn=None,
    ):
        # Interpolate the eef toward the target pose.
        waypoint_reach = self.waypoint_reach_cls(
            np.array(self.ctrl_config["output_max"]),
            target_pos,
            target_euler,
            self.cfg.waypoint_reach,
        )

        for _ in range(self.cfg.waypoint_max_step):
            proprio = self.observe_proprio()
            delta_pos, delta_euler, reached = waypoint_reach.step(
                proprio.eef_pos, proprio.eef_euler
            )

            if reached:
                break

            # record before apply action
            need_obs = recorder is not None or stream_fn is not None
            obs_for_step = self.observe() if need_obs else None

            self.apply_action(delta_pos, delta_euler, self.curr_gripper_open)

            if recorder is not None:
                assert obs_for_step is not None
                action = np.concatenate([delta_pos, delta_euler, [self.curr_gripper_open]])
                recorder.record(ActMode.Interpolate, obs_for_step, action, reward=self.reward)

            if stream_fn is not None:
                # stream to the browser
                stream_fn(obs_for_step)

            if render and self.has_renderer:
                self.env.render()

            if self.terminal:
                return

    def update_gripper(
        self,
        gripper_open: float,
        recorder: Optional[EpisodeRecorder],
        render: bool = False,
        stream_fn=None,
    ):
        # gripper_open is the binary commanded state: 1.0 = open, 0.0 = closed.
        # it is not the continuous normalized jaw width from observe_proprio().
        if self.verbose:
            logger.info(f"[gripper] from {self.curr_gripper_open:.1f} to {gripper_open:.1f}")

        # Closing stops against an object before width hits 0.0; treat it done
        # once the jaw width holds steady for this many consecutive steps.
        MIN_CLOSE_STABLE_COUNT = 5

        # observe_proprio().gripper_open is normalized jaw width: 0.0 closed,
        # 1.0 max width. Not the binary commanded open/closed state.
        prev_width = self.observe_proprio().gripper_open
        stable_count = 0
        # waypoint_max_step bounds the gripper move, same cap as the pose loop.
        for _ in range(self.cfg.waypoint_max_step):
            need_obs = recorder is not None or stream_fn is not None
            obs_for_step = self.observe() if need_obs else None

            action = np.concatenate([np.zeros(3), np.zeros(3), [gripper_open]])
            self.apply_action(action[:3], action[3:6], action[6])

            if recorder is not None:
                assert obs_for_step is not None
                recorder.record(ActMode.Interpolate, obs_for_step, action, reward=self.reward)

            if stream_fn is not None:
                stream_fn(obs_for_step)

            curr_width = self.observe_proprio().gripper_open
            if gripper_open == 1:
                done = np.abs(curr_width - 1) < 0.01
            else:
                stable = np.abs(curr_width - prev_width) < 0.002
                stable_count = stable_count + 1 if stable else 0
                done = stable_count >= MIN_CLOSE_STABLE_COUNT
            prev_width = curr_width

            if render and self.has_renderer:
                self.env.render()

            if done or self.terminal:
                return

    def get_point_cloud(self, obs, crop_table):
        pc = self.pc_config
        points_list = []
        colors_list = []
        for view in [cam for cam in self.cfg.cameras if "eye_in_hand" not in cam]:
            rgb_frame = obs[f"{view}_image"]
            depth_frame = obs[f"{view}_depth"]
            depth_frame = camera_utils.get_real_depth_map(self.env.sim, depth_frame)

            agent_intrinsics = self.get_camera_intrinsics(view)
            camera_extrinsics = self.get_camera_extrinsics(view)
            points = deproject(
                depth_frame.squeeze(), agent_intrinsics, camera_extrinsics, base_units=0
            )

            colors = rgb_frame.reshape(points.shape) / 255.0

            x_min, x_max = pc.x_min, pc.x_max
            y_min, y_max = pc.y_min, pc.y_max
            # raise the lower z bound to the table top when cropping it out
            z_min = pc.z_min_crop_table if crop_table else pc.z_min
            z_max = pc.z_max

            valid = (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )
            points = points[valid]
            colors = colors[valid]

            points_list.append(points)
            colors_list.append(colors)

        merged_points = np.vstack(points_list)
        merged_colors = np.vstack(colors_list)
        return merged_points, merged_colors


def render_random_episode(env: SimEnv):
    """Roll out zero-velocity actions, dumping observation shapes each step."""
    env.reset()
    if env.has_renderer:
        env.env.render()

    for _ in range(30):
        env.apply_action(np.zeros(3), np.zeros(3), 1)

        prop = env.observe_proprio()
        logger.info(f"prop: {prop.gripper_open}")
        obs = env.observe()
        for k, v in obs.items():
            logger.info(f"{k} {v.shape} {v.dtype}")

        if env.has_renderer:
            env.env.render()

        if env.terminal:
            break


def render_pc(env: SimEnv):
    """Reset and scatter-plot the merged point cloud."""
    env.reset()
    points, colors = env.get_point_cloud(
        env.observe(),
        crop_table=bool(env.cfg.crop_table),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    vis_points = points
    ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], c=colors, s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def render_cameras(env: SimEnv):
    """Reset and save every camera's RGB to one figure for eyeballing / tuning.

    Use this to tune the extra cameras: edit _DEFAULT_ORBIT_ANGLES (or
    _orbit_agentview), rerun, and inspect the saved PNG.
    """
    env.reset()
    obs = env.observe_camera()
    cams = env.cfg.cameras

    fig, axes = plt.subplots(1, len(cams), figsize=(4 * len(cams), 4))
    axes = np.atleast_1d(axes)
    for ax, cam in zip(axes, cams):
        ax.imshow(obs[f"{cam}_image"])
        ax.set_title(cam, fontsize=9)
        ax.axis("off")

    out = "camera_views.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    logger.info(f"saved {out} with cameras: {cams}")


TESTS = {
    "pc": render_pc,
    "random": render_random_episode,
    "cams": render_cameras,
}


def list_supported_tasks():
    """Log the robomimic presets and every LIBERO suite's task list."""
    logger.info("Robomimic presets:")
    for t in ROBOMIMIC_TASKS:
        logger.info(f"  {t}")
    logger.info("LIBERO (<suite>/<id>):")
    try:
        from libero.libero import benchmark

        bm_dict = benchmark.get_benchmark_dict()
        for suite in bm_dict:
            bm = bm_dict[suite]()
            for i in range(bm.n_tasks):
                logger.info(f"  {suite}/{i}  ->  {bm.get_task(i).language}")
    except Exception as e:
        logger.warning(f"  <unavailable: {type(e).__name__}: {e}>")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list supported tasks and exit.",
    )
    parser.add_argument(
        "--task",
        default="square",
        help="env task: a robomimic preset (lift/square/stack) or a LIBERO key "
        "(<suite>/<id>, e.g. libero_spatial/0, libero_goal/2). See --list.",
    )
    parser.add_argument(
        "--test",
        choices=list(TESTS),
        default="pc",
        help="which test to run: pc (point cloud), random (zero-action rollout), "
        "cams (save every camera's RGB). Default: pc.",
    )

    args = parser.parse_args()
    setup_logging()
    if args.list:
        list_supported_tasks()
        return

    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    env = SimEnv(args.task, on_screen_render=True, verbose=True)
    TESTS[args.test](env)


if __name__ == "__main__":
    main()
