from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt


@dataclass
class Proprio:
    # supplied as arguments
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_open: float  # gripper_width

    # computed in __init__
    gripper_open_np: np.ndarray  # gripper_width converted to array
    eef_euler: np.ndarray  # rotation in euler
    eef_pos_euler: np.ndarray

    def __init__(
        self,
        eef_pos: list[float],
        eef_quat: list[float],
        gripper_open: float,
    ):
        self.eef_pos = np.array(eef_pos)  # , dtype=np.float32)
        self.eef_quat = np.array(eef_quat)  # , dtype=np.float32)
        self.gripper_open = gripper_open

        self.gripper_open_np = np.array([self.gripper_open])  # , dtype=np.float32)
        self.eef_euler = Rotation.from_quat(self.eef_quat).as_euler("xyz")  # .astype(np.float32)
        self.eef_pos_euler = np.concatenate([self.eef_pos, self.eef_euler, self.gripper_open_np])


def position_action_to_delta_action(
    curr_pos: np.ndarray, curr_euler: np.ndarray, new_pos: np.ndarray, new_euler: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta_pos = new_pos - curr_pos
    curr_rot = Rotation.from_euler("xyz", curr_euler)
    target_rot = Rotation.from_euler("xyz", new_euler)
    delta_rot = target_rot * curr_rot.inv()
    delta_euler = delta_rot.as_euler("xyz")
    return delta_pos, delta_euler


# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta):
    total_delta = target_pt - start_pt
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = np.linalg.norm(total_delta) % max_delta
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps

    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)

    return gen_waypoint, int(num_steps)


# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = Rotation.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    if diff < 0.02 or num_steps < 2:

        def gen_ori(i):
            return initial_euler

    else:
        slerp = Slerp([1, num_steps], ori_chg)

        def gen_ori(i):
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler

    return gen_ori


def _p_step(step_size, max_norm, delta):
    delta = step_size * delta
    delta_norm = np.linalg.norm(delta)
    delta = delta / delta_norm * min(delta_norm, max_norm)
    return delta


def _pi_step(step_size, ki, max_norm, band, integral_max, delta, integral):
    """Proportional-integral version of _p_step.

    Returns (output_delta, updated_integral). The proportional part is identical
    to _p_step's `step_size * delta`; the integral adds `ki * integral`
    to overcome the steady-state error a pure-P loop plateaus on near the target.
    Anti-windup via conditional integration: only accumulate inside the
    final-approach band (||delta|| < band) -- the deadband regime where P alone
    under-acts. Outside the band the integral is held (so it can't wind up over a
    long approach), and its norm is clamped to integral_max as a hard bound.
    Additionally, when the error flips to oppose the accumulated integral
    (delta . integral < 0) the target has just been crossed, so the wound-up
    integral is dumped to stop it from driving overshoot past the target.
    """
    p_term = step_size * delta
    if np.linalg.norm(delta) < band:
        if np.dot(delta, integral) < 0:  # crossed the target: dump wound-up integral
            integral = np.zeros_like(integral)
        integral = integral + delta
        integral_norm = np.linalg.norm(integral)
        if integral_norm > integral_max:
            integral = integral / integral_norm * integral_max
    else:
        integral = np.zeros_like(integral)
    out = p_term + ki * integral
    out_norm = np.linalg.norm(out)
    out = out / out_norm * min(out_norm, max_norm)
    return out, integral


@dataclass
class WaypointReachConfig:
    pos_threshold: float = 0.002  # 2mm err tolerance
    pos_step_size: float = 0.5
    pos_max_norm: float = 0.1
    rot_threshold: float = 0.02
    rot_step_size: float = 0.5
    rot_max_norm: float = 0.2
    # Integral gains for WaypointReachPI; ignored by the P-only WaypointReach.
    pos_ki: float = 0.1
    rot_ki: float = 0.0  # rotation shows no under-shoot; integral only hurts there
    integral_max: float = 1.0
    # Only integrate within integral_band_scale * threshold of the target (the
    # final-approach deadband regime), to keep the integral from winding up.
    integral_band_scale: float = 3.0
    # Absolute cap (meters) on the position integration band. Without it a loose
    # pos_threshold opens a wide band the integral winds up over, adding approach
    # speed and overshoot. With the cap below typical thresholds, the loop reaches
    # before entering the band (integral never engages -> behaves like P) unless
    # sub-cap precision is requested -- exactly the regime where P under-shoots.
    integral_band_max: float = 0.003


class WaypointReach:
    def __init__(
        self,
        max_delta_action: np.ndarray,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        cfg: WaypointReachConfig,
    ):
        assert max_delta_action.shape == (6,)
        self.max_delta_pos = max_delta_action[:3]
        self.max_delta_euler = max_delta_action[3:]
        self.target_pos = target_pos
        self.target_euler = target_euler
        self.cfg = cfg

    def step(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        delta_pos = self.target_pos - curr_pos
        pos_reached = np.linalg.norm(delta_pos) < self.cfg.pos_threshold
        if pos_reached:
            delta_pos_action = np.zeros_like(curr_pos)
        else:
            delta_pos = _p_step(self.cfg.pos_step_size, self.cfg.pos_max_norm, delta_pos)
            delta_pos_action = (delta_pos / self.max_delta_pos).clip(min=-1, max=1)

        # next, process rot
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        target_rot = Rotation.from_euler("xyz", self.target_euler)
        delta_euler = (target_rot * curr_rot.inv()).as_euler("xyz")

        rot_reached = np.linalg.norm(delta_euler) < self.cfg.rot_threshold
        if rot_reached:
            delta_euler_action = np.zeros_like(delta_euler)
        else:
            delta_euler = _p_step(self.cfg.rot_step_size, self.cfg.rot_max_norm, delta_euler)
            delta_euler_action = (delta_euler / self.max_delta_euler).clip(min=-1, max=1)

        reached = rot_reached and pos_reached
        return delta_pos_action, delta_euler_action, reached


class WaypointReachPI(WaypointReach):
    """PI variant of WaypointReach.

    Adds an integral term to kill the steady-state under-shoot a pure-P loop
    plateaus on near the target (controller deadband / friction). Structurally
    identical to WaypointReach -- it only swaps _p_step for
    _pi_step and threads the integral state -- and reduces exactly to the
    P controller when cfg.pos_ki == cfg.rot_ki == 0.
    """

    def __init__(
        self,
        max_delta_action: np.ndarray,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        cfg: WaypointReachConfig,
    ):
        super().__init__(max_delta_action, target_pos, target_euler, cfg)
        self.pos_integral = np.zeros(3)
        self.euler_integral = np.zeros(3)

    def step(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        delta_pos = self.target_pos - curr_pos
        pos_reached = np.linalg.norm(delta_pos) < self.cfg.pos_threshold
        if pos_reached:
            delta_pos_action = np.zeros_like(curr_pos)
        else:
            delta_pos, self.pos_integral = _pi_step(
                self.cfg.pos_step_size,
                self.cfg.pos_ki,
                self.cfg.pos_max_norm,
                min(
                    self.cfg.integral_band_scale * self.cfg.pos_threshold,
                    self.cfg.integral_band_max,
                ),
                self.cfg.integral_max,
                delta_pos,
                self.pos_integral,
            )
            delta_pos_action = (delta_pos / self.max_delta_pos).clip(min=-1, max=1)

        # next, process rot
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        target_rot = Rotation.from_euler("xyz", self.target_euler)
        delta_euler = (target_rot * curr_rot.inv()).as_euler("xyz")

        rot_reached = np.linalg.norm(delta_euler) < self.cfg.rot_threshold
        if rot_reached:
            delta_euler_action = np.zeros_like(delta_euler)
        else:
            delta_euler, self.euler_integral = _pi_step(
                self.cfg.rot_step_size,
                self.cfg.rot_ki,
                self.cfg.rot_max_norm,
                self.cfg.integral_band_scale * self.cfg.rot_threshold,
                self.cfg.integral_max,
                delta_euler,
                self.euler_integral,
            )
            delta_euler_action = (delta_euler / self.max_delta_euler).clip(min=-1, max=1)

        reached = rot_reached and pos_reached
        return delta_pos_action, delta_euler_action, reached


class MoveErrorPlot:
    def __init__(self, target):
        self.final_target = target
        self.pos = []
        self.target = []
        self.action = []

    def add(self, pos, target, action):
        self.pos.append(pos)
        self.target.append(target)
        self.action.append(action)

    def plot(self, labels=("x", "y", "z"), value_name="value", title=""):
        # top row: actual vs desired per component; bottom row: the action delta.
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex="col", squeeze=False)

        poses = np.array(self.pos)
        targets = np.array(self.target)
        actions = np.array(self.action)

        for i in range(3):
            ax[0][i].plot(poses[:, i], label="actual")
            ax[0][i].plot(targets[:, i], "--", label="desired")
            ax[1][i].plot(actions[:, i], color="tab:green", label="action")
            ax[0][i].set_title(labels[i])
            for r in (0, 1):
                ax[r][i].grid(True, alpha=0.3)
            ax[1][i].set_xlabel("control step")
            # Tight y-limits around the actual+target range so small overshoot is
            # visible; the absolute offset (e.g. z ~ 1.0) would otherwise dominate.
            lo = min(poses[:, i].min(), targets[:, i].min())
            hi = max(poses[:, i].max(), targets[:, i].max())
            pad = max((hi - lo) * 0.15, 1e-4)
            ax[0][i].set_ylim(lo - pad, hi + pad)

        ax[0][0].set_ylabel(value_name)
        ax[1][0].set_ylabel("action delta")
        ax[0][0].legend(loc="best")
        ax[1][0].legend(loc="best")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()
