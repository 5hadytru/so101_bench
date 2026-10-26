"""Isaac-side pieces shared by the LLM-harness evaluators (GPT-Policy, PyRUA-Lean).

The episode setup follows ``scripts/groot_eval.py`` (same reset path, object pool and
initial hold); ``SimArm`` owns one episode's control stream, images, camera calibration
and terminal capture. ``PUBLIC_TASK_RULES`` is the success/failure text every harness
gives its model, generated from the benchmark constants.

Import only after Isaac Sim's ``AppLauncher`` has started.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from so101_bench.benchmark import (
    BETWEEN_LINE_TOLERANCE_M,
    BIN_DISPLACEMENT_LIMIT_M,
    BOUNDARY_DISPLACEMENT_LIMIT_M,
    MOVE_BOUNDARY_SUCCESS_DISTANCE_M,
    MOVE_NO_BOUNDARY_MIN_PROGRESS_M,
    MOVE_STRAIGHTNESS_TOLERANCE_M,
    NON_BIN_TARGET_MIN_DISPLACEMENT_M,
    NON_TARGET_DISPLACEMENT_LIMIT_M,
    ROBOT_CONTACT_FORCE_THRESHOLD_N,
    SPATIAL_SUCCESS_DISTANCE_M,
    TASK_BETWEEN,
    TASK_BIN,
    TASK_MOVE,
    TASK_NEXT_TO,
    BenchmarkEpisodeSpec,
    episode_length_s,
)
from so101_bench.layouts import normalize_layout_object_slots
from so101_bench.mdp.terminations import (
    DEFAULT_CONTACT_GRACE_TIME_S,
    DEFAULT_FAILURE_CONFIRM_TIME_S,
    POSTMORTEM_NOT_APPLICABLE,
)
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import SUCCESS_CONFIRM_TIME_S
from so101_bench.utils import so101_kinematics as kin
from so101_bench.utils.lerobot_calibration import (
    LEROBOT_INITIAL_JOINT_POS,
    lerobot_pose_to_sim_joint_pos,
)

ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
INITIAL_ROBOT_JOINT_POS = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)
CAMERAS = ("overhead", "wrist")


def _inch(meters: float) -> str:
    return f"{meters * 100:.2f} cm"


#: Bin and between success are judged on each object's root, which is placed at (roughly) the
#: centre of the object's top-down bounding box; the text says so.
REFERENCE_POINT_NOTE = (
    "An object's reference point is the centre of the rectangle that bounds it as seen from above (its "
    "top-down outline's bounding box), to within about a millimetre for most objects and up to about 3 cm "
    "for a few long ones."
)

#: Rules every task shares: how conditions are timed and measured, and the time-out extension.
_COMMON_RULES = (
    "How judging works: positions and displacements are measured in the tabletop plane (height ignored) "
    "from where each object and the bin rest when you take control. The robot 'touches' an object when "
    f"any robot link presses on it with more than {ROBOT_CONTACT_FORCE_THRESHOLD_N:.2f} N. Every success "
    f"condition must hold continuously for {SUCCESS_CONFIRM_TIME_S:.0f} s of simulated time; any break "
    "restarts that count. Every failure condition ends the episode once it has held continuously for "
    f"{DEFAULT_FAILURE_CONFIRM_TIME_S:.0f} s (undoing it sooner, e.g. putting a nudged object back, avoids "
    "the failure). If the time limit arrives while success is being confirmed, the episode is extended "
    "until the confirmation finishes; if success breaks during that extension, the episode fails."
)

PUBLIC_TASK_RULES = {
    TASK_BIN: (
        "Success: every object on the table ends with its reference point inside the plastic bin's footprint "
        "seen from above (height does not matter), with the robot touching none of the objects. "
        f"{REFERENCE_POINT_NOTE} Failure: the bin moved more than {_inch(BIN_DISPLACEMENT_LIMIT_M)} from its "
        "start (the robot may brush the bin but must not shift it, directly or by pushing an object into it). "
        "In this task every object is a target, so objects may touch each other "
        "and other objects may be moved while you work; only the bin's position is limited. " + _COMMON_RULES
    ),
    TASK_NEXT_TO: (
        "Success: the target's top-down outline ends within "
        f"{_inch(SPATIAL_SUCCESS_DISTANCE_M)} (surface to surface) of the referent's outline, the target has "
        f"moved at least {_inch(NON_BIN_TARGET_MIN_DISPLACEMENT_M)} from its start, and the robot is not touching "
        "the target. Failures: (1) any object other than the target, the referent included, moved more than "
        f"{_inch(NON_TARGET_DISPLACEMENT_LIMIT_M)} from its start; (2) the target, or any object held in the "
        "gripper, touching any other object (the table, the bin and the robot do not count) for more than "
        f"{DEFAULT_CONTACT_GRACE_TIME_S:.0f} s without a break. Only the target may be grasped or moved on "
        "purpose; touching other objects with the robot is not scored by itself, but it is how they get "
        "displaced, so avoid it. " + _COMMON_RULES
    ),
    TASK_BETWEEN: (
        "Success: the target's reference point ends within "
        f"{_inch(BETWEEN_LINE_TOLERANCE_M)} of the straight line through the two referents' reference points, "
        "and closer to each referent's reference point than those two are to each other (exact midpoint not "
        f"required). {REFERENCE_POINT_NOTE} Only that point of the target is judged against the line, not "
        "where the rest of the object lies. The target must also have moved at least "
        f"{_inch(NON_BIN_TARGET_MIN_DISPLACEMENT_M)}, and the robot must not be touching it. Failures: (1) any "
        "object other than the target, both referents included, moved more than "
        f"{_inch(NON_TARGET_DISPLACEMENT_LIMIT_M)} from its start; (2) the target, or any object held in the "
        "gripper, touching any other object (the table, the bin and the robot do not count) for more than "
        f"{DEFAULT_CONTACT_GRACE_TIME_S:.0f} s without a break. Only the target may be grasped or moved on "
        "purpose; touching other objects with the robot is not scored by itself, but it is how they get "
        "displaced, so avoid it. " + _COMMON_RULES
    ),
    TASK_MOVE: (
        "Directions are in the robot's frame: left = +x, right = -x, forward = -y (away from the robot base), "
        "backward = +y (toward the base). At the start of the episode the benchmark picks a blocker: the "
        "NEAREST other object ahead of the target in that direction whose outline overlaps at least 10% of the "
        "target's sideways extent; it stays the blocker for the whole episode. Success with a blocker: the gap "
        "from the target's leading edge to the blocker's facing edge, measured along the direction, ends at most "
        f"{_inch(MOVE_BOUNDARY_SUCCESS_DISTANCE_M)} (no minimum gap, and ending past that edge is allowed, but the "
        "target resting against the blocker is contact, failure 4 below); moving "
        "the target well along the direction is not enough if it stops further than that from the blocker. "
        "Success with no blocker: the middle of the target's outline progresses at least "
        f"{_inch(MOVE_NO_BOUNDARY_MIN_PROGRESS_M)} in the direction. In both cases the middle of its outline must "
        f"be within {_inch(MOVE_STRAIGHTNESS_TOLERANCE_M)} sideways of the straight path from its start, the "
        f"target must have moved at least {_inch(NON_BIN_TARGET_MIN_DISPLACEMENT_M)} in total, and the robot "
        f"must not be touching it. Failures: (1) the blocker's facing edge moved more than "
        f"{_inch(BOUNDARY_DISPLACEMENT_LIMIT_M)} along the direction; (2) any object other than the target moved "
        f"more than {_inch(NON_TARGET_DISPLACEMENT_LIMIT_M)} from its start; (3) the target's sideways deviation "
        f"over {_inch(MOVE_STRAIGHTNESS_TOLERANCE_M)}; (4) the target, or any object held in the gripper, "
        "touching any other object (the table, the bin and the robot do not count; the blocker does) for more "
        f"than {DEFAULT_CONTACT_GRACE_TIME_S:.0f} s without a break. Only the target may be grasped or moved on "
        "purpose; touching other objects with the robot is not scored by itself, but it is how they get "
        "displaced, so avoid it. " + _COMMON_RULES
    ),
}


def load_layouts(episodes: list[BenchmarkEpisodeSpec], path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_trial = {str(row["trial_id"]): row for row in rows}
    layouts = []
    for index, episode in enumerate(episodes):
        trial_id = (episode.metadata or {}).get("trial_id", index)
        if str(trial_id) not in by_trial:
            raise ValueError(f"{path} has no layout for trial_id={trial_id!r}.")
        layouts.append(normalize_layout_object_slots(by_trial[str(trial_id)], episode.objects, episode_index=index))
    return layouts


def object_pool(episodes: list[BenchmarkEpisodeSpec]) -> list[str]:
    pool: list[str] = []
    for episode in episodes:
        pool.extend(name for name in episode.objects if name not in pool)
    return pool


def reset_params(episode: BenchmarkEpisodeSpec, layout: dict, pool: list[str], asset_names: list[str]) -> dict:
    slot = {name: index for index, name in enumerate(pool)}
    payload = episode.reset_payload()
    local_to_pool = [slot[name] for name in episode.objects]
    payload["active_object_ids"] = local_to_pool
    payload["target_object_id"] = local_to_pool[episode.target_object_id]
    payload["referent_object_ids"] = [local_to_pool[i] for i in episode.referent_object_ids]
    remapped = dict(layout)
    remapped["objects"] = []
    for entry in layout.get("objects", []):
        entry = dict(entry)
        pool_slot = slot[str(entry.get("name") or episode.objects[int(entry["slot"])])]
        entry["slot"] = pool_slot
        entry["asset_name"] = f"object_{pool_slot + 1}"
        remapped["objects"].append(entry)
    return {
        "object_asset_names": asset_names,
        "object_labels": pool,
        "task_family": episode.task_family,
        "object_count_range": (len(episode.objects), len(episode.objects)),
        "active_object_selection": "fixed",
        "fixed_active_object_ids": tuple(local_to_pool),
        "shuffle_object_labels": False,
        "force_bin_all_objects_instruction": False,
        "episode_spec": payload,
        "episode_layout": remapped,
    }


def initial_action(env) -> torch.Tensor:
    return torch.tensor(
        [[INITIAL_ROBOT_JOINT_POS[name] for name in ACTION_JOINT_NAMES]], dtype=torch.float32, device=env.unwrapped.device
    )


def reset_env(env) -> dict:
    with torch.inference_mode(False):
        env.reset()
        robot = env.unwrapped.scene["robot"]
        ids = [robot.joint_names.index(name) for name in ACTION_JOINT_NAMES]
        joint_pos = initial_action(env)
        robot.data.default_joint_pos[:, ids] = joint_pos
        robot.data.default_joint_vel[:, ids] = 0.0
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), joint_ids=ids)
        robot.set_joint_position_target(joint_pos, joint_ids=ids)
        robot.write_data_to_sim()
        unwrapped = env.unwrapped
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.forward()
        for _ in range(getattr(unwrapped.cfg, "num_rerenders_on_reset", 0) if unwrapped.sim.has_rtx_sensors() else 0):
            unwrapped.sim.render()
        obs = unwrapped.observation_manager.compute(update_history=True)
        unwrapped.obs_buf = obs
        return obs


def quat_wxyz_matrix(q) -> np.ndarray:
    return kin._quat_wxyz_to_matrix(q)


class SimArm:
    """Owns one episode's control stream: commanded targets, frames, and terminal capture."""

    def __init__(self, env, episode: BenchmarkEpisodeSpec, output: Path, obs: dict):
        self.env = env
        self.unwrapped = env.unwrapped
        self.episode = episode
        self.robot = self.unwrapped.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(name) for name in ACTION_JOINT_NAMES]
        self.base_body = self.robot.body_names.index("base")
        self.origin = self.unwrapped.scene.env_origins[0].detach().cpu().numpy().astype(float)
        self.control_dt = float(self.unwrapped.step_dt)
        self.time_limit_s = episode_length_s(episode.task_family, len(episode.objects))
        self.action = initial_action(env)
        self.pose_tracker: NonTargetPoseTracker | None = None
        self.step_id = 0
        self.sim_loop_wall_s = 0.0
        self.terminal: dict[str, Any] | None = None
        self.output = output
        self.video = imageio.get_writer(
            output / "rollout_overhead_wrist.mp4", fps=round(1.0 / self.control_dt), codec="libx264", quality=7
        )
        self.obs = obs

    # -- state ---------------------------------------------------------------
    def joints(self) -> np.ndarray:
        return self.robot.data.joint_pos[0, self.joint_ids].detach().cpu().numpy().astype(float)

    def world_from_base(self) -> np.ndarray:
        pos = self.robot.data.body_pos_w[0, self.base_body].detach().cpu().numpy().astype(float) - self.origin
        quat = self.robot.data.body_quat_w[0, self.base_body].detach().cpu().numpy().astype(float)
        result = np.eye(4)
        result[:3, :3] = quat_wxyz_matrix(quat)
        result[:3, 3] = pos
        return result

    def images(self) -> dict[str, np.ndarray]:
        visual = self.obs["visual"]
        return {name: np.ascontiguousarray(visual[f"rgb_{name}"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)) for name in CAMERAS}

    def calibration(self, images: dict[str, np.ndarray]) -> dict[str, Any]:
        settings = (getattr(self.unwrapped, "_so101_domain_randomization", None) or [{}])[0]
        result = {}
        for name in CAMERAS:
            data = self.unwrapped.scene[f"camera_{name}"].data
            intrinsic = data.intrinsic_matrices[0].detach().cpu().numpy().astype(float).copy()
            height, width = images[name].shape[:2]
            # apply_visual_camera_postprocessing shifts content by offset * image size.
            intrinsic[0, 2] += float(settings.get(f"{name}_camera_principal_point_x_offset", 0.0)) * width
            intrinsic[1, 2] += float(settings.get(f"{name}_camera_principal_point_y_offset", 0.0)) * height
            transform = np.eye(4)
            transform[:3, :3] = quat_wxyz_matrix(data.quat_w_ros[0].detach().cpu().numpy().astype(float))
            transform[:3, 3] = data.pos_w[0].detach().cpu().numpy().astype(float) - self.origin
            result[name] = {
                "intrinsic": intrinsic.round(6).tolist(),
                "resolution": [int(width), int(height)],
                "camera_to_environment": transform.round(6).tolist(),
                "distortion_model": "pinhole",
                "frame": "environment",
                "optical_axes": "x_right_y_down_z_forward",
            }
        return result

    def robot_state(self) -> dict[str, Any]:
        q = self.joints()
        pose = kin.tool_pose(self.world_from_base(), q[:5])
        return {
            "joint_pos_rad": {name: round(float(v), 4) for name, v in zip(ACTION_JOINT_NAMES, q, strict=True)},
            "tcp_xyz": pose.tcp.round(4).tolist(),
            "fingertip_xyz": pose.fingertip.round(4).tolist(),
            "pitch_deg": round(pose.pitch_deg, 1),
            "jaw_yaw_deg": None if pose.jaw_yaw_deg is None else round(pose.jaw_yaw_deg, 1),
            "tool_axis": pose.tool_axis.round(3).tolist(),
            "jaw_axis": pose.jaw_axis.round(3).tolist(),
            "measured_gripper_opening_m": round(kin.opening_for_jaw_angle(float(q[5])), 4),
            "commanded_gripper_opening_m": round(kin.opening_for_jaw_angle(float(self.action[0, 5])), 4),
            "base_xyz": self.world_from_base()[:3, 3].round(4).tolist(),
        }

    def elapsed_s(self) -> float:
        return float(self.unwrapped.episode_length_buf[0].item()) * self.control_dt

    # -- stepping ------------------------------------------------------------
    def step(self) -> bool:
        """Advance one control with the current action. Returns True once the episode ended."""

        started = time.monotonic()
        try:
            return self._step_native()
        finally:
            self.sim_loop_wall_s += time.monotonic() - started

    def _step_native(self) -> bool:

        obs, _rewards, terminated, truncated, info = self.env.step(self.action)
        self.step_id += 1
        if bool(terminated.any().item() or truncated.any().item()):
            # the env already auto-reset: the pose tracker keeps the last pre-reset values
            self.terminal = self._terminal_record(terminated, truncated, info)
            self._write_frame(self.terminal_frame)
            return True
        self.obs = obs
        if self.pose_tracker is not None:
            self.pose_tracker.update(self.unwrapped)
        self._write_frame(self.images())
        return False

    def _write_frame(self, images: dict[str, np.ndarray]) -> None:
        self.video.append_data(np.concatenate([images["overhead"], images["wrist"]], axis=1))

    def _terminal_record(self, terminated, truncated, info) -> dict[str, Any]:
        # env.step() already auto-reset the scene: keep the last pre-reset frame and
        # read the postmortem stored on the terminating step (as groot_eval does).
        self.terminal_frame = self.images()
        term_log = info.get("log", {})
        success = bool(term_log.get("Episode_Termination/success", 0.0) > 0.0)
        failure_reasons = getattr(self.unwrapped, "_so101_failure_reasons", None) or ["none"]
        if success:
            reason = "success"
        elif failure_reasons[0] != "none":
            reason = failure_reasons[0]
        elif bool(term_log.get("Episode_Termination/failure", 0.0) > 0.0):
            reason = "failure"
        elif bool(truncated.any().item()):
            reason = "time_out"
        else:
            reason = "unknown"
        diagnostics = getattr(self.unwrapped, "_so101_postmortem_failure_diagnostics", None)
        postmortem = diagnostics[0] if diagnostics else None
        record = {
            "success": success,
            "termination_reason": reason,
            "episode_length_s": self.step_id * self.control_dt,
            "agent_control_steps": self.step_id,
            "postmortem": asdict(postmortem) if postmortem is not None else None,
        }
        if self.pose_tracker is not None:
            record["non_target_pose_changes"] = self.pose_tracker.report()
        if postmortem is not None:
            record["failure_type"] = (
                None if success or postmortem.failure_type == POSTMORTEM_NOT_APPLICABLE else postmortem.failure_type
            )
            if postmortem.bin_progress is not None:
                record["objects_in_bin"], record["bin_object_count"] = postmortem.bin_progress
        return record

    def close(self) -> None:
        self.video.close()


class NonTargetPoseTracker:
    """Records how far each non-target object's orientation strays from where it rested at robot
    start (non-bin tasks): tilt = angle between its initial and current up axes, yaw = rotation
    about the world vertical.  Diagnostic only; it never changes success or failure."""

    def __init__(self, env, asset_names: list[str], names: list[str], object_ids: list[int]) -> None:
        from so101_bench.mdp.resets import benchmark_object_quats

        self._quats = benchmark_object_quats
        self.asset_names, self.ids = asset_names, object_ids
        self.names = {i: names[i] for i in object_ids}
        q0 = self._quats(env, asset_names)[0]
        self.start = {i: q0[i].detach().cpu().numpy() for i in object_ids}
        self.stats = {i: {"max_tilt_deg": 0.0, "final_tilt_deg": 0.0, "max_yaw_deg": 0.0, "final_yaw_deg": 0.0}
                      for i in object_ids}

    @staticmethod
    def _angles(q0: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        r0, r = quat_wxyz_matrix(q0), quat_wxyz_matrix(q)
        tilt = math.degrees(math.acos(float(np.clip(np.dot(r0[:, 2], r[:, 2]), -1.0, 1.0))))
        rel = r @ r0.T
        yaw = abs(math.degrees(math.atan2(rel[1, 0], rel[0, 0])))
        return tilt, yaw

    def update(self, env) -> None:
        q = self._quats(env, self.asset_names)[0].detach().cpu().numpy()
        for i in self.ids:
            if np.isnan(self.start[i]).any() or np.isnan(q[i]).any():
                continue
            tilt, yaw = self._angles(self.start[i], q[i])
            st = self.stats[i]
            st["max_tilt_deg"], st["final_tilt_deg"] = max(st["max_tilt_deg"], tilt), tilt
            st["max_yaw_deg"], st["final_yaw_deg"] = max(st["max_yaw_deg"], yaw), yaw

    def report(self) -> dict:
        return {self.names[i]: {k: round(v, 1) for k, v in self.stats[i].items()} for i in self.ids}


# --------------------------------------------------------------------------- #
