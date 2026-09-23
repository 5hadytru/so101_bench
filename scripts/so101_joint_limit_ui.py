# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive SO-101 joint-limit viewer for Isaac Sim.

Run with Isaac Lab, for example:

    /home/truman/IsaacLab/isaaclab.sh -p scripts/so101_joint_limit_ui.py --task So101Bench-Bin-v0

Then type commands into the launch terminal, such as:

    shoulder_pan min
    wrist_roll max
    gripper min
    list
    current
"""

from __future__ import annotations

import argparse
import math
import queue
import random
import threading

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Launch Isaac Sim UI and move SO-101 joints to their USD limits.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument(
    "--limit_margin_deg",
    type=float,
    default=0.0,
    help="Optional margin to stay inside min/max joint limits, in degrees.",
)
parser.add_argument(
    "--print_after_command",
    action="store_true",
    default=False,
    help="Print current joint positions after each command.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.utils.lerobot_calibration import LEROBOT_TO_USD_JOINT_NAMES


ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")

USD_TO_LEROBOT_JOINT_NAMES = {usd: lerobot for lerobot, usd in LEROBOT_TO_USD_JOINT_NAMES.items()}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _command_loop(commands: queue.SimpleQueue[str]) -> None:
    while True:
        try:
            command = input("so101> ")
        except EOFError:
            commands.put("quit")
            return
        commands.put(command)


class JointLimitCommander:
    """Converts terminal commands into absolute joint-position actions."""

    def __init__(self, env):
        self.env = env
        self.robot = env.unwrapped.scene["robot"]
        self.device = env.unwrapped.device
        self.action_joint_ids = [self.robot.joint_names.index(name) for name in ACTION_JOINT_NAMES]
        self.joint_index_by_usd_name = {name: self.robot.joint_names.index(name) for name in ACTION_JOINT_NAMES}
        self.usd_name_by_alias = self._build_aliases()
        self.limit_margin_rad = math.radians(args_cli.limit_margin_deg)
        self.target = self._initial_action()

    def _build_aliases(self) -> dict[str, str]:
        aliases = {}
        for usd_name in ACTION_JOINT_NAMES:
            aliases[_normalize_name(usd_name)] = usd_name

        for lerobot_name, usd_name in LEROBOT_TO_USD_JOINT_NAMES.items():
            aliases[_normalize_name(lerobot_name)] = usd_name

        aliases.update(
            {
                "shoulder": "Rotation",
                "pan": "Rotation",
                "shoulder_pan_joint": "Rotation",
                "shoulder_lift_joint": "Pitch",
                "lift": "Pitch",
                "elbow": "Elbow",
                "wrist": "Wrist_Pitch",
                "wrist_pitch": "Wrist_Pitch",
                "wrist_flex_joint": "Wrist_Pitch",
                "roll": "Wrist_Roll",
                "jaw": "Jaw",
            }
        )
        return aliases

    def _initial_action(self) -> torch.Tensor:
        joint_pos = self.robot.data.default_joint_pos[0, self.action_joint_ids].clone()
        actions = torch.zeros(self.env.action_space.shape, device=self.device)
        if actions.ndim == 1:
            actions[:] = joint_pos
        else:
            actions[:] = joint_pos.unsqueeze(0)
        return actions

    def action(self) -> torch.Tensor:
        return self.target

    def _target_view(self) -> torch.Tensor:
        if self.target.ndim == 1:
            return self.target
        return self.target[0]

    def _set_action_value(self, action_index: int, value: float) -> None:
        if self.target.ndim == 1:
            self.target[action_index] = value
        else:
            self.target[:, action_index] = value

    def _resolve_joint_name(self, token: str) -> str | None:
        return self.usd_name_by_alias.get(_normalize_name(token))

    def _limits_for_usd_joint(self, usd_name: str) -> torch.Tensor:
        joint_index = self.joint_index_by_usd_name[usd_name]
        return self.robot.data.joint_pos_limits[0, joint_index]

    def _format_joint(self, usd_name: str, value_rad: float) -> str:
        lerobot_name = USD_TO_LEROBOT_JOINT_NAMES.get(usd_name, usd_name)
        return f"{lerobot_name:14s} ({usd_name:11s}) {math.degrees(value_rad):8.3f} deg  {value_rad:8.4f} rad"

    def print_help(self) -> None:
        print(
            "\nCommands:\n"
            "  <joint> min         Move one joint to its live USD/PhysX minimum limit\n"
            "  <joint> max         Move one joint to its live USD/PhysX maximum limit\n"
            "  <joint> mid         Move one joint to midpoint of its live limits\n"
            "  <joint> <deg>deg    Move one joint to an explicit degree value\n"
            "  list                Print live joint limits parsed from the loaded USD\n"
            "  current             Print current measured joint positions\n"
            "  target              Print current commanded target positions\n"
            "  reset               Return to the environment default joint pose\n"
            "  help                Show this help\n"
            "  quit                Exit\n"
            "\nJoint names can be LeRobot names like shoulder_pan/wrist_flex/gripper or USD names like Rotation/Jaw.\n"
        )

    def print_limits(self) -> None:
        print("\nLive joint limits:")
        for usd_name in ACTION_JOINT_NAMES:
            lower, upper = self._limits_for_usd_joint(usd_name).detach().cpu().tolist()
            lerobot_name = USD_TO_LEROBOT_JOINT_NAMES.get(usd_name, usd_name)
            print(
                f"  {lerobot_name:14s} ({usd_name:11s}) "
                f"min={math.degrees(lower):8.3f} deg  max={math.degrees(upper):8.3f} deg"
            )
        if self.limit_margin_rad:
            print(f"  using margin: {args_cli.limit_margin_deg:.3f} deg")
        print()

    def print_current(self) -> None:
        print("\nCurrent measured joint positions:")
        for usd_name in ACTION_JOINT_NAMES:
            joint_index = self.joint_index_by_usd_name[usd_name]
            value = float(self.robot.data.joint_pos[0, joint_index].detach().cpu())
            print(f"  {self._format_joint(usd_name, value)}")
        print()

    def print_target(self) -> None:
        print("\nCurrent commanded joint targets:")
        target = self._target_view().detach().cpu().tolist()
        for action_index, usd_name in enumerate(ACTION_JOINT_NAMES):
            print(f"  {self._format_joint(usd_name, float(target[action_index]))}")
        print()

    def reset_target(self) -> None:
        self.target = self._initial_action()
        print("[INFO]: Reset commanded target to the environment default joint pose.")

    def _parse_position(self, usd_name: str, token: str) -> float | None:
        normalized = _normalize_name(token)
        lower, upper = self._limits_for_usd_joint(usd_name).detach().cpu().tolist()
        lower += self.limit_margin_rad
        upper -= self.limit_margin_rad

        if normalized in {"min", "minimum", "lower", "closed", "close"}:
            return lower
        if normalized in {"max", "maximum", "upper", "open"}:
            return upper
        if normalized in {"mid", "middle", "center", "centre"}:
            return 0.5 * (lower + upper)
        if normalized in {"zero", "home"}:
            return 0.0

        try:
            if normalized.endswith("deg"):
                return math.radians(float(normalized[:-3]))
            if normalized.endswith("rad"):
                return float(normalized[:-3])
            return math.radians(float(normalized))
        except ValueError:
            return None

    def handle_command(self, command: str) -> bool:
        command = command.strip()
        if not command:
            return True

        normalized = _normalize_name(command)
        if normalized in {"q", "quit", "exit"}:
            print("[INFO]: Exiting joint-limit UI.")
            return False
        if normalized in {"h", "help", "?"}:
            self.print_help()
            return True
        if normalized in {"list", "limits"}:
            self.print_limits()
            return True
        if normalized in {"current", "pos", "position", "positions"}:
            self.print_current()
            return True
        if normalized in {"target", "targets", "command", "commands"}:
            self.print_target()
            return True
        if normalized in {"reset", "default"}:
            self.reset_target()
            return True

        parts = command.split()
        if len(parts) != 2:
            print(f"[WARN]: Expected '<joint> <target>', got {command!r}. Type 'help' for examples.")
            return True

        usd_name = self._resolve_joint_name(parts[0])
        if usd_name is None:
            print(f"[WARN]: Unknown joint {parts[0]!r}. Type 'list' to see accepted joints.")
            return True

        value = self._parse_position(usd_name, parts[1])
        if value is None:
            print(f"[WARN]: Unknown target {parts[1]!r}. Use min, max, mid, zero, 42deg, or 0.73rad.")
            return True

        lower, upper = self._limits_for_usd_joint(usd_name).detach().cpu().tolist()
        clamped = min(upper, max(lower, value))
        action_index = ACTION_JOINT_NAMES.index(usd_name)
        self._set_action_value(action_index, clamped)
        print(f"[INFO]: Commanded {self._format_joint(usd_name, clamped)}")
        if clamped != value:
            print(
                f"[INFO]: Requested value was clamped to live limits "
                f"[{math.degrees(lower):.3f}, {math.degrees(upper):.3f}] deg."
            )
        if args_cli.print_after_command:
            self.print_current()
        return True


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    commander = JointLimitCommander(env)
    commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    threading.Thread(target=_command_loop, args=(commands,), daemon=True).start()

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("[INFO]: Isaac Sim is running. Type 'help' in this terminal for commands.")
    commander.print_limits()

    keep_running = True
    while simulation_app.is_running() and keep_running:
        while not commands.empty():
            keep_running = commander.handle_command(commands.get())
            if not keep_running:
                break

        with torch.inference_mode():
            env.step(commander.action())

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
