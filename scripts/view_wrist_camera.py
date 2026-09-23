# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""View or export SO-101 Bench camera observations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View or save SO-101 Bench camera frames.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument(
    "--camera",
    choices=("wrist", "overhead", "overhead_init", "all"),
    default="all",
    help="Camera observation to inspect. Use 'all' to save wrist, overhead, and overhead_init.",
)
parser.add_argument("--steps", type=int, default=240, help="Number of sim steps to run before exiting.")
parser.add_argument("--save_every", type=int, default=30, help="Save every N steps. Use 0 to disable saving.")
parser.add_argument("--output_dir", type=Path, default=Path("logs/camera_view"), help="Directory for saved frames.")
parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window when cv2 is available.")
parser.add_argument(
    "--policy_image_width",
    type=int,
    default=640,
    help="Resize saved/displayed camera frames to this policy input width. Use 0 to disable resizing.",
)
parser.add_argument(
    "--policy_image_height",
    type=int,
    default=480,
    help="Resize saved/displayed camera frames to this policy input height. Use 0 to disable resizing.",
)
parser.add_argument(
    "--overhead_init_time_s",
    type=float,
    default=1.0,
    help="Seconds to hold the initial joint pose before capturing the fixed overhead_init frame.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401

ACTION_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")


def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image tensor, got shape {array.shape}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected image channel dimension of 3 or 4, got shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    return np.ascontiguousarray(array.astype(np.uint8))


def _resize_rgb(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return image

    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image

    try:
        import cv2

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)
    except Exception:
        pass

    try:
        from PIL import Image

        resized = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
    except Exception:
        pass

    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return np.ascontiguousarray(image[y_idx][:, x_idx])


def _policy_rgb(image: torch.Tensor, image_size: tuple[int, int] | None) -> np.ndarray:
    return _resize_rgb(_to_uint8_rgb(image), image_size)


def _camera_names(selected_camera: str) -> tuple[str, ...]:
    if selected_camera == "all":
        return ("wrist", "overhead", "overhead_init")
    return (selected_camera,)


def _camera_obs_key(camera_name: str) -> str:
    if camera_name == "overhead_init":
        return "rgb_overhead"
    return f"rgb_{camera_name}"


def _camera_cfg_name(camera_name: str) -> str:
    if camera_name == "overhead_init":
        return "camera_overhead"
    return f"camera_{camera_name}"


def _write_image(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        png_path = path.with_suffix(".png")
        cv2.imwrite(str(png_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return png_path
    except Exception:
        ppm_path = path.with_suffix(".ppm")
        with ppm_path.open("wb") as file:
            file.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
            file.write(rgb.tobytes())
        return ppm_path


def _maybe_display(window_name: str, rgb: np.ndarray) -> bool:
    if not args_cli.display:
        return False
    try:
        import cv2

        cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return cv2.waitKey(1) == 27
    except Exception as exc:
        print(f"[WARN]: OpenCV display unavailable: {exc}")
        args_cli.display = False
        return False


def _initial_robot_action(env) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.joint_names.index(joint_name) for joint_name in ACTION_JOINT_NAMES]
    return robot.data.default_joint_pos[0, joint_ids].clone()


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    image_size = (
        (args_cli.policy_image_width, args_cli.policy_image_height)
        if args_cli.policy_image_width > 0 and args_cli.policy_image_height > 0
        else None
    )
    if image_size is not None:
        print(f"[INFO]: Saving/displaying policy video frames resized to {image_size[0]}x{image_size[1]}")
    else:
        print("[INFO]: Saving/displaying raw camera frames without policy resizing.")

    camera_names = _camera_names(args_cli.camera)
    hold_action = _initial_robot_action(env)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    actions[:] = hold_action

    if "overhead_init" in camera_names:
        overhead_init_steps = max(0, math.ceil(args_cli.overhead_init_time_s / float(env.unwrapped.step_dt)))
        for _ in range(overhead_init_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                obs, _, _, _, _ = env.step(actions)

    overhead_init_rgb = None
    for camera_name in camera_names:
        camera_key = _camera_obs_key(camera_name)
        if camera_key not in obs["visual"]:
            raise KeyError(f"Expected visual observation key {camera_key!r}; got: {list(obs['visual'].keys())}")

        camera_cfg = getattr(env.unwrapped.scene.cfg, _camera_cfg_name(camera_name))
        print(f"[INFO]: Viewing camera '{camera_name}' ({camera_cfg.width}x{camera_cfg.height})")
        print(f"[INFO]: Observation tensor: {tuple(obs['visual'][camera_key].shape)} {obs['visual'][camera_key].dtype}")
        if camera_name == "overhead_init":
            overhead_init_rgb = _policy_rgb(obs["visual"][camera_key][0], image_size)

    if args_cli.display and len(camera_names) > 1:
        print("[INFO]: Displaying one OpenCV window per selected camera.")

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            obs, _, _, _, _ = env.step(actions)

        should_save = args_cli.save_every > 0 and (step == 0 or step % args_cli.save_every == 0)
        should_exit = False
        for camera_name in camera_names:
            if camera_name == "overhead_init":
                rgb = overhead_init_rgb
            else:
                rgb = _policy_rgb(obs["visual"][_camera_obs_key(camera_name)][0], image_size)

            if rgb is None:
                raise RuntimeError("Expected an overhead_init frame captured after the initial hold.")
            if should_save:
                saved_path = _write_image(args_cli.output_dir / f"{camera_name}_{step:05d}", rgb)
                print(f"[INFO]: Saved {saved_path}")
            should_exit = _maybe_display(f"SO-101 {camera_name} camera", rgb) or should_exit
        if should_exit:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
