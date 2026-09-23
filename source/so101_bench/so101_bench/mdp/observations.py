"""Observation terms for the SO-101 benchmark environments."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the end-effector frame pose in the robot root frame."""

    robot = env.scene[robot_cfg.name]
    robot_root_pos = robot.data.root_pos_w
    robot_root_quat = robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    return torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)


def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera_overhead"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """Return a camera tensor without normalization or reshaping."""

    sensor = env.scene[sensor_cfg.name]
    return sensor.data.output[data_type].clone()


def _gaussian_blur_rgb(images: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply a small separable Gaussian blur to an NCHW RGB tensor."""

    if sigma_px <= 0.0:
        return images
    radius = max(1, int(round(3.0 * sigma_px)))
    coordinates = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma_px * sigma_px))
    kernel /= kernel.sum()
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    images = functional.pad(images, (radius, radius, 0, 0), mode="replicate")
    images = functional.conv2d(images, horizontal, groups=channels)
    images = functional.pad(images, (0, 0, radius, radius), mode="replicate")
    return functional.conv2d(images, vertical, groups=channels)


def _shift_principal_point(
    images: torch.Tensor,
    x_offset: float,
    y_offset: float,
) -> torch.Tensor:
    """Shift image content by fractional image dimensions without wraparound."""

    if x_offset == 0.0 and y_offset == 0.0:
        return images
    theta = torch.tensor(
        [[[1.0, 0.0, -2.0 * x_offset], [0.0, 1.0, -2.0 * y_offset]]],
        dtype=images.dtype,
        device=images.device,
    )
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def apply_visual_camera_postprocessing(
    env: ManagerBasedEnv,
    images: torch.Tensor,
    camera_name: str,
) -> torch.Tensor:
    """Apply task-row camera exposure, optical-center, blur, and noise values.

    Exposure is measured in stops (EV), principal-point offsets are fractions
    of image width/height, noise standard deviation is normalized to the
    ``[0, 1]`` RGB range, and blur sigma is in pixels.  With all-zero defaults
    the source render is returned byte-for-byte apart from the required clone.
    """

    if camera_name.startswith("camera_"):
        camera_name = camera_name.removeprefix("camera_")
    if camera_name not in {"overhead", "wrist"} or images.ndim != 4:
        return images.clone()

    settings_by_env = getattr(env, "_so101_domain_randomization", None)
    if not isinstance(settings_by_env, list) or len(settings_by_env) != images.shape[0]:
        return images.clone()

    prefix = f"{camera_name}_camera"
    output = images.clone()
    for env_id, settings in enumerate(settings_by_env):
        exposure_ev = float(settings.get(f"{prefix}_exposure_ev", 0.0))
        x_offset = float(settings.get(f"{prefix}_principal_point_x_offset", 0.0))
        y_offset = float(settings.get(f"{prefix}_principal_point_y_offset", 0.0))
        noise_std = float(settings.get(f"{prefix}_noise_std", 0.0))
        blur_sigma_px = float(settings.get(f"{prefix}_blur_sigma_px", 0.0))
        if exposure_ev == 0.0 and x_offset == 0.0 and y_offset == 0.0 and noise_std == 0.0 and blur_sigma_px == 0.0:
            continue

        source = images[env_id : env_id + 1, ..., :3]
        source_is_float = torch.is_floating_point(source)
        rgb = source.to(dtype=torch.float32)
        if not source_is_float:
            rgb = rgb / 255.0
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = _shift_principal_point(rgb, x_offset, y_offset)
        rgb = _gaussian_blur_rgb(rgb, blur_sigma_px)
        if exposure_ev != 0.0:
            rgb = rgb * (2.0**exposure_ev)
        if noise_std != 0.0:
            rgb = rgb + torch.randn_like(rgb) * noise_std
        rgb = rgb.clamp_(0.0, 1.0).permute(0, 2, 3, 1)
        if source_is_float:
            output[env_id : env_id + 1, ..., :3] = rgb.to(dtype=source.dtype)
        else:
            output[env_id : env_id + 1, ..., :3] = (rgb * 255.0).round().to(dtype=source.dtype)
    return output


def visual_randomized_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera_overhead"),
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    """Return a camera image with the current episode's visual settings."""

    sensor = env.scene[sensor_cfg.name]
    images = sensor.data.output[data_type]
    if data_type != "rgb":
        return images.clone()
    images = apply_visual_camera_postprocessing(env, images, sensor_cfg.name)
    if normalize:
        images = images.float() / 255.0
        images -= torch.mean(images, dim=(1, 2), keepdim=True)
    return images
