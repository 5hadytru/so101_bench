"""SO-101-specific action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JawErrorLimitedJointPositionAction(JointPositionAction):
    """Absolute joint targets with the jaw drive error bounded around its measured position."""

    cfg: JawErrorLimitedJointPositionActionCfg

    def __init__(self, cfg: JawErrorLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        try:
            self._jaw_action_index = self._joint_names.index(cfg.jaw_joint_name)
        except ValueError as exc:
            raise ValueError(
                f"Jaw joint {cfg.jaw_joint_name!r} is not part of this action term's joints: {self._joint_names}."
            ) from exc
        if cfg.max_jaw_position_error <= 0.0:
            raise ValueError(
                "max_jaw_position_error must be positive, got "
                f"{cfg.max_jaw_position_error!r}."
            )

    def apply_actions(self) -> None:
        targets = self.processed_actions.clone()
        jaw_position = self._asset.data.joint_pos[:, self._joint_ids][:, self._jaw_action_index]
        jaw_target = targets[:, self._jaw_action_index]
        targets[:, self._jaw_action_index] = torch.clamp(
            jaw_target,
            min=jaw_position - self.cfg.max_jaw_position_error,
            max=jaw_position + self.cfg.max_jaw_position_error,
        )
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class JawErrorLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`JawErrorLimitedJointPositionAction`."""

    class_type: type[ActionTerm] = JawErrorLimitedJointPositionAction

    jaw_joint_name: str = "Jaw"
    """Name of the jaw joint whose drive error is limited."""

    max_jaw_position_error: float = 0.30
    """Maximum absolute jaw target-position error in radians.

    Re-applied every physics step, so it acts as a torque cap of
    ``jaw_stiffness * max_jaw_position_error`` rather than a per-control-step limit.
    """
