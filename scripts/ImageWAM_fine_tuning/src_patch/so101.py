from __future__ import annotations

import torch


class SO101ToCanonicalEE16:
    """Pad SO-101 6D joint vectors into the 16D layout of the InternData-A1-EE pretrain weight.

    Like RobotWin14ToCanonicalEE16, this does not convert joints into end-effector poses. The five
    arm joints fill the left-arm pose slots [0, 5) and the gripper the left-gripper slot 7, so the
    gripper lands where every pretrain embodiment kept its gripper. The other ten dimensions are
    zero-filled and masked (the normalizer zeroes them, the action loss ignores them).
    """

    ARM = slice(0, 5)
    GRIPPER = 7

    def __init__(self, key: str = "default"):
        self.key = str(key)

    @classmethod
    def _to_canonical(cls, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 6:
            raise ValueError(f"Expected 6D SO-101 vector, got {x.shape[-1]}D.")
        out = x.new_zeros(*x.shape[:-1], 16)
        out[..., cls.ARM] = x[..., 0:5]
        out[..., cls.GRIPPER] = x[..., 5]
        return out

    @classmethod
    def _from_canonical(cls, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 16:
            raise ValueError(f"Expected 16D canonical EE vector, got {x.shape[-1]}D.")
        return torch.cat([x[..., cls.ARM], x[..., cls.GRIPPER : cls.GRIPPER + 1]], dim=-1)

    @classmethod
    def _canonical_dim_mask(cls, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.ones(16, dtype=torch.bool, device=device)
        mask[cls.ARM] = False
        mask[cls.GRIPPER] = False
        return mask

    def forward(self, batch: dict) -> dict:
        if "action" in batch and self.key in batch["action"]:
            batch["action"][self.key] = self._to_canonical(batch["action"][self.key])
            batch["action_dim_is_pad"] = self._canonical_dim_mask(batch["action"][self.key].device)

        if "state" in batch and self.key in batch["state"]:
            batch["state"][self.key] = self._to_canonical(batch["state"][self.key])
            batch["state_dim_is_pad"] = self._canonical_dim_mask(batch["state"][self.key].device)

        batch["embodiment"] = batch.get("embodiment", "so101_ee16")
        return batch

    def backward(self, batch: dict) -> dict:
        if "action" in batch and self.key in batch["action"]:
            batch["action"][self.key] = self._from_canonical(batch["action"][self.key])
        if "state" in batch and self.key in batch["state"]:
            batch["state"][self.key] = self._from_canonical(batch["state"][self.key])
        return batch
