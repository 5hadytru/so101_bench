"""GR00T N1.7 modality config for fine-tuning on so101_bench_sim_WM_v2.1.

Pass to Isaac-GR00T's launch_finetune.py with
``--modality-config-path <this file> --embodiment-tag NEW_EMBODIMENT``.

Keys must match the dataset's meta/modality.json. The layout mirrors the N1.6
WM v7 fine-tune (16-step chunk, relative arm joints, absolute gripper) plus the
settled wrist frame (``wrist_init``) that v2.1 records. scripts/groot_eval.py
sends exactly these keys with its defaults (no --rename_map, --use_overhead_init
true, --use_wrist_init true) and checks them against the served checkpoint.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so101_sim_wm_v2_1_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["overhead", "wrist", "overhead_init", "wrist_init"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(so101_sim_wm_v2_1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
