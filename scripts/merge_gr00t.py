#!/usr/bin/env python3
"""Merge a pretrained GR00T checkpoint with a fine-tuned GR00T checkpoint.

The RETAIN-style merge used here is a uniform linear interpolation over every
matching model tensor:

    merged = (1 - alpha) * pretrained + alpha * finetuned

By default this script merges the local GR00T-N1.6-3B base checkpoint with the
SO-101 working-memory fine-tune and writes one inference checkpoint per alpha
under ``checkpoints/``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_PRETRAINED = Path("checkpoints/GR00T-N1.6-3B")
DEFAULT_FINETUNED = Path("checkpoints/so101_GR00T_N1.6-3B_WM_v7_50k/checkpoint-52000")
DEFAULT_OUTPUT_TEMPLATE = "so101_GR00T_N1.6-3B_WM_v7_50k_merge_alpha_{alpha_tag}"

TRAINING_ONLY_FILES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "wandb_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=DEFAULT_PRETRAINED,
        help=f"Base/pretrained GR00T checkpoint. Defaults to {DEFAULT_PRETRAINED}.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=DEFAULT_FINETUNED,
        help=f"Fine-tuned GR00T checkpoint. Defaults to {DEFAULT_FINETUNED}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where merged checkpoint directories are written.",
    )
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.6, 0.8],
        help="Merge coefficients alpha. Alpha weights the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default=DEFAULT_OUTPUT_TEMPLATE,
        help=(
            "Output directory name template. Available fields: {alpha}, {alpha_tag}, "
            "{pretrained_name}, {finetuned_name}."
        ),
    )
    parser.add_argument(
        "--save-dtype",
        choices=("finetuned", "float32", "bfloat16"),
        default="finetuned",
        help=(
            "Dtype for saved tensors. 'finetuned' preserves each fine-tuned tensor dtype, "
            "which is the default and keeps the checkpoint layout closest to the input fine-tune."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing merged checkpoint directories with the same names.",
    )
    return parser.parse_args()


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def output_dir_for(args: argparse.Namespace, alpha: float) -> Path:
    name = args.output_template.format(
        alpha=f"{alpha:g}",
        alpha_tag=alpha_tag(alpha),
        pretrained_name=args.pretrained_checkpoint.name,
        finetuned_name=args.finetuned_checkpoint.name,
    )
    return args.output_root / name


def load_index(checkpoint: Path) -> dict[str, Any]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text())
    if "weight_map" not in index:
        raise ValueError(f"Safetensors index has no weight_map: {index_path}")
    return index


def validate_inputs(
    pretrained_dir: Path, finetuned_dir: Path, pretrained_index: dict[str, Any], finetuned_index: dict[str, Any]
) -> None:
    pretrained_map = pretrained_index["weight_map"]
    finetuned_map = finetuned_index["weight_map"]

    pretrained_keys = set(pretrained_map)
    finetuned_keys = set(finetuned_map)
    if pretrained_keys != finetuned_keys:
        missing_from_pretrained = sorted(finetuned_keys - pretrained_keys)
        missing_from_finetuned = sorted(pretrained_keys - finetuned_keys)
        raise ValueError(
            "Checkpoint tensor names do not match. "
            f"Missing from pretrained: {missing_from_pretrained[:10]}; "
            f"missing from finetuned: {missing_from_finetuned[:10]}"
        )

    for root, weight_map in ((pretrained_dir, pretrained_map), (finetuned_dir, finetuned_map)):
        for shard_name in sorted(set(weight_map.values())):
            shard_path = root / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard referenced by index: {shard_path}")

    with ExitStack() as stack:
        pretrained_readers = {
            shard_name: stack.enter_context(safe_open(pretrained_dir / shard_name, framework="pt", device="cpu"))
            for shard_name in sorted(set(pretrained_map.values()))
        }
        finetuned_readers = {
            shard_name: stack.enter_context(safe_open(finetuned_dir / shard_name, framework="pt", device="cpu"))
            for shard_name in sorted(set(finetuned_map.values()))
        }

        mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for key in sorted(finetuned_map):
            pretrained_tensor = pretrained_readers[pretrained_map[key]].get_tensor(key)
            finetuned_tensor = finetuned_readers[finetuned_map[key]].get_tensor(key)
            if pretrained_tensor.shape != finetuned_tensor.shape:
                mismatches.append((key, tuple(pretrained_tensor.shape), tuple(finetuned_tensor.shape)))

    if mismatches:
        preview = "\n".join(
            f"  {key}: {pretrained_shape} vs {finetuned_shape}"
            for key, pretrained_shape, finetuned_shape in mismatches[:10]
        )
        raise ValueError(f"Checkpoint tensor shapes do not match:\n{preview}")


def target_dtype(finetuned_tensor: torch.Tensor, save_dtype: str) -> torch.dtype:
    if save_dtype == "finetuned":
        return finetuned_tensor.dtype
    if save_dtype == "float32":
        return torch.float32
    if save_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported save dtype: {save_dtype}")


def merge_tensor(
    pretrained_tensor: torch.Tensor,
    finetuned_tensor: torch.Tensor,
    alpha: float,
    save_dtype: str,
) -> torch.Tensor:
    if not pretrained_tensor.is_floating_point() or not finetuned_tensor.is_floating_point():
        if torch.equal(pretrained_tensor, finetuned_tensor):
            return finetuned_tensor.clone().contiguous()
        raise TypeError("Cannot linearly merge non-floating tensors unless they are identical.")

    merged = torch.lerp(pretrained_tensor.to(torch.float32), finetuned_tensor.to(torch.float32), alpha)
    return merged.to(dtype=target_dtype(finetuned_tensor, save_dtype)).contiguous()


def copy_inference_files(source_dir: Path, output_dir: Path) -> None:
    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.name == "model.safetensors.index.json" or source_path.suffix == ".safetensors":
            continue
        if source_path.name in TRAINING_ONLY_FILES:
            continue
        shutil.copy2(source_path, output_dir / source_path.name)


def write_merge_config(
    output_dir: Path,
    pretrained_dir: Path,
    finetuned_dir: Path,
    alpha: float,
    save_dtype: str,
    tensor_count: int,
    shard_count: int,
) -> None:
    merge_config = {
        "alpha": alpha,
        "formula": "merged = (1 - alpha) * pretrained + alpha * finetuned",
        "pretrained_checkpoint": str(pretrained_dir.resolve()),
        "finetuned_checkpoint": str(finetuned_dir.resolve()),
        "save_dtype": save_dtype,
        "tensor_count": tensor_count,
        "shard_count": shard_count,
    }
    (output_dir / "merge_config.json").write_text(json.dumps(merge_config, indent=2) + "\n")


def bytes_for_tensor(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def write_index(output_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def validate_output(output_dir: Path, weight_map: dict[str, str], expected_total_size: int) -> None:
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        keys_by_shard[shard_name].append(key)

    actual_total_size = 0
    for shard_name, expected_keys in sorted(keys_by_shard.items()):
        shard_path = output_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Expected output shard missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            actual_keys = set(reader.keys())
            expected_key_set = set(expected_keys)
            if actual_keys != expected_key_set:
                missing = sorted(expected_key_set - actual_keys)
                extra = sorted(actual_keys - expected_key_set)
                raise ValueError(f"{shard_path} key mismatch. Missing: {missing[:10]}; extra: {extra[:10]}")
            for key in expected_keys:
                actual_total_size += bytes_for_tensor(reader.get_tensor(key))

    if actual_total_size != expected_total_size:
        raise ValueError(f"Output total tensor size mismatch: expected {expected_total_size}, got {actual_total_size}")


def merge_checkpoint(
    args: argparse.Namespace,
    alpha: float,
    pretrained_index: dict[str, Any],
    finetuned_index: dict[str, Any],
) -> Path:
    output_dir = output_dir_for(args, alpha)
    tmp_dir = output_dir.with_name(f"{output_dir.name}.tmp")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    pretrained_map = pretrained_index["weight_map"]
    output_weight_map = dict(finetuned_index["weight_map"])
    keys_by_output_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in output_weight_map.items():
        keys_by_output_shard[shard_name].append(key)

    print(f"[merge] alpha={alpha:g} -> {output_dir}")
    try:
        copy_inference_files(args.finetuned_checkpoint, tmp_dir)

        total_size = 0
        with ExitStack() as stack:
            pretrained_readers = {
                shard_name: stack.enter_context(
                    safe_open(args.pretrained_checkpoint / shard_name, framework="pt", device="cpu")
                )
                for shard_name in sorted(set(pretrained_map.values()))
            }
            finetuned_readers = {
                shard_name: stack.enter_context(
                    safe_open(args.finetuned_checkpoint / shard_name, framework="pt", device="cpu")
                )
                for shard_name in sorted(set(output_weight_map.values()))
            }

            for shard_index, (output_shard_name, keys) in enumerate(sorted(keys_by_output_shard.items()), start=1):
                print(
                    f"[merge]   shard {shard_index}/{len(keys_by_output_shard)}: "
                    f"{output_shard_name} ({len(keys)} tensors)"
                )
                shard_tensors: dict[str, torch.Tensor] = {}
                for key in sorted(keys):
                    pretrained_tensor = pretrained_readers[pretrained_map[key]].get_tensor(key)
                    finetuned_tensor = finetuned_readers[output_weight_map[key]].get_tensor(key)
                    merged_tensor = merge_tensor(pretrained_tensor, finetuned_tensor, alpha, args.save_dtype)
                    shard_tensors[key] = merged_tensor
                    total_size += bytes_for_tensor(merged_tensor)

                save_file(shard_tensors, tmp_dir / output_shard_name)
                del shard_tensors
                gc.collect()

        write_index(tmp_dir, output_weight_map, total_size)
        write_merge_config(
            tmp_dir,
            args.pretrained_checkpoint,
            args.finetuned_checkpoint,
            alpha,
            args.save_dtype,
            len(output_weight_map),
            len(keys_by_output_shard),
        )
        validate_output(tmp_dir, output_weight_map, total_size)
        tmp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"[merge] wrote {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    args.pretrained_checkpoint = args.pretrained_checkpoint.expanduser()
    args.finetuned_checkpoint = args.finetuned_checkpoint.expanduser()
    args.output_root = args.output_root.expanduser()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for alpha in args.coefficients:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Merge coefficient alpha must be in [0, 1], got {alpha}")

    pretrained_index = load_index(args.pretrained_checkpoint)
    finetuned_index = load_index(args.finetuned_checkpoint)
    validate_inputs(args.pretrained_checkpoint, args.finetuned_checkpoint, pretrained_index, finetuned_index)

    outputs = [
        merge_checkpoint(args, alpha, pretrained_index=pretrained_index, finetuned_index=finetuned_index)
        for alpha in args.coefficients
    ]
    print("[merge] complete")
    for output in outputs:
        print(f"[merge]   {output}")


if __name__ == "__main__":
    main()
