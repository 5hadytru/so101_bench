# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactively edit SO-101 Bench episode layouts in the Isaac Sim viewport.

This is a layout-editing counterpart to ``scripts/so101_follower_teleop.py``: it
loads an ``episode_layouts_jsonl`` file, spawns each episode's objects at their
saved poses, and lets you translate/rotate the objects (and the plastic bin) with
the keyboard.  Pressing Enter writes the edited poses back into the layout JSONL
and advances to the next episode.

Physics is never stepped, so objects stay exactly where you put them.  Poses are
applied through the same write path the reset event uses to replay a layout
(:func:`so101_bench.mdp.resets._write_pose`), and the values written to the JSONL
are the exact poses handed to that write path, so an edited layout replays
identically in teleop / eval.

Keyboard controls (click the Isaac viewport first):

    Tab              select next target (objects, then the bin)
    1-9              select object by number (IDs are printed per episode)
    B                select the plastic bin
    Up/Down/Left/Right  translate along the world X/Y axes; the current mapping is
                     shown on the HUD "arrows:" line and adjusted live with X / M
    X                rotate the arrow->world mapping 90 degrees
    M                mirror the left/right arrows
    S                swap positions: press once to anchor the selected object,
                     select another object, then press S again to swap (yaws kept)
    , / .            rotate yaw CCW / CW
    [ / ]            smaller / bigger translation step
    - / =            smaller / bigger rotation step
    R                revert this episode to its last-saved poses (and instruction)
    P                print the selected pose (tracked vs. sim)
    Enter            save this episode's poses -> next episode
    N                skip to the next episode without saving
    Q                finish and exit

The same commands are accepted as words typed in the launch terminal
(``save``/``next``/``skip``/``revert``/``print``/``quit``, ``sel N``, ``bin``, ``ep N`` / ``goto N``).
Type ``swap A B`` to swap object A's and object B's positions while keeping each
object's yaw; ``swap B`` swaps the selected object with object B.
Type ``ep <N>`` or ``goto <N>`` to jump to episode N (1-based) without saving.
Type ``instr <text>`` (or ``instruction <text>``) in the terminal to rewrite the
current episode's instruction; on the next save the new text is written into
*both* the layout row and the matching ``--episodes_jsonl`` row (paired by
``trial_id``).  Instruction editing therefore requires ``--episodes_jsonl``:
eval/teleop take the instruction from that file, so editing only the layout row
would not change what actually gets replayed.  Free-text has no keyboard
shortcut, so instruction editing is terminal-only.  The new text is validated as
a loadable episode before it is accepted; explicit ``target``/``referents``/
``direction`` metadata is rewritten from the new instruction at save time.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
from typing import Any

from isaaclab.app import AppLauncher


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


parser = argparse.ArgumentParser(description="Interactively edit SO-101 Bench episode layouts in Isaac Sim.")
parser.add_argument(
    "--episode_layouts_jsonl",
    "--layouts_jsonl",
    dest="episode_layouts_jsonl",
    type=Path,
    required=True,
    help="Layout JSONL file to edit. Each row is one episode layout (objects + bin).",
)
parser.add_argument(
    "--episodes_jsonl",
    type=Path,
    default=None,
    help=(
        "Optional benchmark episodes JSONL (matching so101_follower_teleop). When given, layout rows are "
        "matched to episodes by trial_id. When omitted, the episode plan is derived from the layout rows "
        "themselves (object names, task_family, and instruction)."
    ),
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Where to write edited layouts. Defaults to editing --episode_layouts_jsonl in place.",
)
parser.add_argument(
    "--episodes_output",
    type=Path,
    default=None,
    help=(
        "Where to write edited instructions on the episodes side. Only used when --episodes_jsonl is "
        "given and an instruction is edited. Defaults to editing --episodes_jsonl in place."
    ),
)
parser.add_argument(
    "--no_backup",
    action="store_true",
    default=False,
    help="Do not write a one-time .bak copy of the layout file before the first in-place save.",
)
parser.add_argument("--task", type=str, default="So101Bench-Bin-v0", help="Isaac Lab task name.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument(
    "--start_episode",
    type=int,
    default=1,
    help="1-based layout row to start editing from.",
)
parser.add_argument(
    "--translate_step",
    type=float,
    default=0.005,
    help="Initial translation nudge in meters (adjust live with [ and ]).",
)
parser.add_argument(
    "--rotate_step_deg",
    type=float,
    default=5.0,
    help="Initial yaw nudge in degrees (adjust live with - and =).",
)
parser.add_argument(
    "--edit_bin",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Include the plastic bin as an editable target. Pass '--edit_bin false' to lock the bin.",
)
parser.add_argument(
    "--keep_stale_metadata",
    action="store_true",
    default=False,
    help=(
        "Keep the original sampler provenance verbatim on edited rows. By default, editing a row's poses "
        "nulls the now-inaccurate 'placement' stats (marking layout_selection='hand_edited'), drops the "
        "moved bin's stale pose_index, and stamps 'edited_at'."
    ),
)
parser.add_argument("--invert_x", action="store_true", default=False, help="Invert Up/Down world-X translation.")
parser.add_argument("--invert_y", action="store_true", default=False, help="Invert Left/Right world-Y translation.")
parser.add_argument(
    "--swap_xy",
    action="store_true",
    default=False,
    help="Swap the arrow-key axes so Up/Down move world Y and Left/Right move world X.",
)
parser.add_argument(
    "--terminal_control_stdin",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Accept edit commands typed in the launch terminal.",
)
parser.add_argument(
    "--keyboard_debug",
    action="store_true",
    default=False,
    help="Print every Isaac keyboard press seen by the editor.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Editing is inherently a single-scene activity; force one environment regardless of the task default.
args_cli.num_envs = 1
# Match the known-good teleop rendering path (RTX cameras + viewport) so lighting/materials converge.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import (
    TASK_FAMILIES,
    BenchmarkEpisodeSpec,
    episode_spec_from_json,
    infer_task_family,
    load_episode_jsonl,
    row_with_canonical_instruction_metadata,
)
from so101_bench.layouts import normalize_layout_object_slots
from so101_bench.mdp import benchmark_object_positions, benchmark_object_yaws
from so101_bench.mdp.resets import _rpy_quat, _write_pose, _yaw_quat
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import (
    configure_env_cfg_for_object_pool,
)


BIN_NAME = "plastic_bin"
MIN_TRANSLATE_STEP = 0.0005
MAX_TRANSLATE_STEP = 0.1
MIN_ROTATE_STEP_DEG = 0.5
MAX_ROTATE_STEP_DEG = 90.0


# --------------------------------------------------------------------------- #
# Layout / episode loading
# --------------------------------------------------------------------------- #
def _load_layout_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line.")
            if not isinstance(row.get("objects"), list) or not row["objects"]:
                raise ValueError(f"{path}:{line_no}: layout row is missing an 'objects' list.")
            if not isinstance(row.get("bin"), dict):
                raise ValueError(f"{path}:{line_no}: layout row is missing a 'bin' entry.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No layout rows found in {path}.")
    return rows


def _row_object_names(row: dict) -> tuple[str, ...]:
    entries = sorted(row["objects"], key=lambda entry: int(entry["slot"]))
    names = [str(entry["name"]) for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"Layout row for trial_id={row.get('trial_id')!r} has duplicate object names: {names}.")
    return tuple(names)


def _spec_from_layout_row(row: dict) -> BenchmarkEpisodeSpec:
    """Build a minimal episode spec from a layout row.

    Only ``objects``/``task_family``/``instruction`` affect object placement; the
    target/referent ids are placeholders that never move an object, so we fill
    them with safe defaults rather than re-deriving task semantics.
    """

    names = _row_object_names(row)
    recorded_task_family = str(row.get("task_family") or "").strip()
    if recorded_task_family not in TASK_FAMILIES:
        raise ValueError(
            f"Layout row for trial_id={row.get('trial_id')!r} has task_family={recorded_task_family!r}; "
            f"expected one of {sorted(TASK_FAMILIES)}. Pass --episodes_jsonl to supply episode metadata."
        )
    instruction = str(row.get("instruction", ""))
    # Legacy teleop_3 layouts labelled named-object bin rows as plain ``bin``.
    # The instruction is authoritative now that named-bin is a first-class family.
    task_family = infer_task_family(instruction)
    count = len(names)
    referents = (1 % count, 2 % count) if count > 1 else (0, 0)
    return BenchmarkEpisodeSpec(
        objects=names,
        instruction=instruction,
        task_family=task_family,
        target_object_id=0,
        referent_object_ids=referents,
        direction=None,
        metadata={"trial_id": row.get("trial_id", row.get("episode_index"))},
    )


def _match_specs_to_rows(rows: list[dict], episodes_jsonl: Path) -> list[BenchmarkEpisodeSpec]:
    specs = load_episode_jsonl(episodes_jsonl)
    specs_by_trial_id: dict[str, BenchmarkEpisodeSpec] = {}
    for index, spec in enumerate(specs):
        trial_id = (spec.metadata or {}).get("trial_id", index)
        specs_by_trial_id[str(trial_id)] = spec

    matched: list[BenchmarkEpisodeSpec] = []
    missing: list[Any] = []
    for row in rows:
        if "trial_id" not in row:
            raise ValueError(
                f"{episodes_jsonl} was provided, but a layout row has no trial_id to match against."
            )
        spec = specs_by_trial_id.get(str(row["trial_id"]))
        if spec is None:
            missing.append(row["trial_id"])
            continue
        if tuple(spec.objects) != _row_object_names(row):
            raise ValueError(
                f"Layout trial_id={row['trial_id']!r} objects {_row_object_names(row)} do not match "
                f"episode objects {tuple(spec.objects)}."
            )
        matched.append(spec)
    if missing:
        raise ValueError(f"{episodes_jsonl} is missing episodes for layout trial_id(s): {missing}.")
    return matched


def _build_episode_plan(rows: list[dict]) -> list[BenchmarkEpisodeSpec]:
    if args_cli.episodes_jsonl is not None:
        plan = _match_specs_to_rows(rows, args_cli.episodes_jsonl)
        print(f"[INFO]: Matched {len(plan)} layout row(s) to episodes in {args_cli.episodes_jsonl}.")
        return plan
    plan = [_spec_from_layout_row(row) for row in rows]
    print(f"[INFO]: Derived {len(plan)} episode plan(s) from the layout rows themselves.")
    return plan


def _load_episodes_raw_rows(path: Path) -> list[dict]:
    """Load the episodes JSONL as raw dicts we can edit and write back verbatim.

    ``load_episode_jsonl`` returns validated specs, not the original rows; to
    rewrite one episode's instruction in place (preserving every other field and
    every untouched row) we need the raw dicts.
    """

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line.")
            rows.append(row)
    return rows


def _match_episode_rows_to_layout(layout_rows: list[dict], episode_rows: list[dict]) -> list[dict | None]:
    """Align raw episode rows to layout rows by ``trial_id`` (same keying as specs).

    ``_match_specs_to_rows`` has already validated that every layout row has a
    matching episode, so a ``None`` here only guards against an unexpected keying
    mismatch (handled defensively at edit/save time).
    """

    by_trial_id: dict[str, dict] = {}
    for index, row in enumerate(episode_rows):
        by_trial_id[str(row.get("trial_id", index))] = row
    return [by_trial_id.get(str(row.get("trial_id"))) for row in layout_rows]


def _canonical_layout_instruction_row(row: dict, instruction: str) -> dict:
    task_row = {
        "objects": list(_row_object_names(row)),
        "instruction": instruction,
    }
    if "task_family" in row:
        task_row["task_family"] = row["task_family"]
    for key in ("target", "referents", "direction"):
        if key in row:
            task_row[key] = row[key]
    return row_with_canonical_instruction_metadata(
        task_row,
        source=f"layout trial_id={row.get('trial_id')!r}",
    )


def _apply_instruction_fields(row: dict, canonical: dict) -> None:
    for key in ("target", "referents", "direction"):
        row.pop(key, None)
    row["instruction"] = canonical["instruction"]
    if "task_family" in row and "task_family" in canonical:
        row["task_family"] = canonical["task_family"]
    for key in ("target", "referents", "direction"):
        if key in canonical:
            row[key] = canonical[key]


# --------------------------------------------------------------------------- #
# Object pool / per-episode env configuration (mirrors so101_follower_teleop)
# --------------------------------------------------------------------------- #
def _episode_object_pool(episode_plan: list[BenchmarkEpisodeSpec]) -> list[str]:
    object_pool: list[str] = []
    seen: set[str] = set()
    for episode in episode_plan:
        for object_name in episode.objects:
            if object_name in seen:
                continue
            seen.add(object_name)
            object_pool.append(object_name)
    return object_pool


def _episode_pool_payload(episode: BenchmarkEpisodeSpec, pool_index_by_name: dict[str, int]) -> dict[str, Any]:
    payload = episode.reset_payload()
    local_to_pool = [pool_index_by_name[object_name] for object_name in episode.objects]
    payload["active_object_ids"] = local_to_pool
    payload["target_object_id"] = local_to_pool[episode.target_object_id]
    payload["referent_object_ids"] = [local_to_pool[object_id] for object_id in episode.referent_object_ids]
    return payload


def _episode_pool_layout(
    episode: BenchmarkEpisodeSpec,
    episode_layout: dict,
    pool_index_by_name: dict[str, int],
) -> dict:
    remapped_layout = dict(episode_layout)
    remapped_objects = []
    for entry in episode_layout.get("objects", []):
        remapped_entry = dict(entry)
        local_slot = int(remapped_entry["slot"])
        object_name = str(remapped_entry.get("name") or episode.objects[local_slot])
        pool_slot = pool_index_by_name[object_name]
        remapped_entry["slot"] = pool_slot
        remapped_entry["asset_name"] = f"object_{pool_slot + 1}"
        remapped_objects.append(remapped_entry)
    remapped_layout["objects"] = remapped_objects
    return remapped_layout


def _configure_env_for_episode(
    env,
    episode: BenchmarkEpisodeSpec,
    layout_row: dict,
    object_pool: list[str],
    object_asset_names: list[str],
) -> None:
    pool_index_by_name = {name: object_id for object_id, name in enumerate(object_pool)}
    normalized = normalize_layout_object_slots(layout_row, episode.objects)
    pool_layout = _episode_pool_layout(episode, normalized, pool_index_by_name)
    payload = _episode_pool_payload(episode, pool_index_by_name)
    params = {
        "object_asset_names": object_asset_names,
        "object_labels": object_pool,
        "task_family": episode.task_family,
        "object_count_range": (len(episode.objects), len(episode.objects)),
        "active_object_selection": "fixed",
        "fixed_active_object_ids": tuple(payload["active_object_ids"]),
        "shuffle_object_labels": False,
        "force_bin_all_objects_instruction": False,
        "episode_spec": payload,
        "episode_layout": pool_layout,
    }
    env.unwrapped.cfg.events.reset_benchmark_scene.params.update(params)
    env.unwrapped.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)


def _make_env(object_pool: list[str]) -> tuple[gym.Env, list[str]]:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    object_asset_names = configure_env_cfg_for_object_pool(env_cfg, object_pool)
    # Nothing should auto-terminate while editing (we never step physics anyway).
    env_cfg.terminations.success = None
    env_cfg.terminations.failure = None
    env_cfg.terminations.time_out = None
    return gym.make(args_cli.task, cfg=env_cfg), object_asset_names


def _reset_env(env) -> None:
    env.reset()
    unwrapped = env.unwrapped
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    num_rerenders = getattr(unwrapped.cfg, "num_rerenders_on_reset", 0)
    if unwrapped.sim.has_rtx_sensors() and num_rerenders > 0:
        for _ in range(num_rerenders):
            unwrapped.sim.render()


# --------------------------------------------------------------------------- #
# Editable targets
# --------------------------------------------------------------------------- #
class _EditTarget:
    """One movable thing in the scene, backed by a layout JSONL entry."""

    def __init__(
        self,
        *,
        label: str,
        asset_name: str,
        asset,
        entry: dict,
        is_bin: bool,
        position: list[float],
        rpy: list[float],
    ):
        self.label = label
        self.asset_name = asset_name
        self.asset = asset
        self.entry = entry  # reference into the original layout row, mutated on save
        self.is_bin = is_bin
        self.position = [float(position[0]), float(position[1]), float(position[2])]
        # Objects only carry a yaw; the bin keeps its authored roll/pitch and edits yaw.
        self.rpy = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
        self.modified = False  # set once this target's pose is nudged

    @property
    def yaw(self) -> float:
        return self.rpy[2]

    def quat(self, device: str) -> torch.Tensor:
        if self.is_bin:
            return _rpy_quat((self.rpy[0], self.rpy[1], self.rpy[2]), device)
        return _yaw_quat(self.rpy[2], device)

    def commit_to_entry(self) -> None:
        self.entry["position"] = [self.position[0], self.position[1], self.position[2]]
        self.entry["yaw"] = self.rpy[2]
        self.entry["rpy"] = [self.rpy[0], self.rpy[1], self.rpy[2]]
        if self.is_bin:
            self.entry["rpy_deg"] = [math.degrees(angle) for angle in self.rpy]


def _entry_position(entry: dict) -> list[float]:
    position = entry.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError(f"Layout entry is missing a 3D position: {entry!r}.")
    return [float(position[0]), float(position[1]), float(position[2])]


def _entry_rpy(entry: dict) -> list[float]:
    if isinstance(entry.get("rpy"), (list, tuple)) and len(entry["rpy"]) >= 3:
        return [float(entry["rpy"][0]), float(entry["rpy"][1]), float(entry["rpy"][2])]
    if "yaw" in entry:
        return [0.0, 0.0, float(entry["yaw"])]
    raise ValueError(f"Layout entry is missing a yaw/rpy rotation: {entry!r}.")


def _build_targets(env, row: dict, object_pool: list[str], object_asset_names: list[str]) -> list[_EditTarget]:
    pool_index_by_name = {name: object_id for object_id, name in enumerate(object_pool)}
    targets: list[_EditTarget] = []
    for entry in sorted(row["objects"], key=lambda item: int(item["slot"])):
        name = str(entry["name"])
        asset_name = object_asset_names[pool_index_by_name[name]]
        targets.append(
            _EditTarget(
                label=name,
                asset_name=asset_name,
                asset=env.unwrapped.scene[asset_name],
                entry=entry,
                is_bin=False,
                position=_entry_position(entry),
                rpy=_entry_rpy(entry),
            )
        )
    if args_cli.edit_bin:
        bin_entry = row["bin"]
        targets.append(
            _EditTarget(
                label="plastic bin",
                asset_name=BIN_NAME,
                asset=env.unwrapped.scene[BIN_NAME],
                entry=bin_entry,
                is_bin=True,
                position=_entry_position(bin_entry),
                rpy=_entry_rpy(bin_entry),
            )
        )
    return targets


def _apply_target_to_sim(env, target: _EditTarget) -> None:
    _write_pose(
        env.unwrapped,
        target.asset,
        0,
        (target.position[0], target.position[1], target.position[2]),
        target.quat(env.unwrapped.device),
        asset_name=None if target.is_bin else target.asset_name,
    )


def _prim_path_for_asset(env, asset_name: str) -> str | None:
    asset_cfg = getattr(env.unwrapped.scene.cfg, asset_name, None)
    if asset_cfg is None:
        return None
    matches = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
    return matches[0] if matches else None


def _select_prim_in_viewport(prim_path: str | None) -> None:
    if prim_path is None:
        return
    try:
        import omni.usd

        omni.usd.get_context().get_selection().set_selected_prim_paths([prim_path], True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Controls: Isaac keyboard + terminal stdin
# --------------------------------------------------------------------------- #
def _norm_key(name: str) -> str:
    return name.strip().upper().replace("KEY_", "").replace("NUMPAD_", "")


class _EditorControls:
    def __init__(self, *, terminal_enabled: bool, debug: bool):
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._key_press_type = None
        self._key_repeat_type = None
        self._debug = debug
        self._start_isaac_keyboard_listener()
        if terminal_enabled:
            self._start_stdin_listener()

    # -- Isaac keyboard -----------------------------------------------------
    def _start_isaac_keyboard_listener(self) -> None:
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                print("[WARN]: No Isaac app window found; keyboard controls are unavailable.")
                return
            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._key_press_type = carb.input.KeyboardEventType.KEY_PRESS
            self._key_repeat_type = carb.input.KeyboardEventType.KEY_REPEAT
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            print(
                "[INFO]: Keyboard: Tab select, 1-9 pick object, B bin, arrows translate, X/M remap arrows, "
                "S swap, ',' '.' rotate, '[' ']' move step, '-' '=' rotate step, R revert, P print, "
                "Enter save+next, N skip, Q quit."
            )
        except Exception as exc:
            print(f"[WARN]: Keyboard controls unavailable: {exc}")

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        is_press = event.type == self._key_press_type
        is_repeat = event.type == self._key_repeat_type
        if not is_press and not is_repeat:
            return False
        name = _norm_key(getattr(getattr(event, "input", None), "name", ""))
        if self._debug and is_press:
            print(f"[DEBUG]: Isaac key press: {name!r}")

        # Held-key auto-repeat drives continuous nudging for translate/rotate/step only.
        repeatable = {
            "UP": "translate:up",
            "DOWN": "translate:down",
            "LEFT": "translate:left",
            "RIGHT": "translate:right",
            "COMMA": "rotate:+",
            "PERIOD": "rotate:-",
            "LEFT_BRACKET": "trans_step:-",
            "RIGHT_BRACKET": "trans_step:+",
            "MINUS": "rot_step:-",
            "EQUAL": "rot_step:+",
        }
        if name in repeatable:
            self._events.put(repeatable[name])
            return True
        if not is_press:
            return False

        if name.isdigit() and name != "0":
            self._events.put(f"select_index:{int(name) - 1}")
            return True
        press_only = {
            "TAB": "select_next",
            "B": "select_bin",
            "S": "swap_anchor",
            "X": "arrow_rotate",
            "M": "arrow_mirror",
            "R": "revert",
            "P": "print",
            "N": "skip",
            "Q": "quit",
            "ENTER": "save_next",
            "RETURN": "save_next",
        }
        if name in press_only:
            self._events.put(press_only[name])
            return True
        return False

    # -- Terminal stdin -----------------------------------------------------
    def _start_stdin_listener(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return

        def _read_stdin():
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    return
                if line == "":
                    return
                self._queue_terminal_command(line)

        threading.Thread(target=_read_stdin, daemon=True).start()
        print(
            "[INFO]: Terminal: save/next, skip, revert, print, quit, 'sel N', 'bin', "
            "'ep N' / 'goto N' (jump to episode), 'swap A B', 'instr <text>' (type then Enter; blank Enter saves)."
        )

    def _queue_terminal_command(self, line: str) -> None:
        raw = line.strip()
        # Instruction editing needs the original casing, so branch before lowercasing.
        head, _, rest = raw.partition(" ")
        if head.lower() in ("instr", "instruction"):
            new_instruction = rest.strip()
            if not new_instruction:
                print("[WARN]: Usage: instr <new instruction text>")
                return
            self._events.put(f"set_instruction:{new_instruction}")
            return
        command = raw.lower()
        if command.startswith("sel"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                self._events.put(f"select_index:{int(parts[1]) - 1}")
                return
        if command.startswith("ep") or command.startswith("goto"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                self._events.put(f"goto_episode:{int(parts[1]) - 1}")
                return
            if command.startswith("ep"):
                print("[WARN]: Usage: ep <episode#>  (1-based episode number)")
            else:
                print("[WARN]: Usage: goto <episode#>  (1-based episode number)")
            return
        if command.startswith("swap"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                self._events.put(f"swap_selected:{int(parts[1]) - 1}")
                return
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                self._events.put(f"swap_indices:{int(parts[1]) - 1}:{int(parts[2]) - 1}")
                return
            print("[WARN]: Usage: swap A B  (or swap B to swap selected object with object B)")
            return
        aliases = {
            "": "save_next",
            "save": "save_next",
            "next": "save_next",
            "save_next": "save_next",
            "n": "skip",
            "skip": "skip",
            "r": "revert",
            "revert": "revert",
            "reset": "revert",
            "p": "print",
            "print": "print",
            "tab": "select_next",
            "bin": "select_bin",
            "b": "select_bin",
            "q": "quit",
            "quit": "quit",
            "exit": "quit",
            "finish": "quit",
        }
        event = aliases.get(command)
        if event is None:
            print(
                "[WARN]: Unknown command. Use save/next, skip, revert, print, quit, "
                "'sel N', 'bin', or 'swap A B'."
            )
            return
        self._events.put(event)

    def poll(self) -> list[str]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def close(self) -> None:
        if self._input is None or self._keyboard is None or self._keyboard_sub is None:
            return
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None


# --------------------------------------------------------------------------- #
# On-screen HUD
# --------------------------------------------------------------------------- #
class _EditorHUD:
    def __init__(self):
        self._window = None
        self._labels: dict[str, Any] = {}
        try:
            import omni.ui as ui

            self._window = ui.Window("SO-101 Layout Editor", width=360, height=200)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    for key in ("episode", "objects", "instruction", "selected", "pose", "steps", "arrows", "status"):
                        self._labels[key] = ui.Label("", word_wrap=True)
        except Exception as exc:
            print(f"[WARN]: HUD unavailable: {exc}")
            self._window = None

    def update(self, lines: dict[str, str]) -> None:
        for key, text in lines.items():
            label = self._labels.get(key)
            if label is not None:
                label.text = text

    def close(self) -> None:
        if self._window is not None:
            self._window.visible = False
        self._window = None
        self._labels = {}


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def _mark_layout_row_stale(row: dict, *, any_pose_changed: bool, bin_changed: bool, edited_at: str) -> None:
    """Invalidate sampler provenance that a hand edit makes inaccurate.

    Editing object/bin poses invalidates the ``placement`` stats (attempts,
    achieved surface distances, rejection tallies) and the bin's ``pose_index``.
    Nothing that replays or scores a layout reads these, but leaving them intact
    would misrepresent a hand-tuned layout as a sampler output.  We null the
    ``placement`` values (keeping the dict so consumers' ``.get`` calls still
    work) rather than dropping the block, but *delete* the bin ``pose_index``
    because the reset event does ``int(bin["pose_index"])`` whenever the key is
    present -- a ``None`` there would crash replay.
    """

    if not any_pose_changed:
        return
    row["edited_at"] = edited_at
    placement = row.get("placement")
    if isinstance(placement, dict):
        for key in list(placement):
            if key != "layout_selection":
                placement[key] = None
        placement["layout_selection"] = "hand_edited"
    if bin_changed:
        bin_entry = row.get("bin")
        if isinstance(bin_entry, dict):
            bin_entry.pop("pose_index", None)


def _write_layout_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(tmp_path, output_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if args_cli.translate_step <= 0.0:
        raise ValueError(f"--translate_step must be positive, got {args_cli.translate_step}.")
    if args_cli.rotate_step_deg <= 0.0:
        raise ValueError(f"--rotate_step_deg must be positive, got {args_cli.rotate_step_deg}.")

    layout_path = args_cli.episode_layouts_jsonl
    output_path = args_cli.output or layout_path
    layout_rows = _load_layout_rows(layout_path)
    print(f"[INFO]: Loaded {len(layout_rows)} layout row(s) from {layout_path}.")

    if args_cli.start_episode < 1 or args_cli.start_episode > len(layout_rows):
        raise ValueError(
            f"--start_episode must be in [1, {len(layout_rows)}], got {args_cli.start_episode}."
        )

    episode_plan = _build_episode_plan(layout_rows)
    object_pool = _episode_object_pool(episode_plan)
    print(f"[INFO]: Pre-spawning {len(object_pool)} object asset(s): {', '.join(object_pool)}")

    # Instruction editing requires --episodes_jsonl so the change lands where eval/teleop
    # actually reads it; we keep the raw episode rows to rewrite that file in place.
    instruction_editing_enabled = args_cli.episodes_jsonl is not None
    episodes_full_rows: list[dict] = []
    episodes_row_for_layout: list[dict | None] = [None] * len(layout_rows)
    episodes_path = args_cli.episodes_jsonl
    episodes_output_path = args_cli.episodes_output or episodes_path
    if instruction_editing_enabled:
        episodes_full_rows = _load_episodes_raw_rows(episodes_path)
        episodes_row_for_layout = _match_episode_rows_to_layout(layout_rows, episodes_full_rows)
        print(
            f"[INFO]: Instruction editing enabled: 'instr <text>' rewrites the layout row and the "
            f"matching episode in {episodes_path} (-> {episodes_output_path})."
        )
    else:
        print("[INFO]: Instruction editing disabled; pass --episodes_jsonl to enable it. 'instr' will be ignored.")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env, object_asset_names = _make_env(object_pool)
    env_origin = env.unwrapped.scene.env_origins[0].detach().cpu().tolist()
    print(f"[INFO]: Gym action space: {env.action_space}")

    controls = _EditorControls(terminal_enabled=args_cli.terminal_control_stdin, debug=args_cli.keyboard_debug)
    hud = _EditorHUD()

    translate_step = float(min(max(args_cli.translate_step, MIN_TRANSLATE_STEP), MAX_TRANSLATE_STEP))
    rotate_step = math.radians(float(min(max(args_cli.rotate_step_deg, MIN_ROTATE_STEP_DEG), MAX_ROTATE_STEP_DEG)))

    # Arrow -> world mapping is unit (dx, dy) for the "up" and "right" arrows (down/left
    # are their negatives). Default: vertical keys (Up/Down) move world Y, horizontal keys
    # (Left/Right) move world X. The CLI invert/swap flags choose the starting orientation
    # (--swap_xy restores the old up/down=X, left/right=Y mapping); X (rotate 90 deg) and
    # M (mirror left/right) then adjust it live so the arrows can be made to match whatever
    # the viewport camera shows without restarting.
    _sign_x = -1.0 if args_cli.invert_x else 1.0
    _sign_y = -1.0 if args_cli.invert_y else 1.0
    if args_cli.swap_xy:
        arrow_up_unit = (_sign_x, 0.0)
        arrow_right_unit = (0.0, -_sign_y)
    else:
        arrow_up_unit = (0.0, _sign_y)
        arrow_right_unit = (_sign_x, 0.0)

    backup_done = args_cli.no_backup or output_path != layout_path
    episodes_backup_done = (
        True
        if not instruction_editing_enabled
        else (args_cli.no_backup or episodes_output_path != episodes_path)
    )

    state: dict[str, Any] = {
        "episode_index": args_cli.start_episode - 1,
        "targets": [],
        "selected": 0,
        "swap_anchor": None,
        "dirty": False,
        "instruction": "",
        "instruction_dirty": False,
        "status": "",
        "quit": False,
    }

    def _selected_target() -> _EditTarget | None:
        targets = state["targets"]
        if not targets:
            return None
        return targets[state["selected"]]

    def _axis_label(vec: tuple[float, float]) -> str:
        if abs(vec[0]) >= abs(vec[1]):
            return "+X" if vec[0] > 0 else "-X"
        return "+Y" if vec[1] > 0 else "-Y"

    def _object_id_text() -> str:
        entries = [t for t in state["targets"] if not t.is_bin]
        if not entries:
            return "objects: -"
        return "objects: " + "   ".join(f"{index + 1}={target.label}" for index, target in enumerate(entries))

    def _refresh_hud() -> None:
        episode_index = state["episode_index"]
        episode = episode_plan[episode_index]
        target = _selected_target()
        pose_text = "-"
        if target is not None:
            pose_text = (
                f"pos=({target.position[0]:.4f}, {target.position[1]:.4f}, {target.position[2]:.4f})  "
                f"yaw={math.degrees(target.yaw):.1f} deg"
            )
        ux, uy = arrow_up_unit
        rx, ry = arrow_right_unit
        arrows_text = (
            f"arrows: up={_axis_label((ux, uy))} down={_axis_label((-ux, -uy))} "
            f"left={_axis_label((-rx, -ry))} right={_axis_label((rx, ry))}"
        )
        hud.update(
            {
                "episode": f"Episode {episode_index + 1}/{len(layout_rows)}   trial_id={layout_rows[episode_index].get('trial_id')}",
                "objects": _object_id_text(),
                "instruction": f"task={episode.task_family}: {state['instruction']}",
                "selected": f"selected: {target.label if target else '-'}  "
                f"({state['selected'] + 1}/{len(state['targets'])})",
                "pose": pose_text,
                "steps": f"move step={translate_step * 1000:.1f} mm   rotate step={math.degrees(rotate_step):.1f} deg",
                "arrows": arrows_text,
                "status": state["status"],
            }
        )

    def _select(index: int) -> None:
        targets = state["targets"]
        if not targets:
            return
        state["selected"] = index % len(targets)
        target = targets[state["selected"]]
        _select_prim_in_viewport(_prim_path_for_asset(env, target.asset_name))
        print(f"[INFO]: Selected {target.label} ({state['selected'] + 1}/{len(targets)}).")

    def _load_episode(episode_index: int) -> None:
        state["episode_index"] = episode_index
        episode = episode_plan[episode_index]
        _configure_env_for_episode(env, episode, layout_rows[episode_index], object_pool, object_asset_names)
        _reset_env(env)
        state["targets"] = _build_targets(env, layout_rows[episode_index], object_pool, object_asset_names)
        state["selected"] = 0
        state["swap_anchor"] = None
        state["dirty"] = True  # flush the reset-applied poses on the next frame
        state["instruction"] = episode.instruction
        state["instruction_dirty"] = False
        state["status"] = "loaded"
        print(
            f"[INFO]: Editing episode {episode_index + 1}/{len(layout_rows)} "
            f"(trial_id={layout_rows[episode_index].get('trial_id')}): {episode.instruction}"
        )
        print(f"[INFO]: Object IDs -> {_object_id_text()[len('objects: '):]}  (select with number keys 1-9)")
        _select(0)
        _refresh_hud()

    def _translate(dx_world: float, dy_world: float) -> None:
        target = _selected_target()
        if target is None:
            return
        target.position[0] += dx_world
        target.position[1] += dy_world
        target.modified = True
        _apply_target_to_sim(env, target)
        state["dirty"] = True
        state["status"] = "edited (unsaved)"

    def _rotate(dyaw: float) -> None:
        target = _selected_target()
        if target is None:
            return
        target.rpy[2] = math.atan2(math.sin(target.rpy[2] + dyaw), math.cos(target.rpy[2] + dyaw))
        target.modified = True
        _apply_target_to_sim(env, target)
        state["dirty"] = True
        state["status"] = "edited (unsaved)"

    def _swap_object_positions(first_index: int, second_index: int) -> None:
        targets = state["targets"]
        object_count = sum(1 for target in targets if not target.is_bin)
        if first_index < 0 or second_index < 0 or first_index >= object_count or second_index >= object_count:
            print(
                f"[WARN]: Swap object indices must be in [1, {object_count}], "
                f"got {first_index + 1} and {second_index + 1}."
            )
            state["status"] = "swap rejected (index)"
            return
        first = targets[first_index]
        second = targets[second_index]
        if first is second:
            print("[WARN]: Swap needs two different objects.")
            state["status"] = "swap rejected (same object)"
            return

        first.position, second.position = list(second.position), list(first.position)
        first.modified = True
        second.modified = True
        _apply_target_to_sim(env, first)
        _apply_target_to_sim(env, second)
        state["dirty"] = True
        state["status"] = "swapped positions (unsaved)"
        print(
            f"[INFO]: Swapped positions: {first_index + 1}={first.label} <-> "
            f"{second_index + 1}={second.label} (yaws unchanged)."
        )

    def _swap_anchor() -> None:
        """Keyboard swap: press S to anchor the selected object, then S again on another."""
        target = _selected_target()
        if target is None:
            return
        if target.is_bin:
            print("[WARN]: Swap works on objects, not the bin; select an object first.")
            state["status"] = "swap: select an object (not the bin)"
            return
        index = state["selected"]
        anchor = state["swap_anchor"]
        if anchor is None:
            state["swap_anchor"] = index
            print(f"[INFO]: Swap anchored on {index + 1}={target.label}; select another object and press S.")
            state["status"] = f"swap: anchored {index + 1}={target.label} (pick another, press S)"
            return
        if anchor == index:
            state["swap_anchor"] = None
            print("[INFO]: Swap anchor cleared.")
            state["status"] = "swap anchor cleared"
            return
        _swap_object_positions(anchor, index)
        state["swap_anchor"] = None

    def _set_instruction(text: str) -> None:
        if not instruction_editing_enabled:
            print(
                "[WARN]: Instruction editing requires --episodes_jsonl (so the change is written to both "
                "the layout and episodes files). Ignoring."
            )
            state["status"] = "instruction editing needs --episodes_jsonl"
            return
        episodes_row = episodes_row_for_layout[state["episode_index"]]
        if episodes_row is None:
            trial_id = layout_rows[state["episode_index"]].get("trial_id")
            print(f"[WARN]: No episodes_jsonl row matched trial_id={trial_id!r}; cannot edit instruction.")
            state["status"] = "no matching episodes row"
            return
        # Reject anything that would make the episodes file fail to load next time.
        try:
            candidate = row_with_canonical_instruction_metadata(
                episodes_row,
                instruction=text,
                source="edited instruction",
            )
            _canonical_layout_instruction_row(layout_rows[state["episode_index"]], candidate["instruction"])
        except ValueError as exc:
            print(f"[WARN]: Rejected instruction (not a loadable episode): {exc}")
            state["status"] = "instruction rejected (invalid)"
            return
        state["instruction"] = candidate["instruction"]
        state["instruction_dirty"] = True
        state["status"] = "instruction edited (unsaved)"
        print(f"[INFO]: Instruction set to: {candidate['instruction']}")

    def _arrow_delta(direction: str) -> tuple[float, float]:
        # Map arrows to world X/Y using the live-adjustable up/right unit vectors
        # (rotate with X, mirror with M). down/left are the negatives of up/right.
        ux, uy = arrow_up_unit
        rx, ry = arrow_right_unit
        vecs = {
            "up": (ux, uy),
            "down": (-ux, -uy),
            "right": (rx, ry),
            "left": (-rx, -ry),
        }
        vx, vy = vecs[direction]
        return vx * translate_step, vy * translate_step

    def _print_selected() -> None:
        target = _selected_target()
        if target is None:
            return
        try:
            _print_selected_unsafe(target)
        except Exception as exc:  # diagnostics must never kill the edit loop
            print(f"[WARN]: Could not read back pose for {target.label}: {exc}")

    def _print_selected_unsafe(target: _EditTarget) -> None:
        positions = benchmark_object_positions(env.unwrapped, object_asset_names)
        yaws = benchmark_object_yaws(env.unwrapped, object_asset_names)
        if target.is_bin:
            bin_asset = env.unwrapped.scene[BIN_NAME]
            sim_pos = bin_asset.data.root_pos_w[0].detach().cpu().tolist()
            sim_local = [sim_pos[i] - env_origin[i] for i in range(3)]
            print(
                f"[INFO]: {target.label}: tracked pos={target.position} yaw={math.degrees(target.yaw):.2f} deg | "
                f"sim(local) pos=({sim_local[0]:.4f}, {sim_local[1]:.4f}, {sim_local[2]:.4f})"
            )
        else:
            pool_slot = object_asset_names.index(target.asset_name)
            sim_pos = positions[0, pool_slot].detach().cpu().tolist()
            sim_local = [sim_pos[i] - env_origin[i] for i in range(3)]
            sim_yaw = math.degrees(float(yaws[0, pool_slot].item()))
            print(
                f"[INFO]: {target.label}: tracked pos={target.position} yaw={math.degrees(target.yaw):.2f} deg | "
                f"sim(local) pos=({sim_local[0]:.4f}, {sim_local[1]:.4f}, {sim_local[2]:.4f}) yaw={sim_yaw:.2f} deg"
            )

    def _save_current() -> None:
        nonlocal backup_done, episodes_backup_done
        episode_index = state["episode_index"]
        row = layout_rows[episode_index]
        for target in state["targets"]:
            target.commit_to_entry()
        instruction_changed = bool(state["instruction_dirty"])
        episodes_row = episodes_row_for_layout[episode_index] if instruction_editing_enabled else None
        if instruction_changed:
            # Instruction editing is gated on --episodes_jsonl, so write both sides in lock-step
            # and keep the in-memory spec so a later revert restores the saved text (not the original).
            canonical_layout_row = _canonical_layout_instruction_row(row, state["instruction"])
            _apply_instruction_fields(row, canonical_layout_row)
            if episodes_row is not None:
                canonical_episode_row = row_with_canonical_instruction_metadata(
                    episodes_row,
                    instruction=state["instruction"],
                    source=f"episodes trial_id={row.get('trial_id')!r}",
                )
                episodes_row.clear()
                episodes_row.update(canonical_episode_row)
            episode_plan[episode_index] = episode_spec_from_json(
                canonical_layout_row,
                source=f"edited layout trial_id={row.get('trial_id')!r}",
            )
        if not args_cli.keep_stale_metadata:
            edited_at = datetime.now().astimezone().isoformat(timespec="seconds")
            _mark_layout_row_stale(
                row,
                any_pose_changed=any(target.modified for target in state["targets"]),
                bin_changed=any(target.modified and target.is_bin for target in state["targets"]),
                edited_at=edited_at,
            )
            # A pose-only edit stamps edited_at via _mark_layout_row_stale; an
            # instruction-only edit does not change poses, so stamp it here too.
            if instruction_changed:
                row["edited_at"] = edited_at
        state["instruction_dirty"] = False
        if not backup_done:
            backup_path = layout_path.with_suffix(layout_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(layout_path, backup_path)
                print(f"[INFO]: Wrote one-time backup: {backup_path}")
            backup_done = True
        _write_layout_rows(layout_rows, output_path)
        if instruction_changed and episodes_row is not None:
            if not episodes_backup_done:
                episodes_backup_path = episodes_path.with_suffix(episodes_path.suffix + ".bak")
                if not episodes_backup_path.exists():
                    shutil.copy2(episodes_path, episodes_backup_path)
                    print(f"[INFO]: Wrote one-time episodes backup: {episodes_backup_path}")
                episodes_backup_done = True
            _write_layout_rows(episodes_full_rows, episodes_output_path)
            print(f"[INFO]: Updated instruction in episodes file -> {episodes_output_path}")
        state["status"] = "saved"
        print(
            f"[INFO]: Saved episode {episode_index + 1}/{len(layout_rows)} "
            f"(trial_id={row.get('trial_id')}) -> {output_path}"
        )

    def _advance() -> bool:
        next_index = state["episode_index"] + 1
        if next_index >= len(layout_rows):
            print("[INFO]: Reached the last layout row.")
            return False
        _load_episode(next_index)
        return True

    def _handle(event: str) -> None:
        nonlocal translate_step, rotate_step, arrow_up_unit, arrow_right_unit
        if event == "select_next":
            _select(state["selected"] + 1)
        elif event == "select_bin":
            for index, target in enumerate(state["targets"]):
                if target.is_bin:
                    _select(index)
                    break
        elif event.startswith("select_index:"):
            _select(int(event.split(":", 1)[1]))
        elif event.startswith("set_instruction:"):
            _set_instruction(event.split(":", 1)[1])
        elif event == "swap_anchor":
            _swap_anchor()
        elif event.startswith("swap_selected:"):
            _swap_object_positions(state["selected"], int(event.split(":", 1)[1]))
        elif event.startswith("swap_indices:"):
            _head, first_index, second_index = event.split(":", 2)
            _swap_object_positions(int(first_index), int(second_index))
        elif event == "arrow_rotate":
            # Rotate both unit vectors 90 deg CCW in world XY: (x, y) -> (-y, x).
            arrow_up_unit = (-arrow_up_unit[1], arrow_up_unit[0])
            arrow_right_unit = (-arrow_right_unit[1], arrow_right_unit[0])
            state["status"] = "rotated arrow mapping 90 deg"
            print("[INFO]: Rotated arrow->world mapping 90 deg.")
        elif event == "arrow_mirror":
            arrow_right_unit = (-arrow_right_unit[0], -arrow_right_unit[1])
            state["status"] = "mirrored left/right arrows"
            print("[INFO]: Mirrored the left/right arrows.")
        elif event.startswith("goto_episode:"):
            ep_index = int(event.split(":", 1)[1])
            if ep_index < 0 or ep_index >= len(layout_rows):
                print(f"[WARN]: Episode {ep_index + 1} is out of range [1, {len(layout_rows)}].")
                state["status"] = "invalid episode number"
            else:
                _load_episode(ep_index)
        elif event.startswith("translate:"):
            _translate(*_arrow_delta(event.split(":", 1)[1]))
        elif event == "rotate:+":
            _rotate(rotate_step)
        elif event == "rotate:-":
            _rotate(-rotate_step)
        elif event == "trans_step:+":
            translate_step = min(translate_step * 2.0, MAX_TRANSLATE_STEP)
        elif event == "trans_step:-":
            translate_step = max(translate_step / 2.0, MIN_TRANSLATE_STEP)
        elif event == "rot_step:+":
            rotate_step = min(rotate_step * 2.0, math.radians(MAX_ROTATE_STEP_DEG))
        elif event == "rot_step:-":
            rotate_step = max(rotate_step / 2.0, math.radians(MIN_ROTATE_STEP_DEG))
        elif event == "revert":
            print("[INFO]: Reverting to last-saved poses.")
            _load_episode(state["episode_index"])
        elif event == "print":
            _print_selected()
        elif event == "save_next":
            _save_current()
            if not _advance():
                state["quit"] = True
        elif event == "skip":
            if not _advance():
                state["quit"] = True
        elif event == "quit":
            state["quit"] = True
        _refresh_hud()

    _load_episode(state["episode_index"])

    try:
        while simulation_app.is_running():
            for event in controls.poll():
                _handle(event)
                if state["quit"]:
                    break
            if state["quit"]:
                break
            if state["dirty"]:
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.forward()
                state["dirty"] = False
            env.unwrapped.sim.render()
            time.sleep(0.01)
    finally:
        controls.close()
        hud.close()
        env.close()
        print(f"[INFO]: Layout editor finished. File: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
