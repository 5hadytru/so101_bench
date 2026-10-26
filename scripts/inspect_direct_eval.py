"""Run the upstream Inspect Robots agent on SO-101 RGB and proprioception.

This is a native benchmark evaluation loop around unmodified LLMAgentPolicy.
Only the embodiment/controller and budget transport are project-specific.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes_jsonl", type=Path, required=True)
parser.add_argument("--episode_layouts_jsonl", type=Path, required=True)
parser.add_argument("--episode_indices", required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--inspect_packages", type=Path,
                    default=Path.home()/"workspace/inspect_robots_deps")
parser.add_argument("--key_file", type=Path,
                    default=Path.home()/".config/so101_bench/jay_openai_api_key")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_calls", type=int, default=60)
parser.add_argument("--budget_usd", type=float, default=15)
parser.add_argument("--trial_budget_usd", type=float, default=2)
parser.add_argument("--trial_wall_s", type=float, default=1800)
parser.add_argument("--resource_profile", choices=("stock", "extended"), default="stock")
parser.add_argument("--recover_tool_failures", type=int, default=0)
parser.add_argument("--scripted_smoke", action="store_true")
parser.add_argument("--provider", choices=("openai", "deepseek", "gemini"), default="openai")
parser.add_argument("--base_url", default="http://127.0.0.1:8210/v1")
parser.add_argument("--effort", default="high")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.headless and not args.experience:
    args.experience = "isaaclab.python.rendering.kit"
app = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import so101_bench.tasks  # noqa: F401
from so101_bench.benchmark import load_episode_jsonl
from so101_bench.mdp import mark_benchmark_robot_start
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import configure_env_cfg_for_object_pool
from so101_bench.utils import so101_kinematics as kin
from so101_bench.utils.llm_harness import (
    INITIAL_ROBOT_JOINT_POS, PUBLIC_TASK_RULES, SimArm, load_layouts,
    object_pool, reset_env, reset_params,
)
from inspect_so101_controller import PoseController
from inspect_budget_transport import BudgetTransport, BudgetExceeded
from inspect_outcomes import classify_exception
from inspect_retry import act_with_recovery
from eval_efficiency import efficiency_metrics
from so101_bench.utils.task_guidance import task_rules_and_guidance, REVISION as GUIDANCE_REVISION

sys.path.insert(0, str(args.inspect_packages))
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.task import TaskEnvelope
from inspect_robots.types import Observation

LABELS = ("x_m", "y_m", "z_m", "pitch_deg", "wrist_roll_deg", "gripper_opening_m")
LOW = np.array([-.5, -.5, .014, 0, -166, 0.0])
HIGH = np.array([.5, .4, .5, 180, 166, kin.MAX_GRIPPER_OPENING_M])
OPERATING_NOTES = """You control a SIMULATED SO-101 with five arm joints and a one-sided jaw.
Commands are absolute TABLE/WORLD-frame grasp-point targets: x_m,y_m,z_m in metres;
pitch_deg is 0 pointing vertically DOWN, 90 horizontal; wrist_roll_deg is the actual
fifth joint's rotation in degrees, not a world heading. Gripper opening is metres (0 squeezes
closed, 0.0809 fully open); omitted targets hold measured values. The base faces -y:
robot left is +x, forward is -y, table top z=0. The overhead camera sees the robot
at the top; +x is image right. The TCP is between the finger pads, 1.2 cm above
the fingertips. The fixed finger's face is 8 mm from TCP toward -jaw_axis.
Each camera's intrinsic is its pixel calibration K. camera_to_world maps ROS optical
coordinates (+x image right, +y image down, +z forward) into the table/world frame.
Pixels are (u=column,v=row); wrist calibration changes with robot motion.
Five joints cannot realize arbitrary six-DoF orientations. A conventional IK
controller checks every Cartesian waypoint; errors report the closest robot pose.
It has no object-state information or object collision planner. Use short motions
near objects and fresh images to assess acquisition and placement. Separate jaw
changes from approach motions. The arm may sag under load.
The closing line can become vertical when tilted, making top-down jaw yaw undefined;
that is why the fifth target uses wrist_roll_deg. At a top-down pose the wrist roll
turns the closing line in the table plane. The measured tool_axis and jaw_axis state
vectors give the resulting world directions. Observe them and the images after turning.
Each move interpolates the selected target at 30 Hz (up to 0.005 m, 3 degrees and
0.01 m jaw opening per step); joint changes are limited to 0.06 rad/native step,
which can extend a Cartesian chunk. Read the ACTUAL env_step after each move.
Only executed motion/holding consumes simulated time; physics pauses during reasoning.
All original RGB history is retained within this episode; there are no demonstrations,
past-trial memories, segmentation tools, code execution, or learned motor policies.
Native success/failure and timeout rules remain authoritative. On done/give_up or
call exhaustion the arm HOLDS its final command; neither signal opens the gripper
or homes it. Release and withdraw yourself, then done lets the benchmark confirm
the final state. You cannot reset the episode.
"""


def spaces():
    """Declare a reduced SO-101 Cartesian pose, with explicitly labeled angles."""
    # rotation_repr=none means no canonical quaternion/rot6d packing. The two
    # reachable orientation coordinates are explicitly named scalar channels.
    action = Box(shape=(6,), low=LOW, high=HIGH, semantics=ActionSemantics(
        control_mode="eef_abs_pose", rotation_repr="none", gripper="continuous",
        frame="world", dim_labels=LABELS, max_step=(.005, .005, .005, 3., 3., .01)))
    obs = ObservationSpace(cameras=(CameraSpec("overhead", 480, 640), CameraSpec("wrist", 480, 640)),
                           state=StateSpec(fields=(StateField("eef_state", (6,), "mixed"),
                                                   StateField("joint_pos", (6,), "rad"))))
    return action, obs


def observe(arm, instruction, feedback):
    """Expose only rendered images, robot sensors, calibration and execution feedback."""
    images = arm.images()
    robot = arm.robot_state()
    values = [*robot["tcp_xyz"], robot["pitch_deg"], float(np.degrees(arm.joints()[4])),
              robot["measured_gripper_opening_m"]]
    # State vectors carry calibration too; Inspect serializes state fields but
    # intentionally ignores arbitrary extra metadata in its model observations.
    state = {"eef_state": np.array(values), "joint_pos": arm.joints(),
             "robot_base_xyz": np.array(robot["base_xyz"]),
             "tool_axis_world": np.array(robot["tool_axis"]),
             "jaw_axis_world": np.array(robot["jaw_axis"])}
    for camera, data in arm.calibration(images).items():
        state[camera+"_intrinsic"] = np.array(data["intrinsic"])
        state[camera+"_camera_to_world"] = np.array(data["camera_to_environment"])
    return Observation(images=images, state=state, instruction=instruction,
                       extra={"env_step": arm.step_id, "approvals": feedback})


def timeout_handler(_sig, _frame):
    raise TimeoutError("Inspect trial wall-time limit reached")


from primary_eval_exclusions import EXCLUDED_TRIAL_IDS, primary_episode_excluded
from so101_bench.benchmark import episode_is_executable

def main():
    model = {'deepseek':'deepseek-flash','gemini':'gemini-4-argon','openai':'gpt-6.1-sol'}[args.provider]
    adapter_version = {'deepseek':'direct-v0.4-deepseek-32k','gemini':'direct-v0.4-gemini-chat-32k','openai':'direct-v0.3'}[args.provider]
    if args.resource_profile=='extended':adapter_version='direct-v0.6-'+args.provider+'-extended'
    image_horizon=2 if args.resource_profile=='extended' else None
    tasks = load_episode_jsonl(args.episodes_jsonl)
    indices = [int(x) for x in args.episode_indices.split(',')]
    indices = [i for i in indices if not primary_episode_excluded(tasks[i]) and episode_is_executable(tasks[i])]
    if not indices:
        raise ValueError("No eligible episodes remain after exclusions")
    plan = [tasks[i] for i in indices]
    layouts = load_layouts(plan, args.episode_layouts_jsonl)
    pool = object_pool(plan)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {"started_at": datetime.now().astimezone().isoformat(),
                  "adapter_version": adapter_version, "guidance_revision": GUIDANCE_REVISION,
                  "max_output_tokens": 4096 if args.provider == 'openai' else 32768,
                  "guidance_sha256": hashlib.sha256(Path(task_rules_and_guidance.__code__.co_filename).read_bytes()).hexdigest(),
                  "stock_consecutive_failure_limit": 3,
                  "resource_profile": args.resource_profile, "recovery_retries": args.recover_tool_failures,
                  "image_horizon": image_horizon,
                  "credential_label": os.environ.get("SO101_CREDENTIAL_LABEL", "jay"), "model": model, "effort": args.effort,
                  "arguments": {k: str(v) for k, v in vars(args).items()},
                  "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.episodes_jsonl, args.episode_layouts_jsonl)}}
    (args.output_dir/'run_config.json').write_text(json.dumps(provenance, indent=2))
    cfg = parse_env_cfg("So101Bench-Bin-v0", device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.scene.robot.init_state.joint_pos = dict(INITIAL_ROBOT_JOINT_POS)
    cfg.scene.camera_wrist.update_latest_camera_pose = True
    cfg.scene.camera_overhead.update_latest_camera_pose = True
    names = configure_env_cfg_for_object_pool(cfg, pool)
    cfg.events.reset_benchmark_scene.params.update(reset_params(plan[0], layouts[0], pool, names))
    env = gym.make("So101Bench-Bin-v0", cfg=cfg)
    if args.provider == 'gemini':
        from inspect_gemini_transport import GeminiBudgetTransport
        meter = GeminiBudgetTransport(args.output_dir/'usage.jsonl', args.budget_usd, args.trial_budget_usd)
    elif args.provider == 'deepseek':
        from inspect_deepseek_transport import DeepSeekBudgetTransport
        meter = DeepSeekBudgetTransport(args.output_dir/'usage.jsonl', args.budget_usd, args.trial_budget_usd)
    else:
        meter = BudgetTransport(args.output_dir/'usage.jsonl', args.budget_usd, args.trial_budget_usd)
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        for index, episode, layout in zip(indices, plan, layouts, strict=True):
            params = reset_params(episode, layout, pool, names)
            env.unwrapped.cfg.events.reset_benchmark_scene.params.update(params)
            env.unwrapped.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)
            obs = reset_env(env)
            folder = args.output_dir/f'ep{index:03d}'
            folder.mkdir()
            arm = SimArm(env, episode, folder, obs)
            for _ in range(45):
                if arm.step():
                    raise RuntimeError("Episode terminated during initial hold")
            mark_benchmark_robot_start(env.unwrapped, object_asset_names=names,
                                       bin_name="plastic_bin", force_robot_start_time=True)
            controller = PoseController(kin, arm)
            started = time.monotonic()
            sim_wall_baseline, step_baseline = arm.sim_loop_wall_s, arm.step_id
            if args.scripted_smoke:
                packet = observe(arm, episode.instruction, [])
                pose = packet.state['eef_state'].copy()
                assert controller.check(np.array([pose])) is None
                for command in controller.commands:
                    arm.action[0] = torch.as_tensor(command, dtype=arm.action.dtype, device=arm.action.device)
                    arm.step()
                (folder/'smoke.json').write_text(json.dumps({"ok": True, "native_steps": arm.step_id,
                    "rgb_shapes": {k: list(v.shape) for k, v in packet.images.items()}}))
                arm.close()
                print('[SMOKE] Native reset, RGB, calibration, IK and action execution passed', flush=True)
                return
            action, observation = spaces()
            task_text = task_rules_and_guidance(episode.task_family, PUBLIC_TASK_RULES)
            (folder/'task_prompt.txt').write_text(task_text)
            info = EmbodimentInfo(name="so101_bench", action_space=action,
                observation_space=observation, control_hz=30., is_simulated=True,
                docs=OPERATING_NOTES+'\nPublic task conditions:\n'+task_text)
            provider_options = dict(service_tier='default') if args.provider == 'openai' else {}
            # Four-object bin episodes last 75 s rather than 30 s. Give them
            # the same call allowance per native second, within the unchanged
            # per-episode dollar and wall-time guards.
            trial_max_calls=(max(args.max_calls,int(round(args.max_calls*arm.time_limit_s/30)))
                if args.resource_profile=='extended' else args.max_calls)
            key_env = {'deepseek':'DEEPSEEK_API_KEY','gemini':'GEMINI_API_KEY','openai':'OPENAI_API_KEY'}[args.provider]
            if args.provider == 'gemini':
                provider_options.update(base_url=args.base_url,api_key_env=key_env)
            policy = LLMAgentPolicy(model=model if args.provider == 'gemini' else args.provider+'/'+model,
                wire="chat" if args.provider == 'gemini' else "responses", effort=args.effort,
                **provider_options, max_llm_calls=trial_max_calls, max_speed_frac=.25,
                images="always", depth="off", image_horizon=image_horizon, prior_learnings=None,
                transport=meter, env={key_env: args.key_file.read_text().strip()},
                pre_check=controller.check, max_retries=2, backoff_s=2., transcript_echo=True)
            policy.bind(info)
            limit = int(round(arm.time_limit_s/arm.control_dt))
            policy.bind_task(TaskEnvelope(name='primary_eval_v2', max_steps=limit))
            policy.reset(Scene(id=folder.name, instruction=episode.instruction))
            policy.on_trial_start(folder.name, 0, str(args.output_dir), 'inspect_'+args.provider)
            (folder/'policy_config.json').write_text(json.dumps(asdict(policy.config), indent=2))
            meter.trial = folder.name
            stop, error, feedback = 'native_terminal', None, []
            policy_failure = None
            resource_limit = None
            recoveries=[]
            def on_recovery(message):
                recoveries.append(dict(native_step=arm.step_id,error=message))
                with (folder/'recovery.jsonl').open('a') as f:f.write(json.dumps(recoveries[-1])+'\n')
                print('[RECOVERY] Continuing same conversation after rejected tool calls; native step',arm.step_id,flush=True)
            signal.alarm(int(args.trial_wall_s))
            try:
                while arm.terminal is None:
                    chunk = act_with_recovery(policy,lambda:observe(arm,episode.instruction,feedback),
                        max(0,args.recover_tool_failures-len(recoveries)),on_recovery)
                    if any(a.meta.get('request_stop') for a in chunk.actions):
                        stop = chunk.actions[-1].meta.get('stop_reason', 'policy_stop')
                        detail=str(chunk.actions[-1].meta.get('stop_detail',''))
                        if 'LLM call budget exhausted' in detail:resource_limit='LLM call budget exhausted'
                        break
                    if not controller.commands:
                        raise RuntimeError('Inspect motion had no validated joint plan')
                    executed = 0
                    for command in controller.commands:
                        arm.action[0] = torch.as_tensor(command, dtype=arm.action.dtype, device=arm.action.device)
                        executed += 1
                        if arm.step():
                            break
                    feedback = ([{"t": arm.step_id, "detail":
                        f"joint speed bound extended Cartesian chunk by {controller.extension_steps} native controls"}]
                        if controller.extension_steps else [])
                    with (folder/'motion.jsonl').open('a') as f:
                        f.write(json.dumps({"native_step": arm.step_id, "executed": executed,
                                            "requested_cartesian_steps": len(chunk.actions)})+'\n')
            except (BudgetExceeded, TimeoutError) as exc:
                # Predeclared per-trial resources bound the policy. These are
                # censored policy outcomes, not infrastructure errors to retry.
                stop, resource_limit = type(exc).__name__, str(exc)
            except Exception as exc:
                stop, error = classify_exception(exc)
                # Only the recognized stock-policy error text is safe to retain;
                # arbitrary provider errors may contain sensitive response data.
                if stop == 'policy_error':
                    policy_failure = str(exc)
            finally:
                signal.alarm(0)
                stopped_at_native_step = arm.step_id
                control_wall_s = time.monotonic()-started
                sim_loop_wall_s = arm.sim_loop_wall_s-sim_wall_baseline
                control_sim_s = (arm.step_id-step_baseline)*arm.control_dt
                while arm.terminal is None:
                    arm.step()
                (folder/'transcript.json').write_text(json.dumps(policy.transcript(), indent=2))
                arm.close()
            result = {**arm.terminal, "trial_id": index, "episode": folder.name,
                      "instruction": episode.instruction, "task_family": episode.task_family,
                      "harness": "inspect_direct", "model": model, "effort": args.effort,
                      "agent_stop": stop, "infrastructure_error": error,
                      "adapter_version": adapter_version, "guidance_revision": GUIDANCE_REVISION, "policy_failure": policy_failure,
                      "max_output_tokens": meter.output_cap,
                      "resource_profile": args.resource_profile, "max_calls": trial_max_calls,
                      "image_horizon": image_horizon,
                      "trial_budget_usd": args.trial_budget_usd, "trial_wall_s": args.trial_wall_s,
                      "tool_failure_recoveries": len(recoveries),
                      "resource_limit": resource_limit,
                      "stopped_at_native_step": stopped_at_native_step,
                      "wall_time_s": round(time.monotonic()-started, 1),
                      "estimated_usd": meter.spent(folder.name)}
            result['efficiency'] = efficiency_metrics(control_wall_s, sim_loop_wall_s, control_sim_s,
                                                       result['estimated_usd'])
            (folder/'result.json').write_text(json.dumps(result, indent=2, default=str))
            with (args.output_dir/'results.jsonl').open('a') as f:
                f.write(json.dumps(result, default=str)+'\n')
            print('[RESULT]', folder.name, result['success'], stop,
                  f"estimated spend ${meter.spent():.3f}", flush=True)
            if stop in ('BudgetExceeded', 'infrastructure_error'):
                break
    finally:
        meter.close()
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.app.close()
