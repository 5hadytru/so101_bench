#!/usr/bin/env python3
"""Test the real gripper's close with the runner's limit (sim's -12.75 deg jaw limit), gripper torque only.

Only the gripper motor is powered; the rest of the arm stays limp, so leave it resting in the dock.
At each prompt the gripper moves to the commanded sim angle and reports where it stopped and how hard
it is pushing (Present_Load, in 0.1% of max torque). Commands never go below the jaw floor.

    /home/truman/env_isaaclab_51/bin/python scripts/so101_gripper_test.py

Prompt commands:
    c          close to the jaw floor (what a policy's full close sends)
    o          open to 30 deg
    <number>   go to that sim angle in degrees (clamped to the floor)
    t          measure the stall: close in 2-tick steps past the touch point until the jaw reaches it (stops
               at most 6 deg past touch), then save it as gripper_stall_ticks in the alignment file; the
               runner then aims that far past closing goals so the jaw lands where sim's does
    q          open, torque off, quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_real_policy_runner as runner  # noqa: E402

OPEN_DEG = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="follower_arm_1")
    parser.add_argument("--gripper_min_tick", type=int, default=None, help="Default: the runner's (sim jaw limit).")
    args = parser.parse_args()

    alignment = runner.alignment_lib.RealSimAlignment.load(runner.alignment_lib.default_alignment_path(args.robot_id))
    arm = runner.RealArm(args.port, alignment)
    bus = arm.bus
    if any(bus.sync_read("Torque_Enable", normalize=False).values()):
        arm.close(torque_off=False)
        raise SystemExit("Torque is on (is the runner open?). Quit the runner with q first.")
    path = runner.alignment_lib.default_alignment_path(args.robot_id)
    tick = args.gripper_min_tick or runner.gripper_min_tick(alignment)
    arm.set_gripper_min_tick(tick)
    touch = alignment.sim_degrees_to_ticks("gripper", -12.0, clamp=False)
    print(f"Jaw floor {runner.JAW_FLOOR_DEG:.2f} deg; EEPROM min tick {tick}; tips touch at -12.00 deg = tick {touch:.0f}; "
          f"stall compensation {arm.stall_ticks():.0f} ticks.")

    def find_stall() -> None:
        hard_floor = int(round(touch - 6.0 * runner.alignment_lib.TICKS_PER_DEGREE))
        bus.disable_torque("gripper", num_retry=runner.SERIAL_RETRIES)
        bus.write("Min_Position_Limit", "gripper", hard_floor, normalize=False, num_retry=runner.SERIAL_RETRIES)
        bus.write("Goal_Position", "gripper", int(round(touch)), normalize=False)
        bus.enable_torque("gripper", num_retry=runner.SERIAL_RETRIES)
        found = None
        for goal in range(int(round(touch)), hard_floor - 1, -2):
            bus.write("Goal_Position", "gripper", goal, normalize=False, num_retry=runner.SERIAL_RETRIES)
            time.sleep(0.25)
            pos = bus.read("Present_Position", "gripper", normalize=False)
            print(f"\r  goal {goal}  at {pos}  ({alignment.ticks_to_sim_degrees('gripper', pos):6.2f} deg)", end="", flush=True)
            if pos <= touch + 1:
                found = touch - goal
                break
        print()
        bus.write("Goal_Position", "gripper", int(round(touch + 100)), normalize=False)
        time.sleep(0.8)
        if found is None:
            print("  The jaw never reached the touch point within 6 deg of extra goal: check the jaws for binding.")
        else:
            alignment.notes["gripper_stall_ticks"] = round(float(found), 1)
            alignment.save(path)
            print(f"  Stall = {found:.0f} ticks ({found / runner.alignment_lib.TICKS_PER_DEGREE:.2f} deg); saved to {path}.")
        bus.disable_torque("gripper", num_retry=runner.SERIAL_RETRIES)
        new_tick = runner.gripper_min_tick(alignment)
        bus.write("Min_Position_Limit", "gripper", new_tick, normalize=False, num_retry=runner.SERIAL_RETRIES)
        arm.set_gripper_min_tick(new_tick)
        bus.write("Goal_Position", "gripper", bus.read("Present_Position", "gripper", normalize=False), normalize=False)
        bus.enable_torque("gripper", num_retry=runner.SERIAL_RETRIES)
        print(f"  EEPROM min tick now {new_tick}. Press c to check the runner's full close.")

    def go(deg: float) -> None:
        deg = max(deg, runner.JAW_FLOOR_DEG)
        goal = arm.gripper_ticks(deg)  # exactly what the runner sends
        bus.write("Goal_Position", "gripper", goal, normalize=False, num_retry=runner.SERIAL_RETRIES)
        time.sleep(1.2)
        for _ in range(3):
            pos = bus.read("Present_Position", "gripper", normalize=False)
            load = bus.read("Present_Load", "gripper", normalize=False)
            current = bus.read("Present_Current", "gripper", normalize=False)
            temp = bus.read("Present_Temperature", "gripper", normalize=False)
            # STS load: bit 10 is direction, low 10 bits magnitude in 0.1 % of max torque.
            pct = (load & 0x3FF) / 10.0
            print(f"  goal {deg:7.2f} deg (tick {goal})  at {alignment.ticks_to_sim_degrees('gripper', pos):7.2f} deg "
                  f"(tick {pos})  load {pct:5.1f}%  current {current * 6.5:5.0f} mA  temp {temp} C")
            time.sleep(0.3)

    bus.write("Goal_Position", "gripper", bus.read("Present_Position", "gripper", normalize=False), normalize=False)
    bus.enable_torque("gripper", num_retry=runner.SERIAL_RETRIES)
    try:
        go(OPEN_DEG)
        while True:
            command = input("c close / o open / t measure stall / <sim deg> / q quit > ").strip().lower()
            if command == "q":
                break
            if command == "t":
                find_stall()
            elif command == "c":
                go(runner.JAW_FLOOR_DEG)
            elif command == "o":
                go(OPEN_DEG)
            else:
                try:
                    deg = float(command)
                except ValueError:
                    print("  ?")
                    continue
                if not -15.0 <= deg <= 100.0:
                    print("  Numbers are sim DEGREES (-12.75 closed .. ~100 open), not motor ticks.")
                    continue
                go(deg)
    finally:
        try:
            go(OPEN_DEG)
        finally:
            bus.disable_torque("gripper", num_retry=runner.SERIAL_RETRIES)
            bus.port_handler.closePort()
            print("Gripper open, torque off.")


if __name__ == "__main__":
    main()
