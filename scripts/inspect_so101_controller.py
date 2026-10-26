"""Non-learned SO-101 execution of Inspect's Cartesian target chunks.

Only robot kinematics, joint state, and the table plane enter the controller.
There is no task-object state or perception service here.
"""

from __future__ import annotations

import math
import numpy as np


class PoseController:
    """Prevalidate a Cartesian chunk and retain its bounded joint commands."""

    def __init__(self, kin, arm):
        self.kin = kin
        self.arm = arm
        self.commands = []
        self.extension_steps = 0

    def check(self, waypoints):
        """Reject unreachable/table-penetrating paths before any action executes."""
        self.commands = []
        self.extension_steps = 0
        measured = self.arm.joints()
        previous = np.asarray(self.arm.action[0].detach().cpu().numpy(), dtype=float)
        seed = measured[:5].copy()
        base = self.arm.world_from_base()
        for index, pose in enumerate(waypoints):
            sol = self.kin.solve_ik(base, pose[:3], seed, pitch_deg=float(pose[3]))
            # The wrist's rotation is a well-defined fifth pose coordinate even
            # when the jaw closes vertically and its top-down yaw is undefined.
            solved_q = sol.arm_q.copy()
            solved_q[4] = math.radians(float(pose[4]))
            achieved = self.kin.tool_pose(base, solved_q)
            low, high = self.kin.ARM_LIMITS_RAD[4]
            if (np.linalg.norm(achieved.tcp - pose[:3]) > 0.01 or
                (sol.pitch_error_deg is not None and sol.pitch_error_deg > 5.0) or
                not low <= solved_q[4] <= high or achieved.fingertip[2] < 0):
                self.commands = []
                return (f"waypoint {index} is unreachable or puts fingertips below the table; "
                        f"closest TCP={sol.achieved.tcp.round(4).tolist()}, "
                        f"pitch_deg={sol.achieved.pitch_deg:.1f}, "
                        f"jaw_yaw_deg={sol.achieved.jaw_yaw_deg}")
            goal = np.r_[solved_q, self.kin.jaw_angle_for_opening(float(pose[5]))]
            count = max(1, math.ceil(float(np.max(np.abs(goal - previous))) / 0.06))
            for k in range(1, count + 1):
                command = previous + (goal - previous) * (k / count)
                # Check robot/table geometry on interpolation as well as at Cartesian targets.
                if self.kin.tool_pose(base, command[:5]).fingertip[2] < 0:
                    self.commands = []
                    return f"joint interpolation at waypoint {index} puts fingertips below table"
                self.commands.append(command)
            self.extension_steps += count - 1
            previous, seed = goal, solved_q
        return None
