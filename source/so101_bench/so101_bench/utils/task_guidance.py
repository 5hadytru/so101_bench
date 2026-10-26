"""Shared task-level advice, separate from authoritative benchmark scoring rules.

Only robot/control interfaces differ between Direct and PyRUA. This text names
neither Python/SAM3 primitives nor hidden simulator object state.
"""

REVISION = 'placement-v1-gap12'
COMMON = (
    'Judge grasp acquisition, clearance and placement from fresh camera images. '
    'A closed jaw alone does not prove a grasp. Keep the gripper commanded closed '
    'while carrying; release deliberately. Leave enough simulated time for the '
    'required five-second success confirmation. Do not claim success solely '
    'because a motion command returned successfully.'
)
SPATIAL = (
    'Be VERY slow and patient during spatial placement. The held object, gripper '
    'and arm must avoid both referents/blockers and all other non-target objects. '
    'Carry above obstacles, descend above the intended position, and inspect fresh '
    'images between small moves. Below a grasp-point height of 0.10 m, lower by '
    'at most 1 cm per command before looking again. Stop with the object just '
    'clear of the table, release gently, then withdraw vertically before moving '
    'sideways. Check the released placement and correct it carefully if time allows. '
)
TASK_PLACEMENT_GUIDANCE = {
    'bin': 'Grasp and place every object in the bin. Verify acquisition and release '
           'using both views. Avoid shifting the bin. Once all objects are in, '
           'release and withdraw the arm so it contacts no object. ' + COMMON,
    'next_to': SPATIAL + 'Target a 1-2 cm surface-to-surface gap from the referent '
               'without contact. Use facing edges, not center distance; check the '
               'gap before and after release. ' + COMMON,
    'between': SPATIAL + "Aim the target's reference point onto the straight line "
               "through the referents' centers, near the midpoint: zero offset, "
               'not merely within tolerance. Orient long objects so both ends '
               'remain clear of both referents. After release, re-estimate all '
               'three positions and the perpendicular offset; correct an offset '
               'over about 1 cm if time allows without contact. ' + COMMON,
    'move': SPATIAL + 'Find the blocker specified by the task rules. Target a '
            '1-2 cm gap from the leading edge of the target to the facing edge '
            'of the blocker along the move direction, without contact. Keep the '
            'target centered on its straight path. Check edge gaps before and '
            'after release. With no blocker ahead, aim for about 8 cm progress '
            'in the instructed direction while staying within the path tolerance. ' + COMMON,
}


def task_rules_and_guidance(family, public_rules):
    """The identical task block supplied to Direct and the latest PyRUA profiles."""
    return public_rules[family] + '\nShared task guidance:\n' + TASK_PLACEMENT_GUIDANCE[family]
