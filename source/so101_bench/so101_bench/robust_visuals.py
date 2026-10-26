"""Explicit evaluation-only visual holdouts; never add these to training samplers."""

TRAINING_BACKGROUND_INDICES = tuple(range(1, 19))
ROBUST_BACKGROUND_INDICES = (19, 20, 21, 22)

# Linear RGB diffuse material values, converted from the proposed sRGB swatches.
# Keep separate from mdp.resets.ROBOT_COLORS (the existing training palette).
ROBUST_ROBOT_COLORS = {
    "royal_blue": (0.0307, 0.0908, 0.6939),  # #3155D9
    "leaf_green": (0.0168, 0.2961, 0.0630),  # #239447
    "violet": (0.2346, 0.0437, 0.5520),      # #853BC4
    "magenta_pink": (0.7011, 0.0452, 0.2831),  # #DA3C91
}
