"""Scene, camera and motion constants shared by every stage."""

import numpy as np

# ------------------------------ Scene ----------------------------------------
TABLE_Z = 0.80
APPLE_RADIUS = 0.025
APPLE_Z = TABLE_Z + APPLE_RADIUS + 0.002
# 3 red + 2 green. Rows are 10 cm apart in x and apples >= 12 cm apart in y so
# the open Panda fingers (~8 cm) never touch a neighbouring apple.
APPLE_POSITIONS = {
    "red": [
        np.array([-0.08, -0.12, APPLE_Z]),
        np.array([-0.08, 0.00, APPLE_Z]),
        np.array([-0.08, 0.12, APPLE_Z]),
    ],
    "green": [
        np.array([0.02, -0.06, APPLE_Z]),
        np.array([0.02, 0.06, APPLE_Z]),
    ],
}
APPLE_RGBA = {
    "red": [0.90, 0.04, 0.04, 1.0],
    "green": [0.08, 0.70, 0.12, 1.0],
}
COLORS = tuple(APPLE_POSITIONS)

BOX_CENTER = np.array([0.18, 0.0, TABLE_Z])
BOX_HALF_X = 0.075
BOX_HALF_Y = 0.070
BOX_WALL = 0.008
BOX_HEIGHT = 0.060
# Drop slots inside the box so several apples do not stack on one spot.
BOX_SLOTS = [
    np.array([-0.03, -0.03]),
    np.array([0.03, 0.03]),
    np.array([-0.03, 0.03]),
    np.array([0.03, -0.03]),
]

# ------------------------------ Camera ---------------------------------------
# Top-down camera for vision: the oblique frontview sees the apple's shaded
# underside poorly, which biased the estimate ~14 mm and caused off-center grasps.
CAMERA = "birdview"
RENDER_CAMERA = "frontview"
CAMERA_WIDTH = 512
CAMERA_HEIGHT = 512
MIN_BLOB_PIXELS = 10

# ------------------------------ Motion ---------------------------------------
OSC_TRANSLATION_LIMIT = 0.05  # meters per normalized OSC_POSE action
GRIPPER_OPEN = -1.0  # robosuite gripper: -1 open, +1 close
GRIPPER_CLOSE = 1.0
# The robot base and the box both sit on y=0, so from the top-down camera the arm
# hides anything on the table's centreline. Park the gripper folded back near the
# base (behind the apple rows at x >= -0.105) before every image capture.
CAMERA_CLEAR_POSE = np.array([-0.32, 0.0, TABLE_Z + 0.35])
