"""The robosuite scene: Panda robot, table, open box and red/green apples."""

import xml.etree.ElementTree as ET

import numpy as np

from . import sim_setup  # noqa: F401  (must run before the robosuite imports below)
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.tasks import ManipulationTask

from .config import (
    APPLE_POSITIONS,
    APPLE_RADIUS,
    APPLE_RGBA,
    BOX_CENTER,
    BOX_HALF_X,
    BOX_HALF_Y,
    BOX_HEIGHT,
    BOX_WALL,
    CAMERA,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    RENDER_CAMERA,
    TABLE_Z,
)


class ApplePickPlaceVision(Lift):
    """Custom scene reusing robosuite's Panda robot and control stack."""

    target_color = "red"
    target_count = 1

    def _load_model(self):
        ManipulationEnv._load_model(self)
        robot = self.robots[0]
        robot.robot_model.set_base_xpos(
            robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
        )

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self._add_box_walls(arena.worldbody)

        self.apples = {
            color: [
                BallObject(name=f"{color}_apple_{i}", size=[APPLE_RADIUS], rgba=APPLE_RGBA[color])
                for i in range(len(positions))
            ]
            for color, positions in APPLE_POSITIONS.items()
        }
        # Lift expects a `cube` object for its default observable setup.
        self.cube = self.apples["red"][0]
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[obj for objs in self.apples.values() for obj in objs],
        )

    @staticmethod
    def _add_box_walls(worldbody):
        """Make an open box; the table surface is its bottom."""
        cx, cy, _ = BOX_CENTER
        z = TABLE_Z + BOX_HEIGHT / 2
        wall_specs = [
            ("box_x_neg", [cx - BOX_HALF_X, cy, z], [BOX_WALL, BOX_HALF_Y, BOX_HEIGHT / 2]),
            ("box_x_pos", [cx + BOX_HALF_X, cy, z], [BOX_WALL, BOX_HALF_Y, BOX_HEIGHT / 2]),
            ("box_y_neg", [cx, cy - BOX_HALF_Y, z], [BOX_HALF_X, BOX_WALL, BOX_HEIGHT / 2]),
            ("box_y_pos", [cx, cy + BOX_HALF_Y, z], [BOX_HALF_X, BOX_WALL, BOX_HEIGHT / 2]),
        ]
        for name, pos, size in wall_specs:
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "pos": " ".join(map(str, pos)),
                    "size": " ".join(map(str, size)),
                    "rgba": "0.18 0.28 0.38 1",
                    "group": "1",  # visible while retaining collision geometry
                    "friction": "1.0 0.005 0.0001",
                },
            )

    def _setup_references(self):
        # Keep Lift's standard robot references, then track every apple body.
        ManipulationEnv._setup_references(self)
        self.apple_body_ids = {
            color: [self.sim.model.body_name2id(obj.root_body) for obj in objs]
            for color, objs in self.apples.items()
        }
        self.cube_body_id = self.apple_body_ids["red"][0]

    def _reset_internal(self):
        ManipulationEnv._reset_internal(self)
        for color, objs in self.apples.items():
            for obj, pos in zip(objs, APPLE_POSITIONS[color]):
                qpos = np.array([*pos, 1.0, 0.0, 0.0, 0.0])  # xyz + identity quaternion
                self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    # ---- Evaluation only: privileged simulator state, never used for planning ----
    def apple_positions_for_evaluation(self, color):
        return [np.array(self.sim.data.body_xpos[i]).copy() for i in self.apple_body_ids[color]]

    @staticmethod
    def is_in_box(pos):
        inside_x = abs(pos[0] - BOX_CENTER[0]) < BOX_HALF_X - BOX_WALL
        inside_y = abs(pos[1] - BOX_CENTER[1]) < BOX_HALF_Y - BOX_WALL
        low_enough = TABLE_Z + APPLE_RADIUS - 0.015 <= pos[2] <= TABLE_Z + BOX_HEIGHT + 0.04
        return bool(inside_x and inside_y and low_enough)

    def count_in_box(self, color):
        return sum(self.is_in_box(p) for p in self.apple_positions_for_evaluation(color))

    def _check_success(self):
        # Enough target-colour apples in the box and none of the other colours.
        target_in = self.count_in_box(self.target_color)
        wrong_in = sum(self.count_in_box(c) for c in self.apples if c != self.target_color)
        return target_in >= self.target_count and wrong_in == 0

    def reward(self, action=None):
        return float(self._check_success())


def make_env(render=True, record=False):
    """Build the scene with the Panda OSC controller and the vision camera enabled.

    With `record`, the scene camera is also rendered into every observation so
    step snapshots can be saved (slower: two offscreen renders per step).
    """
    camera_names = [CAMERA, RENDER_CAMERA] if record else [CAMERA]
    return ApplePickPlaceVision(
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
        camera_depths=False,
        use_object_obs=True,
        render_camera=RENDER_CAMERA,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        reward_shaping=False,
    )
