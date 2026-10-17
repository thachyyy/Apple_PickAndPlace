"""Scripted waypoint motion and the pick-and-place task loop."""

import numpy as np

from .config import (
    APPLE_POSITIONS,
    APPLE_RADIUS,
    BOX_CENTER,
    BOX_HEIGHT,
    BOX_SLOTS,
    CAMERA,
    CAMERA_CLEAR_POSE,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    OSC_TRANSLATION_LIMIT,
    RENDER_CAMERA,
    TABLE_Z,
)
from .vision import estimate_targets_from_camera


def snapshot(recorder, obs, name):
    """Save the scene camera image for this step when recording is on."""
    if recorder is not None:
        recorder.save_camera(name, obs, RENDER_CAMERA)


# ------------------------------ Motion primitives ----------------------------
def step_toward(env, obs, target_xyz, gripper, max_steps=100, tolerance=0.012, render=True):
    """Use current end-effector proprioception to move toward a waypoint."""
    target_xyz = np.asarray(target_xyz)
    for _ in range(max_steps):
        delta = target_xyz - np.asarray(obs["robot0_eef_pos"])
        if np.linalg.norm(delta) < tolerance:
            return obs
        action = np.zeros(env.action_dim, dtype=float)
        action[:3] = np.clip(delta / OSC_TRANSLATION_LIMIT, -1.0, 1.0)
        action[-1] = gripper
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
    # Not fatal (e.g. a closed gripper pressing on an apple), but worth knowing when tuning poses.
    error = np.linalg.norm(target_xyz - np.asarray(obs["robot0_eef_pos"]))
    print(f"Motion: waypoint {np.round(target_xyz, 3)} not reached, {error * 1000:.0f} mm off.")
    return obs


def hold(env, obs, gripper, steps=15, render=True):
    """Keep the arm still for `steps` control steps while commanding the gripper."""
    action = np.zeros(env.action_dim, dtype=float)
    action[-1] = gripper
    for _ in range(steps):
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
    return obs


def move_out_of_camera_view(env, obs, render=True):
    """Park the arm away from the apples, then let it settle so the next image is unoccluded."""
    obs = step_toward(env, obs, CAMERA_CLEAR_POSE, GRIPPER_OPEN, render=render)
    return hold(env, obs, GRIPPER_OPEN, steps=10, render=render)


def pick_and_place_one(env, obs, apple_pos, release_xy, render=True, recorder=None):
    """Scripted pick of one apple at `apple_pos`, dropped at `release_xy` in the box."""
    approach = apple_pos + np.array([0.0, 0.0, 0.10])
    lift = np.array([apple_pos[0], apple_pos[1], TABLE_Z + 0.28])
    box_approach = np.array([release_xy[0], release_xy[1], TABLE_Z + 0.25])
    box_release = np.array(
        [release_xy[0], release_xy[1], TABLE_Z + BOX_HEIGHT + APPLE_RADIUS + 0.02]
    )

    obs = step_toward(env, obs, approach, GRIPPER_OPEN, render=render)
    snapshot(recorder, obs, "1_approach")
    obs = step_toward(env, obs, apple_pos, GRIPPER_OPEN, render=render)
    obs = hold(env, obs, GRIPPER_CLOSE, steps=25, render=render)
    snapshot(recorder, obs, "2_grasp")
    obs = step_toward(env, obs, lift, GRIPPER_CLOSE, render=render)
    snapshot(recorder, obs, "3_lift")
    obs = step_toward(env, obs, box_approach, GRIPPER_CLOSE, render=render)
    obs = step_toward(env, obs, box_release, GRIPPER_CLOSE, render=render)
    snapshot(recorder, obs, "4_over_box")
    obs = hold(env, obs, GRIPPER_OPEN, steps=35, render=render)
    snapshot(recorder, obs, "5_release")
    # Retreat upward so the arm does not knock the apple on the next approach.
    return step_toward(env, obs, box_approach, GRIPPER_OPEN, render=render)


# ------------------------------ Task ------------------------------------------
def run_pick_place(env, obs, color, count=1, render=True, recorder=None):
    """Pick `count` apples of `color` (None = all) into the box, re-detecting before each pick."""
    available = len(APPLE_POSITIONS[color])
    count = available if count is None else min(count, available)
    if count > len(BOX_SLOTS):
        raise ValueError(f"Box has only {len(BOX_SLOTS)} slots; cannot place {count} apples.")
    env.target_color = color
    env.target_count = count
    print(f"Target selected by language: {count} {color} apple(s)")
    snapshot(recorder, obs, "start")

    for i in range(count):
        if recorder is not None:
            recorder.stage = f"pick{i + 1}"
        # Fresh, unoccluded image each time: earlier picks may have nudged apples.
        obs = move_out_of_camera_view(env, obs, render=render)
        targets = estimate_targets_from_camera(env, obs, color, recorder=recorder)
        if not targets:
            print(f"Vision: no more {color} apples found on the table.")
            break
        if len(targets) < count - i:
            print(f"Vision WARNING: expected {count - i} {color} apple(s), saw {len(targets)} (occluded?).")
        eef = np.asarray(obs["robot0_eef_pos"])
        apple_pos = min(targets, key=lambda p: np.linalg.norm(p[:2] - eef[:2]))
        release_xy = BOX_CENTER[:2] + (BOX_SLOTS[i] if count > 1 else 0.0)
        print(f"Picking {color} apple {i + 1}/{count} at", np.round(apple_pos, 3))
        obs = pick_and_place_one(env, obs, apple_pos, release_xy, render=render, recorder=recorder)

    if recorder is not None:
        recorder.stage = ""
    obs = move_out_of_camera_view(env, obs, render=render)
    snapshot(recorder, obs, "final")
    if recorder is not None:
        recorder.save_camera("final_top", obs, CAMERA)
    success = env._check_success()
    placed = env.count_in_box(color)
    print(
        f"SUCCESS: {placed} {color} apple(s) in the box."
        if success
        else f"NOT VERIFIED: {placed}/{count} {color} apple(s) in the box (or a wrong colour got in)."
    )
    return success
