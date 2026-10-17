"""RGB color segmentation and pixel -> table-plane projection.

The controller never reads the apples' simulator XYZ positions; targets come
only from the camera image plus the known table height.
"""

import numpy as np
from scipy import ndimage

from . import sim_setup  # noqa: F401  (must run before the robosuite imports below)
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from .config import (
    APPLE_Z,
    BOX_CENTER,
    BOX_HALF_X,
    BOX_HALF_Y,
    BOX_WALL,
    CAMERA,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MIN_BLOB_PIXELS,
)


def color_mask(rgb, color):
    """Simple per-pixel color thresholds; adequate for this controlled teaching scene."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "red":
        return (red > 65) & (red > 1.35 * green) & (red > 1.35 * blue)
    if color == "green":
        return (green > 55) & (green > 1.25 * red) & (green > 1.15 * blue)
    raise ValueError(f"No color mask defined for {color!r}.")


def camera_rgb(obs):
    """The vision camera's current image as an HxWx3 uint8 array."""
    key = f"{CAMERA}_image"
    if key not in obs:
        raise RuntimeError(
            f"Camera observation '{key}' is missing. Check use_camera_obs and camera_names."
        )
    rgb = np.asarray(obs[key])
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError(f"Unexpected camera image shape: {rgb.shape}")
    return rgb[..., :3].astype(np.uint8)


def detect_apple_pixels(rgb, color):
    """Find the pixel centroid of every apple of `color`; returns (centroids, mask)."""
    mask = color_mask(rgb.astype(np.float32), color)

    # Split the mask into one blob per apple (8-connected components).
    labels, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    centroids = []
    for label in range(1, num + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < MIN_BLOB_PIXELS:
            continue  # stray pixels, not an apple
        # Median is less sensitive than a mean to a few stray matching pixels.
        u, v = float(np.median(xs)), float(np.median(ys))
        print(f"Vision: {color} blob at image pixel (u={u:.1f}, v={v:.1f}); pixels={len(xs)}")
        centroids.append((u, v))
    return centroids, mask


def pixel_to_world_on_plane(sim, u, v, plane_z):
    """Project a camera pixel to a known horizontal plane using calibration."""
    intrinsic = get_camera_intrinsic_matrix(
        sim=sim, camera_name=CAMERA, camera_height=CAMERA_HEIGHT, camera_width=CAMERA_WIDTH
    )
    camera_to_world = get_camera_extrinsic_matrix(sim, CAMERA)
    ray_world = camera_to_world[:3, :3] @ (np.linalg.inv(intrinsic) @ np.array([u, v, 1.0]))
    camera_origin = camera_to_world[:3, 3]

    if abs(ray_world[2]) < 1e-8:
        raise RuntimeError("Camera ray is almost parallel to the table plane.")
    distance = (plane_z - camera_origin[2]) / ray_world[2]
    if distance <= 0:
        raise RuntimeError("Detected pixel projects behind the camera.")
    return camera_origin + distance * ray_world


def is_over_box(xyz):
    """True if a table-plane point lies inside the box's outer footprint."""
    return (
        abs(xyz[0] - BOX_CENTER[0]) < BOX_HALF_X + BOX_WALL
        and abs(xyz[1] - BOX_CENTER[1]) < BOX_HALF_Y + BOX_WALL
    )


def estimate_targets_from_camera(env, obs, color, recorder=None):
    """Estimate the centers of all `color` apples still on the table (not in the box)."""
    rgb = camera_rgb(obs)
    centroids, mask = detect_apple_pixels(rgb, color)
    detections = []
    for u, v in centroids:
        xyz = pixel_to_world_on_plane(env.sim, u, v, APPLE_Z)
        detections.append((u, v, xyz, not is_over_box(xyz)))
    if recorder is not None:
        recorder.save_detection(color, rgb, mask, detections)

    targets = [xyz for _, _, xyz, kept in detections if kept]
    print(f"Vision-estimated {color} apples on table (m):", [np.round(t, 3).tolist() for t in targets])
    return targets
