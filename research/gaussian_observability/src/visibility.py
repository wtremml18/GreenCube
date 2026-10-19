"""Coordinate conventions matching the pinned ScalarSyn camera loader."""

import numpy as np


def world_to_voxel(world, metadata):
    matrix = np.array(metadata["voxel_matrix"])[:, [2, 1, 0, 3]]
    smoke = ((world - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T) / np.array(
        metadata["voxel_scale"]
    )
    return smoke * np.array([100, 150, 100])


def project_points(world, camera_to_world_opengl, width, height, fovx):
    c2w = np.array(camera_to_world_opengl, dtype=float).copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    focal = width / (2 * np.tan(fovx / 2))
    uv = focal * camera[:, :2] / camera[:, 2:3] + np.array(
        [(width - 1) / 2, (height - 1) / 2]
    )
    return uv, camera[:, 2]
