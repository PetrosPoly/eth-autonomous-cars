"""Project LiDAR points and 3-D bounding boxes into the camera-2 image.

Loads a KITTI-style pickle via :func:`autonomous_cars.visualization.load_data`,
projects velodyne points via the calibration matrices, and overlays 3-D
bounding boxes (one per annotated object). The output is saved to
``./results/visualize_2d.pdf`` by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autonomous_cars.visualization.data_loader import load_data


def compute_box_in_image2(dimension, location, ry, intrinsic_matrix_2):
    h, w, l = dimension
    x, y, z = location

    rotation_matrix = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    vertices = np.array(
        [
            [l / 2, 0, w / 2],
            [-l / 2, 0, w / 2],
            [-l / 2, 0, -w / 2],
            [l / 2, 0, -w / 2],
            [l / 2, -h, w / 2],
            [-l / 2, -h, w / 2],
            [-l / 2, -h, -w / 2],
            [l / 2, -h, -w / 2],
        ]
    )
    vertices_in_camera0 = (vertices @ rotation_matrix.T) + np.array([x, y, z])
    vertices_in_camera2 = vertices_in_camera0 - np.array(
        [0.06, 0, 0]
    )  # 0.6(m) is written in handout Fig1.
    vertices_in_image2 = vertices_in_camera2 @ intrinsic_matrix_2.T

    return vertices_in_image2


def compute_points_in_image2(points_in_world, intrinsic_matrix_2, extrinsic_matrix_2):
    points_homo_in_world = np.concatenate(
        (points_in_world, np.ones((points_in_world.shape[0], 1))), axis=1
    )
    points_in_camera2 = points_homo_in_world @ extrinsic_matrix_2.T
    points_in_image2 = points_in_camera2 @ intrinsic_matrix_2.T
    return points_in_image2


def visualize_in_image2(data):
    # set points and colors
    points_in_world = data["velodyne"][:, :3]
    points_color = [data["color_map"][x][::-1] for x in data["sem_label"][:, 0]]

    # set matrix for ponnt clouds
    intrinsic_matrix_2_with_rect = data["P_rect_20"]
    extrinsic_matrix_2 = data["T_cam2_velo"]

    # caluclate point clouds
    points_in_image2 = compute_points_in_image2(
        points_in_world, intrinsic_matrix_2_with_rect, extrinsic_matrix_2
    )

    # set image and prepare for the visulaization
    image2 = data["image_2"].astype(np.uint8)
    image2 = Image.fromarray(image2, mode="RGB")
    draw = ImageDraw.Draw(image2)

    # visualize the points
    for i in range(len(points_in_image2)):
        x, y, z = points_in_image2[i]
        if z > 0:
            draw.point((x // z, y // z), fill=tuple(points_color[i]))

    # calculate bounding boxes
    objects = data["objects"]
    intrinsic_matrix_2 = data["K_cam2"]
    boxes_in_image2 = [
        compute_box_in_image2(object[8:11], object[11:14], object[14], intrinsic_matrix_2)
        for object in objects
    ]

    # connection between the vertices in bounding box.
    connect = np.asarray(
        [
            [0, 1],
            [0, 3],
            [0, 4],
            [2, 1],
            [2, 3],
            [2, 6],
            [5, 1],
            [5, 4],
            [5, 6],
            [7, 3],
            [7, 4],
            [7, 6],
        ]
    )

    # visualize bounding box
    for vertices in boxes_in_image2:
        for vertex in vertices:
            x, y, z = vertex
            draw.point((x // z, y // z))
        for i1, i2 in connect:
            x1, y1, z1 = vertices[i1]
            x2, y2, z2 = vertices[i2]
            draw.line([(x1 // z1, y1 // z1), (x2 // z2, y2 // z2)])

    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    image2.save(output_dir / "visualize_2d.pdf")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    data = load_data(repo_root / "data" / "autonomous_cars" / "demo.p")
    visualize_in_image2(data)
