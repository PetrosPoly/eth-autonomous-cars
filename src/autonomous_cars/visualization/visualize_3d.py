"""Interactive 3-D viewer for KITTI point clouds and bounding boxes.

Adapted from the handout material for *Computer Vision and AI for Autonomous
Cars* (Project 2, Problem 1). Requires ``vispy`` and a usable OpenGL backend
(install via ``poetry install --extras vis``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vispy
from vispy.scene import SceneCanvas, visuals

from autonomous_cars.visualization.data_loader import load_data


class Visualizer:
    def __init__(self):
        self.canvas = SceneCanvas(keys="interactive", show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color="white", parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up="z", azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)

        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray(
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

    def update(self, points, colors):
        """
        :param points: point cloud data
                        shape (N, 3)
        Task: Change this function such that each point
        is colored depending on its semantic label
        """
        self.sem_vis.set_data(points, size=3, face_color=colors)

    def update_boxes(self, corners):
        """
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        """
        for i in range(corners.shape[0]):
            connect = (
                np.concatenate((connect, self.connect + 8 * i), axis=0) if i > 0 else self.connect
            )
        self.obj_vis.set_data(corners.reshape(-1, 3), connect=connect, width=2, color=[0, 1, 0, 1])


def compute_box_in_wolrd(dimension, location, ry, extrinsic_matrix_2):
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
    vertices_in_camera2 = vertices_in_camera0 - np.array([0.06, 0, 0])

    rotation_matrix_ex = extrinsic_matrix_2[:3, :3]
    translation_ex = extrinsic_matrix_2[3, :3]

    vertices_in_world = (vertices_in_camera2 - translation_ex) @ rotation_matrix_ex
    return vertices_in_world


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    data = load_data(repo_root / "data" / "autonomous_cars" / "demo.p")
    visualizer = Visualizer()
    points_color = [data["color_map"][x][::-1] for x in data["sem_label"][:, 0]]
    points_color = [[color[0] / 256, color[1] / 256, color[2] / 256] for color in points_color]
    visualizer.update(data["velodyne"][:, :3], points_color)
    """
    Task: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    """
    boxes = np.array(
        [
            compute_box_in_wolrd(object[8:11], object[11:14], object[14], data["T_cam2_velo"])
            for object in data["objects"]
        ]
    )
    visualizer.update_boxes(boxes)

    vispy.app.run()
