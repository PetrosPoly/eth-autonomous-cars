"""2-D and 3-D KITTI visualisation (Project 2 / Problem 1).

Utilities to project LiDAR point clouds and 3-D bounding boxes into the
camera-2 image plane (:mod:`.visualize_2d`) and a :mod:`vispy`-based
interactive 3-D viewer (:mod:`.visualize_3d`).
"""

from autonomous_cars.visualization.data_loader import load_data

__all__ = ["load_data"]
