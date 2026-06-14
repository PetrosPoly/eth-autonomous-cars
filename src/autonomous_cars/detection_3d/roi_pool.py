"""RoI pooling for the second-stage 3-D object detector.

Given predicted 3-D bounding boxes from the stage-1 network, enlarge them
by ``delta`` metres on every side, gather all LiDAR points inside the
enlarged box, and sample/repeat up to ``max_points`` points per RoI so
that downstream layers operate on fixed-size tensors. Empty boxes are
discarded.
"""

from __future__ import annotations

import numpy as np


def roi_pool(pred, xyz, feat, config):
    """
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    """
    delta = config["delta"]
    max_points = config["max_points"]

    valid_pred, pooled_xyz, pooled_feat = [], [], []

    for box in pred:
        enlarged_box = box.copy()
        enlarged_box = enlarge_bounding_boxes(box, delta)  # x, y, z, h, w, l, ry

        in_box = points_in_box(enlarged_box, xyz)
        box_points = xyz[in_box]
        box_features = feat[in_box]

        if len(box_points) == 0:
            continue  # Skip empty boxes

        if len(box_points) > max_points:
            # Randomly sample points
            chosen_indices = np.random.choice(len(box_points), max_points, replace=False)
            box_points = box_points[chosen_indices]
            box_features = box_features[chosen_indices]

        elif len(box_points) < max_points:
            # Randomly repeat points
            chosen_indices = np.random.choice(len(box_points), max_points, replace=True)
            box_points = box_points[chosen_indices]
            box_features = box_features[chosen_indices]

        valid_pred.append(box)
        pooled_xyz.append(box_points)
        pooled_feat.append(box_features)

    return np.array(valid_pred), np.array(pooled_xyz), np.array(pooled_feat)


def enlarge_bounding_boxes(box, delta):
    # Enlarge each bounding box by delta in all directions (width, length, height)
    box[1] += delta  # y-coordinate
    box[3] += 2 * delta  # height
    box[4] += 2 * delta  # width
    box[5] += 2 * delta  # length
    return box


def points_in_box(box, xyz):
    """Check if points are inside the given bounding box."""
    x, y, z, h, w, l, ry = box

    xyz_rdf = np.zeros_like(xyz)

    # Convert point cloud coordinates from FLU to RDF coordinate system
    xyz_rdf[:, 0] = -xyz[:, 1]  # FLU y (left)  becomes RDF x (right)
    xyz_rdf[:, 1] = -xyz[:, 2]  # FLU z (up)    becomes RDF y (down)
    xyz_rdf[:, 2] = xyz[:, 0] + 0.27  # FLU x (front) becomes RDF z (front)

    rotation_matrix = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    # Apply rotation to points
    xyz_rdf_rotated = np.dot(xyz_rdf, rotation_matrix.T)

    # Check if points are inside the box
    condition_1 = xyz_rdf_rotated >= [x - w / 2, y - h, z - l / 2]
    condition_2 = xyz_rdf_rotated <= [x + w / 2, y, z + l / 2]
    logical_test = np.logical_and(condition_1, condition_2)

    in_box = np.all(logical_test, axis=1)

    return in_box


#  condition_1 =  xyz_rdf_rotated >= [x-l/2, y-h, z-w/2]
#  condition_2 =  xyz_rdf_rotated <= [x+l/2, y,   z+w/2]
