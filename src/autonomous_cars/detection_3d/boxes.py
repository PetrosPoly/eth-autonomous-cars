"""3-D bounding-box geometry: corner extraction, IoU, and recall.

Labels are :math:`(x, y, z, h, w, l, r_y)` in the KITTI rectified
reference frame (y points down, 3-D box centred at ``(x, y, z)`` on the
bottom face, rotated by yaw ``ry`` about the y-axis).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def label2corners(label):
    """
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    """
    # pass
    x, y, z, h, w, l, ry = np.split(label, 7, axis=1)
    rotation_matrix = (
        np.array(
            [
                [np.cos(ry), np.zeros(ry.shape), np.sin(ry)],
                [np.zeros(ry.shape), np.ones(ry.shape), np.zeros(ry.shape)],
                [-np.sin(ry), np.zeros(ry.shape), np.cos(ry)],
            ]
        )
        .transpose((2, 0, 1, 3))
        .reshape(-1, 3, 3)
    )

    vertices = (
        np.array(
            [
                [l / 2, np.zeros(w.shape), w / 2],
                [-l / 2, np.zeros(w.shape), w / 2],
                [-l / 2, np.zeros(w.shape), -w / 2],
                [l / 2, np.zeros(w.shape), -w / 2],
                [l / 2, -h, w / 2],
                [-l / 2, -h, w / 2],
                [-l / 2, -h, -w / 2],
                [l / 2, -h, -w / 2],
            ]
        )
        .transpose((2, 0, 1, 3))
        .reshape(-1, 8, 3)
    )
    corners = np.einsum("ijk,ikl->ijl", rotation_matrix, vertices.transpose(0, 2, 1))
    corners += np.concatenate([x, y, z], axis=1)[:, :, np.newaxis]
    return corners.transpose(0, 2, 1)


def get_iou(pred, target):
    """
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    """
    pred_box, target_box = label2corners(pred), label2corners(target)

    # For area of intersection
    N = len(pred)
    M = len(target)
    pred_polygon = [Polygon(pred_box[i, :4, [0, 2]].transpose(1, 0)) for i in range(N)]
    target_polygon = [Polygon(target_box[i, :4, [0, 2]].transpose(1, 0)) for i in range(M)]
    intersection_area = np.array(
        [[pred_polygon[i].intersection(target_polygon[j]).area for j in range(M)] for i in range(N)]
    )

    # For height of intersection
    start = np.array(
        [[max(pred_box[i, 4, 1], target_box[j, 4, 1]) for j in range(M)] for i in range(N)]
    )
    end = np.array(
        [[min(pred_box[i, 0, 1], target_box[j, 0, 1]) for j in range(M)] for i in range(N)]
    )
    intersection_height = np.maximum(0, end - start)

    # Volume
    pred_volume = pred[:, 3] * pred[:, 4] * pred[:, 5]
    target_volume = target[:, 3] * target[:, 4] * target[:, 5]
    intersection_volume = intersection_area * intersection_height
    union_volume = pred_volume[:, np.newaxis] + target_volume - intersection_volume

    iou_3d = np.where(union_volume > 0, intersection_volume / union_volume, 0)
    return iou_3d


def compute_recall(pred, target, threshold):
    """
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene

    """
    # pass
    iou = get_iou(pred, target)
    tp = np.sum(np.any(iou >= threshold, axis=0))
    fn = np.sum(np.all(iou < threshold, axis=0))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall
