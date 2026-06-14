import numpy as np
from autonomous_cars.detection_3d import (
    compute_recall,
    get_iou,
    label2corners,
)


def _unit_cube(center=(0.0, 0.0, 0.0), ry=0.0):
    x, y, z = center
    return np.array([[x, y, z, 1.0, 1.0, 1.0, ry]])  # h=w=l=1


def test_label2corners_shape():
    labels = np.vstack([_unit_cube(), _unit_cube(center=(2.0, 0.0, 0.0))])
    corners = label2corners(labels)
    assert corners.shape == (2, 8, 3)


def test_iou_identical_boxes_is_one():
    box = _unit_cube()
    iou = get_iou(box, box)
    assert np.isclose(iou[0, 0], 1.0, atol=1e-6)


def test_iou_non_overlapping_boxes_is_zero():
    a = _unit_cube(center=(0.0, 0.0, 0.0))
    b = _unit_cube(center=(10.0, 0.0, 0.0))
    iou = get_iou(a, b)
    assert np.isclose(iou[0, 0], 0.0)


def test_recall_perfect_matches():
    boxes = np.vstack([_unit_cube(), _unit_cube(center=(5.0, 0.0, 0.0))])
    assert compute_recall(boxes, boxes, threshold=0.5) == 1.0
