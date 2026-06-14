# Project 2 / Problem 2 — Two-stage 3-D object detection building blocks

**Course:** Computer Vision and Artificial Intelligence for Autonomous
Cars, ETH Zürich (autumn 2023).

## Problem

Implement two core building blocks for the second stage of a PointRCNN-
style detector on KITTI data:

1. **Bounding-box geometry.** Given KITTI-format labels
   $(x, y, z, h, w, l, r_y)$:
   - Convert to the 8 corners of the 3-D box.
   - Compute pairwise 3-D IoU between a set of predictions and targets.
   - Compute recall at a chosen IoU threshold.
2. **RoI pooling.** Given stage-1 predictions plus the raw LiDAR cloud
   and per-point features, for each box:
   - Enlarge by $\delta$ metres in all directions.
   - Gather the points (and features) inside the enlarged box.
   - Resample to exactly `max_points` points via random draw (too many
     points) or random repeat (too few). Discard empty boxes.

## Key implementation notes

- The 2-D intersection polygon uses `shapely.geometry.Polygon` on the
  projection onto the $xz$ ground plane; the height intersection is the
  1-D overlap along $y$.
- Coordinate convention: the bounding-box centre $(x,y,z)$ sits on the
  bottom face with the y-axis pointing down (KITTI rectified camera
  frame). The point cloud arrives in the LiDAR FLU frame; `points_in_box`
  converts FLU → RDF before testing membership.

## Run

This module exposes library functions consumed by a wider PointRCNN
pipeline (not included in this repository):

```python
from autonomous_cars.detection_3d import (
    compute_recall, get_iou, label2corners, roi_pool,
)
```

There's no standalone CLI; unit tests for IoU/recall live in `tests/`.
