# Case-study overview

Short, one-paragraph summaries of each sub-project, with pointers to the
relevant module. For full problem statements and implementation details
see the README inside each subpackage.

## Autonomous Cars

### Problem 1 — KITTI visualisation
`autonomous_cars.visualization` — project LiDAR points into the
camera-2 image plane with KITTI calibration matrices, colour by semantic
label, and draw 3-D bounding-box wireframes both in 2-D (`PIL`) and 3-D
(`vispy`).

### Problem 2 — 3-D object detection blocks
`autonomous_cars.detection_3d` — the numerical core of the
second-stage detector: 3-D IoU between bounding boxes, recall at a given
threshold, and RoI pooling that enlarges boxes by $\delta$ and resamples
the contained points to a fixed count.
