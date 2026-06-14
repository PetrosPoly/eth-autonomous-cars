# Project 2 / Problem 1 — KITTI point-cloud & bounding-box visualisation

**Course:** Computer Vision and Artificial Intelligence for Autonomous
Cars, ETH Zürich (autumn 2023).

## Problem

Given a KITTI-style frame (RGB image, LiDAR point cloud with semantic
labels, camera intrinsics/extrinsics, and a list of annotated 3-D
bounding boxes):

1. Project the LiDAR points into the camera-2 image plane and colour
   each point by its semantic label.
2. Draw the 3-D bounding boxes on top of the projected image.
3. Additionally, render the same scene interactively in 3-D.

## Pipeline

- `visualize_2d.visualize_in_image2(frame)` — perspective-project points
  through $\mathbf{K}_2 \cdot \mathbf{T}_{\text{cam2}\leftarrow\text{velo}}$
  and draw bounding-box wireframes.
- `visualize_3d.Visualizer` — `vispy` scene with a turntable camera and
  ``visuals.Markers`` for the semantic point cloud plus ``visuals.Line``
  for the 3-D boxes.

Both files share `data_loader.load_data`, which unpickles a dict with
keys ``velodyne``, ``sem_label``, ``color_map``, ``image_2``, ``K_cam2``,
``P_rect_20``, ``T_cam2_velo``, ``objects``.

## Run

```bash
# 2-D overlay (saves a PDF under ./results/)
poetry run python -m autonomous_cars.visualization.visualize_2d

# 3-D interactive viewer (requires a desktop OpenGL context)
poetry install --extras vis
poetry run python -m autonomous_cars.visualization.visualize_3d
```

Place the handout's `demo.p` / `data.p` pickle under
`data/autonomous_cars/`.
