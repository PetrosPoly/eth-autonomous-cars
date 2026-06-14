"""Two-stage 3-D object detection (Project 2 / Problem 2).

Building blocks for the second stage of a PointNet-style detector:

* :mod:`.boxes` — ``label2corners``, pairwise 3-D IoU, recall.
* :mod:`.roi_pool` — enlarge + gather + fixed-size sampling of points
  inside each proposed 3-D bounding box.
"""

from autonomous_cars.detection_3d.boxes import (
    compute_recall,
    get_iou,
    label2corners,
)
from autonomous_cars.detection_3d.roi_pool import roi_pool

__all__ = ["compute_recall", "get_iou", "label2corners", "roi_pool"]
