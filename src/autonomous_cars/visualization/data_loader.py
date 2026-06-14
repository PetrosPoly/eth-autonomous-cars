"""Loader for the KITTI-style dataset pickle shipped with Project 2.

The course distribution of *Computer Vision and AI for Autonomous Cars* ships
a pickle file (``data.p`` / ``demo.p``) containing calibrated point clouds,
images, and 3-D bounding-box labels for a single KITTI-like frame. This
module simply unpickles the file into a dict.

The pickle is not included in this repository; place it at
``data/autonomous_cars/<name>.p`` and load with::

    from autonomous_cars.visualization import load_data
    frame = load_data("data/autonomous_cars/demo.p")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_data(path: str | Path) -> dict[str, Any]:
    """Return the dictionary stored inside the given pickle file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; see data/autonomous_cars/README.md for the "
            "expected files and where to obtain them."
        )
    with path.open("rb") as f:
        return pickle.load(f)
