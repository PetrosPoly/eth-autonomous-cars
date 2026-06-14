# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this is

`autonomous-cars` — production-style implementations of ETH Zürich *Computer
Vision and AI for Autonomous Cars* coursework (2023). A single Poetry-managed
Python package under `src/autonomous_cars/`.

## Layout

```
src/autonomous_cars/
├── visualization/      # KITTI 2D/3D viz: LiDAR→image projection, 3D box wireframes
│   ├── data_loader.py  # load_data() for the KITTI-style pickle
│   ├── visualize_2d.py # 2D overlay via PIL
│   └── visualize_3d.py # 3D scene via vispy (optional `vis` extra)
└── detection_3d/       # numerical core of a two-stage 3D detector
    ├── boxes.py        # 3D IoU between boxes, recall at threshold
    └── roi_pool.py     # enlarge boxes by delta, resample contained points
tests/                  # pytest; covers detection_3d
```

Each subpackage has a local `README.md` with problem statement and run details.

## Commands

```bash
poetry install            # core deps (numpy, pillow, shapely)
poetry install --extras vis   # add vispy + pyqt5 for 3D visualisation
make check                # ruff lint + ruff format --check + pytest  (run before committing)
make format               # auto-format
make test                 # pytest
```

Run the visualisations (needs `data/autonomous_cars/demo.p` — see `data/README.md`):

```bash
poetry run python -m autonomous_cars.visualization.visualize_2d
poetry run python -m autonomous_cars.visualization.visualize_3d   # needs the vis extra
```

## Conventions

- Python ≥3.10, line length 100, formatted and linted with **ruff** (config in
  `pyproject.toml`).
- Import paths are flat: `from autonomous_cars.<subpkg> import ...` (no `mlcs`
  prefix).
- Path resolution in entry points uses `Path(__file__).resolve().parents[N]` to
  find the repo root — update `N` if you move a file between directory levels.
- `detection_3d` keeps legacy `np.random` as part of its public API (NPY002 is
  silenced there); `visualize_3d.py` references vispy symbols guarded at import
  time (F821 silenced). Adjust `per-file-ignores` rather than rewriting that code.
- No torch / scipy / sklearn here — this domain is numpy + shapely + PIL (+ vispy).

## Data

Datasets are **not** committed (see `data/README.md`). The single `demo.p`
pickle bundles calibration, 3D annotations, and labelled LiDAR for one frame.

## Before committing

Always run `make check` and make sure it is green. Do not commit unless asked.
