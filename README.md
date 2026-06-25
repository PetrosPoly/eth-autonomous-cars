# ETH Zürich Computer Vision & AI for Autonomous Cars (2023)

Production-style implementations of selected coursework from the *Computer
Vision and Artificial Intelligence for Autonomous Cars* course at ETH Zürich,
packaged as a single Python project.

## Scope

- **KITTI 2D/3D visualisation** — project LiDAR points into the camera image
  plane with KITTI calibration matrices, colour by semantic label, and draw
  3D bounding-box wireframes both in 2D (`PIL`) and 3D (`vispy`)
- **3D object detection geometry** — the numerical core of a two-stage
  detector: 3D IoU between bounding boxes, recall at a threshold, and RoI
  pooling that enlarges boxes and resamples the contained points

## Repository structure

```text
.
├── src/autonomous_cars/            # Python package
│   ├── visualization/              # KITTI 2D/3D visualisation
│   └── detection_3d/               # 3D IoU, recall, RoI pooling
├── tests/                          # automated tests
├── data/                           # dataset placeholders + docs
├── docs/                           # high-level technical docs
├── .github/                        # CI, templates, ownership metadata
├── pyproject.toml                  # Poetry project config
└── Makefile                        # common dev commands
```

Each subpackage includes a local `README.md` describing the problem,
methodology, and execution details.

## Quickstart

### Prerequisites

- Python 3.10–3.12
- [Poetry](https://python-poetry.org/)
- macOS system deps (for shapely): `brew install geos`

### Install

```bash
poetry install
poetry run pre-commit install
```

The 3D visualisation backend (`vispy` + `pyqt5`) is optional:

```bash
poetry install --extras vis
```

## Common commands

```bash
make check      # lint + format-check + tests
make format     # auto-format source and tests
make test       # run pytest
```

## Data

Datasets are intentionally excluded from version control.
See [`data/README.md`](data/README.md) for required files and paths.

## CI/CD

GitHub Actions runs Ruff lint, Ruff formatting check, and Pytest on every push
and pull request. See `.github/workflows/ci.yml`.

## License

MIT. See [`LICENSE`](LICENSE).
