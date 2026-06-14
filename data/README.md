# Dataset files

Data files are **not** committed to this repository — they are covered by the
ETH course's internal-use policy. Place the files yourself at the paths below
and the task entry points will find them automatically.

## Directory layout

```
data/
└── autonomous_cars/
    └── demo.p   # KITTI-style pickle (single frame)
```

## Sources

| Dataset | Origin |
|---------|--------|
| `autonomous_cars/demo.p` | CVAI Cars 2023, project 2 handout — a single KITTI frame with calibration, annotations, and semantically-labelled LiDAR. |

## Tips

- The single `demo.p` pickle bundles calibration matrices, 3D annotations, and
  a semantically-labelled LiDAR scan for one frame.
- If you want to version the data too, enable Git LFS and remove the
  blanket ignores in `.gitignore`.
