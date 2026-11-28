# Mask benchmark

The cases are deliberately small and reproducible.  Point the environment
variables at the local source videos; missing videos are skipped rather than
failing the whole run.

```powershell
$env:FACE_VIDEO="C:\path\face.mp4"
$env:SUBWAY_VIDEO="C:\path\subway.mp4"
$env:DANCE_VIDEO="C:\path\dance.mp4"
$env:STREET_VIDEO="C:\path\street.mp4"
.\.venv\Scripts\python.exe scripts\benchmark_masks.py
```

Outputs:

- `report_assets/benchmark_masks.json` — machine-readable results
- `report_assets/benchmark_masks.md` — small table for review

Metrics per face case:

- `coverage` — polygon area / detector-box area
- `box_ratio` — polygon bbox area / detector-box area
- `polygon_points`
- `glasses_frac` — parser `eye_g` fraction (where available)
- `pass` — `_poly_face_ok` at the configured threshold

`profile_proxy` is a side/ear-visible frame selected from the subway clip; it
is not a substitute for a manually labelled profile set.  Add more cases to
`benchmarks/mask_cases.json` when a labelled profile/occlusion clip exists.

The `false_positive` case checks that a background patch is not detected as a
face, and is reported separately.
