# Recording Analyzer

Tiny standalone version of the four-camera notebook viewer.

## Setup

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the venv like this instead:

```bash
.venv\Scripts\activate
```

## Four-camera viewer

```bash
python analyze_recordings.py /path/to/recording_root
```

The recording folder should contain the four camera folders:

```text
depth_front_left/
depth_front_right/
depth_rear_left/
depth_rear_right/
```

Each camera folder needs a `color/` folder with `color_*.png` files and a `depth/` folder with `depth_*.png` files.

The viewer opens a separate **Depth camera navigation** window. It previews
the `DepthCameraNavigation` content for the selected recording time by combining
the median deviations from the two front cameras and the two rear cameras. The
same confidence, timestamp-difference, angle-difference, and
horizontal-difference thresholds as the ROS2 implementation are applied. The
age check accepts the nearest recorded frame on either side of the selected
time. The panel shows the configured 10 Hz / 100 ms output cycle; the offline
preview itself is refreshed whenever the selected frame changes.

The **Edge pipeline debug** window has centered, labeled camera buttons and a
labeled pipeline-stage slider. Click a camera, stage name, or slider tick to
show raw depth, cropped ROI depth, depth after cutoff, visualized depth, Canny
edges, filtered edges, or the Hough result for the current master frame. The
Hough view numbers the current candidate pool in blue and highlights the chosen
line in yellow.

The four `*_depth_processing.yaml` files mirror the current production camera
intrinsics, ROIs, reference offsets, and line-selection settings. Horizontal
deviation uses the same interpolated per-pixel depth sum as the ROS2 pipeline.

### Controls

- Left/Right or A/D: move one frame
- Up/Down or W/S: move ten frames
- type a number + Enter: jump to a master frame
- Home/End: first/last frame
- Q/Esc: close

## Edge-parameter stability optimizer

Use the headless optimizer to search for stable front/rear navigation validity
over recording frame numbers 260–1650 inclusive:

```bash
python optimize_edge_parameters.py ../kaunitz-rec/20260804_081059
```

By default it uses seed `42`, up to eight worker processes, and writes to
`tuning_results/20260804_081059/`. Use `--start-frame`, `--end-frame`,
`--output-dir`, `--seed`, or `--workers` to override those defaults. Frame
arguments are PNG recording frame numbers, not viewer master-frame indices.

The search varies only Canny minimum/maximum, zero-mask dilation, Hough
threshold, minimum line length, and maximum line gap. One setting is shared by
both front cameras and another by both rear cameras. Navigation validity gates,
ROI/depth preprocessing, reference and median-line settings, left/right
selection, and the current Hough candidate-pool behavior remain frozen. The
optimizer does not rewrite the camera YAML files and verifies protected-file
hashes before writing results.

Before searching, it compares the current configuration with the planned
baseline (front: 78.50% valid, 167 transitions, longest invalid run 96; rear:
68.08%, 303, 163). A mismatch stops before the search. If tuning the changed
working state is intentional, pass `--allow-baseline-drift`; the current
configuration then becomes the baseline and its actual values are recorded.

Outputs include `report.md`, machine-readable `results.json`, per-frame
`winner_frames.csv`, and `contact_sheets/` images comparing baseline and winner
on representative test frames and the longest dropouts. Inspect the contact
sheets before applying parameters; higher validity alone does not prove that
the detected line is the rail.

## Folder processing script

Use `process_depth_images.py` to process paired color/depth PNGs from one camera
and save the processed visualizations as images.

Example for front left:
```bash
python process_depth_images.py \
  --config front_left_depth_processing.yaml \
  --input-dir /path/to/depth_front_left \
  --output-dir /path/to/processed_output
```

The input folder should contain `color/` and `depth/` subfolders with matching
PNG names such as `color_000001.png` and `depth_000001.png`.

The output folder is created if needed. For each input image, the script writes a
processed PNG with the same name plus `_processed`: `depth_000001_processed.png`.


The saved image contains the processed depth visualization, color image,
reference line, current and median detected lines, center point, and a metadata
panel on the right. The per-camera `*_depth_processing.yaml` files contain the
camera, ROI, reference-line, median-line, and edge-detection settings used by
the processing pipeline.


![alt text](images/depth_000583_processed.png)
