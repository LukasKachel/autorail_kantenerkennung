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

## Run

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

## Controls

- Left/Right or A/D: move one frame
- Up/Down or W/S: move ten frames
- type a number + Enter: jump to a master frame
- Home/End: first/last frame
- Q/Esc: close
# autorail_kantenerkennung
