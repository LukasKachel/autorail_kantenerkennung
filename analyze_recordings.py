from __future__ import annotations
from typing import Callable

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import cv2
import numpy as np
import yaml

from helpers import (
    apply_canny_edge_detection,
    apply_depth_cutoff,
    calculate_line_deviation,
    calculate_median_line,
    calculate_pixel_area,
    calculate_pixel_validity_score,
    calculate_temporal_line_score,
    draw_long_line,
    draw_reference_line,
    filter_out_zero_boundaries,
    find_longest_line_right,
    process_depth_image,
)


FPS = 15.0

# RealSense D435f values used for px/mm conversion.
IMG_WIDTH = 848
IMG_HEIGHT = 480
HFOV = 87
VFOV = 58
THETA_HORIZONTAL = HFOV / IMG_WIDTH
THETA_VERTICAL = VFOV / IMG_HEIGHT
DEFAULT_DEPTH_SCALE = 0.0010000000474974513


@dataclass
class CameraConfig:
    """Per-camera pipeline parameters tuned for a specific mounting position.

    All fields are tunable at runtime via the ``ConfigPanel`` sliders.
    The dataclass holds both algorithmic parameters (depth cutoff, Canny,
    Hough) and visualization settings (alpha, info panel, reference line).

    Attributes:
        label: Human-readable camera name shown in the UI (e.g. "Front left").
        folder: Sub-directory name under the recording root that holds this
            camera's ``color/`` and ``depth/`` frame pairs.
        offset_seconds: Time offset applied when synchronizing this camera
            against the master timeline. Positive values shift the camera
            later; negative values shift it earlier.
        depth_scale: Multiplier that converts raw depth units to meters
            (RealSense D435f default: ``0.001``).
        roi_enabled: When True, only a rectangular region of interest is
            processed instead of the full frame.
        roi_x: Left edge of the ROI in pixels (from the original frame).
        roi_y: Top edge of the ROI in pixels.
        roi_width: Width of the ROI in pixels.
        roi_height: Height of the ROI in pixels.
        depth_alpha: Contrast factor for the depth colormap visualization
            (range 0–1, higher = more contrast).
        depth_cutoff_enabled: When True, depth values outside
            ``[depth_cutoff_min_m, depth_cutoff_max_m]`` are zeroed out.
        depth_cutoff_min_m: Closest allowed depth in meters.
        depth_cutoff_max_m: Furthest allowed depth in meters.
        canny_min_val: Lower hysteresis threshold for Canny edge detection.
        canny_max_val: Upper hysteresis threshold for Canny edge detection.
        dilate_size: Kernel size for dilating the zero-depth mask before
            filtering edges that touch invalid pixels (odd number; 1 = off).
        min_line_length: Minimum line length (px) accepted by the
            probabilistic Hough transform.
        max_line_gap: Maximum gap (px) between segments that Hough will
            consider part of the same line.
        hough_threshold: Minimum accumulator votes for Hough; higher values
            produce fewer, more confident line detections.
        find_rightmost_line: When True, pick the rightmost among the longest
            Hough candidates; otherwise pick the leftmost. Set according to
            which side of the vehicle the camera is mounted.
        median_line_enabled: When True, compute a median reference line from
            the recent detection history for temporal scoring.
        median_line_window_size: Number of past frames to include in the
            median-line history window.
        median_line_min_detections: Minimum number of valid detections
            required in the history window for the median to be computed.
        ref_offset_x: Horizontal offset (px) of the reference line from the
            image center. Used for deviation calculations and visualization.
        ref_angle_deg: Angle of the reference line in degrees (90 = vertical).
        info_panel_width: Pixel width of the info panel rendered below each
            camera row. Pure visualization; does not affect the pipeline.
        display_depth_roi_in_full_frame: When True and ROI is enabled, the
            processed depth tile is placed at its ROI position inside a
            full-frame-sized canvas instead of showing the cropped region.
    """
    label: str
    folder: str
    offset_seconds: float = 0.0
    depth_scale: float = DEFAULT_DEPTH_SCALE

    # ROI: Region Of Interest
    # Use a specified region of the original frame.
    roi_enabled: bool = True
    roi_x: int = 100
    roi_y: int = 100
    roi_width: int = 700
    roi_height: int = 120

    # alpha value for visualization of depth image [0,1] (higher -> more contrast)
    depth_alpha: float = 0.55
    # cutoff distance in meters. Distances outside the [min, max] range will be ignored.
    depth_cutoff_enabled: bool = True
    depth_cutoff_min_m: float = 0.2
    depth_cutoff_max_m: float = 1.0

    # Canny edge detection parameters.
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#theory
    canny_min_val: int = 130
    canny_max_val: int = 150
    # kernel size for dilating / removing edges that are created by zero-value (invalid) pixels in the depth image
    # 1 => no dilation, 3 => dilate with a 3x3 kernel, ... (has to be an odd number)
    dilate_size: int = 1

    # Paramters for houghlinesp transform
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html#theory
    min_line_length: int = 50
    max_line_gap: int = 20
    # Minimum number of votes for a line to be detected. Higher means fewer detections.
    hough_threshold: int = 20

    # When True, the line with the rightmost position among detected lines will be selected.
    # Important for left or right side of the vehicle, bc the rails look different
    find_rightmost_line: bool = False

    # When enabled, a median line will be calculated based on the history of detected lines in previous frames.
    median_line_enabled: bool = True
    median_line_window_size: int = 15
    median_line_min_detections: int = 8
    # Tolerances for the temporal confidence score: differences at or beyond
    # these values score 0 for that part. Match the ROS2 config.
    confidence_max_center_shift_px: float = 40.0
    confidence_max_angle_dev: float = 20.0

    # Reference line parameters for deviation calculation and visualization.
    # The reference line is defined by an offset in pixels from the center of the image and an angle in degrees.
    ref_offset_x: int = 0
    ref_angle_deg: float = 90
    # parameters just for visualization of the reference line and info panel, not affecting the processing pipeline
    info_panel_width: int = 560
    display_depth_roi_in_full_frame: bool = True


# ------------------------------------------------------------------
# Parameter definitions for the config panel
# ------------------------------------------------------------------

@dataclass
class ParamDef:
    """Definition of a single tunable parameter rendered in the ``ConfigPanel``.

    Attributes:
        group: Section header under which the parameter is grouped
            (e.g. ``"Depth"``, ``"Canny"``).
        label: Human-readable label shown next to the slider.
        attr: Name of the corresponding attribute on :class:`CameraConfig`
            that this slider reads from and writes to.
        kind: Data type of the parameter: ``"int"``, ``"float"``, or ``"bool"``.
        vmin: Minimum slider value.
        vmax: Maximum slider value.
        step: Increment/decrement amount used by the ``+``/``-`` fine-tune
            buttons.
        fmt: Python format string used to display the current value
            (e.g. ``"{:.2f} m"``).
    """
    group: str
    label: str
    attr: str
    kind: str
    vmin: float
    vmax: float
    step: float
    fmt: str


CONFIG_PARAMS: list[ParamDef] = [
    # ---- Depth ----
    ParamDef("Depth",     "Alpha",        "depth_alpha",
             "float", 0.0, 1.0, 0.01, "{:.2f}"),
    ParamDef("Depth",     "Cutoff On",    "depth_cutoff_enabled",
             "bool",  0,   1,   1,    "{}"),
    ParamDef("Depth",     "Cutoff Min",   "depth_cutoff_min_m",
             "float", 0.0, 5.0, 0.01, "{:.2f} m"),
    ParamDef("Depth",     "Cutoff Max",   "depth_cutoff_max_m",
             "float", 0.0, 5.0, 0.01, "{:.2f} m"),
    # ---- Canny ----
    ParamDef("Canny",     "Min Val",      "canny_min_val",
             "int",   0,   500, 1,    "{}"),
    ParamDef("Canny",     "Max Val",      "canny_max_val",
             "int",   0,   500, 1,    "{}"),
    ParamDef("Canny",     "Dilate",       "dilate_size",
             "int",   1,   21,  1,    "{}"),
    # ---- Hough ----
    ParamDef("Hough",     "Threshold",    "hough_threshold",
             "int",   0,   200, 1,    "{}"),
    ParamDef("Hough",     "Min Length",   "min_line_length",
             "int",   0,   800, 1,    "{} px"),
    ParamDef("Hough",     "Max Gap",      "max_line_gap",
             "int",   0,   200, 1,    "{} px"),
    # ---- Line ----
    ParamDef("Line",      "Rightmost",    "find_rightmost_line",
             "bool",  0,   1,   1,    "{}"),
    ParamDef("Line",      "Median",       "median_line_enabled",
             "bool",  0,   1,   1,    "{}"),
    ParamDef("Line",      "Ref Offset",   "ref_offset_x",
             "int",   -200, 200, 1,  "{} px"),
    ParamDef("Line",      "Ref Angle",    "ref_angle_deg",
             "int",   0,   180, 1,    "{:.0f} deg"),
    # ---- Confidence ----
    ParamDef("Confidence", "Max Shift",   "confidence_max_center_shift_px",
             "float", 0.0, 200.0, 1.0, "{:.0f} px"),
    ParamDef("Confidence", "Max Angle",   "confidence_max_angle_dev",
             "float", 0.0, 90.0, 1.0,  "{:.0f} deg"),
    # ---- ROI ----
    ParamDef("ROI",       "ROI On",       "roi_enabled",
             "bool",  0,   1,   1,    "{}"),
    ParamDef("ROI",       "ROI X",        "roi_x",
             "int",   0,   848, 1,    "{} px"),
    ParamDef("ROI",       "ROI Y",        "roi_y",
             "int",   0,   480, 1,    "{} px"),
    ParamDef("ROI",       "ROI Width",    "roi_width",
             "int",   10,  848, 1,    "{} px"),
    ParamDef("ROI",       "ROI Height",   "roi_height",
             "int",   10,  480, 1,    "{} px"),
]


class ConfigPanel:
    """Custom-drawn OpenCV window with labelled sliders for real-time tuning.

    Uses mouse click-and-drag on slider bars plus +/- click zones for fine
    adjustment.  No trackbars – everything is rendered with cv2 primitives.
    """

    WIDTH = 660
    ROW_H = 26
    HEADER_H = 32
    MARGIN_L = 10
    LABEL_W = 138
    SLIDER_W = 320
    SLIDER_H = 16
    GAP = 6
    SLIDER_Y_OFF = 5   # vertical offset of slider bar inside the row
    BTN_W = 16         # width of +/- buttons

    def __init__(self, window_name: str, dirty_callback: Callable[[], None] | None = None):
        """Create the config panel.

        Args:
            window_name: OpenCV window name used for ``imshow`` / mouse
                callbacks.
            dirty_callback: Optional callable invoked whenever any slider
                value changes, so the parent viewer can mark its display
                as needing a redraw.
        """
        self.window_name = window_name
        self.selected_camera = 0
        self._dirty_callback = dirty_callback
        self._dragging: int | None = None  # row index currently being dragged
        # row layout info: {y, is_header, param?}
        self._rows: list[dict] = []
        self._canvas_h = 800

    @property
    def config(self) -> CameraConfig:
        """The :class:`CameraConfig` currently selected via the camera tabs."""
        return CAMERA_CONFIGS[CAMERA_ORDER[self.selected_camera]]

    # -- value helpers ---------------------------------------------------

    def _get(self, p: ParamDef) -> float:
        """Read the current value of *p* from the active camera config."""
        return float(getattr(self.config, p.attr))

    def _set(self, p: ParamDef, value: float) -> None:
        """Write *value* back to the active config, casting to the correct type.

        Fires the dirty callback (if set) so the viewer re-renders.
        """
        if p.kind == "bool":
            setattr(self.config, p.attr, value >= 0.5)
        elif p.kind == "int":
            setattr(self.config, p.attr, int(round(value)))
        else:
            setattr(self.config, p.attr, value)
        if self._dirty_callback:
            self._dirty_callback()

    # -- coordinate mapping ----------------------------------------------

    def _slider_left(self) -> int:
        """X-coordinate of the left edge of the slider track."""
        return self.MARGIN_L + self.LABEL_W + self.GAP + self.BTN_W

    def _to_frac(self, p: ParamDef) -> float:
        """Map the current value of *p* to a 0–1 fraction for the slider fill."""
        denom = p.vmax - p.vmin
        return (self._get(p) - p.vmin) / denom if denom != 0 else 0.5

    def _x_to_val(self, p: ParamDef, mx: int) -> float:
        """Convert a mouse x-coordinate on the slider track back to a raw value."""
        rel = mx - self._slider_left()
        frac = max(0.0, min(1.0, rel / self.SLIDER_W))
        return p.vmin + frac * (p.vmax - p.vmin)

    # -- row hit-testing -------------------------------------------------

    def _hit_row(self, my: int) -> int | None:
        """Return the parameter row index under mouse y, or ``None`` if no
        slider row was hit (skipping section headers)."""
        for idx, r in enumerate(self._rows):
            if r["is_header"]:
                continue
            ry = r["y"]
            if ry <= my < ry + self.ROW_H:
                return idx
        return None

    def _hit_camera_btn(self, mx: int, my: int) -> int | None:
        """Return the camera tab index (0–3) under the mouse, or ``None``."""
        btn_y = 44
        btn_h = 24
        if not (btn_y <= my < btn_y + btn_h):
            return None
        total_gap = (4 - 1) * 6
        btn_w = (self.WIDTH - self.MARGIN_L * 2 - total_gap) // 4
        for i in range(4):
            bx = self.MARGIN_L + i * (btn_w + 6)
            if bx <= mx < bx + btn_w:
                return i
        return None

    # -- mouse callback --------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        """OpenCV mouse callback: handles slider drags, +/- buttons, and
        camera tab clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Camera selector buttons?
            cam = self._hit_camera_btn(x, y)
            if cam is not None and cam != self.selected_camera:
                self.selected_camera = cam
                self.render()
                if self._dirty_callback:
                    self._dirty_callback()
                return

            row = self._hit_row(y)
            if row is not None:
                p = self._rows[row]["param"]
                sx = self._slider_left()
                # +/- button zones
                if sx - self.BTN_W <= x < sx:
                    self._set(p, max(p.vmin, self._get(p) - p.step))
                elif sx + self.SLIDER_W <= x < sx + self.SLIDER_W + self.BTN_W:
                    self._set(p, min(p.vmax, self._get(p) + p.step))
                else:
                    self._set(p, self._x_to_val(p, x))
                    self._dragging = row
                self.render()

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._dragging is not None:
                p = self._rows[self._dragging]["param"]
                self._set(p, self._x_to_val(p, x))
                self.render()

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = None

    # -- render ----------------------------------------------------------

    def render(self) -> None:
        W = self.WIDTH
        canvas = np.full((self._canvas_h, W, 3), 32, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 8
        self._rows.clear()

        # -- title bar ---------------------------------------------------
        cv2.putText(canvas, f"Config - {CAMERA_ORDER[self.selected_camera]}",
                    (self.MARGIN_L, y + 22), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

        # -- camera selector buttons -------------------------------------
        btn_y = y + 6
        total_gap = (4 - 1) * 6
        btn_w = (W - self.MARGIN_L * 2 - total_gap) // 4
        btn_h = 24
        for i, ck in enumerate(CAMERA_ORDER):
            bx = self.MARGIN_L + i * (btn_w + 6)
            fill = (70, 150, 70) if i == self.selected_camera else (55, 55, 55)
            border = (100, 200, 100) if i == self.selected_camera else (
                75, 75, 75)
            cv2.rectangle(canvas, (bx, btn_y),
                          (bx + btn_w, btn_y + btn_h), fill, -1)
            cv2.rectangle(canvas, (bx, btn_y),
                          (bx + btn_w, btn_y + btn_h), border, 1)
            short = ck.replace("_", " ").title()[:12]
            cv2.putText(canvas, short, (bx + 6, btn_y + 17), font, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
        y = btn_y + btn_h + 14

        # -- parameter sections ------------------------------------------
        current_group: str | None = None
        for pdef in CONFIG_PARAMS:
            # Section header
            if pdef.group != current_group:
                current_group = pdef.group
                cv2.rectangle(canvas, (6, y), (W - 6, y +
                              self.HEADER_H), (42, 42, 48), -1)
                cv2.putText(canvas, f"-- {pdef.group} --",
                            (self.MARGIN_L + 4, y + 23), font, 0.55, (160, 190, 240), 2, cv2.LINE_AA)
                self._rows.append(
                    {"y": y, "is_header": True, "group": pdef.group})
                y += self.HEADER_H + 2

            row_y = y
            self._rows.append({"y": row_y, "is_header": False, "param": pdef})

            # alternating row background
            bg = (40, 40, 40) if len(self._rows) % 2 == 0 else (35, 35, 35)
            cv2.rectangle(canvas, (6, row_y),
                          (W - 6, row_y + self.ROW_H), bg, -1)

            # label
            cv2.putText(canvas, pdef.label, (self.MARGIN_L, row_y + 19),
                        font, 0.52, (215, 215, 215), 1, cv2.LINE_AA)

            sx = self._slider_left()
            sy = row_y + self.SLIDER_Y_OFF

            # slider track
            cv2.rectangle(canvas, (sx, sy), (sx + self.SLIDER_W, sy + self.SLIDER_H),
                          (65, 65, 65), -1)
            cv2.rectangle(canvas, (sx, sy), (sx + self.SLIDER_W, sy + self.SLIDER_H),
                          (48, 48, 48), 1)

            # filled portion
            frac = self._to_frac(pdef)
            fill_w = int(frac * self.SLIDER_W)
            if fill_w > 0:
                c = (55, 135, 210) if pdef.kind != "bool" else (55, 175, 95)
                cv2.rectangle(canvas, (sx, sy),
                              (sx + fill_w, sy + self.SLIDER_H), c, -1)

            # thumb
            tx = sx + fill_w - 2
            cv2.rectangle(canvas, (max(sx - 1, tx), sy - 1),
                          (min(sx + self.SLIDER_W + 1, tx + 5),
                           sy + self.SLIDER_H + 1),
                          (240, 240, 240), 2)

            # +/- buttons
            cv2.rectangle(canvas, (sx - self.BTN_W, sy),
                          (sx, sy + self.SLIDER_H), (52, 52, 52), -1)
            cv2.putText(canvas, "-", (sx - self.BTN_W + 4, sy + 13),
                        font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (sx + self.SLIDER_W, sy),
                          (sx + self.SLIDER_W + self.BTN_W, sy + self.SLIDER_H), (52, 52, 52), -1)
            cv2.putText(canvas, "+", (sx + self.SLIDER_W + 4, sy + 13),
                        font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            # value display
            val = self._get(pdef)
            vx = sx + self.SLIDER_W + self.BTN_W + 12
            if pdef.kind == "bool":
                txt = "ON" if val >= 0.5 else "OFF"
                vc = (110, 240, 110) if val >= 0.5 else (240, 110, 110)
            elif pdef.kind == "float":
                txt = pdef.fmt.format(val)
                vc = (195, 225, 255)
            else:
                txt = pdef.fmt.format(int(val))
                vc = (195, 225, 255)
            cv2.putText(canvas, txt, (vx, row_y + 19),
                        font, 0.50, vc, 1, cv2.LINE_AA)

            y += self.ROW_H

        # -- footer ------------------------------------------------------
        footer_y = y + 8
        needed_h = max(footer_y + 40, 800)
        if canvas.shape[0] < needed_h:
            pad = np.full(
                (needed_h - canvas.shape[0], W, 3), 32, dtype=np.uint8)
            canvas = np.vstack((canvas, pad))
        self._canvas_h = canvas.shape[0]
        cv2.rectangle(canvas, (0, footer_y),
                      (W, self._canvas_h), (22, 22, 28), -1)
        cv2.putText(canvas, "Drag sliders  |  +/- fine-tune  |  Click camera tabs  |  P=toggle  R=reset",
                    (self.MARGIN_L, footer_y + 28), font, 0.44, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)

    # -- public API ------------------------------------------------------

    def open(self) -> None:
        """Create (or re-create) the config-panel OpenCV window and render it."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.WIDTH, self._canvas_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self.render()

    def close(self) -> None:
        """Destroy the config-panel window (safe to call even if already closed)."""
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def is_visible(self) -> bool:
        """Return ``True`` if the config-panel window currently exists and is visible."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1.0
        except cv2.error:
            return False


@dataclass(frozen=True)
class FrameRecord:
    """Immutable reference to one matched color+depth frame pair.

    Each frame is identified by its integer ``frame_index`` and assigned a
    ``base_time`` derived from ``(frame_index - 1) / FPS``.  The record
    stores the on-disk paths to both the color and depth PNG files.

    Attributes:
        camera_key: Logical camera name (e.g. ``"front_left"``).
        frame_index: Integer frame number parsed from the filename.
        color_path: Absolute path to the color PNG.
        depth_path: Absolute path to the depth PNG (16-bit).
        base_time: Nominal capture time in seconds, computed from the frame
            index under the constant-FPS assumption.
    """
    camera_key: str
    frame_index: int
    color_path: Path
    depth_path: Path
    base_time: float


CAMERA_ORDER = ["front_left", "front_right", "rear_left", "rear_right"]
CAMERA_GRID = [
    ["front_left", "front_right"],
    ["rear_left", "rear_right"],
]

# Per-camera YAML config files (next to this script). These are the source of
# truth for the processing pipeline; fields not present in the YAMLs (folder,
# offset_seconds, confidence tolerances, info panel) keep the CameraConfig
# defaults below.
CAMERA_CONFIG_FILES: dict[str, str] = {
    "front_left": "front_left_depth_processing.yaml",
    "front_right": "front_right_depth_processing.yaml",
    "rear_left": "rear_left_depth_processing.yaml",
    "rear_right": "rear_right_depth_processing.yaml",
}

# Recording sub-directory per camera (not stored in the YAML files).
CAMERA_FOLDERS: dict[str, str] = {
    "front_left": "depth_front_left",
    "front_right": "depth_front_right",
    "rear_left": "depth_rear_left",
    "rear_right": "depth_rear_right",
}

# Maps YAML ``processing_config`` section/key pairs to CameraConfig attributes.
_YAML_TO_CONFIG: dict[tuple[str, str], str] = {
    ("camera", "depth_scale"): "depth_scale",
    ("processing", "depth_alpha"): "depth_alpha",
    ("depth_cutoff", "enabled"): "depth_cutoff_enabled",
    ("depth_cutoff", "min_depth_m"): "depth_cutoff_min_m",
    ("depth_cutoff", "max_depth_m"): "depth_cutoff_max_m",
    ("edge_detection", "canny_min_val"): "canny_min_val",
    ("edge_detection", "canny_max_val"): "canny_max_val",
    ("edge_detection", "dilation_size"): "dilate_size",
    ("edge_detection", "min_line_length"): "min_line_length",
    ("edge_detection", "max_line_gap"): "max_line_gap",
    ("edge_detection", "hough_threshold"): "hough_threshold",
    ("edge_detection", "right"): "find_rightmost_line",
    ("median_line", "enabled"): "median_line_enabled",
    ("median_line", "window_size"): "median_line_window_size",
    ("median_line", "min_detections"): "median_line_min_detections",
    ("reference_line", "offset_x"): "ref_offset_x",
    ("reference_line", "angle_deg"): "ref_angle_deg",
    ("roi", "enabled"): "roi_enabled",
    ("roi", "x"): "roi_x",
    ("roi", "y"): "roi_y",
    ("roi", "width"): "roi_width",
    ("roi", "height"): "roi_height",
}

# YAML camera-section keys that feed the module-level geometry constants.
_YAML_GEOMETRY: dict[str, str] = {
    "img_width": "IMG_WIDTH",
    "img_height": "IMG_HEIGHT",
    "fps": "FPS",
    "hfov": "HFOV",
    "vfov": "VFOV",
}


def _sensor_label(raw_config: dict, camera_key: str) -> str:
    """Return the human-readable label from the YAML ``sensors`` list."""
    sensors = raw_config.get("sensors")
    if isinstance(sensors, list) and sensors and isinstance(sensors[0], dict):
        name = sensors[0].get("name")
        if name:
            return str(name)
    return camera_key.replace("_", " ").title()


def load_camera_config(
    camera_key: str,
    config_path: Path,
) -> tuple[CameraConfig, dict[str, object]]:
    """Load one camera's processing config from its YAML file.

    Values missing from the YAML keep the :class:`CameraConfig` defaults
    (``folder``, ``offset_seconds``, confidence tolerances, info panel).
    The camera-section geometry values are returned separately so the module
    constants can be kept in sync with the YAML.
    """
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path}: config root must be a mapping")

    processing = raw_config.get("processing_config") or {}
    if not isinstance(processing, dict):
        raise ValueError(f"{config_path}: processing_config must be a mapping")

    values: dict[str, object] = {}
    for (section, key), attr in _YAML_TO_CONFIG.items():
        section_data = processing.get(section)
        if isinstance(section_data, dict) and key in section_data:
            values[attr] = section_data[key]

    geometry: dict[str, object] = {}
    camera = processing.get("camera")
    if isinstance(camera, dict):
        for yaml_key, const_name in _YAML_GEOMETRY.items():
            if yaml_key in camera:
                geometry[const_name] = camera[yaml_key]

    config = CameraConfig(
        label=_sensor_label(raw_config, camera_key),
        folder=CAMERA_FOLDERS[camera_key],
        **values,
    )
    return config, geometry


def _apply_camera_geometry(geometry: dict[str, object]) -> None:
    """Sync the module-level geometry constants from the YAML camera section."""
    for const_name, value in geometry.items():
        globals()[const_name] = value
    if "IMG_WIDTH" in geometry and "HFOV" in geometry:
        globals()["THETA_HORIZONTAL"] = HFOV / IMG_WIDTH
    if "IMG_HEIGHT" in geometry and "VFOV" in geometry:
        globals()["THETA_VERTICAL"] = VFOV / IMG_HEIGHT


def load_camera_configs() -> dict[str, CameraConfig]:
    """Load all four camera configs from the YAML files next to this script."""
    config_dir = Path(__file__).resolve().parent
    configs: dict[str, CameraConfig] = {}
    geometry: dict[str, object] = {}
    for camera_key, filename in CAMERA_CONFIG_FILES.items():
        config_path = config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing camera config file: {config_path}")
        config, camera_geometry = load_camera_config(camera_key, config_path)
        configs[camera_key] = config
        geometry = camera_geometry or geometry
    _apply_camera_geometry(geometry)
    return configs


CAMERA_CONFIGS = load_camera_configs()


def frame_number(path: Path) -> int:
    """Extract the integer frame number from a filename like ``color_000042.png``.

    Args:
        path: Path whose filename is expected to contain an underscore
            followed by digits immediately before the ``.png`` extension.

    Returns:
        Parsed integer frame number.

    Raises:
        ValueError: If no ``_<digits>.png`` pattern is found in the filename.
    """
    match = re.search(r"_(\d+)\.png$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse frame number from {path.name}")
    return int(match.group(1))


@lru_cache(maxsize=768)
def load_color(path: str) -> np.ndarray:
    """Load a BGR color image from disk with LRU caching.

    Repeated requests for the same path return the cached array without
    re-reading from disk.  The cache is bounded at 768 entries to limit
    memory usage during long interactive sessions.

    Args:
        path: Filesystem path to a color PNG.

    Returns:
        BGR image as a numpy array (``height × width × 3``, ``uint8``).

    Raises:
        FileNotFoundError: If the file cannot be read by OpenCV.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


@lru_cache(maxsize=768)
def load_depth(path: str) -> np.ndarray:
    """Load a 16-bit depth image from disk with LRU caching.

    Same caching strategy as :func:`load_color`.

    Args:
        path: Filesystem path to a depth PNG.

    Returns:
        Depth image as a numpy array (``uint16``).

    Raises:
        FileNotFoundError: If the file cannot be read by OpenCV.
    """
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(path)
    return image


class RecordingAnalyzer:
    def __init__(self, recording_root: Path):
        """Create an analyzer for a four-camera recording directory.

        Args:
            recording_root: Top-level folder containing the per-camera
                sub-directories (``depth_front_left/``, etc.), each with
                ``color/`` and ``depth/`` sub-folders of timestamped PNGs.
        """
        self.recording_root = recording_root.expanduser().resolve()
        self.frame_index: dict[str, list[FrameRecord]] = {}
        self.frame_by_number: dict[str, dict[int, FrameRecord]] = {}
        self.position_by_frame: dict[str, dict[int, int]] = {}
        self.base_time_arrays: dict[str, np.ndarray] = {}
        self.current_line_cache: dict[tuple,
                                      tuple[int, int, int, int] | None] = {}

    def index_all_cameras(self) -> None:
        """Scan all four cameras and build the in-memory frame indices.

        Must be called once before any other method.  Prints a summary of
        frame counts and missing pairs per camera.
        """
        print(f"Recording root: {self.recording_root}")
        print(f"Nominal FPS: {FPS:g}")
        for camera_key in CAMERA_ORDER:
            records, missing_color, missing_depth = self.index_camera(
                camera_key, CAMERA_CONFIGS[camera_key]
            )
            self.frame_index[camera_key] = records
            self.frame_by_number[camera_key] = {
                record.frame_index: record for record in records
            }
            self.position_by_frame[camera_key] = {
                record.frame_index: pos for pos, record in enumerate(records)
            }
            self.base_time_arrays[camera_key] = np.array(
                [record.base_time for record in records], dtype=float
            )
            first_idx = records[0].frame_index
            last_idx = records[-1].frame_index
            print(
                f"{CAMERA_CONFIGS[camera_key].label:12s}: {len(records):4d} pairs "
                f"({first_idx}..{last_idx}), missing color={len(missing_color)}, "
                f"missing depth={len(missing_depth)}"
            )

    def index_camera(
        self,
        camera_key: str,
        config: CameraConfig,
    ) -> tuple[list[FrameRecord], list[int], list[int]]:
        """Index one camera's ``color/`` and ``depth/`` sub-directories.

        Pairs files by their frame number.  Only frames with both a color
        and a depth PNG are included in the returned record list.

        Args:
            camera_key: Logical camera name.
            config: Camera configuration (used only for the folder path).

        Returns:
            Tuple of ``(records, missing_color, missing_depth)`` where
            *records* is the list of paired :class:`FrameRecord` objects in
            frame-number order, and the two ``missing_*`` lists contain frame
            numbers that only exist in one of the two directories.
        """
        camera_root = self.recording_root / config.folder
        color_dir = camera_root / "color"
        depth_dir = camera_root / "depth"
        if not color_dir.exists() or not depth_dir.exists():
            raise FileNotFoundError(
                f"Missing color/depth folders below {camera_root}")

        color_paths = {
            frame_number(path): path for path in sorted(color_dir.glob("color_*.png"))
        }
        depth_paths = {
            frame_number(path): path for path in sorted(depth_dir.glob("depth_*.png"))
        }
        common_indices = sorted(set(color_paths) & set(depth_paths))
        if not common_indices:
            raise ValueError(
                f"No matching color/depth PNG pairs found below {camera_root}")

        records = [
            FrameRecord(
                camera_key=camera_key,
                frame_index=idx,
                color_path=color_paths[idx],
                depth_path=depth_paths[idx],
                base_time=(idx - 1) / FPS,
            )
            for idx in common_indices
        ]
        missing_color = sorted(set(depth_paths) - set(color_paths))
        missing_depth = sorted(set(color_paths) - set(depth_paths))
        return records, missing_color, missing_depth

    def config_offsets(self) -> dict[str, float]:
        """Return the per-camera time offsets as a ``{camera_key: seconds}`` dict."""
        return {
            camera_key: CAMERA_CONFIGS[camera_key].offset_seconds
            for camera_key in CAMERA_ORDER
        }

    def camera_time(self, record: FrameRecord, offsets: dict[str, float]) -> float:
        """Compute the synchronized time of *record* by applying its camera's offset."""
        return record.base_time + offsets.get(record.camera_key, 0.0)

    def camera_bounds(
        self,
        camera_key: str,
        offsets: dict[str, float],
    ) -> tuple[float, float]:
        """Return the (start, end) synchronized time range for one camera."""
        records = self.frame_index[camera_key]
        offset = offsets.get(camera_key, 0.0)
        return records[0].base_time + offset, records[-1].base_time + offset

    def common_time_bounds(
        self,
        offsets: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Return the time range where **all four** cameras have data.

        This is the intersection of each camera's synchronized time span.

        Raises:
            ValueError: If the offsets produce no overlap at all.
        """
        offsets = self.config_offsets() if offsets is None else offsets
        starts, ends = zip(
            *(self.camera_bounds(camera_key, offsets) for camera_key in CAMERA_ORDER)
        )
        start = max(starts)
        end = min(ends)
        if end < start:
            raise ValueError("Camera offsets leave no overlapping time range")
        return start, end

    def master_frame_count(self, offsets: dict[str, float] | None = None) -> int:
        """Number of master timeline frames covering the common overlap.

        One master frame per ``1/FPS`` second.
        """
        start, end = self.common_time_bounds(offsets)
        return max(1, int(np.floor((end - start) * FPS)) + 1)

    def master_time_for_frame(
        self,
        master_frame: int,
        offsets: dict[str, float] | None = None,
    ) -> float:
        """Convert a master frame index to an absolute synchronized time."""
        start, _ = self.common_time_bounds(offsets)
        return start + master_frame / FPS

    def nearest_record(
        self,
        camera_key: str,
        target_time: float,
        offsets: dict[str, float],
    ) -> tuple[FrameRecord, float]:
        """Find the :class:`FrameRecord` whose synchronized time is closest to
        *target_time*.

        Searches among up to three candidate indices (the insertion point ± 1)
        to handle the case where the target falls between frames.

        Returns:
            Tuple of ``(record, delta_ms)`` where *delta_ms* is the signed
            difference ``record_time - target_time`` in milliseconds.
        """
        times = self.base_time_arrays[camera_key] + \
            offsets.get(camera_key, 0.0)
        pos = int(np.searchsorted(times, target_time))
        candidates = [min(max(pos, 0), len(times) - 1)]
        if pos > 0:
            candidates.append(pos - 1)
        if pos + 1 < len(times):
            candidates.append(pos + 1)
        best_pos = min(set(candidates), key=lambda idx: abs(
            times[idx] - target_time))
        record = self.frame_index[camera_key][best_pos]
        delta_ms = (times[best_pos] - target_time) * 1000.0
        return record, delta_ms

    def print_sync_summary(self) -> None:
        """Print the common overlap range and a few sample sync points to stdout."""
        offsets = self.config_offsets()
        overlap_start, overlap_end = self.common_time_bounds(offsets)
        master_count = self.master_frame_count(offsets)
        print(
            f"Common overlap: {overlap_start:.3f}s .. {overlap_end:.3f}s "
            f"({master_count} master frames)"
        )
        for probe in [0, master_count // 2, master_count - 1]:
            t = self.master_time_for_frame(probe, offsets)
            summary = []
            for camera_key in CAMERA_ORDER:
                record, delta_ms = self.nearest_record(camera_key, t, offsets)
                summary.append(
                    f"{camera_key}=#{record.frame_index:06d} ({delta_ms:+.1f} ms)")
            print(f"master {probe:04d} @ {t:.3f}s: " + ", ".join(summary))

    def pipeline_signature(self, config: CameraConfig) -> tuple:
        """Return a hashable tuple of all pipeline-affecting config values.

        Used as part of the :meth:`detect_line` cache key so that changing
        any parameter (Canny thresholds, ROI, etc.) invalidates cached
        detections.
        """
        return (
            config.depth_scale,
            config.roi_enabled,
            config.roi_x,
            config.roi_y,
            config.roi_width,
            config.roi_height,
            config.depth_alpha,
            config.depth_cutoff_enabled,
            config.depth_cutoff_min_m,
            config.depth_cutoff_max_m,
            config.canny_min_val,
            config.canny_max_val,
            config.dilate_size,
            config.min_line_length,
            config.max_line_gap,
            config.hough_threshold,
            config.find_rightmost_line,
            config.ref_offset_x,
            config.ref_angle_deg,
        )

    def crop_depth(
        self,
        depth_image: np.ndarray,
        config: CameraConfig,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Optionally extract the ROI from *depth_image*.

        Returns:
            Tuple of ``(cropped_image, roi)`` where *roi* is
            ``(x, y, width, height)`` if cropping was applied, or ``None``
            if :attr:`CameraConfig.roi_enabled` is ``False`` (in which case
            a full copy of the input is returned).
        """
        if not config.roi_enabled:
            return depth_image.copy(), None
        height, width = depth_image.shape[:2]
        x0 = max(0, config.roi_x)
        y0 = max(0, config.roi_y)
        x1 = min(width, config.roi_x + config.roi_width)
        y1 = min(height, config.roi_y + config.roi_height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"Invalid ROI for {config.label}: "
                f"{(config.roi_x, config.roi_y, config.roi_width, config.roi_height)}"
            )
        return depth_image[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)

    def build_edge_inputs(self, depth_image: np.ndarray, config: CameraConfig):
        """Run the full pre-processing chain on a raw depth image.

        Steps: ROI crop → depth cutoff → colormap → Canny edges → zero-boundary
        filtering.

        Returns:
            Tuple of ``(depth_work, applied_roi, edge_depth, depth_colormap,
            filtered_edges)`` where:

            * *depth_work* — cropped (or full) depth array.
            * *applied_roi* — ROI as ``(x, y, w, h)`` or ``None``.
            * *edge_depth* — depth array after cutoff (used for pixel score).
            * *depth_colormap* — JET-coloured depth visualization (BGR).
            * *filtered_edges* — binary edge image ready for Hough.
        """
        depth_work, applied_roi = self.crop_depth(depth_image, config)
        edge_depth = apply_depth_cutoff(
            depth_work,
            depth_scale=config.depth_scale,
            min_depth_m=config.depth_cutoff_min_m,
            max_depth_m=config.depth_cutoff_max_m,
            enabled=config.depth_cutoff_enabled,
        )
        depth_colormap = process_depth_image(
            edge_depth, depth_alpha=config.depth_alpha)
        # Production edge input for comparison:
        depth_grayscale = cv2.convertScaleAbs(
            edge_depth, alpha=config.depth_alpha)
        canny_input = depth_grayscale
        canny_input = depth_colormap
        canny_edges = apply_canny_edge_detection(
            canny_input,
            min_val=config.canny_min_val,
            max_val=config.canny_max_val,
        )
        filtered_edges = filter_out_zero_boundaries(
            canny_edges, edge_depth, dilate_size=config.dilate_size
        )
        return depth_work, applied_roi, edge_depth, depth_colormap, filtered_edges

    def detect_line(
        self,
        camera_key: str,
        frame_index: int,
    ) -> tuple[int, int, int, int] | None:
        """Run the full detection pipeline for a single frame and return the
        selected Hough line.

        Results are cached keyed on ``(camera, frame, pipeline_signature)``
        so repeated requests (e.g. for the median window) are cheap.

        Returns:
            Detected line as ``(x1, y1, x2, y2)`` in ROI coordinates, or
            ``None`` if no line was found.
        """
        config = CAMERA_CONFIGS[camera_key]
        cache_key = (camera_key, frame_index, self.pipeline_signature(config))
        if cache_key in self.current_line_cache:
            return self.current_line_cache[cache_key]

        record = self.frame_by_number[camera_key][frame_index]
        depth_image = load_depth(str(record.depth_path))
        _, _, _, _, filtered_edges = self.build_edge_inputs(
            depth_image, config)
        line = find_longest_line_right(
            filtered_edges,
            min_line_length=config.min_line_length,
            max_line_gap=config.max_line_gap,
            threshold=config.hough_threshold,
            right=config.find_rightmost_line,
        )
        if line is not None:
            line = tuple(int(value) for value in line)
        self.current_line_cache[cache_key] = line
        return line

    def median_line_for_frame(
        self,
        camera_key: str,
        record: FrameRecord,
        image_width: int,
        image_height: int,
    ):
        """Compute the median reference line from the detection history
        preceding the given frame.

        The current frame is intentionally excluded so the temporal
        confidence score compares against an independent reference.

        Returns:
            Tuple of ``(median_line, status, valid_count)`` where
            *median_line* is ``(x1, y1, x2, y2)`` or ``None``, *status* is a
            human-readable label (``"disabled"``, ``"warming up …"``,
            ``"no majority …"``, ``"ready …"``, ``"unavailable"``), and
            *valid_count* is the number of valid detections in the window.
        """
        config = CAMERA_CONFIGS[camera_key]
        if not config.median_line_enabled:
            return None, "disabled", 0
        position = self.position_by_frame[camera_key][record.frame_index]
        start = max(0, position - config.median_line_window_size)
        # Exclude the current frame so the temporal confidence is measured
        # against an independent reference built from previous detections.
        history_records = self.frame_index[camera_key][start:position]
        line_history = [self.detect_line(
            camera_key, item.frame_index) for item in history_records]
        valid_count = sum(line is not None for line in line_history)
        if len(line_history) < config.median_line_window_size:
            return None, f"warming up {len(line_history)}/{config.median_line_window_size}", valid_count
        if valid_count < config.median_line_min_detections:
            return None, f"no majority {valid_count}/{config.median_line_window_size}", valid_count
        median_line = calculate_median_line(
            line_history,
            image_width=image_width,
            image_height=image_height,
            window_size=config.median_line_window_size,
            min_detections=config.median_line_min_detections,
        )
        status = f"ready {valid_count}/{config.median_line_window_size}" if median_line is not None else "unavailable"
        return median_line, status, valid_count

    def render_camera_row(
        self,
        camera_key: str,
        record: FrameRecord,
        target_time: float,
        delta_ms: float,
        offsets: dict[str, float],
    ) -> np.ndarray:
        """Render the full visualization tile for one camera.

        Produces a vertical stack of:

        1. A side-by-side pair: color frame (with ROI rectangle) | processed
           depth tile (with detected, median, and reference lines drawn).
        2. A three-column info panel showing status, config, and detection
           metrics (including the pixel and temporal confidence scores).

        Returns:
            A BGR image suitable for arrangement in the overview grid.
        """
        config = CAMERA_CONFIGS[camera_key]
        color_image = load_color(str(record.color_path)).copy()
        depth_image = load_depth(str(record.depth_path))
        depth_work, applied_roi, edge_depth, depth_colormap, _ = self.build_edge_inputs(
            depth_image, config
        )

        if applied_roi is not None:
            x, y, width, height = applied_roi
            cv2.rectangle(color_image, (x, y),
                          (x + width, y + height), (0, 255, 0), 2)

        camera_t = self.camera_time(record, offsets)
        current_line = self.detect_line(camera_key, record.frame_index)
        median_line, median_status, _ = self.median_line_for_frame(
            camera_key,
            record,
            image_width=depth_work.shape[1],
            image_height=depth_work.shape[0],
        )

        result_img = draw_reference_line(
            depth_colormap.copy(), config.ref_offset_x, config.ref_angle_deg
        )
        current_color = (255, 100, 100)
        median_color = (0, 0, 0)
        current_text_color = (80, 220, 255)
        median_text_color = (210, 255, 210)
        dim_color = (205, 205, 205)

        angle_deviation = horizontal_deviation = detected_depth_at_center_y = None
        horizontal_deviation_mm = None
        median_angle_deviation = median_horizontal_deviation = median_depth_at_center_y = None
        median_horizontal_deviation_mm = None

        if current_line is not None:
            result_img = draw_long_line(result_img, *current_line)
            angle_deviation, horizontal_deviation, detected_depth_at_center_y = (
                calculate_line_deviation(
                    current_line,
                    depth_work,
                    config.ref_offset_x,
                    config.ref_angle_deg,
                )
            )
            if horizontal_deviation is None:
                horizontal_deviation = -1.0
            if angle_deviation is not None:
                horizontal_deviation_mm = -1.0
                if detected_depth_at_center_y is not None:
                    pixel_width, _, _ = calculate_pixel_area(
                        depth_in_mm=detected_depth_at_center_y,
                        theta_horizontal=THETA_HORIZONTAL,
                        theta_vertical=THETA_VERTICAL,
                    )
                    horizontal_deviation_mm = horizontal_deviation * pixel_width

        if median_line is not None:
            result_img = draw_long_line(
                result_img, *median_line, color=median_color, thickness=2
            )
            median_angle_deviation, median_horizontal_deviation, median_depth_at_center_y = (
                calculate_line_deviation(
                    median_line,
                    depth_work,
                    config.ref_offset_x,
                    config.ref_angle_deg,
                )
            )
            if median_horizontal_deviation is None:
                median_horizontal_deviation = -1.0
            if median_angle_deviation is not None:
                median_horizontal_deviation_mm = -1.0
                if median_depth_at_center_y is not None:
                    median_pixel_width, _, _ = calculate_pixel_area(
                        depth_in_mm=median_depth_at_center_y,
                        theta_horizontal=THETA_HORIZONTAL,
                        theta_vertical=THETA_VERTICAL,
                    )
                    median_horizontal_deviation_mm = (
                        median_horizontal_deviation * median_pixel_width
                    )

        h, w = depth_work.shape[:2]
        center_x, center_y = w // 2, h // 2
        center_depth = int(depth_work[center_y, center_x])
        pixel_score, valid_pixels, total_pixels = calculate_pixel_validity_score(
            edge_depth)
        temporal_score, center_x_diff_px, ang_diff_deg = calculate_temporal_line_score(
            current_line=current_line,
            median_line=median_line,
            image_height=depth_work.shape[0],
            max_tolerated_center_shift_px=config.confidence_max_center_shift_px,
            max_tolerated_angle_deviation_deg=config.confidence_max_angle_dev,
        )
        cv2.circle(result_img, (center_x, center_y), radius=3,
                   color=current_color, thickness=-1)
        _, _, center_pixel_area = calculate_pixel_area(
            depth_in_mm=center_depth,
            theta_horizontal=THETA_HORIZONTAL,
            theta_vertical=THETA_VERTICAL,
        )

        roi_str = (
            f"on ({config.roi_x},{config.roi_y}) {config.roi_width}x{config.roi_height}"
            if config.roi_enabled
            else "off"
        )
        line_side = "rightmost" if config.find_rightmost_line else "leftmost"
        current_mm = f"{horizontal_deviation_mm:.1f} mm" if horizontal_deviation_mm is not None else "N/A"
        median_mm = f"{median_horizontal_deviation_mm:.1f} mm" if median_horizontal_deviation_mm is not None else "N/A"

        status_lines = [
            (f"{config.label} frame {record.frame_index:06d}", (255, 255, 255)),
            (f"Target {target_time:.3f}s   Camera {camera_t:.3f}s", dim_color),
            (f"Delta {delta_ms:+.1f} ms   Offset {offsets.get(camera_key, 0.0):+.3f}s", dim_color),
            (f"Center depth {center_depth} mm", current_text_color),
            (f"Pixel area {center_pixel_area:.2f} mm^2", current_text_color),
            (
                f"Pixel score    {pixel_score:5.1f}%  ({valid_pixels}/{total_pixels} valid)",
                current_text_color,
            ),
            (
                f"Temporal score {temporal_score:5.1f}%  "
                + (
                    f"(Δcx={center_x_diff_px:.1f}px Δang={ang_diff_deg:.2f}deg)"
                    if center_x_diff_px is not None else
                    "(no current/median line)"
                ),
                current_text_color,
            ),
        ]

        config_lines = [
            ("Config", (255, 255, 255)),
            (f"ROI {roi_str}", dim_color),
            (f"Cutoff {'on' if config.depth_cutoff_enabled else 'off'} {config.depth_cutoff_min_m:.2f}-{config.depth_cutoff_max_m:.2f} m", dim_color),
            (f"Canny {config.canny_min_val}/{config.canny_max_val}   filter {config.dilate_size}", dim_color),
            (f"Hough len {config.min_line_length} gap {config.max_line_gap} thr {config.hough_threshold}", dim_color),
            (f"Line {line_side}   Ref {config.ref_offset_x}px {config.ref_angle_deg:g}deg", dim_color),
            (f"Median {'on' if config.median_line_enabled else 'off'} win {config.median_line_window_size} min {config.median_line_min_detections}", dim_color),
        ]

        detection_lines = [("Detected line", current_text_color)]
        if current_line is not None and angle_deviation is not None:
            detection_lines += [
                (f"Current angle {angle_deviation:.2f} deg",
                 current_text_color),
                (f"Current horiz {horizontal_deviation:.1f} px",
                 current_text_color),
                (f"Current horiz {current_mm}", current_text_color),
            ]
        else:
            detection_lines.append(
                ("Current: no detection", current_text_color))

        if config.median_line_enabled:
            if median_line is not None and median_angle_deviation is not None:
                detection_lines += [
                    (f"Median angle {median_angle_deviation:.2f} deg",
                     median_text_color),
                    (f"Median horiz {median_horizontal_deviation:.1f} px",
                     median_text_color),
                    (f"Median horiz {median_mm}", median_text_color),
                ]
            else:
                detection_lines.append(
                    (f"Median: {median_status}", median_text_color))

        depth_tile = result_img
        if applied_roi is not None and config.display_depth_roi_in_full_frame:
            depth_tile = place_roi_in_full_frame(
                result_img, depth_image.shape[:2], applied_roi)
        depth_tile = add_label(
            depth_tile,
            [
                f"{config.label} processed depth",
                "ROI shown in full-frame position"
                if applied_roi is not None and config.display_depth_roi_in_full_frame
                else "Full processed depth frame",
            ],
        )
        image_pair = hstack_padded([color_image, depth_tile], gap=10)
        info_panel = make_info_panel_columns(
            [status_lines, config_lines, detection_lines],
            width=max(image_pair.shape[1], config.info_panel_width),
            min_height=270,
        )
        return vstack_padded([image_pair, info_panel], gap=0)

    def render_overview_frame(
        self,
        target_time: float,
        offsets: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Render the full four-camera grid for a single master time point.

        Arranges the four camera rows in a 2×2 grid (front row above,
        rear row below) with a header bar showing the target time and
        overlap range.

        Returns:
            A single BGR image combining all four cameras.
        """
        offsets = self.config_offsets() if offsets is None else offsets
        grid_rows = []
        for row_keys in CAMERA_GRID:
            row_panels = []
            for camera_key in row_keys:
                record, delta_ms = self.nearest_record(
                    camera_key, target_time, offsets)
                row_panels.append(
                    self.render_camera_row(
                        camera_key, record, target_time, delta_ms, offsets)
                )
            grid_rows.append(hstack_padded(row_panels, gap=12))
        body = vstack_padded(grid_rows, gap=12)
        start, end = self.common_time_bounds(offsets)
        header = np.zeros((70, body.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            "Four-camera recording",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"target={target_time:.3f}s  overlap={start:.3f}s..{end:.3f}s  fps={FPS:g}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        return vstack_padded([header, body], gap=0)

    def run_opencv_keyboard_viewer(
        self,
        start_frame: int = 0,
        display_scale: float = 0.58,
        resize_window_on_frame_change: bool = False,
        window_name: str = "Four-camera recording viewer",
    ) -> None:
        """Launch the interactive OpenCV viewer (blocking main loop).

        Keyboard controls:

        * ``A`` / ``D`` or Left / Right — step one master frame.
        * ``W`` / ``S`` or Up / Down — step ±10 frames.
        * ``Space`` — advance one frame.
        * ``Home`` / ``End`` — jump to first / last frame.
        * Digits + ``Enter`` — jump to a specific frame number.
        * ``Backspace`` — edit the typed frame number.
        * ``P`` — toggle the config panel.
        * ``R`` — reset all config sliders to their original values.
        * ``Q`` / ``Esc`` — quit.

        Args:
            start_frame: Initial master frame index.
            display_scale: Scale factor applied to the OpenCV window
                (1.0 = full rendered resolution).
            resize_window_on_frame_change: When True, re-apply the scale on
                every frame change (can be jarring; default is once).
            window_name: Title of the main OpenCV window.
        """
        offsets = self.config_offsets()
        max_frame = self.master_frame_count(offsets) - 1
        state = {
            "frame": int(np.clip(start_frame, 0, max_frame)),
            "dirty": True,
            "typed": "",
        }

        # Snapshot original config values so the user can reset with 'r'.
        import dataclasses as _dc
        _original_configs: dict[str, dict[str, object]] = {
            ck: {f.name: getattr(cfg, f.name)
                 for f in _dc.fields(CameraConfig)}
            for ck, cfg in CAMERA_CONFIGS.items()
        }

        def clamp_frame(value: int) -> int:
            return int(np.clip(value, 0, max_frame))

        def set_frame(value: int, update_trackbar: bool = True) -> None:
            value = clamp_frame(value)
            if value != state["frame"]:
                state["frame"] = value
                state["dirty"] = True
            if update_trackbar:
                cv2.setTrackbarPos("master_frame", window_name, value)

        def on_trackbar(value: int) -> None:
            state["frame"] = clamp_frame(value)
            state["dirty"] = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("master_frame", window_name,
                           state["frame"], max_frame, on_trackbar)

        # ---- Config panel ----
        config_panel = ConfigPanel(
            "Config Panel", dirty_callback=lambda: state.update(dirty=True))
        config_visible = True
        config_panel.open()
        # -----------------------

        last_image = None
        window_size_initialized = False
        try:
            while True:
                # Recreate config window if it was closed externally.
                if config_visible and not config_panel.is_visible():
                    config_panel.open()

                if state["dirty"] or last_image is None:
                    target_time = self.master_time_for_frame(
                        state["frame"], offsets)
                    overview = self.render_overview_frame(target_time, offsets)
                    last_image = add_opencv_status_bar(
                        overview,
                        master_frame=state["frame"],
                        max_frame=max_frame,
                        target_time=target_time,
                        typed_index=state["typed"],
                    )
                    if (
                        display_scale is not None
                        and display_scale > 0
                        and (not window_size_initialized or resize_window_on_frame_change)
                    ):
                        cv2.resizeWindow(
                            window_name,
                            max(320, int(last_image.shape[1] * display_scale)),
                            max(240, int(last_image.shape[0] * display_scale)),
                        )
                        window_size_initialized = True
                    cv2.setWindowTitle(
                        window_name, f"{window_name} - frame {state['frame']}/{max_frame}"
                    )
                    cv2.imshow(window_name, last_image)
                    state["dirty"] = False

                key = cv2.waitKeyEx(30)
                if key == -1:
                    continue

                ascii_key = key & 0xFF
                if key in (27,) or ascii_key == ord("q"):
                    break
                if key in (81, 2424832, 65361) or ascii_key == ord("a"):
                    set_frame(state["frame"] - 1)
                elif key in (83, 2555904, 65363) or ascii_key in (ord("d"), ord(" ")):
                    set_frame(state["frame"] + 1)
                elif key in (82, 2490368, 65362) or ascii_key == ord("w"):
                    set_frame(state["frame"] + 10)
                elif key in (84, 2621440, 65364) or ascii_key == ord("s"):
                    set_frame(state["frame"] - 10)
                elif key in (36, 2359296, 65360):
                    set_frame(0)
                elif key in (35, 2293760, 65367):
                    set_frame(max_frame)
                elif ord("0") <= ascii_key <= ord("9"):
                    state["typed"] += chr(ascii_key)
                    state["dirty"] = True
                elif ascii_key in (8, 127):
                    state["typed"] = state["typed"][:-1]
                    state["dirty"] = True
                elif ascii_key in (10, 13):
                    if state["typed"]:
                        set_frame(int(state["typed"]))
                        state["typed"] = ""
                        state["dirty"] = True
                elif ascii_key == ord("c"):
                    state["typed"] = ""
                    state["dirty"] = True
                elif ascii_key == ord("p"):
                    # Toggle config panel.
                    config_visible = not config_visible
                    if config_visible:
                        config_panel.open()
                    else:
                        config_panel.close()
                elif ascii_key == ord("r"):
                    # Reset all configs to their original values.
                    for ck, cfg in CAMERA_CONFIGS.items():
                        saved = _original_configs[ck]
                        for fname, fval in saved.items():
                            setattr(cfg, fname, fval)
                    config_panel.render()
                    state["dirty"] = True
        finally:
            cv2.destroyWindow(window_name)
            config_panel.close()


def pad_to_height(
    image: np.ndarray,
    height: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Vertically pad *image* to *height* with solid *color*, or return it
    unchanged if already tall enough."""
    if image.shape[0] >= height:
        return image
    pad = np.full(
        (height - image.shape[0], image.shape[1], 3), color, dtype=np.uint8)
    return np.vstack((image, pad))


def pad_to_width(
    image: np.ndarray,
    width: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Horizontally pad *image* to *width* with solid *color*, or return it
    unchanged if already wide enough."""
    if image.shape[1] >= width:
        return image
    pad = np.full(
        (image.shape[0], width - image.shape[1], 3), color, dtype=np.uint8)
    return np.hstack((image, pad))


def hstack_padded(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """Horizontally stack images, padding shorter ones to equal height and
    inserting a black *gap* between them."""
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if padded and gap > 0:
            padded.append(np.zeros((height, gap, 3), dtype=np.uint8))
        padded.append(pad_to_height(image, height))
    return np.hstack(padded)


def vstack_padded(images: list[np.ndarray], gap: int = 10) -> np.ndarray:
    """Vertically stack images, padding narrower ones to equal width and
    inserting a black *gap* between them."""
    width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        if padded and gap > 0:
            padded.append(np.zeros((gap, width, 3), dtype=np.uint8))
        padded.append(pad_to_width(image, width))
    return np.vstack(padded)


def add_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Overlay a semi-transparent title bar with *lines* of text at the top
    of *image*.  Returns a new array (the original is not modified)."""
    labeled = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.78
    line_h = 32
    panel_h = 22 + line_h * len(lines)
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, labeled, 0.38, 0, dst=labeled)
    for i, text in enumerate(lines):
        cv2.putText(
            labeled,
            text,
            (12, 30 + i * line_h),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return labeled


def make_info_panel_columns(
    columns: list[list[tuple[str, tuple[int, int, int]]]],
    width: int,
    min_height: int = 240,
    font_scale: float = 0.72,
) -> np.ndarray:
    """Render a multi-column text panel.

    Each column is a list of ``(text, bgr_color)`` tuples; the first row of
    each column is rendered slightly larger as a section title.  Columns are
    separated by thin vertical lines.

    Returns:
        A BGR image of exactly *width* pixels wide.
    """
    margin_x = 16
    margin_y = 16
    col_gap = 20
    line_h = 34
    column_count = max(1, len(columns))
    usable_width = width - 2 * margin_x - col_gap * (column_count - 1)
    col_width = max(160, usable_width // column_count)
    max_lines = max((len(column) for column in columns), default=1)
    height = max(min_height, 2 * margin_y + max_lines * line_h + 6)
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for col_idx, column in enumerate(columns):
        x = margin_x + col_idx * (col_width + col_gap)
        if col_idx > 0:
            sep_x = x - col_gap // 2
            cv2.line(panel, (sep_x, margin_y), (sep_x, height -
                     margin_y), (58, 58, 58), 1, cv2.LINE_AA)

        for line_idx, (text, color) in enumerate(column):
            if not text:
                continue
            is_title = line_idx == 0
            scale = 0.82 if is_title else font_scale
            thickness = 2 if is_title else 1
            y = margin_y + 28 + line_idx * line_h
            cv2.putText(panel, text, (x, y), font, scale,
                        color, thickness, cv2.LINE_AA)
    return panel


def place_roi_in_full_frame(
    roi_image: np.ndarray,
    full_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Paste *roi_image* into a black canvas the size of *full_shape* at the
    position given by *roi* ``(x, y, w, h)``, drawing a green border around it."""
    full_h, full_w = full_shape[:2]
    x, y, width, height = roi
    canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    paste_w = min(width, roi_image.shape[1], full_w - x)
    paste_h = min(height, roi_image.shape[0], full_h - y)
    if paste_w > 0 and paste_h > 0:
        canvas[y:y + paste_h, x:x + paste_w] = roi_image[:paste_h, :paste_w]
        cv2.rectangle(canvas, (x, y), (x + paste_w,
                      y + paste_h), (0, 255, 0), 2)
    return canvas


def add_opencv_status_bar(
    image: np.ndarray,
    master_frame: int,
    max_frame: int,
    target_time: float,
    typed_index: str,
) -> np.ndarray:
    """Append a dark status bar below *image* showing the current frame
    position, target time, typed jump buffer, and keyboard shortcut hints."""
    bar_h = 86
    output = np.vstack(
        (image, np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)))
    typed = typed_index if typed_index else "-"
    cv2.putText(
        output,
        f"master_frame={master_frame}/{max_frame}  target={target_time:.3f}s  typed_jump={typed}",
        (14, image.shape[0] + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Keys: Left/Right or A/D step  |  Up/Down or W/S +/-10  |  digits+Enter jump  |  Backspace edit  |  Home/End  |  Q/Esc close",
        (14, image.shape[0] + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the recording analyzer."""
    parser = argparse.ArgumentParser(
        description="Open the four-camera recording analyzer.")
    parser.add_argument(
        "recording_root",
        type=Path,
        help="Folder containing depth_front_left, depth_front_right, depth_rear_left, depth_rear_right.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Master frame to show first.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.58,
        help="Initial OpenCV window scale. Use 1.0 for full rendered size.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: index the recording, print a sync summary, then launch
    the interactive OpenCV viewer."""
    args = parse_args()
    analyzer = RecordingAnalyzer(args.recording_root)
    analyzer.index_all_cameras()
    analyzer.print_sync_summary()
    analyzer.run_opencv_keyboard_viewer(
        start_frame=args.start_frame,
        display_scale=args.display_scale,
    )


if __name__ == "__main__":
    main()
