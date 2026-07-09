from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from helpers import (
    apply_canny_edge_detection,
    apply_depth_cutoff,
    calculate_line_deviation,
    calculate_median_line,
    calculate_pixel_area,
    draw_long_line,
    draw_reference_line,
    filter_out_zero_boundaries,
    find_longest_line_right,
    process_depth_image,
)

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when deps are missing.
    yaml = None


DEFAULT_DEPTH_SCALE = 0.0010000000474974513


@dataclass(frozen=True)
class SingleCameraConfig:
    img_width: int = 848
    img_height: int = 480
    fps: float = 15.0
    hfov: float = 87.0
    vfov: float = 58.0
    depth_scale: float = DEFAULT_DEPTH_SCALE

    roi_enabled: bool = True
    roi_x: int = 100
    roi_y: int = 100
    roi_width: int = 700
    roi_height: int = 120

    depth_alpha: float = 0.55
    depth_cutoff_enabled: bool = True
    depth_cutoff_min_m: float = 0.2
    depth_cutoff_max_m: float = 1.0

    canny_min_val: int = 130
    canny_max_val: int = 150
    dilate_size: int = 1
    min_line_length: int = 50
    max_line_gap: int = 20
    hough_threshold: int = 20

    find_rightmost_line: bool = False
    median_line_enabled: bool = True
    median_line_window_size: int = 15
    median_line_min_detections: int = 8

    ref_offset_x: int = 0
    ref_angle_deg: float = 90.0
    info_panel_width: int = 560


@dataclass(frozen=True)
class FrameResult:
    rendered_image: np.ndarray
    current_line: tuple[int, int, int, int] | None


def load_config(config_path: Path) -> SingleCameraConfig:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read the config file. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    processing_config = (
        raw_config.get("depth_camera", {}).get("processing_config", {})
        if isinstance(raw_config, dict)
        else {}
    )
    camera = _as_dict(processing_config.get("camera"))
    processing = _as_dict(processing_config.get("processing"))
    depth_cutoff = _as_dict(processing_config.get("depth_cutoff"))
    roi = _as_dict(processing_config.get("roi"))
    edge_detection = _as_dict(processing_config.get("edge_detection"))
    line_selection = _as_dict(processing_config.get("line_selection"))
    reference_line = _as_dict(processing_config.get("reference_line"))
    median_line = _as_dict(processing_config.get("median_line"))

    return SingleCameraConfig(
        img_width=_get_int(camera, "img_width", SingleCameraConfig.img_width),
        img_height=_get_int(camera, "img_height", SingleCameraConfig.img_height),
        fps=_get_float(camera, "fps", SingleCameraConfig.fps),
        hfov=_get_float(camera, "hfov", SingleCameraConfig.hfov),
        vfov=_get_float(camera, "vfov", SingleCameraConfig.vfov),
        depth_alpha=_get_float(
            processing, "depth_alpha", SingleCameraConfig.depth_alpha
        ),
        depth_cutoff_enabled=_get_bool(
            depth_cutoff, "enabled", SingleCameraConfig.depth_cutoff_enabled
        ),
        depth_cutoff_min_m=_get_float(
            depth_cutoff, "min_m", SingleCameraConfig.depth_cutoff_min_m
        ),
        depth_cutoff_max_m=_get_float(
            depth_cutoff, "max_m", SingleCameraConfig.depth_cutoff_max_m
        ),
        roi_enabled=_get_bool(roi, "enabled", SingleCameraConfig.roi_enabled),
        roi_x=_get_int(roi, "x", SingleCameraConfig.roi_x),
        roi_y=_get_int(roi, "y", SingleCameraConfig.roi_y),
        roi_width=_get_int(roi, "width", SingleCameraConfig.roi_width),
        roi_height=_get_int(roi, "height", SingleCameraConfig.roi_height),
        canny_min_val=_get_int(
            edge_detection, "canny_min_val", SingleCameraConfig.canny_min_val
        ),
        canny_max_val=_get_int(
            edge_detection, "canny_max_val", SingleCameraConfig.canny_max_val
        ),
        dilate_size=max(
            1,
            _get_int(edge_detection, "dilation_size", SingleCameraConfig.dilate_size),
        ),
        min_line_length=_get_int(
            edge_detection, "min_line_length", SingleCameraConfig.min_line_length
        ),
        max_line_gap=_get_int(
            edge_detection, "max_line_gap", SingleCameraConfig.max_line_gap
        ),
        hough_threshold=_get_int(
            edge_detection, "hough_threshold", SingleCameraConfig.hough_threshold
        ),
        find_rightmost_line=_get_bool(
            line_selection,
            "find_rightmost_line",
            SingleCameraConfig.find_rightmost_line,
        ),
        median_line_enabled=_get_bool(
            median_line, "enabled", SingleCameraConfig.median_line_enabled
        ),
        median_line_window_size=max(
            1,
            _get_int(
                median_line,
                "window_size",
                SingleCameraConfig.median_line_window_size,
            ),
        ),
        median_line_min_detections=max(
            1,
            _get_int(
                median_line,
                "min_detections",
                SingleCameraConfig.median_line_min_detections,
            ),
        ),
        ref_offset_x=_get_int(
            reference_line, "offset_x", SingleCameraConfig.ref_offset_x
        ),
        ref_angle_deg=_get_float(
            reference_line, "angle_deg", SingleCameraConfig.ref_angle_deg
        ),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def find_depth_images(depth_image_dir: Path) -> list[Path]:
    images = sorted(depth_image_dir.glob("*.png"), key=natural_sort_key)
    if not images:
        raise FileNotFoundError(f"No PNG depth images found in {depth_image_dir}")
    return images


def load_depth_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel depth image, got {path}")
    return image


def crop_depth(
    depth_image: np.ndarray,
    config: SingleCameraConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if not config.roi_enabled:
        return depth_image.copy(), None

    height, width = depth_image.shape[:2]
    x0 = max(0, config.roi_x)
    y0 = max(0, config.roi_y)
    x1 = min(width, config.roi_x + config.roi_width)
    y1 = min(height, config.roi_y + config.roi_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Invalid ROI "
            f"{(config.roi_x, config.roi_y, config.roi_width, config.roi_height)} "
            f"for image shape {depth_image.shape[:2]}"
        )
    return depth_image[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def build_edge_inputs(
    depth_image: np.ndarray,
    config: SingleCameraConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray, np.ndarray]:
    depth_work, applied_roi = crop_depth(depth_image, config)
    edge_depth = apply_depth_cutoff(
        depth_work,
        depth_scale=config.depth_scale,
        min_depth_m=config.depth_cutoff_min_m,
        max_depth_m=config.depth_cutoff_max_m,
        enabled=config.depth_cutoff_enabled,
    )
    depth_colormap = process_depth_image(edge_depth, depth_alpha=config.depth_alpha)
    # Production edge input for comparison:
    # depth_grayscale = cv2.convertScaleAbs(edge_depth, alpha=config.depth_alpha)
    # canny_input = depth_grayscale
    canny_input = depth_colormap
    canny_edges = apply_canny_edge_detection(
        canny_input,
        min_val=config.canny_min_val,
        max_val=config.canny_max_val,
    )
    filtered_edges = filter_out_zero_boundaries(
        canny_edges, edge_depth, dilate_size=config.dilate_size
    )
    return depth_work, applied_roi, depth_colormap, filtered_edges


def detect_line(
    filtered_edges: np.ndarray,
    config: SingleCameraConfig,
) -> tuple[int, int, int, int] | None:
    line = find_longest_line_right(
        filtered_edges,
        min_line_length=config.min_line_length,
        max_line_gap=config.max_line_gap,
        threshold=config.hough_threshold,
        right=config.find_rightmost_line,
    )
    return tuple(int(value) for value in line) if line is not None else None


def median_line_for_history(
    line_history: list[tuple[int, int, int, int] | None],
    image_width: int,
    image_height: int,
    config: SingleCameraConfig,
) -> tuple[tuple[int, int, int, int] | None, str, int]:
    if not config.median_line_enabled:
        return None, "disabled", 0

    history_window = line_history[-config.median_line_window_size :]
    valid_count = sum(line is not None for line in history_window)
    if len(history_window) < config.median_line_window_size:
        return (
            None,
            f"warming up {len(history_window)}/{config.median_line_window_size}",
            valid_count,
        )
    if valid_count < config.median_line_min_detections:
        return (
            None,
            f"no majority {valid_count}/{config.median_line_window_size}",
            valid_count,
        )

    median_line = calculate_median_line(
        history_window,
        image_width=image_width,
        image_height=image_height,
        window_size=config.median_line_window_size,
        min_detections=config.median_line_min_detections,
    )
    status = (
        f"ready {valid_count}/{config.median_line_window_size}"
        if median_line is not None
        else "unavailable"
    )
    return median_line, status, valid_count


def render_frame(
    image_path: Path,
    depth_image: np.ndarray,
    frame_position: int,
    total_frames: int,
    line_history: list[tuple[int, int, int, int] | None],
    config: SingleCameraConfig,
) -> FrameResult:
    depth_work, applied_roi, depth_colormap, filtered_edges = build_edge_inputs(
        depth_image, config
    )
    current_line = detect_line(filtered_edges, config)
    line_history.append(current_line)
    median_line, median_status, _ = median_line_for_history(
        line_history,
        image_width=depth_work.shape[1],
        image_height=depth_work.shape[0],
        config=config,
    )

    result_img = draw_reference_line(
        depth_colormap.copy(), config.ref_offset_x, config.ref_angle_deg
    )
    current_color = (255, 100, 100)
    median_color = (0, 0, 0)
    current_text_color = (80, 220, 255)
    median_text_color = (210, 255, 210)
    dim_color = (205, 205, 205)

    current_metrics = calculate_detection_metrics(current_line, depth_work, config)
    median_metrics = calculate_detection_metrics(median_line, depth_work, config)

    if current_line is not None:
        result_img = draw_long_line(result_img, *current_line)
    if median_line is not None:
        result_img = draw_long_line(
            result_img, *median_line, color=median_color, thickness=2
        )

    height, width = depth_work.shape[:2]
    center_x, center_y = width // 2, height // 2
    center_depth = int(depth_work[center_y, center_x])
    cv2.circle(result_img, (center_x, center_y), 3, current_color, -1)
    _, _, center_pixel_area = calculate_pixel_area(
        depth_in_mm=center_depth,
        theta_horizontal=config.hfov / config.img_width,
        theta_vertical=config.vfov / config.img_height,
    )

    processed_tile = result_img
    if applied_roi is not None:
        processed_tile = place_roi_in_full_frame(
            result_img, depth_image.shape[:2], applied_roi
        )
    processed_tile = add_label(
        processed_tile,
        [
            f"{image_path.name}",
            "processed depth ROI" if applied_roi is not None else "processed depth",
        ],
    )

    info_panel = make_metadata_panel(
        image_path=image_path,
        frame_position=frame_position,
        total_frames=total_frames,
        center_depth=center_depth,
        center_pixel_area=center_pixel_area,
        current_line=current_line,
        current_metrics=current_metrics,
        median_line=median_line,
        median_status=median_status,
        median_metrics=median_metrics,
        config=config,
        height=processed_tile.shape[0],
        width=config.info_panel_width,
        colors={
            "title": (255, 255, 255),
            "dim": dim_color,
            "current": current_text_color,
            "median": median_text_color,
        },
    )
    return FrameResult(
        rendered_image=hstack_padded([processed_tile, info_panel], gap=10),
        current_line=current_line,
    )


def calculate_detection_metrics(
    line: tuple[int, int, int, int] | None,
    depth_work: np.ndarray,
    config: SingleCameraConfig,
) -> dict[str, float | None]:
    angle_deviation, horizontal_deviation, detected_depth = calculate_line_deviation(
        line,
        depth_work,
        config.ref_offset_x,
        config.ref_angle_deg,
    )
    horizontal_deviation_mm = None
    if horizontal_deviation is None and angle_deviation is not None:
        horizontal_deviation = -1.0
    if angle_deviation is not None:
        horizontal_deviation_mm = -1.0
        if detected_depth is not None:
            pixel_width, _, _ = calculate_pixel_area(
                depth_in_mm=detected_depth,
                theta_horizontal=config.hfov / config.img_width,
                theta_vertical=config.vfov / config.img_height,
            )
            horizontal_deviation_mm = horizontal_deviation * pixel_width

    return {
        "angle_deviation": angle_deviation,
        "horizontal_deviation": horizontal_deviation,
        "detected_depth": detected_depth,
        "horizontal_deviation_mm": horizontal_deviation_mm,
    }


def make_metadata_panel(
    image_path: Path,
    frame_position: int,
    total_frames: int,
    center_depth: int,
    center_pixel_area: float,
    current_line: tuple[int, int, int, int] | None,
    current_metrics: dict[str, float | None],
    median_line: tuple[int, int, int, int] | None,
    median_status: str,
    median_metrics: dict[str, float | None],
    config: SingleCameraConfig,
    height: int,
    width: int,
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    roi_str = (
        f"on ({config.roi_x},{config.roi_y}) {config.roi_width}x{config.roi_height}"
        if config.roi_enabled
        else "off"
    )
    line_side = "rightmost" if config.find_rightmost_line else "leftmost"
    current_lines = format_detection_lines(
        "Current",
        current_line,
        current_metrics,
        colors["current"],
    )
    if config.median_line_enabled:
        median_lines = format_detection_lines(
            "Median",
            median_line,
            median_metrics,
            colors["median"],
            fallback_status=median_status,
        )
    else:
        median_lines = [("Median: disabled", colors["median"])]

    panel_lines = [
        ("Single-camera depth pipeline", colors["title"], 0.72, 2),
        (f"Image: {image_path.name}", colors["dim"], 0.55, 1),
        (f"Frame: {frame_position + 1}/{total_frames}", colors["dim"], 0.55, 1),
        (f"Center depth: {center_depth} mm", colors["current"], 0.58, 1),
        (f"Pixel area: {center_pixel_area:.2f} mm^2", colors["current"], 0.58, 1),
        ("", colors["dim"], 0.5, 1),
        *[(line, color, 0.58, 1) for line, color in current_lines],
        *[(line, color, 0.58, 1) for line, color in median_lines],
        ("", colors["dim"], 0.5, 1),
        ("Config", colors["title"], 0.68, 2),
        (f"ROI: {roi_str}", colors["dim"], 0.53, 1),
        (
            "Cutoff: "
            f"{'on' if config.depth_cutoff_enabled else 'off'} "
            f"{config.depth_cutoff_min_m:.2f}-{config.depth_cutoff_max_m:.2f} m",
            colors["dim"],
            0.53,
            1,
        ),
        (
            f"Canny: {config.canny_min_val}/{config.canny_max_val}, "
            f"dilate {config.dilate_size}",
            colors["dim"],
            0.53,
            1,
        ),
        (
            f"Hough: len {config.min_line_length}, gap {config.max_line_gap}, "
            f"thr {config.hough_threshold}",
            colors["dim"],
            0.53,
            1,
        ),
        (
            f"Line: {line_side}, ref {config.ref_offset_x}px "
            f"{config.ref_angle_deg:g}deg",
            colors["dim"],
            0.53,
            1,
        ),
        (
            "Median: "
            f"{'on' if config.median_line_enabled else 'off'} "
            f"win {config.median_line_window_size}, "
            f"min {config.median_line_min_detections}",
            colors["dim"],
            0.53,
            1,
        ),
    ]
    return draw_text_panel(panel_lines, width=width, height=height)


def format_detection_lines(
    label: str,
    line: tuple[int, int, int, int] | None,
    metrics: dict[str, float | None],
    color: tuple[int, int, int],
    fallback_status: str | None = None,
) -> list[tuple[str, tuple[int, int, int]]]:
    if line is None or metrics["angle_deviation"] is None:
        status = fallback_status if fallback_status is not None else "no detection"
        return [(f"{label}: {status}", color)]

    horizontal_mm = metrics["horizontal_deviation_mm"]
    horizontal_mm_text = f"{horizontal_mm:.1f} mm" if horizontal_mm is not None else "N/A"
    return [
        (f"{label} angle: {metrics['angle_deviation']:.2f} deg", color),
        (f"{label} horiz: {metrics['horizontal_deviation']:.1f} px", color),
        (f"{label} horiz: {horizontal_mm_text}", color),
    ]


def draw_text_panel(
    lines: list[tuple[str, tuple[int, int, int], float, int]],
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 16
    y = 28
    line_gap = 22

    for text, color, scale, thickness in lines:
        if text:
            fitted = fit_text(text, font, scale, thickness, width - 2 * x)
            cv2.putText(panel, fitted, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_gap
        if y > height - 14:
            break
    return panel


def fit_text(
    text: str,
    font: int,
    scale: float,
    thickness: int,
    max_width: int,
) -> str:
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text

    ellipsis = "..."
    max_chars = len(text)
    while max_chars > len(ellipsis):
        candidate = text[: max_chars - len(ellipsis)].rstrip() + ellipsis
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
        max_chars -= 1
    return ellipsis


def add_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    labeled = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    line_h = 30
    panel_h = 20 + line_h * len(lines)
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, labeled, 0.38, 0, dst=labeled)
    for index, text in enumerate(lines):
        cv2.putText(
            labeled,
            text,
            (12, 28 + index * line_h),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return labeled


def place_roi_in_full_frame(
    roi_image: np.ndarray,
    full_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    full_h, full_w = full_shape[:2]
    x, y, width, height = roi
    canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    paste_w = min(width, roi_image.shape[1], full_w - x)
    paste_h = min(height, roi_image.shape[0], full_h - y)
    if paste_w > 0 and paste_h > 0:
        canvas[y : y + paste_h, x : x + paste_w] = roi_image[:paste_h, :paste_w]
        cv2.rectangle(canvas, (x, y), (x + paste_w, y + paste_h), (0, 255, 0), 2)
    return canvas


def pad_to_height(
    image: np.ndarray,
    height: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), color, dtype=np.uint8)
    return np.vstack((image, pad))


def hstack_padded(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if padded and gap > 0:
            padded.append(np.zeros((height, gap, 3), dtype=np.uint8))
        padded.append(pad_to_height(image, height))
    return np.hstack(padded)


def process_directory(
    depth_image_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> int:
    config = load_config(config_path)
    image_paths = find_depth_images(depth_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    line_history: list[tuple[int, int, int, int] | None] = []

    for frame_position, image_path in enumerate(image_paths):
        depth_image = load_depth_image(image_path)
        frame_result = render_frame(
            image_path=image_path,
            depth_image=depth_image,
            frame_position=frame_position,
            total_frames=len(image_paths),
            line_history=line_history,
            config=config,
        )
        output_path = output_dir / f"{image_path.stem}_processed.png"
        if not cv2.imwrite(str(output_path), frame_result.rendered_image):
            raise OSError(f"Failed to write output image: {output_path}")

    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the depth image processing pipeline for one camera folder."
    )
    parser.add_argument(
        "depth_image_dir",
        type=Path,
        help="Folder containing raw depth PNG images from one camera.",
    )
    parser.add_argument(
        "config_yaml",
        type=Path,
        help="Path to depth_camera_config.yaml.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where processed images will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_image_dir = args.depth_image_dir.expanduser().resolve()
    config_path = args.config_yaml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        processed_count = process_directory(depth_image_dir, config_path, output_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Processed {processed_count} depth images into {output_dir}")


if __name__ == "__main__":
    main()
