from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import yaml

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


DEFAULT_DEPTH_SCALE = 0.0010000000474974513


@dataclass(frozen=True)
class CameraProcessingConfig:
    label: str = "depth_camera"
    img_width: int = 848
    img_height: int = 480
    fps: float = 15.0
    hfov: float = 87.0
    vfov: float = 58.0
    depth_scale: float = DEFAULT_DEPTH_SCALE

    roi_enabled: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 848
    roi_height: int = 480

    depth_alpha: float = 0.1
    depth_cutoff_enabled: bool = False
    depth_cutoff_min_m: float = 0.2
    depth_cutoff_max_m: float = 1.0

    canny_min_val: int = 50
    canny_max_val: int = 150
    dilation_size: int = 4
    min_line_length: int = 200
    max_line_gap: int = 100
    hough_threshold: int = 50
    right: bool = False

    median_line_enabled: bool = True
    median_line_window_size: int = 15
    median_line_min_detections: int = 8

    ref_offset_x: int = 0
    ref_angle_deg: float = 90.0
    display_depth_roi_in_full_frame: bool = True


@dataclass(frozen=True)
class FramePair:
    frame_index: int
    color_path: Path
    depth_path: Path


@dataclass(frozen=True)
class ProcessedFrame:
    image: np.ndarray
    metadata_lines: list[tuple[str, tuple[int, int, int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process paired color/depth PNGs from one camera and write annotated images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing one camera processing configuration.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing color/ and depth/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where processed PNG images will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_processing_config(args.config)
    frame_pairs = list_frame_pairs(args.input_dir)
    if not frame_pairs:
        raise ValueError(f"No matching color/depth PNG pairs found below {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    line_history: list[tuple[int, int, int, int] | None] = []
    for index, frame_pair in enumerate(frame_pairs, start=1):
        color_image = load_color(frame_pair.color_path)
        depth_image = load_depth(frame_pair.depth_path)
        processed = process_frame_pair(color_image, depth_image, frame_pair, config, line_history)
        output_path = args.output_dir / f"{frame_pair.depth_path.stem}_processed.png"
        if not cv2.imwrite(str(output_path), processed.image):
            raise OSError(f"Failed to write {output_path}")
        print(f"[{index}/{len(frame_pairs)}] wrote {output_path}")


def load_processing_config(config_path: Path) -> CameraProcessingConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration root must be a mapping")

    merged = deepcopy(raw_config.get("processing_config", {}))
    if merged is None:
        merged = {}
    if not isinstance(merged, dict):
        raise ValueError("processing_config must be a mapping")

    sensors = raw_config.get("sensors")
    label = "depth_camera"
    if sensors is not None:
        if not isinstance(sensors, list) or len(sensors) != 1:
            raise ValueError("Config must contain exactly one sensor when sensors is set")
        sensor = sensors[0]
        if not isinstance(sensor, dict):
            raise ValueError("Sensor entry must be a mapping")
        label = str(sensor.get("name") or label)
        sensor_processing = sensor.get("processing_config", {})
        if sensor_processing is None:
            sensor_processing = {}
        if not isinstance(sensor_processing, dict):
            raise ValueError("sensor.processing_config must be a mapping")
        merged = deep_merge(merged, sensor_processing)

    return config_from_mapping(merged, label=label)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def config_from_mapping(
    mapping: dict[str, Any],
    label: str = "depth_camera",
) -> CameraProcessingConfig:
    camera = mapping_section(mapping, "camera")
    processing = mapping_section(mapping, "processing")
    edge_detection = mapping_section(mapping, "edge_detection")
    reference_line = mapping_section(mapping, "reference_line")
    roi = mapping_section(mapping, "roi")
    depth_cutoff = mapping_section(mapping, "depth_cutoff")
    median_line = mapping_section(mapping, "median_line")
    display = mapping_section(mapping, "display")

    img_width = get_int(camera, "img_width", CameraProcessingConfig.img_width)
    img_height = get_int(camera, "img_height", CameraProcessingConfig.img_height)

    return CameraProcessingConfig(
        label=label,
        img_width=img_width,
        img_height=img_height,
        fps=get_float(camera, "fps", CameraProcessingConfig.fps),
        hfov=get_float(camera, "hfov", CameraProcessingConfig.hfov),
        vfov=get_float(camera, "vfov", CameraProcessingConfig.vfov),
        depth_scale=get_float(camera, "depth_scale", CameraProcessingConfig.depth_scale),
        roi_enabled=get_bool(roi, "enabled", CameraProcessingConfig.roi_enabled),
        roi_x=get_int(roi, "x", CameraProcessingConfig.roi_x),
        roi_y=get_int(roi, "y", CameraProcessingConfig.roi_y),
        roi_width=get_int(roi, "width", img_width),
        roi_height=get_int(roi, "height", img_height),
        depth_alpha=get_float(processing, "depth_alpha", CameraProcessingConfig.depth_alpha),
        depth_cutoff_enabled=get_bool(
            depth_cutoff,
            "enabled",
            CameraProcessingConfig.depth_cutoff_enabled,
        ),
        depth_cutoff_min_m=get_float(
            depth_cutoff,
            "min_depth_m",
            CameraProcessingConfig.depth_cutoff_min_m,
        ),
        depth_cutoff_max_m=get_float(
            depth_cutoff,
            "max_depth_m",
            CameraProcessingConfig.depth_cutoff_max_m,
        ),
        canny_min_val=get_int(
            edge_detection,
            "canny_min_val",
            CameraProcessingConfig.canny_min_val,
        ),
        canny_max_val=get_int(
            edge_detection,
            "canny_max_val",
            CameraProcessingConfig.canny_max_val,
        ),
        dilation_size=get_int(
            edge_detection,
            "dilation_size",
            get_int(edge_detection, "dilate_size", CameraProcessingConfig.dilation_size),
        ),
        min_line_length=get_int(
            edge_detection,
            "min_line_length",
            CameraProcessingConfig.min_line_length,
        ),
        max_line_gap=get_int(
            edge_detection,
            "max_line_gap",
            CameraProcessingConfig.max_line_gap,
        ),
        hough_threshold=get_int(
            edge_detection,
            "hough_threshold",
            CameraProcessingConfig.hough_threshold,
        ),
        right=get_bool(edge_detection, "right", CameraProcessingConfig.right),
        median_line_enabled=get_bool(
            median_line,
            "enabled",
            CameraProcessingConfig.median_line_enabled,
        ),
        median_line_window_size=get_int(
            median_line,
            "window_size",
            CameraProcessingConfig.median_line_window_size,
        ),
        median_line_min_detections=get_int(
            median_line,
            "min_detections",
            CameraProcessingConfig.median_line_min_detections,
        ),
        ref_offset_x=get_int(
            reference_line,
            "offset_x",
            CameraProcessingConfig.ref_offset_x,
        ),
        ref_angle_deg=get_float(
            reference_line,
            "angle_deg",
            CameraProcessingConfig.ref_angle_deg,
        ),
        display_depth_roi_in_full_frame=get_bool(
            display,
            "depth_roi_in_full_frame",
            CameraProcessingConfig.display_depth_roi_in_full_frame,
        ),
    )


def mapping_section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def get_int(mapping: dict[str, Any], key: str, default: int) -> int:
    return int(mapping.get(key, default))


def get_float(mapping: dict[str, Any], key: str, default: float) -> float:
    return float(mapping.get(key, default))


def get_bool(mapping: dict[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def list_frame_pairs(input_dir: Path) -> list[FramePair]:
    color_dir = input_dir / "color"
    depth_dir = input_dir / "depth"
    if not color_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"Missing color/depth folders below {input_dir}")
    if not color_dir.is_dir() or not depth_dir.is_dir():
        raise NotADirectoryError(f"color and depth must be folders below {input_dir}")

    color_paths = index_frame_images(color_dir, "color")
    depth_paths = index_frame_images(depth_dir, "depth")
    if set(color_paths) != set(depth_paths):
        missing_color = sorted(set(depth_paths) - set(color_paths))
        missing_depth = sorted(set(color_paths) - set(depth_paths))
        raise ValueError(
            "Color/depth frame mismatch: "
            f"missing color={missing_color[:10]}, missing depth={missing_depth[:10]}"
        )

    return [
        FramePair(frame_index=frame, color_path=color_paths[frame], depth_path=depth_paths[frame])
        for frame in sorted(color_paths)
    ]


def index_frame_images(image_dir: Path, prefix: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(image_dir.glob(f"{prefix}_*.png"), key=depth_sort_key):
        frame = parse_frame_number(path)
        if frame is not None:
            paths[frame] = path
    return paths


def depth_sort_key(path: Path) -> tuple[int, int | str, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match is not None:
        return 0, int(match.group(1)), path.name
    return 1, path.name, path.name


def load_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(path)
    return image


def process_frame_pair(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    frame_pair: FramePair,
    config: CameraProcessingConfig,
    line_history: list[tuple[int, int, int, int] | None],
) -> ProcessedFrame:
    color_image = color_image.copy()
    depth_work, applied_roi, _, depth_colormap, filtered_edges = build_edge_inputs(
        depth_image, config
    )

    if applied_roi is not None:
        x, y, width, height = applied_roi
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    current_line = find_longest_line_right(
        filtered_edges,
        min_line_length=config.min_line_length,
        max_line_gap=config.max_line_gap,
        threshold=config.hough_threshold,
        right=config.right,
    )
    if current_line is not None:
        current_line = tuple(int(value) for value in current_line)
    line_history.append(current_line)

    median_line, _ = median_line_for_history(
        line_history,
        config,
        image_width=depth_work.shape[1],
        image_height=depth_work.shape[0],
    )

    result_img = draw_reference_line(
        depth_colormap.copy(),
        config.ref_offset_x,
        config.ref_angle_deg,
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

    theta_horizontal = config.hfov / config.img_width
    theta_vertical = config.vfov / config.img_height

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
                theta_horizontal=theta_horizontal,
                theta_vertical=theta_vertical,
            )
            horizontal_deviation_mm = horizontal_deviation * pixel_width

    if median_line is not None:
        result_img = draw_long_line(result_img, *median_line, color=median_color, thickness=2)
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
                theta_horizontal=theta_horizontal,
                theta_vertical=theta_vertical,
            )
            median_horizontal_deviation_mm = median_horizontal_deviation * median_pixel_width

    height, width = depth_work.shape[:2]
    center_x, center_y = width // 2, height // 2
    center_depth = int(depth_work[center_y, center_x])
    cv2.circle(result_img, (center_x, center_y), radius=3, color=current_color, thickness=-1)

    _, _, center_pixel_area = calculate_pixel_area(
        depth_in_mm=center_depth,
        theta_horizontal=theta_horizontal,
        theta_vertical=theta_vertical,
    )

    depth_tile = result_img
    if applied_roi is not None and config.display_depth_roi_in_full_frame:
        depth_tile = place_roi_in_full_frame(result_img, depth_image.shape[:2], applied_roi)

    depth_tile = add_label(
        depth_tile,
        [f"{config.label} processed depth"],
    )
    metadata_lines = build_metadata_lines(
        frame_pair=frame_pair,
        depth_shape=depth_image.shape,
        config=config,
        applied_roi=applied_roi,
        center_depth=center_depth,
        center_pixel_area=center_pixel_area,
        current_line=current_line,
        angle_deviation=angle_deviation,
        horizontal_deviation=horizontal_deviation,
        horizontal_deviation_mm=horizontal_deviation_mm,
        median_line=median_line,
        median_angle_deviation=median_angle_deviation,
        median_horizontal_deviation=median_horizontal_deviation,
        median_horizontal_deviation_mm=median_horizontal_deviation_mm,
        text_color=current_text_color,
        median_text_color=median_text_color,
        dim_color=dim_color,
    )
    image_stack = vstack_padded([depth_tile, color_image], gap=10)
    panel = make_metadata_panel(metadata_lines, min_height=image_stack.shape[0])
    return ProcessedFrame(image=hstack_padded([image_stack, panel], gap=10), metadata_lines=metadata_lines)


def build_edge_inputs(
    depth_image: np.ndarray,
    config: CameraProcessingConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray, np.ndarray, np.ndarray]:
    depth_work, applied_roi = crop_depth(depth_image, config)
    edge_depth = apply_depth_cutoff(
        depth_work,
        depth_scale=config.depth_scale,
        min_depth_m=config.depth_cutoff_min_m,
        max_depth_m=config.depth_cutoff_max_m,
        enabled=config.depth_cutoff_enabled,
    )
    depth_colormap = process_depth_image(edge_depth, depth_alpha=config.depth_alpha)
    canny_edges = apply_canny_edge_detection(
        depth_colormap,
        min_val=config.canny_min_val,
        max_val=config.canny_max_val,
    )
    filtered_edges = filter_out_zero_boundaries(
        canny_edges,
        edge_depth,
        dilate_size=max(1, config.dilation_size),
    )
    return depth_work, applied_roi, edge_depth, depth_colormap, filtered_edges


def median_line_for_history(
    line_history: list[tuple[int, int, int, int] | None],
    config: CameraProcessingConfig,
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, int, int, int] | None, int]:
    if not config.median_line_enabled:
        return None, 0

    window = line_history[-config.median_line_window_size:]
    valid_count = sum(line is not None for line in window)
    if len(window) < config.median_line_window_size:
        return None, valid_count
    if valid_count < config.median_line_min_detections:
        return None, valid_count

    median_line = calculate_median_line(
        window,
        image_width=image_width,
        image_height=image_height,
        window_size=config.median_line_window_size,
        min_detections=config.median_line_min_detections,
    )
    return median_line, valid_count


def crop_depth(
    depth_image: np.ndarray,
    config: CameraProcessingConfig,
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
            f"Invalid ROI: {(config.roi_x, config.roi_y, config.roi_width, config.roi_height)}"
        )
    return depth_image[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


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
        canvas[y:y + paste_h, x:x + paste_w] = roi_image[:paste_h, :paste_w]
        cv2.rectangle(canvas, (x, y), (x + paste_w, y + paste_h), (0, 255, 0), 2)
    return canvas


def build_metadata_lines(
    frame_pair: FramePair,
    depth_shape: tuple[int, ...],
    config: CameraProcessingConfig,
    applied_roi: tuple[int, int, int, int] | None,
    center_depth: int,
    center_pixel_area: float,
    current_line: tuple[int, int, int, int] | None,
    angle_deviation: float | None,
    horizontal_deviation: float | None,
    horizontal_deviation_mm: float | None,
    median_line: tuple[int, int, int, int] | None,
    median_angle_deviation: float | None,
    median_horizontal_deviation: float | None,
    median_horizontal_deviation_mm: float | None,
    text_color: tuple[int, int, int],
    median_text_color: tuple[int, int, int],
    dim_color: tuple[int, int, int],
) -> list[tuple[str, tuple[int, int, int]]]:
    current_mm = f"{horizontal_deviation_mm:.1f} mm" if horizontal_deviation_mm is not None else "N/A"
    median_mm = (
        f"{median_horizontal_deviation_mm:.1f} mm"
        if median_horizontal_deviation_mm is not None
        else "N/A"
    )
    roi_text = (
        f"on ({applied_roi[0]},{applied_roi[1]}) {applied_roi[2]}x{applied_roi[3]}"
        if applied_roi is not None
        else "off"
    )
    cutoff_text = (
        f"on {config.depth_cutoff_min_m:.2f}-{config.depth_cutoff_max_m:.2f} m"
        if config.depth_cutoff_enabled
        else "off"
    )
    line_side = "rightmost" if config.right else "leftmost"
    lines = [
        (f"{config.label} frame {frame_pair.frame_index:06d}", (255, 255, 255)),
        (f"Depth {frame_pair.depth_path.name}", dim_color),
        (f"Color {frame_pair.color_path.name}", dim_color),
        (f"Image {depth_shape[1]}x{depth_shape[0]} px", dim_color),
        (f"Center depth {center_depth} mm", text_color),
        (f"Pixel area {center_pixel_area:.2f} mm^2", text_color),
        ("", dim_color),
        ("Config", (255, 255, 255)),
        (f"ROI {roi_text}", dim_color),
        (f"Cutoff {cutoff_text}", dim_color),
        (f"Canny {config.canny_min_val}/{config.canny_max_val} filter {config.dilation_size}", dim_color),
        (f"Hough len {config.min_line_length} gap {config.max_line_gap}", dim_color),
        (f"Hough thr {config.hough_threshold}", dim_color),
        (f"Line {line_side} Ref {config.ref_offset_x}px {config.ref_angle_deg:g}deg", dim_color),
        (
            f"Median {'on' if config.median_line_enabled else 'off'} "
            f"win {config.median_line_window_size} min {config.median_line_min_detections}",
            dim_color,
        ),
        (f"Depth alpha {config.depth_alpha:g}", dim_color),
        ("", dim_color),
        ("Detected line", text_color),
    ]
    if current_line is not None and angle_deviation is not None:
        lines += [
            (f"Current line {current_line}", text_color),
            (f"Current angle {angle_deviation:.2f} deg", text_color),
        ]
        if horizontal_deviation is not None:
            lines.append((f"Current horiz {horizontal_deviation:.1f} px", text_color))
        lines.append((f"Current horiz {current_mm}", text_color))
    else:
        lines.append(("Current: no detection", text_color))

    if (
        config.median_line_enabled
        and median_line is not None
        and median_angle_deviation is not None
    ):
        lines += [
            (f"Median line {median_line}", median_text_color),
            (f"Median angle {median_angle_deviation:.2f} deg", median_text_color),
        ]
        if median_horizontal_deviation is not None:
            lines.append(
                (f"Median horiz {median_horizontal_deviation:.1f} px", median_text_color)
            )
        lines.append((f"Median horiz {median_mm}", median_text_color))
    return lines


def parse_frame_number(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match is not None else None


def add_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
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


def make_metadata_panel(
    lines: list[tuple[str, tuple[int, int, int]]],
    min_height: int,
    width: int = 560,
) -> np.ndarray:
    margin_x = 16
    margin_y = 16
    col_gap = 18
    line_h = 28
    max_lines_per_col = max(1, (min_height - 2 * margin_y - 8) // line_h)
    column_count = max(1, int(np.ceil(len(lines) / max_lines_per_col)))
    col_width = 260
    width = max(width, 2 * margin_x + column_count * col_width + (column_count - 1) * col_gap)
    height = max(min_height, 2 * margin_y + min(len(lines), max_lines_per_col) * line_h + 8)
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (text, color) in enumerate(lines):
        if not text:
            continue
        col_idx = idx // max_lines_per_col
        row_idx = idx % max_lines_per_col
        x = margin_x + col_idx * (col_width + col_gap)
        if row_idx == 0 and col_idx > 0:
            sep_x = x - col_gap // 2
            cv2.line(panel, (sep_x, margin_y), (sep_x, height - margin_y), (58, 58, 58), 1, cv2.LINE_AA)
        is_title = idx == 0 or text in {"Config", "Detected line"}
        scale = 0.74 if is_title else 0.62
        thickness = 2 if is_title else 1
        y = margin_y + 24 + row_idx * line_h
        cv2.putText(panel, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return panel


def hstack_padded(
    images: list[np.ndarray],
    gap: int = 8,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if padded and gap > 0:
            padded.append(np.full((height, gap, 3), color, dtype=np.uint8))
        padded.append(pad_to_height(image, height, color=color))
    return np.hstack(padded)


def vstack_padded(
    images: list[np.ndarray],
    gap: int = 10,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        if padded and gap > 0:
            padded.append(np.full((gap, width, 3), color, dtype=np.uint8))
        padded.append(pad_to_width(image, width, color=color))
    return np.vstack(padded)


def pad_to_height(
    image: np.ndarray,
    height: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), color, dtype=np.uint8)
    return np.vstack((image, pad))


def pad_to_width(
    image: np.ndarray,
    width: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image.shape[1] >= width:
        return image
    pad = np.full((image.shape[0], width - image.shape[1], 3), color, dtype=np.uint8)
    return np.hstack((image, pad))


if __name__ == "__main__":
    main()
