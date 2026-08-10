
"""Live RealSense edge-detection pipeline, mirroring analyze_recordings.

Streams the depth camera, shows the same processed-depth view as the
four-camera viewer (depth cutoff -> JET colormap -> Canny -> zero-boundary
filter -> Hough line) without ROI, and overlays the depth intrinsics (fx/fy),
center depth, and physical pixel size in mm live on the frame.
Edge-detection config hardcoded from front_left_depth_processing.yaml.
The camera's supported profiles are queried first, so the script also works
on USB 2.x (falls back to 640x480). Press q / Esc to quit.
"""

import cv2
import math
import numpy as np
import pyrealsense2 as rs

from helpers import (
    apply_canny_edge_detection,
    apply_depth_cutoff,
    calculate_pixel_area,
    calculate_pixel_area_fov,
    draw_long_line,
    draw_reference_line,
    filter_out_zero_boundaries,
    find_longest_line_right,
    process_depth_image,
)

# --- Hardcoded from front_left_depth_processing.yaml -------------------
DEPTH_SCALE = 0.0010000000474974513
DEPTH_CUTOFF_ENABLED = True
DEPTH_CUTOFF_MIN_M = 0.2
DEPTH_CUTOFF_MAX_M = 1.0
DEPTH_ALPHA = 0.55
CANNY_MIN_VAL = 130
CANNY_MAX_VAL = 150
DILATE_SIZE = 1
MIN_LINE_LENGTH = 150
MAX_LINE_GAP = 40
HOUGH_THRESHOLD = 20
FIND_RIGHTMOST_LINE = True
REF_OFFSET_X = -6
REF_ANGLE_DEG = 90

# Length of the measurement crosshair lines (px) at the image center
VERTICAL_LINE_PX = 100
HORIZONTAL_LINE_PX = 100

# Pixel-size method: False = pinhole (depth/fx, depth/fy), True = old
# FOV-average calculation (2*depth*tan(theta/2)). Toggle to compare results.
USE_OLD_PIXEL_SIZE = False

# Preferred stream profiles; the script falls back to the first supported one.
# DEPTH_PREFERRED = (1280, 720, rs.format.z16, 15)
DEPTH_PREFERRED = (848, 480, rs.format.z16, 15)
DEPTH_FALLBACKS = [
    (848, 480, rs.format.z16, 30),
    (640, 480, rs.format.z16, 30),
    (640, 480, rs.format.z16, 15),
]
# COLOR_PREFERRED = (1280, 720, rs.format.bgr8, 15)
COLOR_PREFERRED = (848, 480, rs.format.bgr8, 15)
COLOR_FALLBACKS = [
    (848, 480, rs.format.bgr8, 30),
    (640, 480, rs.format.bgr8, 30),
    (640, 480, rs.format.bgr8, 15),
]


def stream_profiles(device, stream_type) -> list[tuple[int, int, rs.format, int]]:
    """Return the (width, height, format, fps) profiles a device supports."""
    profiles = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == stream_type and profile.is_video_stream_profile():
                video = profile.as_video_stream_profile()
                profiles.add((video.width(), video.height(), profile.format(), video.fps()))
    return sorted(profiles, key=lambda p: (p[0], p[1], p[3], str(p[2])))


def choose_profile(
    supported: list[tuple[int, int, rs.format, int]],
    preferred: tuple[int, int, rs.format, int],
    fallbacks: list[tuple[int, int, rs.format, int]],
) -> tuple[int, int, rs.format, int] | None:
    """Pick the preferred profile, else the first supported fallback."""
    if preferred in supported:
        return preferred
    for candidate in fallbacks:
        if candidate in supported:
            return candidate
    return supported[0] if supported else None


def main() -> None:
    context = rs.context()
    device = context.query_devices()[0]
    name = device.get_info(rs.camera_info.name)
    try:
        usb = device.get_info(rs.camera_info.usb_type_descriptor)
    except RuntimeError:
        usb = "unknown"
    print(f"Device: {name}   USB: {usb}")

    depth_profiles = stream_profiles(device, rs.stream.depth)
    color_profiles = stream_profiles(device, rs.stream.color)
    print(f"Supported depth profiles: {depth_profiles}")
    print(f"Supported color profiles: {color_profiles}")

    depth_cfg = choose_profile(depth_profiles, DEPTH_PREFERRED, DEPTH_FALLBACKS)
    color_cfg = choose_profile(color_profiles, COLOR_PREFERRED, COLOR_FALLBACKS)
    if depth_cfg is None:
        raise RuntimeError("No depth stream profiles found on the device.")
    print(f"Using depth: {depth_cfg}   color: {color_cfg}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, *depth_cfg)
    if color_cfg is not None:
        config.enable_stream(rs.stream.color, *color_cfg)
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Couldn't start the stream. On USB 2.x, 848x480 is not available - "
            "connect the camera to a USB 3.x port, or check the supported "
            f"profiles printed above. ({exc})"
        ) from exc
    try:
        intrinsics_printed = False
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame:
                continue

            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            cx, cy = intr.width // 2, intr.height // 2
            center_depth_mm = depth_frame.get_distance(cx, cy) * 1000.0
            if USE_OLD_PIXEL_SIZE:
                hfov_deg = 2 * math.degrees(math.atan(intr.width / (2 * intr.fx)))
                vfov_deg = 2 * math.degrees(math.atan(intr.height / (2 * intr.fy)))
                theta_h = hfov_deg / intr.width
                theta_v = vfov_deg / intr.height
                pixel_width_mm, pixel_height_mm, _ = calculate_pixel_area_fov(
                    center_depth_mm, theta_h, theta_v
                )
                px_w_factor = 2 * math.tan(math.radians(theta_h / 2))
                px_h_factor = 2 * math.tan(math.radians(theta_v / 2))
                method_label = "fov"
            else:
                pixel_width_mm, pixel_height_mm, _ = calculate_pixel_area(
                    center_depth_mm, intr.fx, intr.fy
                )
                px_w_factor = 1.0 / intr.fx
                px_h_factor = 1.0 / intr.fy
                method_label = "pinhole"
            depth = np.asanyarray(depth_frame.get_data())
            half_v = VERTICAL_LINE_PX // 2
            half_h = HORIZONTAL_LINE_PX // 2
            v_depths_mm = depth[cy - half_v:cy + half_v + 1, cx] * DEPTH_SCALE * 1000.0
            h_depths_mm = depth[cy, cx - half_h:cx + half_h + 1] * DEPTH_SCALE * 1000.0
            v_valid = v_depths_mm > 0
            h_valid = h_depths_mm > 0
            vertical_line_mm = (
                float((v_depths_mm[v_valid] * px_h_factor).sum())
                if v_valid.any()
                else 0.0
            )
            horizontal_line_mm = (
                float((h_depths_mm[h_valid] * px_w_factor).sum())
                if h_valid.any()
                else 0.0
            )
            if not intrinsics_printed:
                print(
                    f"depth intrinsics: fx={intr.fx:.4f}  fy={intr.fy:.4f}  "
                    f"ppx={intr.ppx:.2f}  ppy={intr.ppy:.2f}  "
                    f"({intr.width}x{intr.height})"
                )
                print(f"pixel-size method: {method_label}")
                intrinsics_printed = True

            color = np.asanyarray(color_frame.get_data()) if color_frame else None

            edge_depth = apply_depth_cutoff(
                depth,
                depth_scale=DEPTH_SCALE,
                min_depth_m=DEPTH_CUTOFF_MIN_M,
                max_depth_m=DEPTH_CUTOFF_MAX_M,
                enabled=DEPTH_CUTOFF_ENABLED,
            )
            depth_colormap = process_depth_image(
                edge_depth, depth_alpha=DEPTH_ALPHA
            )
            canny_edges = apply_canny_edge_detection(
                depth_colormap, min_val=CANNY_MIN_VAL, max_val=CANNY_MAX_VAL
            )
            filtered_edges = filter_out_zero_boundaries(
                canny_edges, edge_depth, dilate_size=DILATE_SIZE
            )
            line = find_longest_line_right(
                filtered_edges,
                min_line_length=MIN_LINE_LENGTH,
                max_line_gap=MAX_LINE_GAP,
                threshold=HOUGH_THRESHOLD,
                right=FIND_RIGHTMOST_LINE,
            )

            result = draw_reference_line(
                depth_colormap.copy(), REF_OFFSET_X, REF_ANGLE_DEG
            )
            if line is not None:
                result = draw_long_line(result, *line)

            view = np.hstack([color, result]) if color is not None else result
            depth_panel_x = color.shape[1] if color is not None else 0

            # Vertical (green) and horizontal (orange) measurement lines,
            # drawn at the center of the color panel and the depth panel
            for panel_x in ([0, depth_panel_x] if color is not None else [0]):
                cv2.line(view, (panel_x + cx, cy - half_v),
                         (panel_x + cx, cy + half_v), (0, 255, 0), 2)
                cv2.line(view, (panel_x + cx - half_h, cy),
                         (panel_x + cx + half_h, cy), (0, 165, 255), 2)
                cv2.circle(view, (panel_x + cx, cy), 3, (0, 255, 255), -1)

            # Live overlay: intrinsics + center depth + pixel size in mm
            info_lines = [
                f"fx={intr.fx:.2f}  fy={intr.fy:.2f}  ({intr.width}x{intr.height})",
                f"center ({cx},{cy})  depth={center_depth_mm:.1f} mm",
                f"pixel [{method_label}] {pixel_width_mm:.3f} x {pixel_height_mm:.3f} mm",
                f"vertical {VERTICAL_LINE_PX}px line: {vertical_line_mm:.2f} mm ({int(v_valid.sum())}/{VERTICAL_LINE_PX + 1} px)",
                f"horizontal {HORIZONTAL_LINE_PX}px line: {horizontal_line_mm:.2f} mm ({int(h_valid.sum())}/{HORIZONTAL_LINE_PX + 1} px)",
            ]
            for i, line in enumerate(info_lines):
                y = 24 + i * 24
                cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("live edge detection (color | processed depth)", view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
