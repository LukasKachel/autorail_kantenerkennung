"""Live RealSense length-calibration screen for a fixed 100px crosshair.

Streams the depth camera and shows the color feed next to a simple JET
colormap of the depth. Two fixed measurement lines (100 px each) are drawn at
the image center, mirroring live_edge_detection.py:

  - horizontal (orange): 100 px along the u axis
  - vertical   (green):  100 px along the v axis

Each line's physical length is computed two ways and shown live, so you can
place a real ruler in front of the camera and validate the result:

  A) per-pixel pinhole sum:  length = sum of depth_i / f over the 100 px
     (each pixel uses its own measured depth; this is the length projected
     onto the image plane, so it under-reads when the surface is tilted)
  B) deprojected endpoints:  rs2_deproject_pixel_to_point on both ends with
     their own depth, then Euclidean distance between the 3D points
     (true 3D length, independent of tilt)

Intrinsics (fx/fy/ppx/ppy) and depth units are read from the device at
runtime. q / Esc quits. The camera's supported profiles are queried first, so
the script also works on USB 2.x (falls back to 640x480).
"""

import math

import cv2
import numpy as np
import pyrealsense2 as rs

from helpers import apply_depth_cutoff, process_depth_image

# --- Hardcoded from front_left_depth_processing.yaml -------------------
DEPTH_CUTOFF_ENABLED = True
DEPTH_CUTOFF_MIN_M = 0.2
DEPTH_CUTOFF_MAX_M = 1.0
DEPTH_ALPHA = 0.55

# Length of the measurement crosshair lines (px) at the image center
VERTICAL_LINE_PX = 100
HORIZONTAL_LINE_PX = 100

# Preferred stream profiles; the script falls back to the first supported one.
DEPTH_PREFERRED = (848, 480, rs.format.z16, 15)
DEPTH_FALLBACKS = [
    (848, 480, rs.format.z16, 30),
    (640, 480, rs.format.z16, 30),
    (640, 480, rs.format.z16, 15),
]
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


def _line_depth_mm(depth, y, x, length, axis, depth_units, fallback_mm):
    """Depth values (mm) along the 100px line; invalid pixels become the fallback."""
    half = length // 2
    if axis == "v":
        depths = depth[y - half:y + half, x]
    else:
        depths = depth[y, x - half:x + half]
    return (depths[depths > 0] * depth_units * 1000.0 if (depths > 0).any()
            else np.array([fallback_mm]))


def measure_line(depth, y, x, length, axis, depth_units, intr, fallback_mm) -> tuple[float, float, int]:
    """Length of a ``length``-px line in mm, two ways.

    Returns ``(per_pixel_sum, deprojected_endpoints, valid_px_count)``.
    ``depth`` is the raw z16 array, ``axis`` is 'v' (vertical) or 'h'
    (horizontal), ``intr`` the depth intrinsics (same resolution).
    """
    depths_mm = _line_depth_mm(depth, y, x, length, axis, depth_units, fallback_mm)
    per_pixel_sum = float(
        (depths_mm / (intr.fy if axis == "v" else intr.fx)).sum()
    )

    half = length // 2
    if axis == "v":
        u1 = u2 = x
        v1, v2 = y - half, y + half
    else:
        v1 = v2 = y
        u1, u2 = x - half, x + half
    z1_raw = depth[v1, u1]
    z2_raw = depth[v2, u2]
    z1_m = (z1_raw * depth_units) if z1_raw > 0 else fallback_mm / 1000.0
    z2_m = (z2_raw * depth_units) if z2_raw > 0 else fallback_mm / 1000.0
    p1 = rs.rs2_deproject_pixel_to_point(intr, [u1, v1], z1_m)
    p2 = rs.rs2_deproject_pixel_to_point(intr, [u2, v2], z2_m)
    deprojected = float(np.linalg.norm(np.array(p2) - np.array(p1))) * 1000.0
    return per_pixel_sum, deprojected, int(depths_mm.size)


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
        depth_units = (
            pipeline.get_active_profile().get_device().first_depth_sensor()
            .get_option(rs.option.depth_units)
        )
        print(f"depth units: {depth_units:.6f} m")
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame:
                continue

            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            cx, cy = intr.width // 2, intr.height // 2
            center_depth_mm = depth_frame.get_distance(cx, cy) * 1000.0
            depth = np.asanyarray(depth_frame.get_data())
            if not intrinsics_printed:
                hfov = 2 * math.degrees(math.atan(intr.width / (2 * intr.fx)))
                vfov = 2 * math.degrees(math.atan(intr.height / (2 * intr.fy)))
                print(
                    f"depth intrinsics: fx={intr.fx:.4f}  fy={intr.fy:.4f}  "
                    f"ppx={intr.ppx:.2f}  ppy={intr.ppy:.2f}  "
                    f"({intr.width}x{intr.height})  fov={hfov:.1f}x{vfov:.1f} deg"
                )
                intrinsics_printed = True

            color = np.asanyarray(color_frame.get_data()) if color_frame else None

            edge_depth = apply_depth_cutoff(
                depth,
                depth_scale=depth_units,
                min_depth_m=DEPTH_CUTOFF_MIN_M,
                max_depth_m=DEPTH_CUTOFF_MAX_M,
                enabled=DEPTH_CUTOFF_ENABLED,
            )
            depth_colormap = process_depth_image(
                edge_depth, depth_alpha=DEPTH_ALPHA
            )

            view = np.hstack([color, depth_colormap]) if color is not None else depth_colormap
            depth_panel_x = color.shape[1] if color is not None else 0

            # Vertical (green) and horizontal (orange) 100px measurement lines
            half_v = VERTICAL_LINE_PX // 2
            half_h = HORIZONTAL_LINE_PX // 2
            for panel_x in ([0, depth_panel_x] if color is not None else [0]):
                cv2.line(view, (panel_x + cx, cy - half_v),
                         (panel_x + cx, cy + half_v), (0, 255, 0), 2)
                cv2.line(view, (panel_x + cx - half_h, cy),
                         (panel_x + cx + half_h, cy), (0, 165, 255), 2)
                cv2.circle(view, (panel_x + cx, cy), 3, (0, 255, 255), -1)

            v_sum, v_deproj, v_valid = measure_line(
                depth, cy, cx, VERTICAL_LINE_PX, "v", depth_units, intr, center_depth_mm
            )
            h_sum, h_deproj, h_valid = measure_line(
                depth, cy, cx, HORIZONTAL_LINE_PX, "h", depth_units, intr, center_depth_mm
            )

            info_lines = [
                f"fx={intr.fx:.2f}  fy={intr.fy:.2f}  ({intr.width}x{intr.height})",
                f"center ({cx},{cy})  depth={center_depth_mm:.1f} mm",
                f"VERTICAL {VERTICAL_LINE_PX}px  sum={v_sum:.2f} mm  deproject={v_deproj:.2f} mm  ({v_valid}/{VERTICAL_LINE_PX} px)",
                f"HORIZONTAL {HORIZONTAL_LINE_PX}px  sum={h_sum:.2f} mm  deproject={h_deproj:.2f} mm  ({h_valid}/{HORIZONTAL_LINE_PX} px)",
            ]
            for i, line in enumerate(info_lines):
                y = 24 + i * 24
                cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("length calibration (color | depth)", view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
