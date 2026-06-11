from __future__ import annotations

import math

import cv2
import numpy as np


def process_depth_image(depth_image: np.ndarray, depth_alpha: float = 0.3) -> np.ndarray:
    """Convert a depth image into a JET color map for visualization.

    Args:
        depth_image: Raw depth image.
        depth_alpha: Scale factor used to convert depth values to 8-bit intensity.

    Returns:
        BGR color image with the OpenCV JET color map applied.
    """
    return cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=depth_alpha),
        cv2.COLORMAP_JET,
    )


def apply_depth_cutoff(
    depth_image: np.ndarray,
    depth_scale: float,
    min_depth_m: float = 0.2,
    max_depth_m: float = 1.0,
    enabled: bool = True,
) -> np.ndarray:
    """Zero depth values outside the configured range (mind_depth_m).

    Args:
        depth_image: Raw depth image whose values use the camera's depth units.
        depth_scale: Conversion factor from raw depth units to meters.
        min_depth_m: Minimum allowed depth in meters.
        max_depth_m: Maximum allowed depth in meters.
        enabled: If false, return a copy of the input without filtering.

    Returns:
        Copy of the depth image with out-of-range non-zero values set to zero.
    """
    cutoff_image = depth_image.copy()
    if not enabled:
        return cutoff_image

    min_raw = int(round(min_depth_m / depth_scale))
    max_raw = int(round(max_depth_m / depth_scale))
    cutoff_mask = (cutoff_image > 0) & (
        (cutoff_image < min_raw) | (cutoff_image > max_raw)
    )
    cutoff_image[cutoff_mask] = 0
    return cutoff_image


def apply_canny_edge_detection(
    depth_image: np.ndarray,
    min_val: int = 50,
    max_val: int = 100,
    aperture_size: int = 3,
) -> np.ndarray:
    """Run Canny edge detection on the given depth image.

    Args:
        depth_image: Image passed to OpenCV's Canny detector.
        min_val: Lower hysteresis threshold.
        max_val: Upper hysteresis threshold.
        aperture_size: Sobel kernel size used by Canny.

    Returns:
        Binary edge image produced by Canny.
    """
    return cv2.Canny(
        image=depth_image,
        threshold1=min_val,
        threshold2=max_val,
        apertureSize=aperture_size,
    )


def filter_out_zero_boundaries(
    canny_edges: np.ndarray,
    depth_image: np.ndarray,
    dilate_size: int = 3,
) -> np.ndarray:
    """Remove edges that touch zero-depth regions after mask dilation.

    Args:
        canny_edges: Binary edge image to filter.
        depth_image: Depth image used to identify invalid zero-depth pixels.
        dilate_size: Width and height of the square dilation kernel.

    Returns:
        Edge image with edges near zero-depth regions masked out.
    """
    zero_mask = (depth_image == 0).astype(np.uint8) * 255
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dilated_zero_mask = cv2.dilate(zero_mask, kernel, iterations=1)
    valid_region_mask = cv2.bitwise_not(dilated_zero_mask)
    return cv2.bitwise_and(canny_edges, valid_region_mask)


def find_longest_line_right(
    edge_img: np.ndarray,
    min_line_length: int = 300,
    max_line_gap: int = 50,
    threshold: int = 50,
    right: bool = True,
) -> tuple[int, int, int, int] | None:
    """Find the rightmost or leftmost line among the longest Hough candidates.

    Args:
        edge_img: Binary edge image used as input for probabilistic Hough lines.
        min_line_length: Minimum accepted line length in pixels.
        max_line_gap: Maximum gap in pixels that can connect line segments.
        threshold: Minimum number of votes required for a line.
        right: If true, select the rightmost candidate; otherwise select the leftmost.

    Returns:
        Detected line as ``(x1, y1, x2, y2)``, or ``None`` if no line was found.
    """
    hough_lines = cv2.HoughLinesP(
        image=edge_img,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if hough_lines is None:
        return None

    lines_with_length = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lines_with_length.append((line_length, (int(x1), int(y1), int(x2), int(y2))))

    longest_candidates = sorted(lines_with_length, reverse=True)[:4]
    if not longest_candidates:
        return None

    key = lambda item: (item[1][0] + item[1][2]) / 2
    selected = max(longest_candidates, key=key) if right else min(longest_candidates, key=key)
    return selected[1]


def draw_long_line(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int] | None = None,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw the detected line extended across the image.

    Args:
        image: Image to draw on. This array is modified in place.
        x1: First line endpoint x-coordinate.
        y1: First line endpoint y-coordinate.
        x2: Second line endpoint x-coordinate.
        y2: Second line endpoint y-coordinate.
        color: Optional BGR line color. Defaults depend on line orientation.
        thickness: Optional line thickness in pixels.

    Returns:
        The same image array with the extended line drawn on it.
    """
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        img_width = image.shape[1]
        p1_y = int(b)
        p2_y = int(m * img_width + b)
        cv2.line(
            img=image,
            pt1=(0, p1_y),
            pt2=(img_width, p2_y),
            color=color if color is not None else (255, 255, 255),
            thickness=thickness if thickness is not None else 3,
        )
    else:
        img_height = image.shape[0]
        cv2.line(
            img=image,
            pt1=(x1, 0),
            pt2=(x1, img_height),
            color=color if color is not None else (0, 255, 0),
            thickness=thickness if thickness is not None else 2,
        )
    return image


def calculate_median_line(
    line_history: list[tuple[int, int, int, int] | None],
    image_width: int,
    image_height: int,
    window_size: int = 15,
    min_detections: int = 8,
) -> tuple[int, int, int, int] | None:
    """Compute a median full-height line from recent detections.

    Args:
        line_history: Recent line detections, with ``None`` for missing detections.
        image_width: Width of the image the returned line belongs to.
        image_height: Height of the image the returned line belongs to.
        window_size: Required history length before a median line is calculated.
        min_detections: Minimum number of valid lines required in the history.

    Returns:
        Median line as ``(top_x, 0, bottom_x, image_height - 1)``, or ``None`` if
        the history is too short or has too few valid detections.
    """
    if len(line_history) < window_size or image_width <= 0 or image_height <= 0:
        return None

    valid_lines = [line for line in line_history if line is not None]
    if len(valid_lines) < min_detections:
        return None

    center_y = (image_height - 1) / 2.0
    center_x_values = []
    dx_dy_values = []

    for x1, y1, x2, y2 in valid_lines:
        dy = y2 - y1
        if dy == 0:
            center_x = (x1 + x2) / 2.0
            dx_dy = 0.0
        else:
            dx_dy = (x2 - x1) / dy
            center_x = x1 + (center_y - y1) * dx_dy

        center_x_values.append(center_x)
        dx_dy_values.append(dx_dy)

    median_center_x = float(np.median(center_x_values))
    median_dx_dy = float(np.median(dx_dy_values))
    top_x = int(round(median_center_x - center_y * median_dx_dy))
    bottom_x = int(round(median_center_x + (image_height - 1 - center_y) * median_dx_dy))
    return top_x, 0, bottom_x, image_height - 1


def draw_reference_line(
    image: np.ndarray,
    offset_x: int = 0,
    angle_deg: float = 90,
) -> np.ndarray:
    """Draw a magenta reference line through the image center with an x offset.

    Args:
        image: Image to draw on. This array is modified in place.
        offset_x: Horizontal offset from the image center in pixels.
        angle_deg: Reference line angle in degrees, where 90 is vertical.

    Returns:
        The same image array with the reference line drawn on it.
    """
    img_height, img_width = image.shape[:2]
    ref_x = img_width // 2 + offset_x
    ref_y = img_height // 2

    if angle_deg == 90:
        cv2.line(image, (ref_x, 0), (ref_x, img_height), (255, 0, 255), 2)
    elif angle_deg == 0:
        cv2.line(image, (0, ref_y), (img_width, ref_y), (255, 0, 255), 2)
    else:
        angle_rad = math.radians(angle_deg)
        cot_a = math.cos(angle_rad) / math.sin(angle_rad)
        x_top = int(ref_x + (0 - ref_y) * cot_a)
        x_bottom = int(ref_x + (img_height - ref_y) * cot_a)
        cv2.line(image, (x_top, 0), (x_bottom, img_height), (255, 0, 255), 2)

    return image


def calculate_line_deviation(
    longest_line: tuple[int, int, int, int] | None,
    depth_image: np.ndarray,
    ref_x_offset: int = 0,
    ref_angle_deg: float = 90,
) -> tuple[float | None, float | None, float | None]:
    """Calculate line angle, horizontal pixel, and depth deviations.

    Args:
        longest_line: Detected line as ``(x1, y1, x2, y2)``, or ``None``.
        depth_image: Depth image used to sample depth at the image center row.
        ref_x_offset: Horizontal reference offset from the image center in pixels.
        ref_angle_deg: Reference line angle in degrees, where 90 is vertical.

    Returns:
        Tuple of ``(angle_deviation, horizontal_deviation, depth_at_center)``.
        Horizontal deviation and depth are ``None`` if the center-row line point
        is outside the image or has no valid nearby depth. All values are
        ``None`` if no line was provided.
    """
    if longest_line is None:
        return None, None, None

    x1, y1, x2, y2 = longest_line
    img_height, img_width = depth_image.shape[:2]
    ref_x = img_width // 2 + ref_x_offset
    angle_deviation = _calculate_line_angle_deviation(
        float(x1), float(y1), float(x2), float(y2), ref_angle_deg
    )
    reference_y_at_center = img_height // 2

    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        detected_x_at_center_y = int((reference_y_at_center - b) / m) if m != 0 else x1
    else:
        detected_x_at_center_y = x1

    if 0 <= detected_x_at_center_y < img_width:
        detected_depth_at_center_y = _find_min_depth_in_region(
            depth_image, detected_x_at_center_y, reference_y_at_center
        )
        if detected_depth_at_center_y > 0:
            horizontal_deviation = detected_x_at_center_y - ref_x
            return angle_deviation, horizontal_deviation, detected_depth_at_center_y

    return angle_deviation, None, None


def calculate_pixel_area(
    depth_in_mm: float,
    theta_horizontal: float,
    theta_vertical: float,
) -> tuple[float, float, float]:
    """Calculate physical pixel width, height, and area at the given depth.

    Args:
        depth_in_mm: Depth from the camera in millimeters.
        theta_horizontal: Horizontal angular size of one pixel in degrees.
        theta_vertical: Vertical angular size of one pixel in degrees.

    Returns:
        Tuple of ``(pixel_width, pixel_height, pixel_area)`` in millimeter units.
    """
    theta_h_rad = math.radians(theta_horizontal / 2)
    theta_v_rad = math.radians(theta_vertical / 2)
    pixel_width = 2 * depth_in_mm * math.tan(theta_h_rad)
    pixel_height = 2 * depth_in_mm * math.tan(theta_v_rad)
    return pixel_width, pixel_height, pixel_width * pixel_height


def _calculate_line_angle_deviation(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ref_angle_deg: float = 90,
) -> float:
    """Return signed line-angle deviation from the reference angle.

    Args:
        x1: First line endpoint x-coordinate.
        y1: First line endpoint y-coordinate.
        x2: Second line endpoint x-coordinate.
        y2: Second line endpoint y-coordinate.
        ref_angle_deg: Reference line angle in degrees, where 90 is vertical.

    Returns:
        Signed angular deviation in degrees, normalized to the ``[-90, 90]`` range.
    """
    if x2 != x1:
        dx = x2 - x1
        dy = y2 - y1
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_deviation = angle_deg - ref_angle_deg
        if angle_deviation > 90:
            angle_deviation -= 180
        elif angle_deviation < -90:
            angle_deviation += 180
        return -angle_deviation

    angle_deviation = -(90.0 - ref_angle_deg)
    if angle_deviation > 90:
        angle_deviation -= 180
    elif angle_deviation < -90:
        angle_deviation += 180
    return angle_deviation


def _find_min_depth_in_region(
    depth_image: np.ndarray,
    center_x: int,
    center_y: int,
) -> float:
    """Find the smallest non-zero depth in the 3x3 region around a pixel.

    Args:
        depth_image: Depth image to sample.
        center_x: Center pixel x-coordinate.
        center_y: Center pixel y-coordinate.

    Returns:
        Minimum non-zero depth in the 3x3 neighborhood, or ``0.0`` if none exists.
    """
    height, width = depth_image.shape
    min_depth = depth_image[center_y, center_x]
    if min_depth == 0:
        min_depth = np.inf

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y = center_y + dy
            x = center_x + dx
            if 0 <= y < height and 0 <= x < width:
                depth = depth_image[y, x]
                if depth != 0 and depth < min_depth:
                    min_depth = depth

    return 0.0 if min_depth == np.inf else float(min_depth)
