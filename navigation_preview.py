"""Depth-camera navigation aggregation and offline OpenCV preview."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    """Thresholds matching the ROS2 depth-camera navigation configuration."""

    publish_rate_hz: float = 10.0
    min_confidence: float = 50.0
    max_sample_age_ms: int = 300
    max_pair_time_difference_ms: int = 150
    max_angle_difference_deg: float = 5.0
    max_horizontal_difference_mm: float = 100.0


@dataclass(frozen=True)
class CameraNavigationSample:
    """Median edge result from one displayed depth-camera frame."""

    timestamp_ms: int
    angle_deviation: float
    horizontal_deviation: float
    depth_at_edge: float
    confidence: float


@dataclass(frozen=True)
class NavigationPair:
    """Combined result from a valid left/right camera pair."""

    angle_deviation: float
    horizontal_deviation: float
    confidence: float


@dataclass(frozen=True)
class NavigationMessage:
    """Offline preview of the ROS2 DepthCameraNavigation message."""

    timestamp_ms: int
    front_valid: bool
    front_angle_deviation: float
    front_horizontal_deviation: float
    front_confidence: float
    rear_valid: bool
    rear_angle_deviation: float
    rear_horizontal_deviation: float
    rear_confidence: float


NAVIGATION_CONFIG = NavigationConfig()


def navigation_sample_is_valid(
    sample: CameraNavigationSample | None,
    target_timestamp_ms: int,
    config: NavigationConfig,
) -> bool:
    """Return whether one recorded camera result is usable for navigation."""
    if sample is None:
        return False
    values = (
        sample.angle_deviation,
        sample.horizontal_deviation,
        sample.depth_at_edge,
        sample.confidence,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    if sample.depth_at_edge <= 0:
        return False
    if not 0 <= sample.confidence <= 100:
        return False
    if sample.confidence < config.min_confidence:
        return False

    # The offline viewer selects the nearest frame, which can be just before
    # or just after the selected master time.
    sample_age_ms = abs(target_timestamp_ms - sample.timestamp_ms)
    return sample_age_ms <= config.max_sample_age_ms


def combine_navigation_pair(
    left_sample: CameraNavigationSample | None,
    right_sample: CameraNavigationSample | None,
    target_timestamp_ms: int,
    config: NavigationConfig,
) -> NavigationPair | None:
    """Combine two valid and mutually consistent median edge results."""
    if not navigation_sample_is_valid(left_sample, target_timestamp_ms, config):
        return None
    if not navigation_sample_is_valid(right_sample, target_timestamp_ms, config):
        return None
    if left_sample is None or right_sample is None:
        return None

    timestamp_difference = abs(left_sample.timestamp_ms - right_sample.timestamp_ms)
    if timestamp_difference > config.max_pair_time_difference_ms:
        return None

    angle_difference = abs(left_sample.angle_deviation - right_sample.angle_deviation)
    horizontal_difference = abs(
        left_sample.horizontal_deviation - right_sample.horizontal_deviation
    )
    if angle_difference > config.max_angle_difference_deg:
        return None
    if horizontal_difference > config.max_horizontal_difference_mm:
        return None

    total_confidence = left_sample.confidence + right_sample.confidence
    if total_confidence <= 0:
        return None

    # Give the more confident camera more influence on the combined values.
    angle_deviation = (
        left_sample.angle_deviation * left_sample.confidence
        + right_sample.angle_deviation * right_sample.confidence
    ) / total_confidence
    horizontal_deviation = (
        left_sample.horizontal_deviation * left_sample.confidence
        + right_sample.horizontal_deviation * right_sample.confidence
    ) / total_confidence

    # Start with the weaker camera confidence and reduce it as disagreement grows.
    angle_agreement = 1.0 - angle_difference / config.max_angle_difference_deg
    horizontal_agreement = (
        1.0 - horizontal_difference / config.max_horizontal_difference_mm
    )
    pair_confidence = min(left_sample.confidence, right_sample.confidence)
    pair_confidence *= (angle_agreement + horizontal_agreement) / 2.0

    return NavigationPair(
        angle_deviation=angle_deviation,
        horizontal_deviation=horizontal_deviation,
        confidence=pair_confidence,
    )


def build_navigation_message(
    samples: dict[str, CameraNavigationSample],
    target_timestamp_ms: int,
    config: NavigationConfig = NAVIGATION_CONFIG,
) -> NavigationMessage:
    """Build front and rear navigation values for the selected recording time."""
    front_pair = combine_navigation_pair(
        samples.get("front_left"),
        samples.get("front_right"),
        target_timestamp_ms,
        config,
    )
    rear_pair = combine_navigation_pair(
        samples.get("rear_left"),
        samples.get("rear_right"),
        target_timestamp_ms,
        config,
    )

    return NavigationMessage(
        timestamp_ms=target_timestamp_ms,
        front_valid=front_pair is not None,
        front_angle_deviation=(
            front_pair.angle_deviation if front_pair is not None else 0.0
        ),
        front_horizontal_deviation=(
            front_pair.horizontal_deviation if front_pair is not None else 0.0
        ),
        front_confidence=front_pair.confidence if front_pair is not None else 0.0,
        rear_valid=rear_pair is not None,
        rear_angle_deviation=(
            rear_pair.angle_deviation if rear_pair is not None else 0.0
        ),
        rear_horizontal_deviation=(
            rear_pair.horizontal_deviation if rear_pair is not None else 0.0
        ),
        rear_confidence=rear_pair.confidence if rear_pair is not None else 0.0,
    )


def render_navigation_panel(
    message: NavigationMessage,
    config: NavigationConfig = NAVIGATION_CONFIG,
) -> np.ndarray:
    """Render the navigation message as a compact, easily readable panel."""
    width = 720
    height = 300
    panel = np.full((height, width, 3), (20, 20, 26), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    dim_color = (165, 165, 165)

    cv2.putText(
        panel,
        "DepthCameraNavigation preview",
        (18, 30),
        font,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    period_ms = 1000.0 / config.publish_rate_hz
    cv2.putText(
        panel,
        f"recording timestamp={message.timestamp_ms} ms   configured rate="
        f"{config.publish_rate_hz:g} Hz ({period_ms:.0f} ms)",
        (18, 54),
        font,
        0.48,
        dim_color,
        1,
        cv2.LINE_AA,
    )

    rows = [
        (
            "FRONT",
            message.front_valid,
            message.front_angle_deviation,
            message.front_horizontal_deviation,
            message.front_confidence,
        ),
        (
            "REAR",
            message.rear_valid,
            message.rear_angle_deviation,
            message.rear_horizontal_deviation,
            message.rear_confidence,
        ),
    ]
    for row_index, (label, valid, angle, horizontal, confidence) in enumerate(rows):
        top = 68 + row_index * 88
        bottom = top + 76
        status_color = (80, 220, 120) if valid else (80, 80, 255)
        cv2.rectangle(panel, (16, top), (width - 16, bottom), status_color, 2)
        cv2.putText(
            panel,
            f"{label}  {'VALID' if valid else 'INVALID'}",
            (30, top + 31),
            font,
            0.68,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if valid:
            value_text = (
                f"angle {angle:+.2f} deg     horizontal {horizontal:+.1f} mm"
                f"     confidence {confidence:.1f}%"
            )
        else:
            value_text = "No fresh, confident and consistent camera pair"
        cv2.putText(
            panel,
            value_text,
            (30, top + 60),
            font,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        f"Median deviations | min confidence {config.min_confidence:.0f}% | "
        f"max difference {config.max_angle_difference_deg:g} deg / "
        f"{config.max_horizontal_difference_mm:g} mm",
        (18, 278),
        font,
        0.46,
        dim_color,
        1,
        cv2.LINE_AA,
    )
    return panel
