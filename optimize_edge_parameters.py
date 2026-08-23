"""Deterministic edge-parameter tuning for recorded depth-camera data.

The optimizer evaluates the existing navigation validity rules without
modifying the camera YAML files or the navigation thresholds.  Frames passed
on the command line are recording/PNG frame numbers (not GUI master indices).
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

from analyze_recordings import (
    CAMERA_CONFIG_FILES,
    CAMERA_CONFIGS,
    FPS,
    CameraConfig,
    frame_number,
)
from helpers import (
    apply_canny_edge_detection,
    apply_depth_cutoff,
    calculate_horizontal_deviation_mm,
    calculate_line_deviation,
    calculate_median_line,
    calculate_temporal_line_score,
    draw_reference_line,
    filter_out_zero_boundaries,
    draw_long_line,
    find_longest_line_right,
    process_depth_image,
)
from navigation_preview import (
    NAVIGATION_CONFIG,
    CameraNavigationSample,
    NavigationConfig,
    combine_navigation_pair,
    navigation_sample_is_valid,
)


PAIR_CAMERAS: dict[str, tuple[str, str]] = {
    "front": ("front_left", "front_right"),
    "rear": ("rear_left", "rear_right"),
}

EXPECTED_BASELINE = {
    "front": {"valid_rate": 78.50467289719626, "transitions": 167, "longest_invalid_run": 96},
    "rear": {"valid_rate": 68.0805176132279, "transitions": 303, "longest_invalid_run": 163},
}
EXPECTED_BASELINE_FRAME_RANGE = (260, 1650)


@dataclass(frozen=True, order=True)
class EdgeParameters:
    """The numeric YAML edge-detection fields searched as one unit."""

    canny_min_val: int
    canny_max_val: int
    dilate_size: int
    hough_threshold: int
    min_line_length: int
    max_line_gap: int

    @classmethod
    def from_config(cls, config: CameraConfig) -> "EdgeParameters":
        return cls(
            canny_min_val=int(config.canny_min_val),
            canny_max_val=int(config.canny_max_val),
            dilate_size=int(config.dilate_size),
            hough_threshold=int(config.hough_threshold),
            min_line_length=int(config.min_line_length),
            max_line_gap=int(config.max_line_gap),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SearchSpace:
    canny_min_values: tuple[int, ...]
    canny_max_values: tuple[int, ...]
    dilate_sizes: tuple[int, ...]
    hough_thresholds: tuple[int, ...]
    min_line_lengths: tuple[int, ...]
    max_line_gaps: tuple[int, ...]

    def contains(self, parameters: EdgeParameters) -> bool:
        return (
            parameters.canny_min_val in self.canny_min_values
            and parameters.canny_max_val in self.canny_max_values
            and parameters.canny_min_val < parameters.canny_max_val
            and parameters.dilate_size in self.dilate_sizes
            and parameters.hough_threshold in self.hough_thresholds
            and parameters.min_line_length in self.min_line_lengths
            and parameters.max_line_gap in self.max_line_gaps
        )

    def all_candidates(self) -> tuple[EdgeParameters, ...]:
        candidates = (
            EdgeParameters(*values)
            for values in itertools.product(
                self.canny_min_values,
                self.canny_max_values,
                self.dilate_sizes,
                self.hough_thresholds,
                self.min_line_lengths,
                self.max_line_gaps,
            )
        )
        return tuple(sorted(item for item in candidates if self.contains(item)))


_COMMON_SPACE = {
    "canny_min_values": (20, 40, 60, 80, 100, 130, 160),
    "canny_max_values": (100, 120, 150, 180, 220, 280, 350),
    "dilate_sizes": (1, 3, 5),
    "hough_thresholds": (10, 15, 20, 25, 30, 40, 50),
}

FRONT_SEARCH_SPACE = SearchSpace(
    **_COMMON_SPACE,
    min_line_lengths=(75, 100, 125, 150, 175, 200, 225, 250),
    max_line_gaps=(10, 20, 30, 40, 50, 65, 80),
)
REAR_SEARCH_SPACE = SearchSpace(
    **_COMMON_SPACE,
    min_line_lengths=(20, 30, 40, 50, 60, 70, 80, 90, 100),
    max_line_gaps=(5, 10, 15, 20, 30, 40, 50),
)


def generate_global_candidates(
    space: SearchSpace,
    count: int,
    seed: int,
    mandatory: Iterable[EdgeParameters] = (),
) -> tuple[EdgeParameters, ...]:
    """Return a deterministic, unique sample while retaining mandatory values."""
    if count <= 0:
        raise ValueError("candidate count must be positive")
    universe = space.all_candidates()
    required = sorted(set(mandatory))
    if len(required) > count:
        raise ValueError("mandatory candidates exceed requested count")
    if count >= len(universe) + len([item for item in required if item not in universe]):
        return tuple(sorted(set(universe) | set(required)))
    remaining = [item for item in universe if item not in set(required)]
    sampled = random.Random(seed).sample(remaining, count - len(required))
    return tuple(sorted((*required, *sampled)))


def _neighbor_values(values: tuple[int, ...], value: int) -> tuple[int, ...]:
    if value not in values:
        nearest = sorted(values, key=lambda candidate: (abs(candidate - value), candidate))[:2]
        return tuple(sorted(nearest))
    index = values.index(value)
    return values[max(0, index - 1):min(len(values), index + 2)]


def generate_local_candidates(
    space: SearchSpace,
    seeds: Sequence[EdgeParameters],
    limit: int,
    random_seed: int = 42,
) -> tuple[EdgeParameters, ...]:
    """Sample the one-grid-step Cartesian neighborhoods of the supplied seeds."""
    if limit <= 0:
        return ()
    candidates: set[EdgeParameters] = set()
    for seed_item in seeds:
        local_space = SearchSpace(
            canny_min_values=_neighbor_values(space.canny_min_values, seed_item.canny_min_val),
            canny_max_values=_neighbor_values(space.canny_max_values, seed_item.canny_max_val),
            dilate_sizes=_neighbor_values(space.dilate_sizes, seed_item.dilate_size),
            hough_thresholds=_neighbor_values(space.hough_thresholds, seed_item.hough_threshold),
            min_line_lengths=_neighbor_values(space.min_line_lengths, seed_item.min_line_length),
            max_line_gaps=_neighbor_values(space.max_line_gaps, seed_item.max_line_gap),
        )
        candidates.update(local_space.all_candidates())
    ordered = sorted(candidates)
    if len(ordered) <= limit:
        return tuple(ordered)
    # A fixed seed avoids a lexicographic bias toward low thresholds.
    return tuple(sorted(random.Random(random_seed).sample(ordered, limit)))


def _frame_blocks(start_frame: int, end_frame: int, block_size: int) -> tuple[tuple[int, ...], ...]:
    if end_frame < start_frame:
        raise ValueError("end frame must not be before start frame")
    if block_size <= 0:
        raise ValueError("block size must be positive")
    return tuple(
        tuple(range(block_start, min(block_start + block_size - 1, end_frame) + 1))
        for block_start in range(start_frame, end_frame + 1, block_size)
    )


def split_frame_block_groups(
    start_frame: int,
    end_frame: int,
    block_size: int = 100,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    groups: dict[str, list[tuple[int, ...]]] = {"train": [], "validation": [], "test": []}
    names = ("train", "validation", "test")
    for index, block in enumerate(_frame_blocks(start_frame, end_frame, block_size)):
        groups[names[index % 3]].append(block)
    return {name: tuple(blocks) for name, blocks in groups.items()}


def split_frame_blocks(
    start_frame: int,
    end_frame: int,
    block_size: int = 100,
) -> dict[str, tuple[int, ...]]:
    """Split inclusive frame numbers into alternating train/validation/test blocks."""
    groups = split_frame_block_groups(start_frame, end_frame, block_size)
    return {
        name: tuple(frame for block in blocks for frame in block)
        for name, blocks in groups.items()
    }


@dataclass(frozen=True)
class FlagMetrics:
    valid_count: int
    total_count: int
    valid_rate: float
    transitions: int
    longest_invalid_run: int
    invalid_runs: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def calculate_flag_metrics(flags: Sequence[bool]) -> FlagMetrics:
    valid_count = sum(bool(value) for value in flags)
    transitions = sum(left != right for left, right in zip(flags, flags[1:]))
    longest = 0
    run = 0
    runs = 0
    for value in flags:
        if value:
            run = 0
        else:
            if run == 0:
                runs += 1
            run += 1
            longest = max(longest, run)
    total = len(flags)
    return FlagMetrics(
        valid_count=valid_count,
        total_count=total,
        valid_rate=(100.0 * valid_count / total) if total else 0.0,
        transitions=transitions,
        longest_invalid_run=longest,
        invalid_runs=runs,
    )


def calculate_block_metrics(block_flags: Sequence[Sequence[bool]]) -> FlagMetrics:
    """Aggregate flags while resetting transition/run state at block boundaries."""
    items = [calculate_flag_metrics(block) for block in block_flags]
    total = sum(item.total_count for item in items)
    valid = sum(item.valid_count for item in items)
    return FlagMetrics(
        valid_count=valid,
        total_count=total,
        valid_rate=(100.0 * valid / total) if total else 0.0,
        transitions=sum(item.transitions for item in items),
        longest_invalid_run=max((item.longest_invalid_run for item in items), default=0),
        invalid_runs=sum(item.invalid_runs for item in items),
    )


@dataclass(frozen=True)
class CameraFrameResult:
    frame_number: int
    sample: CameraNavigationSample
    current_line: tuple[int, int, int, int] | None
    median_line: tuple[int, int, int, int] | None
    center_shift_px: float | None
    angle_difference_deg: float | None


@dataclass(frozen=True)
class PairFrameResult:
    frame_number: int
    valid: bool
    pair_confidence: float
    pair_angle_deviation: float
    pair_horizontal_deviation: float
    left_sample: CameraNavigationSample
    right_sample: CameraNavigationSample
    angle_difference_deg: float
    horizontal_difference_mm: float
    failure_reason: str
    left_current_line: tuple[int, int, int, int] | None
    left_median_line: tuple[int, int, int, int] | None
    right_current_line: tuple[int, int, int, int] | None
    right_median_line: tuple[int, int, int, int] | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return result


@dataclass(frozen=True)
class PairEvaluation:
    pair: str
    parameters: EdgeParameters
    metrics: FlagMetrics
    frame_numbers: tuple[int, ...]
    flags: tuple[bool, ...]
    block_lengths: tuple[int, ...]
    failure_counts: tuple[tuple[str, int], ...]
    frames: tuple[PairFrameResult, ...] = ()

    def rank_key(self) -> tuple[object, ...]:
        """Smaller tuples rank first."""
        return (
            -self.metrics.valid_count,
            self.metrics.transitions,
            self.metrics.longest_invalid_run,
            self.parameters,
        )

    def to_dict(self, include_frames: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "pair": self.pair,
            "parameters": self.parameters.to_dict(),
            "metrics": self.metrics.to_dict(),
            "failure_counts": dict(self.failure_counts),
            "frame_numbers": list(self.frame_numbers),
            "flags": list(self.flags),
            "block_lengths": list(self.block_lengths),
        }
        if include_frames:
            result["frames"] = [item.to_dict() for item in self.frames]
        return result


def _parameters_config(config: CameraConfig, parameters: EdgeParameters) -> CameraConfig:
    return replace(
        config,
        canny_min_val=parameters.canny_min_val,
        canny_max_val=parameters.canny_max_val,
        dilate_size=parameters.dilate_size,
        hough_threshold=parameters.hough_threshold,
        min_line_length=parameters.min_line_length,
        max_line_gap=parameters.max_line_gap,
    )


class HeadlessEvaluator:
    """Evaluate recorded frames with the analyzer's algorithm but no rendering."""

    def __init__(
        self,
        recording_root: Path,
        start_frame: int,
        end_frame: int,
        navigation_config: NavigationConfig = NAVIGATION_CONFIG,
    ) -> None:
        cv2.setNumThreads(1)
        self.recording_root = recording_root.expanduser().resolve()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.navigation_config = replace(navigation_config)
        self.configs = {
            key: replace(config) for key, config in CAMERA_CONFIGS.items()
        }
        self.depth_paths: dict[str, dict[int, Path]] = {}
        self.frame_lists: dict[str, tuple[int, ...]] = {}
        self.positions: dict[str, dict[int, int]] = {}
        for camera_key, config in self.configs.items():
            depth_dir = self.recording_root / config.folder / "depth"
            paths = {
                frame_number(path): path
                for path in sorted(depth_dir.glob("depth_*.png"))
            }
            if not paths:
                raise FileNotFoundError(f"No depth frames found in {depth_dir}")
            frames = tuple(sorted(paths))
            self.depth_paths[camera_key] = paths
            self.frame_lists[camera_key] = frames
            self.positions[camera_key] = {
                frame: position for position, frame in enumerate(frames)
            }
            for required in (start_frame, end_frame):
                if required not in paths:
                    raise ValueError(
                        f"{camera_key} has no recording frame {required}"
                    )

    def _read_depth(self, camera_key: str, frame: int) -> np.ndarray:
        path = self.depth_paths[camera_key].get(frame)
        if path is None:
            raise ValueError(f"{camera_key} has no recording frame {frame}")
        image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise FileNotFoundError(path)
        return image

    @staticmethod
    def _crop_depth(depth_image: np.ndarray, config: CameraConfig) -> np.ndarray:
        if not config.roi_enabled:
            return depth_image.copy()
        height, width = depth_image.shape[:2]
        x0 = max(0, config.roi_x)
        y0 = max(0, config.roi_y)
        x1 = min(width, config.roi_x + config.roi_width)
        y1 = min(height, config.roi_y + config.roi_height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid ROI for {config.label}")
        return depth_image[y0:y1, x0:x1].copy()

    def _detect_frame(
        self,
        camera_key: str,
        frame: int,
        parameters: EdgeParameters,
    ) -> tuple[tuple[int, int, int, int] | None, np.ndarray, np.ndarray]:
        config = _parameters_config(self.configs[camera_key], parameters)
        depth_work = self._crop_depth(self._read_depth(camera_key, frame), config)
        edge_depth = apply_depth_cutoff(
            depth_work,
            depth_scale=config.depth_scale,
            min_depth_m=config.depth_cutoff_min_m,
            max_depth_m=config.depth_cutoff_max_m,
            enabled=config.depth_cutoff_enabled,
        )
        depth_colormap = process_depth_image(
            edge_depth, depth_alpha=config.depth_alpha
        )
        canny_edges = apply_canny_edge_detection(
            depth_colormap,
            min_val=config.canny_min_val,
            max_val=config.canny_max_val,
        )
        filtered_edges = filter_out_zero_boundaries(
            canny_edges, edge_depth, dilate_size=config.dilate_size
        )
        line = find_longest_line_right(
            filtered_edges,
            min_line_length=config.min_line_length,
            max_line_gap=config.max_line_gap,
            threshold=config.hough_threshold,
            right=config.find_rightmost_line,
        )
        if line is not None:
            line = tuple(int(value) for value in line)
        return line, depth_work, depth_colormap

    def _make_camera_result(
        self,
        camera_key: str,
        frame: int,
        parameters: EdgeParameters,
        current_line: tuple[int, int, int, int] | None,
        median_line: tuple[int, int, int, int] | None,
        depth_work: np.ndarray,
    ) -> CameraFrameResult:
        config = _parameters_config(self.configs[camera_key], parameters)
        median_angle = None
        median_horizontal_mm = None
        median_depth = None
        if median_line is not None:
            median_angle, median_horizontal, median_depth = calculate_line_deviation(
                median_line,
                depth_work,
                config.ref_offset_x,
                config.ref_angle_deg,
            )
            if median_horizontal is None:
                median_horizontal = -1.0
            if median_angle is not None:
                median_horizontal_mm = -1.0
                if median_depth is not None:
                    median_horizontal_mm = calculate_horizontal_deviation_mm(
                        horizontal_deviation_px=median_horizontal,
                        depth_image=depth_work,
                        ref_x_offset=config.ref_offset_x,
                        depth_scale=config.depth_scale,
                        focal_length_x=config.focal_length_x,
                    )
                    if median_horizontal_mm is None:
                        median_depth = None

        score, center_shift, angle_difference = calculate_temporal_line_score(
            current_line=current_line,
            median_line=median_line,
            image_height=depth_work.shape[0],
            max_tolerated_center_shift_px=config.confidence_max_center_shift_px,
            max_tolerated_angle_deviation_deg=config.confidence_max_angle_dev,
        )
        timestamp = (frame - 1) / FPS + config.offset_seconds
        sample = CameraNavigationSample(
            timestamp_ms=int(round(timestamp * 1000.0)),
            angle_deviation=float(median_angle if median_angle is not None else 0.0),
            horizontal_deviation=float(
                median_horizontal_mm if median_horizontal_mm is not None else 0.0
            ),
            depth_at_edge=float(median_depth if median_depth is not None else 0.0),
            confidence=float(score),
        )
        return CameraFrameResult(
            frame_number=frame,
            sample=sample,
            current_line=current_line,
            median_line=median_line,
            center_shift_px=center_shift,
            angle_difference_deg=angle_difference,
        )

    def evaluate_camera_sample(
        self,
        camera_key: str,
        frame_number_value: int,
        parameters: EdgeParameters,
    ) -> CameraNavigationSample:
        """Evaluate one sample, including its preceding median history."""
        result = self.evaluate_camera_block(
            camera_key, (frame_number_value,), parameters
        )[0]
        return result.sample

    def evaluate_camera_block(
        self,
        camera_key: str,
        target_frames: Sequence[int],
        parameters: EdgeParameters,
    ) -> tuple[CameraFrameResult, ...]:
        if not target_frames:
            return ()
        config = self.configs[camera_key]
        frames = self.frame_lists[camera_key]
        positions = self.positions[camera_key]
        target_set = set(target_frames)
        start_position = positions[target_frames[0]]
        end_position = positions[target_frames[-1]]
        history_size = (
            config.median_line_window_size if config.median_line_enabled else 0
        )
        process_start = max(0, start_position - history_size)
        history: deque[tuple[int, int, int, int] | None] = deque(
            maxlen=history_size or None
        )
        results: list[CameraFrameResult] = []
        for position in range(process_start, end_position + 1):
            frame = frames[position]
            current_line, depth_work, _ = self._detect_frame(
                camera_key, frame, parameters
            )
            median_line = None
            if config.median_line_enabled and len(history) == history_size:
                median_line = calculate_median_line(
                    list(history),
                    image_width=depth_work.shape[1],
                    image_height=depth_work.shape[0],
                    window_size=history_size,
                    min_detections=config.median_line_min_detections,
                )
            if frame in target_set:
                results.append(
                    self._make_camera_result(
                        camera_key,
                        frame,
                        parameters,
                        current_line,
                        median_line,
                        depth_work,
                    )
                )
            if config.median_line_enabled:
                history.append(current_line)
        result_by_frame = {item.frame_number: item for item in results}
        return tuple(result_by_frame[frame] for frame in target_frames)

    def _sample_failure_reasons(
        self,
        sample: CameraNavigationSample,
        target_ms: int,
    ) -> list[str]:
        reasons: list[str] = []
        values = (
            sample.angle_deviation,
            sample.horizontal_deviation,
            sample.depth_at_edge,
            sample.confidence,
        )
        if not all(math.isfinite(value) for value in values):
            reasons.append("non_finite")
        if sample.depth_at_edge <= 0:
            reasons.append("no_edge_depth")
        if not 0 <= sample.confidence <= 100:
            reasons.append("confidence_range")
        elif sample.confidence < self.navigation_config.min_confidence:
            reasons.append("low_confidence")
        if abs(target_ms - sample.timestamp_ms) > self.navigation_config.max_sample_age_ms:
            reasons.append("sample_age")
        return reasons

    def evaluate_pair(
        self,
        pair: str,
        parameters: EdgeParameters,
        blocks: Sequence[Sequence[int]],
        detailed: bool = False,
    ) -> PairEvaluation:
        left_key, right_key = PAIR_CAMERAS[pair]
        all_flags: list[bool] = []
        all_frames: list[int] = []
        block_flags: list[tuple[bool, ...]] = []
        frame_results: list[PairFrameResult] = []
        failures: Counter[str] = Counter()
        for block in blocks:
            if not block:
                continue
            left_results = self.evaluate_camera_block(left_key, block, parameters)
            right_results = self.evaluate_camera_block(right_key, block, parameters)
            current_flags: list[bool] = []
            for frame, left, right in zip(block, left_results, right_results):
                target_ms = int(round(((frame - 1) / FPS) * 1000.0))
                left_reasons = self._sample_failure_reasons(left.sample, target_ms)
                right_reasons = self._sample_failure_reasons(right.sample, target_ms)
                pair_result = combine_navigation_pair(
                    left.sample,
                    right.sample,
                    target_ms,
                    self.navigation_config,
                )
                angle_difference = abs(
                    left.sample.angle_deviation - right.sample.angle_deviation
                )
                horizontal_difference = abs(
                    left.sample.horizontal_deviation
                    - right.sample.horizontal_deviation
                )
                reasons: list[str] = []
                reasons.extend(f"left_{reason}" for reason in left_reasons)
                reasons.extend(f"right_{reason}" for reason in right_reasons)
                left_valid = navigation_sample_is_valid(
                    left.sample, target_ms, self.navigation_config
                )
                right_valid = navigation_sample_is_valid(
                    right.sample, target_ms, self.navigation_config
                )
                if left_valid and right_valid:
                    if (
                        abs(left.sample.timestamp_ms - right.sample.timestamp_ms)
                        > self.navigation_config.max_pair_time_difference_ms
                    ):
                        reasons.append("pair_timestamp")
                    if angle_difference > self.navigation_config.max_angle_difference_deg:
                        reasons.append("pair_angle")
                    if (
                        horizontal_difference
                        > self.navigation_config.max_horizontal_difference_mm
                    ):
                        reasons.append("pair_horizontal")
                valid = pair_result is not None
                if not valid and not reasons:
                    reasons.append("pair_other")
                failures.update(reasons)
                flag = bool(valid)
                current_flags.append(flag)
                all_flags.append(flag)
                all_frames.append(frame)
                if detailed:
                    frame_results.append(
                        PairFrameResult(
                            frame_number=frame,
                            valid=valid,
                            pair_confidence=(
                                float(pair_result.confidence)
                                if pair_result is not None
                                else 0.0
                            ),
                            pair_angle_deviation=(
                                float(pair_result.angle_deviation)
                                if pair_result is not None
                                else 0.0
                            ),
                            pair_horizontal_deviation=(
                                float(pair_result.horizontal_deviation)
                                if pair_result is not None
                                else 0.0
                            ),
                            left_sample=left.sample,
                            right_sample=right.sample,
                            angle_difference_deg=float(angle_difference),
                            horizontal_difference_mm=float(horizontal_difference),
                            failure_reason=";".join(reasons),
                            left_current_line=left.current_line,
                            left_median_line=left.median_line,
                            right_current_line=right.current_line,
                            right_median_line=right.median_line,
                        )
                    )
            block_flags.append(tuple(current_flags))
        return PairEvaluation(
            pair=pair,
            parameters=parameters,
            metrics=calculate_block_metrics(block_flags),
            frame_numbers=tuple(all_frames),
            flags=tuple(all_flags),
            block_lengths=tuple(len(block) for block in block_flags),
            failure_counts=tuple(sorted(failures.items())),
            frames=tuple(frame_results),
        )


_WORKER_EVALUATOR: HeadlessEvaluator | None = None


def _worker_initialize(recording_root: str, start_frame: int, end_frame: int) -> None:
    global _WORKER_EVALUATOR
    cv2.setNumThreads(1)
    _WORKER_EVALUATOR = HeadlessEvaluator(
        Path(recording_root), start_frame, end_frame
    )


def _worker_evaluate(
    task: tuple[str, EdgeParameters, tuple[tuple[int, ...], ...], bool],
) -> PairEvaluation:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("optimizer worker was not initialized")
    pair, parameters, blocks, detailed = task
    return _WORKER_EVALUATOR.evaluate_pair(
        pair, parameters, blocks, detailed=detailed
    )


def evaluate_candidates(
    recording_root: Path,
    start_frame: int,
    end_frame: int,
    pair: str,
    candidates: Sequence[EdgeParameters],
    blocks: Sequence[Sequence[int]],
    workers: int,
    label: str,
    detailed: bool = False,
) -> tuple[PairEvaluation, ...]:
    """Evaluate candidates in parallel and return a deterministic ordering."""
    unique = tuple(sorted(set(candidates)))
    normalized_blocks = tuple(tuple(block) for block in blocks)
    tasks = tuple((pair, item, normalized_blocks, detailed) for item in unique)
    started = time.monotonic()
    print(f"{label}: evaluating {len(tasks)} {pair} candidates on "
          f"{sum(map(len, normalized_blocks))} frames with {workers} worker(s)")
    if workers <= 1:
        evaluator = HeadlessEvaluator(recording_root, start_frame, end_frame)
        results = [
            evaluator.evaluate_pair(pair, item, normalized_blocks, detailed=detailed)
            for item in unique
        ]
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initialize,
            initargs=(str(recording_root), start_frame, end_frame),
        ) as executor:
            for completed, result in enumerate(
                executor.map(_worker_evaluate, tasks, chunksize=1), start=1
            ):
                results.append(result)
                if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                    elapsed = time.monotonic() - started
                    print(f"  {label}: {completed}/{len(tasks)} complete ({elapsed:.1f}s)")
    return tuple(sorted(results, key=lambda item: item.parameters))


def _rank_pair(evaluations: Iterable[PairEvaluation]) -> list[PairEvaluation]:
    return sorted(evaluations, key=lambda item: item.rank_key())


def _deduplicate_evaluations(
    evaluations: Iterable[PairEvaluation],
) -> tuple[PairEvaluation, ...]:
    by_parameters: dict[EdgeParameters, PairEvaluation] = {}
    for item in evaluations:
        by_parameters[item.parameters] = item
    return tuple(by_parameters[key] for key in sorted(by_parameters))


def _joint_rank_key(
    front: PairEvaluation,
    rear: PairEvaluation,
) -> tuple[object, ...]:
    simultaneous = sum(
        front_flag and rear_flag
        for front_flag, rear_flag in zip(front.flags, rear.flags)
    )
    return (
        -min(front.metrics.valid_count, rear.metrics.valid_count),
        -simultaneous,
        front.metrics.transitions + rear.metrics.transitions,
        max(
            front.metrics.longest_invalid_run,
            rear.metrics.longest_invalid_run,
        ),
        front.parameters,
        rear.parameters,
    )


@dataclass(frozen=True)
class OptimizationRun:
    start_frame: int
    end_frame: int
    seed: int
    workers: int
    recording_root: Path
    split_groups: dict[str, tuple[tuple[int, ...], ...]]
    baseline_parameters: dict[str, EdgeParameters]
    winner_parameters: dict[str, EdgeParameters]
    baseline_full: dict[str, PairEvaluation]
    winner_full: dict[str, PairEvaluation]
    winner_train: dict[str, PairEvaluation]
    winner_test: dict[str, PairEvaluation]
    winner_validation: dict[str, PairEvaluation]
    promoted_validation: dict[str, tuple[PairEvaluation, ...]]
    candidate_parameters: dict[str, dict[str, tuple[EdgeParameters, ...]]]
    global_candidate_count: int
    local_candidate_limit: int
    finalist_count: int
    baseline_drift_allowed: bool
    elapsed_seconds: float


class BaselineMismatchError(RuntimeError):
    """Raised before search when the frozen current pipeline misses the baseline."""


def _shared_current_parameters(pair: str) -> EdgeParameters:
    left_key, right_key = PAIR_CAMERAS[pair]
    left = EdgeParameters.from_config(CAMERA_CONFIGS[left_key])
    right = EdgeParameters.from_config(CAMERA_CONFIGS[right_key])
    if left != right:
        raise ValueError(
            f"{pair} cameras do not currently share numeric edge parameters: "
            f"{left.to_dict()} vs {right.to_dict()}"
        )
    return left


def _production_parameters(pair: str) -> EdgeParameters:
    return EdgeParameters(
        canny_min_val=130,
        canny_max_val=150,
        dilate_size=1,
        hough_threshold=20,
        min_line_length=150 if pair == "front" else 50,
        max_line_gap=40 if pair == "front" else 20,
    )


def run_search(
    recording_root: Path,
    start_frame: int = 260,
    end_frame: int = 1650,
    seed: int = 42,
    workers: int = 4,
    global_count: int = 400,
    local_count: int = 150,
    finalist_count: int = 20,
    allow_baseline_drift: bool = False,
) -> OptimizationRun:
    """Run staged pair searches and select a balanced joint winner."""
    started = time.monotonic()
    split_groups = split_frame_block_groups(start_frame, end_frame)
    spaces = {"front": FRONT_SEARCH_SPACE, "rear": REAR_SEARCH_SPACE}
    baseline_parameters = {
        pair: _shared_current_parameters(pair) for pair in PAIR_CAMERAS
    }
    evaluator = HeadlessEvaluator(recording_root, start_frame, end_frame)
    full_block = (tuple(range(start_frame, end_frame + 1)),)
    print("baseline preflight: evaluating the current working-tree configuration")
    baseline_full = {
        pair: evaluator.evaluate_pair(
            pair, baseline_parameters[pair], full_block, detailed=True
        )
        for pair in PAIR_CAMERAS
    }
    baseline_expectation_applies = (
        start_frame,
        end_frame,
    ) == EXPECTED_BASELINE_FRAME_RANGE
    mismatched_pairs: list[str] = []
    for pair, evaluation in baseline_full.items():
        expected = EXPECTED_BASELINE[pair]
        matches = baseline_expectation_applies and (
            math.isclose(
                evaluation.metrics.valid_rate,
                float(expected["valid_rate"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and evaluation.metrics.transitions == expected["transitions"]
            and evaluation.metrics.longest_invalid_run
            == expected["longest_invalid_run"]
        )
        if not baseline_expectation_applies:
            status = "planned baseline not applicable to this frame range"
        else:
            status = "matches expected" if matches else "DIFFERS from expected"
        if baseline_expectation_applies and not matches:
            mismatched_pairs.append(pair)
        print(
            f"  {pair}: {evaluation.metrics.valid_rate:.3f}% valid, "
            f"{evaluation.metrics.transitions} transitions, longest invalid "
            f"{evaluation.metrics.longest_invalid_run} ({status})"
        )
    if mismatched_pairs and not allow_baseline_drift:
        raise BaselineMismatchError(
            "baseline preflight failed for "
            + ", ".join(mismatched_pairs)
            + "; search was not started. Restore the planned frozen pipeline "
            "or pass --allow-baseline-drift to tune the current state explicitly."
        )
    validation_results: dict[str, tuple[PairEvaluation, ...]] = {}
    candidate_parameters: dict[str, dict[str, tuple[EdgeParameters, ...]]] = {}

    for pair_index, pair in enumerate(("front", "rear")):
        mandatory = (baseline_parameters[pair], _production_parameters(pair))
        global_candidates = generate_global_candidates(
            spaces[pair],
            count=global_count,
            seed=seed + pair_index,
            mandatory=mandatory,
        )
        global_results = evaluate_candidates(
            recording_root,
            start_frame,
            end_frame,
            pair,
            global_candidates,
            split_groups["train"],
            workers,
            label="global search",
        )
        top_global = _rank_pair(global_results)[:5]
        local_candidates = generate_local_candidates(
            spaces[pair],
            seeds=tuple(item.parameters for item in top_global),
            limit=local_count,
            random_seed=seed + pair_index,
        )
        unseen_local = tuple(
            item for item in local_candidates if item not in set(global_candidates)
        )
        local_results = evaluate_candidates(
            recording_root,
            start_frame,
            end_frame,
            pair,
            unseen_local,
            split_groups["train"],
            workers,
            label="local refinement",
        ) if unseen_local else ()
        combined_training = _deduplicate_evaluations(
            (*global_results, *local_results)
        )
        training_shortlist = _rank_pair(combined_training)[:finalist_count]
        shortlist_parameters = {item.parameters for item in training_shortlist}
        shortlist_parameters.update(mandatory)
        validation = evaluate_candidates(
            recording_root,
            start_frame,
            end_frame,
            pair,
            tuple(shortlist_parameters),
            split_groups["validation"],
            workers,
            label="validation",
            detailed=False,
        )
        # Joint-rank every training finalist on validation. Do not independently
        # prune here: simultaneous front/rear validity is part of the selector.
        validation_results[pair] = validation
        candidate_parameters[pair] = {
            "global": global_candidates,
            "local_refinement": unseen_local,
            "validation": tuple(sorted(shortlist_parameters)),
        }

    front_candidates = validation_results["front"]
    rear_candidates = validation_results["rear"]
    winner_front, winner_rear = min(
        itertools.product(front_candidates, rear_candidates),
        key=lambda items: _joint_rank_key(items[0], items[1]),
    )
    winners = {
        "front": winner_front.parameters,
        "rear": winner_rear.parameters,
    }

    winner_full = {
        pair: evaluator.evaluate_pair(pair, winners[pair], full_block, detailed=True)
        for pair in PAIR_CAMERAS
    }
    winner_test = {
        pair: evaluator.evaluate_pair(
            pair, winners[pair], split_groups["test"], detailed=False
        )
        for pair in PAIR_CAMERAS
    }
    winner_train = {
        pair: evaluator.evaluate_pair(
            pair, winners[pair], split_groups["train"], detailed=False
        )
        for pair in PAIR_CAMERAS
    }
    # Retain the exact validation evaluations used for joint selection.
    winner_validation = {
        "front": winner_front,
        "rear": winner_rear,
    }
    return OptimizationRun(
        start_frame=start_frame,
        end_frame=end_frame,
        seed=seed,
        workers=workers,
        recording_root=recording_root.resolve(),
        split_groups=split_groups,
        baseline_parameters=baseline_parameters,
        winner_parameters=winners,
        baseline_full=baseline_full,
        winner_full=winner_full,
        winner_train=winner_train,
        winner_test=winner_test,
        winner_validation=winner_validation,
        promoted_validation=validation_results,
        candidate_parameters=candidate_parameters,
        global_candidate_count=global_count,
        local_candidate_limit=local_count,
        finalist_count=finalist_count,
        baseline_drift_allowed=allow_baseline_drift,
        elapsed_seconds=time.monotonic() - started,
    )


def _protected_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parent
    return tuple(
        [root / filename for filename in CAMERA_CONFIG_FILES.values()]
        + [root / "navigation_preview.py", root / "helpers.py"]
    )


def _file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _selection_pool_size() -> int | None:
    match = re.search(
        r"find_top_hough_lines\([\s\S]*?limit\s*=\s*(\d+)",
        inspect.getsource(find_longest_line_right),
    )
    return int(match.group(1)) if match else None


def _space_to_dict(space: SearchSpace) -> dict[str, list[int]]:
    return {
        key: list(value)
        for key, value in asdict(space).items()
    }


def _baseline_check(run: OptimizationRun) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}
    applicable = (
        run.start_frame,
        run.end_frame,
    ) == EXPECTED_BASELINE_FRAME_RANGE
    for pair, evaluation in run.baseline_full.items():
        expected = EXPECTED_BASELINE[pair]
        actual = evaluation.metrics
        matches = applicable and (
            math.isclose(
                actual.valid_rate,
                float(expected["valid_rate"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and actual.transitions == expected["transitions"]
            and actual.longest_invalid_run == expected["longest_invalid_run"]
        )
        checks[pair] = {
            "applicable": applicable,
            "expected_frame_range": {
                "start": EXPECTED_BASELINE_FRAME_RANGE[0],
                "end": EXPECTED_BASELINE_FRAME_RANGE[1],
                "inclusive": True,
            },
            "matches_expected": matches if applicable else None,
            "expected": expected,
            "actual": actual.to_dict(),
        }
    return checks


def _joint_summary(front: PairEvaluation, rear: PairEvaluation) -> dict[str, object]:
    both = sum(
        front_flag and rear_flag
        for front_flag, rear_flag in zip(front.flags, rear.flags)
    )
    total = min(len(front.flags), len(rear.flags))
    return {
        "minimum_pair_valid_rate": min(
            front.metrics.valid_rate, rear.metrics.valid_rate
        ),
        "both_valid_count": both,
        "both_valid_rate": 100.0 * both / total if total else 0.0,
        "total_transitions": (
            front.metrics.transitions + rear.metrics.transitions
        ),
        "worst_longest_invalid_run": max(
            front.metrics.longest_invalid_run,
            rear.metrics.longest_invalid_run,
        ),
    }


def run_to_json_dict(
    run: OptimizationRun,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> dict[str, object]:
    split_counts = {
        name: sum(map(len, blocks)) for name, blocks in run.split_groups.items()
    }
    return {
        "schema_version": 1,
        "recording_root": str(run.recording_root),
        "frame_range": {"start": run.start_frame, "end": run.end_frame, "inclusive": True},
        "seed": run.seed,
        "workers": run.workers,
        "elapsed_seconds": run.elapsed_seconds,
        "search": {
            "requested": {
                "global_candidates_per_pair": run.global_candidate_count,
                "local_refinements_per_pair_maximum": run.local_candidate_limit,
                "training_finalists_per_pair": run.finalist_count,
            },
            "actual_counts": {
                pair: {
                    stage: len(parameters)
                    for stage, parameters in stages.items()
                }
                for pair, stages in run.candidate_parameters.items()
            },
            "candidate_parameters": {
                pair: {
                    stage: [item.to_dict() for item in parameters]
                    for stage, parameters in stages.items()
                }
                for pair, stages in run.candidate_parameters.items()
            },
            "mandatory_candidates": {
                pair: {
                    "current": run.baseline_parameters[pair].to_dict(),
                    "ros_default": _production_parameters(pair).to_dict(),
                }
                for pair in PAIR_CAMERAS
            },
        },
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "frozen_navigation_config": asdict(NAVIGATION_CONFIG),
        "frozen_camera_configs": {
            key: asdict(config) for key, config in CAMERA_CONFIGS.items()
        },
        "line_selection_candidate_count": _selection_pool_size(),
        "search_spaces": {
            "front": _space_to_dict(FRONT_SEARCH_SPACE),
            "rear": _space_to_dict(REAR_SEARCH_SPACE),
        },
        "split_counts": split_counts,
        "split_blocks": {
            name: [[block[0], block[-1]] for block in blocks]
            for name, blocks in run.split_groups.items()
        },
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_files_unchanged": hashes_before == hashes_after,
        "baseline_check": _baseline_check(run),
        "baseline_drift_allowed": run.baseline_drift_allowed,
        "baseline": {
            pair: evaluation.to_dict() for pair, evaluation in run.baseline_full.items()
        },
        "winner_parameters": {
            pair: parameters.to_dict()
            for pair, parameters in run.winner_parameters.items()
        },
        "winner": {
            "train": {
                pair: item.to_dict() for pair, item in run.winner_train.items()
            },
            "validation": {
                pair: item.to_dict()
                for pair, item in run.winner_validation.items()
            },
            "test": {
                pair: item.to_dict() for pair, item in run.winner_test.items()
            },
            "full": {
                pair: item.to_dict() for pair, item in run.winner_full.items()
            },
            "joint_validation": _joint_summary(
                run.winner_validation["front"], run.winner_validation["rear"]
            ),
            "joint_test": _joint_summary(
                run.winner_test["front"], run.winner_test["rear"]
            ),
            "joint_full": _joint_summary(
                run.winner_full["front"], run.winner_full["rear"]
            ),
        },
        "promoted_validation_candidates": {
            pair: [item.to_dict() for item in _rank_pair(items)]
            for pair, items in run.promoted_validation.items()
        },
    }


def _metrics_table_row(label: str, evaluation: PairEvaluation) -> str:
    metrics = evaluation.metrics
    return (
        f"| {label} | {metrics.valid_count}/{metrics.total_count} "
        f"({metrics.valid_rate:.2f}%) | {metrics.transitions} | "
        f"{metrics.longest_invalid_run} | {metrics.invalid_runs} |"
    )


def render_markdown_report(run: OptimizationRun) -> str:
    checks = _baseline_check(run)
    if not any(check["applicable"] for check in checks.values()):
        baseline_note = (
            "The planned baseline applies only to frames 260–1650; this custom "
            "range was recorded without comparing it to those metrics."
        )
    elif any(not check["matches_expected"] for check in checks.values()):
        baseline_note = (
            "A mismatch means the working camera configuration changed after "
            "the plan was recorded. This completed run used the explicit "
            "baseline-drift override and recorded the current files without "
            "rewriting them."
        )
    else:
        baseline_note = "The required baseline was reproduced before search."
    lines = [
        "# Edge-parameter stability tuning",
        "",
        f"Recording: `{run.recording_root}`  ",
        f"Recording frames: **{run.start_frame}–{run.end_frame} inclusive**  ",
        f"Seed/workers/runtime: `{run.seed}` / `{run.workers}` / `{run.elapsed_seconds:.1f}s`",
        "",
        "Navigation gates, preprocessing, ROI, reference lines, median settings, side selection, and the existing Hough candidate-pool behavior were frozen.",
        "",
        "## Baseline preflight",
        "",
    ]
    for pair in ("front", "rear"):
        check = checks[pair]
        metrics = run.baseline_full[pair].metrics
        if not check["applicable"]:
            lines.append(
                f"- {pair.title()}: {metrics.valid_rate:.2f}% valid, "
                f"{metrics.transitions} transitions, longest invalid run "
                f"{metrics.longest_invalid_run}. The planned baseline applies "
                "only to frames 260–1650."
            )
        else:
            status = "matches" if check["matches_expected"] else "does **not** match"
            lines.append(
                f"- {pair.title()} {status} the planned baseline: "
                f"{metrics.valid_rate:.2f}% valid, {metrics.transitions} transitions, "
                f"longest invalid run {metrics.longest_invalid_run}."
            )
    lines += [
        "",
        baseline_note,
        "",
        "## Selected parameters",
        "",
        "| Pair | Canny min/max | Dilation | Hough threshold | Min length | Max gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in ("front", "rear"):
        item = run.winner_parameters[pair]
        lines.append(
            f"| {pair.title()} | {item.canny_min_val}/{item.canny_max_val} | "
            f"{item.dilate_size} | {item.hough_threshold} | "
            f"{item.min_line_length} | {item.max_line_gap} |"
        )
    lines += [
        "",
        "## Baseline versus winner (full interval)",
        "",
        "| Series | Valid | Transitions | Longest invalid | Invalid runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair in ("front", "rear"):
        lines.append(_metrics_table_row(f"{pair.title()} baseline", run.baseline_full[pair]))
        lines.append(_metrics_table_row(f"{pair.title()} winner", run.winner_full[pair]))
    lines += [
        "",
        "## Winner by split",
        "",
        "| Series | Valid | Transitions | Longest invalid | Invalid runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, evaluations in (
        ("train", run.winner_train),
        ("validation", run.winner_validation),
        ("test", run.winner_test),
    ):
        for pair in ("front", "rear"):
            lines.append(_metrics_table_row(f"{pair.title()} {split}", evaluations[pair]))
    lines += ["", "## Winner failure counts", ""]
    for pair in ("front", "rear"):
        lines += [f"### {pair.title()}", "", "| Reason | Frames |", "|---|---:|"]
        failure_counts = dict(run.winner_full[pair].failure_counts)
        if failure_counts:
            lines.extend(
                f"| `{reason}` | {count} |"
                for reason, count in sorted(failure_counts.items())
            )
        else:
            lines.append("| none | 0 |")
        lines.append("")
    lines += [
        "## Promoted validation candidates",
        "",
        "The JSON file contains every promoted candidate. The five highest-ranked candidates per pair are summarized here.",
        "",
    ]
    for pair in ("front", "rear"):
        lines += [
            f"### {pair.title()}",
            "",
            "| Rank | Parameters | Valid | Transitions | Longest invalid |",
            "|---:|---|---:|---:|---:|",
        ]
        for rank, item in enumerate(_rank_pair(run.promoted_validation[pair])[:5], start=1):
            lines.append(
                f"| {rank} | `{item.parameters.to_dict()}` | "
                f"{item.metrics.valid_rate:.2f}% | {item.metrics.transitions} | "
                f"{item.metrics.longest_invalid_run} |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "The interval contains positive examples only. Higher validity and fewer dropouts do not prove that the selected line is the rail; inspect the generated contact sheets before applying these values to production.",
        "",
        "The four camera YAML files were intentionally left unchanged.",
        "",
    ]
    return "\n".join(lines)


def write_winner_csv(run: OptimizationRun, path: Path) -> None:
    front_by_frame = {
        item.frame_number: item for item in run.winner_full["front"].frames
    }
    rear_by_frame = {
        item.frame_number: item for item in run.winner_full["rear"].frames
    }
    fieldnames = [
        "frame",
        "target_timestamp_ms",
        "front_valid",
        "rear_valid",
        "both_valid",
        "front_pair_confidence",
        "rear_pair_confidence",
        "front_pair_angle_deviation_deg",
        "rear_pair_angle_deviation_deg",
        "front_pair_horizontal_deviation_mm",
        "rear_pair_horizontal_deviation_mm",
        "front_angle_difference_deg",
        "rear_angle_difference_deg",
        "front_horizontal_difference_mm",
        "rear_horizontal_difference_mm",
        *[
            f"{camera}_{field}"
            for camera in (
                "front_left",
                "front_right",
                "rear_left",
                "rear_right",
            )
            for field in (
                "timestamp_ms",
                "confidence",
                "angle_deviation_deg",
                "horizontal_deviation_mm",
                "depth_at_edge",
            )
        ],
        "front_failure_reason",
        "rear_failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(run.start_frame, run.end_frame + 1):
            front = front_by_frame[frame]
            rear = rear_by_frame[frame]
            row: dict[str, object] = {
                "frame": frame,
                "target_timestamp_ms": int(round(((frame - 1) / FPS) * 1000.0)),
                "front_valid": int(front.valid),
                "rear_valid": int(rear.valid),
                "both_valid": int(front.valid and rear.valid),
                "front_pair_confidence": f"{front.pair_confidence:.6f}",
                "rear_pair_confidence": f"{rear.pair_confidence:.6f}",
                "front_pair_angle_deviation_deg": f"{front.pair_angle_deviation:.6f}",
                "rear_pair_angle_deviation_deg": f"{rear.pair_angle_deviation:.6f}",
                "front_pair_horizontal_deviation_mm": f"{front.pair_horizontal_deviation:.6f}",
                "rear_pair_horizontal_deviation_mm": f"{rear.pair_horizontal_deviation:.6f}",
                "front_angle_difference_deg": f"{front.angle_difference_deg:.6f}",
                "rear_angle_difference_deg": f"{rear.angle_difference_deg:.6f}",
                "front_horizontal_difference_mm": f"{front.horizontal_difference_mm:.6f}",
                "rear_horizontal_difference_mm": f"{rear.horizontal_difference_mm:.6f}",
                "front_failure_reason": front.failure_reason,
                "rear_failure_reason": rear.failure_reason,
            }
            samples = {
                "front_left": front.left_sample,
                "front_right": front.right_sample,
                "rear_left": rear.left_sample,
                "rear_right": rear.right_sample,
            }
            for camera, sample in samples.items():
                row[f"{camera}_timestamp_ms"] = sample.timestamp_ms
                row[f"{camera}_confidence"] = f"{sample.confidence:.6f}"
                row[f"{camera}_angle_deviation_deg"] = (
                    f"{sample.angle_deviation:.6f}"
                )
                row[f"{camera}_horizontal_deviation_mm"] = (
                    f"{sample.horizontal_deviation:.6f}"
                )
                row[f"{camera}_depth_at_edge"] = f"{sample.depth_at_edge:.6f}"
            writer.writerow(row)


def _invalid_run_ranges(evaluation: PairEvaluation) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for frame, valid in zip(evaluation.frame_numbers, evaluation.flags):
        if not valid:
            if start is None or (previous is not None and frame != previous + 1):
                if start is not None and previous is not None:
                    runs.append((start, previous))
                start = frame
            previous = frame
        elif start is not None:
            if previous is not None:
                runs.append((start, previous))
            start = None
            previous = None
    if start is not None and previous is not None:
        runs.append((start, previous))
    return sorted(runs, key=lambda item: (-(item[1] - item[0] + 1), item[0]))


def _contact_frames(run: OptimizationRun) -> tuple[int, ...]:
    frames: set[int] = set()
    for block in run.split_groups["test"]:
        frames.update((block[0], block[(len(block) - 1) // 2], block[-1]))
    return tuple(sorted(frames))


def _pair_frame_map(evaluation: PairEvaluation) -> dict[int, PairFrameResult]:
    return {item.frame_number: item for item in evaluation.frames}


def _contact_tile(
    evaluator: HeadlessEvaluator,
    pair: str,
    camera_key: str,
    frame: int,
    parameters: EdgeParameters,
    pair_result: PairFrameResult,
    mode: str,
) -> np.ndarray:
    _, _, image = evaluator._detect_frame(camera_key, frame, parameters)
    config = evaluator.configs[camera_key]
    image = draw_reference_line(
        image.copy(), config.ref_offset_x, config.ref_angle_deg
    )
    is_left = camera_key == PAIR_CAMERAS[pair][0]
    current_line = (
        pair_result.left_current_line if is_left else pair_result.right_current_line
    )
    median_line = (
        pair_result.left_median_line if is_left else pair_result.right_median_line
    )
    sample = pair_result.left_sample if is_left else pair_result.right_sample
    if current_line is not None:
        draw_long_line(image, *current_line, color=(0, 255, 255), thickness=2)
    if median_line is not None:
        draw_long_line(image, *median_line, color=(255, 255, 255), thickness=2)
    target_width = 360
    scale = target_width / image.shape[1]
    resized = cv2.resize(
        image,
        (target_width, max(70, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    bar = np.zeros((72, target_width, 3), dtype=np.uint8)
    output = np.vstack((resized, bar))
    color = (90, 230, 110) if pair_result.valid else (90, 90, 255)
    y0 = resized.shape[0]
    cv2.putText(
        output,
        f"{mode} | {camera_key} | frame {frame}",
        (6, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"pair {'VALID' if pair_result.valid else 'INVALID'} | camera conf {sample.confidence:.1f}%",
        (6, y0 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Canny {parameters.canny_min_val}/{parameters.canny_max_val} D{parameters.dilate_size} H{parameters.hough_threshold} L{parameters.min_line_length} G{parameters.max_line_gap}",
        (6, y0 + 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (175, 175, 175),
        1,
        cv2.LINE_AA,
    )
    return output


def _tile_sheet(tiles: Sequence[np.ndarray], columns: int = 4) -> np.ndarray:
    if not tiles:
        return np.zeros((100, 400, 3), dtype=np.uint8)
    tile_width = max(item.shape[1] for item in tiles)
    tile_height = max(item.shape[0] for item in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3), 18, dtype=np.uint8
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height:row * tile_height + tile.shape[0],
            column * tile_width:column * tile_width + tile.shape[1],
        ] = tile
    return sheet


def write_contact_sheets(run: OptimizationRun, output_dir: Path) -> list[Path]:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    evaluator = HeadlessEvaluator(
        run.recording_root, run.start_frame, run.end_frame
    )
    written: list[Path] = []
    representative = _contact_frames(run)
    for pair, camera_keys in PAIR_CAMERAS.items():
        baseline_map = _pair_frame_map(run.baseline_full[pair])
        winner_map = _pair_frame_map(run.winner_full[pair])
        for camera_key in camera_keys:
            tiles: list[np.ndarray] = []
            for frame in representative:
                tiles.append(
                    _contact_tile(
                        evaluator,
                        pair,
                        camera_key,
                        frame,
                        run.baseline_parameters[pair],
                        baseline_map[frame],
                        "baseline",
                    )
                )
                tiles.append(
                    _contact_tile(
                        evaluator,
                        pair,
                        camera_key,
                        frame,
                        run.winner_parameters[pair],
                        winner_map[frame],
                        "winner",
                    )
                )
            path = contact_dir / f"representative_{camera_key}.png"
            if not cv2.imwrite(str(path), _tile_sheet(tiles)):
                raise RuntimeError(f"Could not write {path}")
            written.append(path)

        boundary_frames: set[int] = set()
        for start, end in _invalid_run_ranges(run.winner_full[pair])[:3]:
            midpoint = (start + end) // 2
            boundary_frames.update(
                frame
                for frame in (start - 1, start, midpoint, end, end + 1)
                if run.start_frame <= frame <= run.end_frame
            )
        boundary_tiles: list[np.ndarray] = []
        for frame in sorted(boundary_frames):
            for camera_key in camera_keys:
                boundary_tiles.append(
                    _contact_tile(
                        evaluator,
                        pair,
                        camera_key,
                        frame,
                        run.baseline_parameters[pair],
                        baseline_map[frame],
                        "baseline",
                    )
                )
                boundary_tiles.append(
                    _contact_tile(
                        evaluator,
                        pair,
                        camera_key,
                        frame,
                        run.winner_parameters[pair],
                        winner_map[frame],
                        "winner",
                    )
                )
        path = contact_dir / f"worst_dropouts_{pair}.png"
        if not cv2.imwrite(str(path), _tile_sheet(boundary_tiles)):
            raise RuntimeError(f"Could not write {path}")
        written.append(path)
    return written


def write_outputs(
    run: OptimizationRun,
    output_dir: Path,
    hashes_before: dict[str, str],
) -> tuple[dict[str, str], list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(render_markdown_report(run), encoding="utf-8")
    csv_path = output_dir / "winner_frames.csv"
    write_winner_csv(run, csv_path)
    contact_paths = write_contact_sheets(run, output_dir)
    hashes_after = _file_hashes(_protected_paths())
    if hashes_after != hashes_before:
        raise RuntimeError(
            "A protected YAML/navigation/helper file changed during optimization"
        )
    results = run_to_json_dict(run, hashes_before, hashes_after)
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashes_after, [results_path, report_path, csv_path, *contact_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search shared front/rear edge parameters for stable navigation "
            "validity without changing YAML or validity thresholds."
        )
    )
    parser.add_argument(
        "recording_root",
        type=Path,
        help="Recording root containing the four depth-camera folders.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=260,
        help="First PNG recording frame number, inclusive (default: 260).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=1650,
        help="Last PNG recording frame number, inclusive (default: 1650).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: tuning_results/<recording-name>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for global sampling and local refinement (default: 42).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Worker processes, 1-8; defaults to min(8, CPU count). "
            "OpenCV uses one thread per worker."
        ),
    )
    parser.add_argument(
        "--global-candidates",
        type=int,
        default=400,
        help="Global candidates per pair (default: 400).",
    )
    parser.add_argument(
        "--local-candidates",
        type=int,
        default=150,
        help="Local-refinement candidates per pair (default: 150).",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=20,
        help="Training finalists promoted to validation (default: 20).",
    )
    parser.add_argument(
        "--allow-baseline-drift",
        action="store_true",
        help=(
            "Continue when the live baseline differs from the planned metrics. "
            "By default, preflight stops before search."
        ),
    )
    args = parser.parse_args()
    if args.end_frame < args.start_frame:
        parser.error("--end-frame must not be before --start-frame")
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.global_candidates < 2:
        parser.error("--global-candidates must be at least 2")
    if args.local_candidates < 0:
        parser.error("--local-candidates must be non-negative")
    if args.finalists < 1:
        parser.error("--finalists must be positive")
    return args


def main() -> None:
    args = parse_args()
    recording_root = args.recording_root.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("tuning_results") / recording_root.name
    hashes_before = _file_hashes(_protected_paths())
    print("Edge-parameter stability optimizer")
    print(f"Recording: {recording_root}")
    print(f"Frames: {args.start_frame}..{args.end_frame} inclusive")
    print(f"Navigation gates (frozen): {asdict(NAVIGATION_CONFIG)}")
    print(f"Current line-selection pool: {_selection_pool_size()}")
    try:
        run = run_search(
            recording_root=recording_root,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            seed=args.seed,
            workers=args.workers,
            global_count=args.global_candidates,
            local_count=args.local_candidates,
            finalist_count=args.finalists,
            allow_baseline_drift=args.allow_baseline_drift,
        )
    except BaselineMismatchError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    _, written = write_outputs(run, output_dir, hashes_before)
    print("Selected parameters:")
    for pair in ("front", "rear"):
        metrics = run.winner_full[pair].metrics
        print(
            f"  {pair}: {run.winner_parameters[pair].to_dict()} -> "
            f"{metrics.valid_rate:.2f}% valid, {metrics.transitions} transitions, "
            f"longest invalid {metrics.longest_invalid_run}"
        )
    print("Outputs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
