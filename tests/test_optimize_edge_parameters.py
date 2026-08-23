from __future__ import annotations

from contextlib import redirect_stdout
import csv
from dataclasses import fields
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import navigation_preview

from analyze_recordings import (
    CAMERA_CONFIG_FILES,
    CAMERA_CONFIGS,
    CAMERA_ORDER,
    RecordingAnalyzer,
)
from optimize_edge_parameters import (
    FRONT_SEARCH_SPACE,
    REAR_SEARCH_SPACE,
    EdgeParameters,
    HeadlessEvaluator,
    PairEvaluation,
    PairFrameResult,
    SearchSpace,
    calculate_block_metrics,
    calculate_flag_metrics,
    generate_global_candidates,
    generate_local_candidates,
    split_frame_blocks,
    write_winner_csv,
)


def _parameters_for_camera(camera_key: str) -> EdgeParameters:
    config = CAMERA_CONFIGS[camera_key]
    return EdgeParameters(
        canny_min_val=config.canny_min_val,
        canny_max_val=config.canny_max_val,
        dilate_size=config.dilate_size,
        hough_threshold=config.hough_threshold,
        min_line_length=config.min_line_length,
        max_line_gap=config.max_line_gap,
    )


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(
            canny_min_values=(20, 40, 100),
            canny_max_values=(100, 120),
            dilate_sizes=(1, 3),
            hough_thresholds=(10, 20),
            min_line_lengths=(75, 100),
            max_line_gaps=(10, 20),
        )

    def test_edge_parameters_to_dict_uses_stable_field_names(self) -> None:
        parameters = EdgeParameters(20, 120, 1, 20, 150, 65)

        self.assertEqual(
            parameters.to_dict(),
            {
                "canny_min_val": 20,
                "canny_max_val": 120,
                "dilate_size": 1,
                "hough_threshold": 20,
                "min_line_length": 150,
                "max_line_gap": 65,
            },
        )
        self.assertEqual(
            tuple(field.name for field in fields(parameters)),
            tuple(parameters.to_dict()),
        )

    def test_global_sampling_is_deterministic_unique_and_keeps_mandatory(self) -> None:
        mandatory = (
            EdgeParameters(20, 100, 1, 10, 75, 10),
            EdgeParameters(40, 120, 3, 20, 100, 20),
        )

        first = generate_global_candidates(
            self.space, count=20, seed=42, mandatory=mandatory
        )
        second = generate_global_candidates(
            self.space, count=20, seed=42, mandatory=mandatory
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(set(mandatory).issubset(first))
        self.assertTrue(all(self.space.contains(item) for item in first))

    def test_global_sampling_returns_whole_space_when_count_is_larger(self) -> None:
        all_candidates = tuple(self.space.all_candidates())

        sampled = generate_global_candidates(
            self.space,
            count=len(all_candidates) + 100,
            seed=42,
        )

        self.assertEqual(set(sampled), set(all_candidates))
        self.assertEqual(len(sampled), len(all_candidates))

    def test_local_refinement_is_deterministic_bounded_and_in_range(self) -> None:
        seeds = (
            EdgeParameters(20, 100, 1, 10, 75, 10),
            EdgeParameters(40, 120, 3, 20, 100, 20),
        )

        first = generate_local_candidates(self.space, seeds=seeds, limit=17)
        second = generate_local_candidates(self.space, seeds=seeds, limit=17)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 17)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(self.space.contains(item) for item in first))

    def test_local_refinement_projects_off_grid_mandatory_seed_to_parent_grid(
        self,
    ) -> None:
        off_grid = EdgeParameters(30, 110, 2, 15, 87, 15)
        global_candidates = generate_global_candidates(
            self.space,
            count=5,
            seed=42,
            mandatory=(off_grid,),
        )
        self.assertIn(off_grid, global_candidates)
        self.assertFalse(self.space.contains(off_grid))

        first = generate_local_candidates(
            self.space,
            seeds=(off_grid,),
            limit=13,
            random_seed=314,
        )
        second = generate_local_candidates(
            self.space,
            seeds=(off_grid,),
            limit=13,
            random_seed=314,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 13)
        self.assertTrue(all(self.space.contains(item) for item in first))
        self.assertNotIn(off_grid, first)


class SearchSpaceTests(unittest.TestCase):
    def test_default_grids_match_the_declared_front_and_rear_ranges(self) -> None:
        common_expected = {
            "canny_min_values": (20, 40, 60, 80, 100, 130, 160),
            "canny_max_values": (100, 120, 150, 180, 220, 280, 350),
            "dilate_sizes": (1, 3, 5),
            "hough_thresholds": (10, 15, 20, 25, 30, 40, 50),
        }
        for name, values in common_expected.items():
            self.assertEqual(getattr(FRONT_SEARCH_SPACE, name), values)
            self.assertEqual(getattr(REAR_SEARCH_SPACE, name), values)

        self.assertEqual(
            FRONT_SEARCH_SPACE.min_line_lengths,
            (75, 100, 125, 150, 175, 200, 225, 250),
        )
        self.assertEqual(
            REAR_SEARCH_SPACE.min_line_lengths,
            (20, 30, 40, 50, 60, 70, 80, 90, 100),
        )
        self.assertEqual(
            FRONT_SEARCH_SPACE.max_line_gaps,
            (10, 20, 30, 40, 50, 65, 80),
        )
        self.assertEqual(
            REAR_SEARCH_SPACE.max_line_gaps,
            (5, 10, 15, 20, 30, 40, 50),
        )

    def test_all_candidates_obey_ranges_and_strict_canny_bound(self) -> None:
        for space in (FRONT_SEARCH_SPACE, REAR_SEARCH_SPACE):
            candidates = tuple(space.all_candidates())

            self.assertTrue(candidates)
            self.assertTrue(
                all(item.canny_min_val < item.canny_max_val for item in candidates)
            )
            self.assertTrue(all(space.contains(item) for item in candidates))
            self.assertEqual(len(candidates), len(set(candidates)))

    def test_contains_rejects_equal_reversed_and_off_grid_canny_values(self) -> None:
        valid_other_values = {
            "dilate_size": FRONT_SEARCH_SPACE.dilate_sizes[0],
            "hough_threshold": FRONT_SEARCH_SPACE.hough_thresholds[0],
            "min_line_length": FRONT_SEARCH_SPACE.min_line_lengths[0],
            "max_line_gap": FRONT_SEARCH_SPACE.max_line_gaps[0],
        }

        self.assertFalse(
            FRONT_SEARCH_SPACE.contains(
                EdgeParameters(100, 100, **valid_other_values)
            )
        )
        self.assertFalse(
            FRONT_SEARCH_SPACE.contains(
                EdgeParameters(160, 100, **valid_other_values)
            )
        )
        self.assertFalse(
            FRONT_SEARCH_SPACE.contains(
                EdgeParameters(21, 100, **valid_other_values)
            )
        )


class SplitAndMetricTests(unittest.TestCase):
    def test_navigation_validity_thresholds_remain_frozen(self) -> None:
        config = navigation_preview.NAVIGATION_CONFIG

        self.assertEqual(config.min_confidence, 50.0)
        self.assertEqual(config.max_sample_age_ms, 300)
        self.assertEqual(config.max_pair_time_difference_ms, 150)
        self.assertEqual(config.max_angle_difference_deg, 8.0)
        self.assertEqual(config.max_horizontal_difference_mm, 150.0)

    def test_block_split_has_exact_coverage_no_overlap_and_expected_rotation(self) -> None:
        split = split_frame_blocks(260, 1650, block_size=100)
        train = set(split["train"])
        validation = set(split["validation"])
        test = set(split["test"])

        self.assertEqual(train | validation | test, set(range(260, 1651)))
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))
        self.assertEqual(
            (len(split["train"]), len(split["validation"]), len(split["test"])),
            (500, 491, 400),
        )
        self.assertEqual(split["train"][:100], tuple(range(260, 360)))
        self.assertEqual(split["validation"][:100], tuple(range(360, 460)))
        self.assertEqual(split["test"][:100], tuple(range(460, 560)))
        self.assertEqual(split["validation"][-91:], tuple(range(1560, 1651)))

    def test_block_split_validates_bounds_and_block_size(self) -> None:
        with self.assertRaises(ValueError):
            split_frame_blocks(10, 9)
        with self.assertRaises(ValueError):
            split_frame_blocks(1, 10, block_size=0)

    def test_flag_metrics_count_transitions_and_longest_invalid_run(self) -> None:
        metrics = calculate_flag_metrics(
            (False, False, True, True, False, True, False, False, False)
        )

        self.assertEqual(metrics.valid_count, 3)
        self.assertEqual(metrics.total_count, 9)
        self.assertAlmostEqual(metrics.valid_rate, 100.0 / 3.0)
        self.assertEqual(metrics.transitions, 4)
        self.assertEqual(metrics.longest_invalid_run, 3)

    def test_flag_metrics_handle_empty_and_constant_sequences(self) -> None:
        empty = calculate_flag_metrics(())
        self.assertEqual(empty.valid_count, 0)
        self.assertEqual(empty.total_count, 0)
        self.assertEqual(empty.valid_rate, 0.0)
        self.assertEqual(empty.transitions, 0)
        self.assertEqual(empty.longest_invalid_run, 0)

        all_valid = calculate_flag_metrics((True, True, True))
        self.assertEqual(all_valid.valid_rate, 100.0)
        self.assertEqual(all_valid.transitions, 0)
        self.assertEqual(all_valid.longest_invalid_run, 0)

        all_invalid = calculate_flag_metrics((False, False, False))
        self.assertEqual(all_invalid.valid_rate, 0.0)
        self.assertEqual(all_invalid.transitions, 0)
        self.assertEqual(all_invalid.longest_invalid_run, 3)

    def test_flag_metrics_reset_at_disjoint_block_boundaries(self) -> None:
        metrics = calculate_block_metrics(((False, False), (False, False)))

        # The two invalid runs are separated by a split-block gap and must not
        # be merged into one four-frame run.
        self.assertEqual(metrics.transitions, 0)
        self.assertEqual(metrics.longest_invalid_run, 2)
        self.assertEqual(metrics.invalid_runs, 2)


class OutputAuditTests(unittest.TestCase):
    def test_pair_evaluation_dict_preserves_split_audit_fields(self) -> None:
        flags = (True, False, True)
        evaluation = PairEvaluation(
            pair="front",
            parameters=EdgeParameters(20, 120, 1, 20, 150, 65),
            metrics=calculate_block_metrics((flags[:2], flags[2:])),
            frame_numbers=(260, 261, 560),
            flags=flags,
            block_lengths=(2, 1),
            failure_counts=(("left_low_confidence", 1),),
        )

        serialized = evaluation.to_dict()

        self.assertEqual(serialized["frame_numbers"], [260, 261, 560])
        self.assertEqual(serialized["flags"], [True, False, True])
        self.assertEqual(serialized["block_lengths"], [2, 1])

    def test_winner_csv_headers_include_pair_and_per_camera_deviations(self) -> None:
        sample = navigation_preview.CameraNavigationSample(
            timestamp_ms=0,
            angle_deviation=1.25,
            horizontal_deviation=12.5,
            depth_at_edge=600.0,
            confidence=80.0,
        )

        def pair_evaluation(pair: str) -> PairEvaluation:
            frame = PairFrameResult(
                frame_number=1,
                valid=True,
                pair_confidence=75.0,
                pair_angle_deviation=1.25,
                pair_horizontal_deviation=12.5,
                left_sample=sample,
                right_sample=sample,
                angle_difference_deg=0.0,
                horizontal_difference_mm=0.0,
                failure_reason="",
                left_current_line=None,
                left_median_line=None,
                right_current_line=None,
                right_median_line=None,
            )
            return PairEvaluation(
                pair=pair,
                parameters=EdgeParameters(20, 120, 1, 20, 50, 20),
                metrics=calculate_flag_metrics((True,)),
                frame_numbers=(1,),
                flags=(True,),
                block_lengths=(1,),
                failure_counts=(),
                frames=(frame,),
            )

        run = SimpleNamespace(
            start_frame=1,
            end_frame=1,
            winner_full={
                "front": pair_evaluation("front"),
                "rear": pair_evaluation("rear"),
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "winner_frames.csv"
            write_winner_csv(run, csv_path)
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                headers = next(csv.reader(csv_file))

        expected_headers = {
            "front_pair_angle_deviation_deg",
            "rear_pair_angle_deviation_deg",
            "front_pair_horizontal_deviation_mm",
            "rear_pair_horizontal_deviation_mm",
        }
        expected_headers.update(
            f"{camera}_{deviation}"
            for camera in (
                "front_left",
                "front_right",
                "rear_left",
                "rear_right",
            )
            for deviation in (
                "angle_deviation_deg",
                "horizontal_deviation_mm",
            )
        )
        self.assertTrue(expected_headers.issubset(headers))


class HeadlessEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _write_synthetic_recording(recording_root: Path, frame_count: int = 8) -> None:
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 600, dtype=np.uint16)
        depth[:, 400:] = 850

        for camera_key in CAMERA_ORDER:
            camera_root = recording_root / CAMERA_CONFIGS[camera_key].folder
            color_dir = camera_root / "color"
            depth_dir = camera_root / "depth"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            for frame_number in range(1, frame_count + 1):
                suffix = f"{frame_number:07d}.png"
                color_path = color_dir / f"color_{suffix}"
                depth_path = depth_dir / f"depth_{suffix}"
                if not cv2.imwrite(str(color_path), color):
                    raise RuntimeError(f"Could not write {color_path}")
                if not cv2.imwrite(str(depth_path), depth):
                    raise RuntimeError(f"Could not write {depth_path}")

    def test_camera_sample_matches_existing_analyzer_without_gui(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        yaml_before = {
            filename: (repository_root / filename).read_bytes()
            for filename in CAMERA_CONFIG_FILES.values()
        }
        navigation_before = navigation_preview.NAVIGATION_CONFIG

        with tempfile.TemporaryDirectory() as temp_dir:
            recording_root = Path(temp_dir)
            self._write_synthetic_recording(recording_root)
            with redirect_stdout(io.StringIO()):
                analyzer = RecordingAnalyzer(recording_root)
                analyzer.index_all_cameras()
                evaluator = HeadlessEvaluator(recording_root, 1, 8)

            camera_key = "front_left"
            frame_number = 7
            record = analyzer.frame_by_number[camera_key][frame_number]
            offsets = analyzer.config_offsets()
            target_time = analyzer.camera_time(record, offsets)
            _, analyzer_sample = analyzer.render_camera_row(
                camera_key,
                record,
                target_time=target_time,
                delta_ms=0.0,
                offsets=offsets,
            )
            parameters = _parameters_for_camera(camera_key)
            headless_result = evaluator.evaluate_camera_block(
                camera_key,
                (frame_number,),
                parameters,
            )[0]
            headless_sample = evaluator.evaluate_camera_sample(
                camera_key,
                frame_number,
                parameters,
            )

            raw_depth = cv2.imread(str(record.depth_path), cv2.IMREAD_ANYDEPTH)
            self.assertIsNotNone(raw_depth)
            depth_work, _ = analyzer.crop_depth(
                raw_depth,
                CAMERA_CONFIGS[camera_key],
            )
            analyzer_current_line = analyzer.detect_line(camera_key, frame_number)
            analyzer_median_line, _, _ = analyzer.median_line_for_frame(
                camera_key,
                record,
                image_width=depth_work.shape[1],
                image_height=depth_work.shape[0],
            )

            self.assertEqual(headless_result.current_line, analyzer_current_line)
            self.assertEqual(headless_result.median_line, analyzer_median_line)
            self.assertEqual(headless_sample.timestamp_ms, analyzer_sample.timestamp_ms)
            self.assertAlmostEqual(
                headless_sample.angle_deviation,
                analyzer_sample.angle_deviation,
                places=9,
            )
            self.assertAlmostEqual(
                headless_sample.horizontal_deviation,
                analyzer_sample.horizontal_deviation,
                places=9,
            )
            self.assertAlmostEqual(
                headless_sample.depth_at_edge,
                analyzer_sample.depth_at_edge,
                places=9,
            )
            self.assertAlmostEqual(
                headless_sample.confidence,
                analyzer_sample.confidence,
                places=9,
            )

            right_key = "front_right"
            right_record = analyzer.frame_by_number[right_key][frame_number]
            _, right_sample = analyzer.render_camera_row(
                right_key,
                right_record,
                target_time=target_time,
                delta_ms=0.0,
                offsets=offsets,
            )
            analyzer_message = navigation_preview.build_navigation_message(
                {camera_key: analyzer_sample, right_key: right_sample},
                target_timestamp_ms=int(round(target_time * 1000.0)),
            )
            pair_evaluation = evaluator.evaluate_pair(
                "front",
                parameters,
                blocks=((frame_number,),),
                detailed=True,
            )
            self.assertEqual(pair_evaluation.flags, (analyzer_message.front_valid,))
            self.assertEqual(pair_evaluation.frames[0].left_sample, analyzer_sample)
            self.assertEqual(pair_evaluation.frames[0].right_sample, right_sample)

        self.assertEqual(
            yaml_before,
            {
                filename: (repository_root / filename).read_bytes()
                for filename in CAMERA_CONFIG_FILES.values()
            },
        )
        self.assertEqual(navigation_preview.NAVIGATION_CONFIG, navigation_before)


if __name__ == "__main__":
    unittest.main()
