"""Small Section 3 checks; no TensorFlow import, CIFAR inference, or SVM fit."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flip_features as features
import flip_fusion_search as fusion
import fusion_search as original


class FakeExtractor:
    def __init__(self, channels=512, inconsistent_repeat=False):
        self.calls = []
        self.channels = channels
        self.inconsistent_repeat = inconsistent_repeat

    def predict(self, images, *, batch_size, verbose):
        internal_batches = [
            images[start:start + batch_size].shape[0]
            for start in range(0, len(images), batch_size)
        ]
        self.calls.append({"images": images.copy(), "batch_size": batch_size,
                           "verbose": verbose, "internal_batches": internal_batches})
        result = np.repeat(images[:, 0, 0, :1], self.channels, axis=1)
        if self.inconsistent_repeat and len(self.calls) == 2:
            result = result.copy()
            result[0, 0] += np.float32(1e-4)
        return result


def sample_images():
    images = np.zeros((2, 32, 32, 3), dtype=np.float32)
    images[:, 0, 0, 0] = [2, 6]
    images[:, 0, 31, 0] = [10, 14]
    return images


def tiny_archives():
    archives = {}
    for offset, name in enumerate(features.MODELS):
        archives[name] = {
            "X_train_features": np.full((2, 512), offset + 1, dtype=np.float32),
            "X_val_features": np.full((1, 512), offset + 1, dtype=np.float32),
            "train_indices": np.array([3, 5]), "val_indices": np.array([4]),
            "y_train": np.array([1, 2]), "y_val": np.array([0]),
        }
    return archives


def mock_metrics(correct):
    return {"correct": correct, "validation_rows": 10000,
            "prediction_seconds": 0.01, "fit_seconds": 0.02,
            "effective_gamma": 0.00025, "support_vectors": 10}


class FlipFeatureTests(unittest.TestCase):
    def test_preprocessing_matches_the_three_v3_formulas(self):
        images = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        images[0, 0, 0] = [50, 100, 200]
        rgb = features.preprocess(images, "RGB")
        avg = features.preprocess(images, "AVG")
        ntsc = features.preprocess(images, "NTSC")
        self.assertEqual(rgb.shape, (1, 32, 32, 3))
        self.assertEqual(avg.shape, ntsc.shape)
        self.assertEqual(avg.shape, (1, 32, 32, 1))
        self.assertEqual(rgb.dtype, np.float32)
        self.assertAlmostEqual(float(avg[0, 0, 0, 0]), (50 + 100 + 200) / 3 / 255, places=6)
        self.assertAlmostEqual(float(ntsc[0, 0, 0, 0]),
                               (0.299 * 50 + 0.587 * 100 + 0.114 * 200) / 255, places=6)

    def test_width_flip_and_embedding_average_uses_full_split_predict(self):
        images = np.repeat(sample_images()[:1], 129, axis=0)
        images[:, 0, 0, 0] = np.arange(129)
        images[:, 0, 31, 0] = np.arange(129) + 10
        # First channel is left pixel; after width flip it is right pixel.
        saved = np.repeat(images[:, 0, 0, :1], 512, axis=1)
        extractor = FakeExtractor()
        averaged, verification = features.infer_averaged(
            extractor, images, saved, 128, representation="RGB")
        self.assertEqual(averaged.shape, (129, 512))
        self.assertEqual(averaged.dtype, np.float32)
        np.testing.assert_array_equal(averaged[:, 0], np.arange(129) + 5)
        self.assertTrue(verification["repeatability"]["exact_match"])
        self.assertEqual(verification["historical_comparison"]["max_absolute_difference"], 0.0)
        self.assertEqual(len(extractor.calls), 3)
        self.assertTrue(all(call["batch_size"] == 128 for call in extractor.calls))
        self.assertTrue(all(call["internal_batches"] == [128, 1] for call in extractor.calls))
        self.assertEqual(extractor.calls[0]["images"].shape[0], 129)
        np.testing.assert_array_equal(extractor.calls[2]["images"][:, 0, 0, 0], np.arange(129) + 10)
        # Height and channel positions have not been exchanged.
        np.testing.assert_array_equal(features.horizontal_flip(images)[:, 0, 31, 0], np.arange(129))

    def test_real_split_final_partial_batch_sizes(self):
        self.assertEqual(features.final_batch_rows(40000), 64)
        self.assertEqual(features.final_batch_rows(10000), 16)
        self.assertEqual(features.final_batch_rows(128), 128)

    def test_historical_difference_is_reported_without_rejection(self):
        images = sample_images()
        extractor = FakeExtractor()
        wrong = np.zeros((2, 512), dtype=np.float32)
        _, verification = features.infer_averaged(
            extractor, images, wrong, 128, representation="RGB")
        self.assertFalse(verification["historical_comparison"]["within_provisional_tolerance"])
        self.assertGreater(verification["historical_comparison"]["violating_feature_values"], 0)
        self.assertEqual(len(extractor.calls), 3)

    def test_repeatability_failure_stops_before_flipped_inference(self):
        images = sample_images()
        saved = np.repeat(images[:, 0, 0, :1], 512, axis=1)
        extractor = FakeExtractor(inconsistent_repeat=True)
        with self.assertRaisesRegex(RuntimeError, "not exactly repeatable"):
            features.infer_averaged(
                extractor, images, saved, 128, representation="NTSC")
        self.assertEqual(len(extractor.calls), 2)

    def test_embedding_shape_and_finite_validation(self):
        images = sample_images()
        saved = np.repeat(images[:, 0, 0, :1], 512, axis=1)
        with self.assertRaisesRegex(ValueError, "512 channels"):
            features.infer_averaged(
                FakeExtractor(channels=511), images, saved, 128, representation="RGB")
        with self.assertRaisesRegex(ValueError, "batch_size=128"):
            features.infer_averaged(
                FakeExtractor(), images, saved, 0, representation="RGB")

    def test_representation_specific_tolerances(self):
        self.assertEqual(features.absolute_tolerance("RGB"), 5e-4)
        self.assertEqual(features.absolute_tolerance("AVG"), 5e-4)
        self.assertEqual(features.absolute_tolerance("NTSC"), 6e-4)
        with self.assertRaisesRegex(ValueError, "Unknown representation"):
            features.absolute_tolerance("OTHER")

        images = np.zeros((2, 32, 32, 3), dtype=np.float32)
        predicted = np.repeat(images[:, 0, 0, :1], 512, axis=1)
        reference = predicted.copy()
        reference[0, 0] += np.float32(0.00056)
        _, rgb = features.infer_averaged(
            FakeExtractor(), images, reference, 128, representation="RGB")
        _, ntsc = features.infer_averaged(
            FakeExtractor(), images, reference, 128, representation="NTSC")
        self.assertEqual(rgb["historical_comparison"]["atol"], 5e-4)
        self.assertFalse(rgb["historical_comparison"]["within_provisional_tolerance"])
        self.assertEqual(ntsc["historical_comparison"]["atol"], 6e-4)
        self.assertTrue(ntsc["historical_comparison"]["within_provisional_tolerance"])

    def test_legacy_rgb_avg_archive_metadata_remains_compatible(self):
        def metadata(atol, include_policy):
            comparison = {
                "all_development_values_verified": True,
                "splits": {
                    split: {
                        "images": rows, "feature_values": rows * 512,
                        "atol": atol, "rtol": features.RTOL,
                        "within_provisional_tolerance": True,
                    }
                    for split, rows in (("train", 40000), ("val", 10000))
                },
            }
            if include_policy:
                comparison.update(absolute_tolerance=atol,
                                  relative_tolerance=features.RTOL)
            return {
                "inference_api": "keras.Model.predict",
                "predict_call_scope": "one call per complete ordered split and view",
                "batch_size": 128,
                "original_comparison": comparison,
            }

        # Existing completed archives do not have the two policy keys.
        features._verify_compatibility_metadata(metadata(5e-4, False), "RGB")
        features._verify_compatibility_metadata(metadata(5e-4, False), "AVG")
        with self.assertRaisesRegex(ValueError, "Only completed legacy RGB/AVG"):
            features._verify_compatibility_metadata(metadata(6e-4, True), "NTSC")

    def test_revised_metadata_accepts_historical_report_violations(self):
        model_hash = "model-hash"
        metadata = {
            "verification_protocol": features.VERIFICATION_PROTOCOL,
            "inference_api": "keras.Model.predict",
            "predict_call_scope": "two original calls and one flipped call per complete ordered split",
            "batch_size": 128,
            "cnn_sha256": model_hash,
            "checkpoint_identity": {
                "current_checkpoint_sha256": model_hash,
                "available_recorded_sha256": None,
                "matches_available_record": None,
                "historical_extraction_checkpoint_identity_verified": False,
            },
            "repeatability_verification": {
                "criterion": features.REPEATABILITY_CRITERION,
                "all_development_splits_repeatable": True,
                "splits": {},
            },
            "original_comparison": {
                "role": "reporting_only",
                "absolute_tolerance": 6e-4,
                "relative_tolerance": features.RTOL,
                "splits": {},
            },
        }
        for split, rows in (("train", 40000), ("val", 10000)):
            metadata["repeatability_verification"]["splits"][split] = {
                "criterion": features.REPEATABILITY_CRITERION,
                "images": rows, "feature_values": rows * 512,
                "max_absolute_difference": 0.0,
                "differing_feature_values": 0, "affected_images": 0,
                "exact_match": True,
            }
            metadata["original_comparison"]["splits"][split] = {
                "images": rows, "feature_values": rows * 512,
                "atol": 6e-4, "rtol": features.RTOL,
                "violating_feature_values": 1 if split == "val" else 0,
                "within_provisional_tolerance": split != "val",
            }
        self.assertEqual(
            features._verify_compatibility_metadata(metadata, "NTSC"),
            features.VERIFICATION_PROTOCOL,
        )

    def test_completed_rgb_avg_archives_still_load(self):
        experiment_id = "fusion_flip_v1_001"
        for representation in ("RGB", "AVG"):
            path = features.feature_path(ROOT, experiment_id, representation)
            arrays, metadata = features._read_feature(path, representation, experiment_id)
            self.assertEqual(arrays["X_train_features"].shape, (40000, 512))
            self.assertNotIn("verification_protocol", metadata)

    def test_diagnostics_cover_values_percentiles_violations_and_images(self):
        reference = np.zeros((3, 512), dtype=np.float32)
        predicted = reference.copy()
        predicted[1, 7] = 0.0004  # Within provisional absolute tolerance.
        predicted[2, 9] = 0.0006  # Violation.
        report = features.numerical_diagnostics(predicted, reference)
        self.assertEqual(report["feature_values"], 3 * 512)
        self.assertEqual(report["images"], 3)
        self.assertAlmostEqual(report["max_absolute_difference"], 0.0006, places=7)
        self.assertGreater(report["mean_absolute_difference"], 0)
        self.assertEqual(set(report["percentile_absolute_difference"]),
                         {"50.0", "90.0", "95.0", "99.0", "99.9", "100.0"})
        self.assertEqual(report["violating_feature_values"], 1)
        self.assertEqual(report["affected_images"], 1)
        self.assertEqual(report["affected_image_offsets"], [2])
        self.assertEqual(report["largest_violation_examples"][0]["feature_index"], 9)
        self.assertFalse(report["within_provisional_tolerance"])

    def test_concatenation_preserves_1536_columns_and_alignment(self):
        archives = tiny_archives()
        combined = features.concatenate_aligned(archives, expected_rows=(2, 1))
        self.assertEqual(combined["X_train_combined"].shape, (2, 1536))
        self.assertEqual(combined["X_val_combined"].shape, (1, 1536))
        np.testing.assert_array_equal(combined["X_train_combined"][0, [0, 512, 1024]], [1, 2, 3])
        np.testing.assert_array_equal(combined["train_indices"], [3, 5])
        np.testing.assert_array_equal(combined["y_val"], [0])
        archives["NTSC"]["y_val"] = np.array([2])
        with self.assertRaisesRegex(ValueError, "y_val differs"):
            features.concatenate_aligned(archives, expected_rows=(2, 1))
        archives["NTSC"]["y_val"] = np.array([0])
        archives["AVG"]["train_indices"] = np.array([5, 3])
        with self.assertRaisesRegex(ValueError, "train_indices differs"):
            features.concatenate_aligned(archives, expected_rows=(2, 1))

    def test_atomic_archive_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.npz"
            features._atomic_npz(path, {"train": np.array([1, 2])})
            before = original.file_digest(path)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                features._atomic_npz(path, {"train": np.array([9, 9])})
            self.assertEqual(original.file_digest(path), before)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["train"], [1, 2])

    def test_train_only_preprocessing_is_reused(self):
        train = np.zeros((4, 1536), dtype=np.float32)
        train[:, 0] = [1, 2, 3, 4]
        validation = np.full((1, 1536), 50, dtype=np.float32)
        prepared_train, prepared_val, ids = original.prepare_features(
            {"X_train_combined": train, "X_val_combined": validation})
        self.assertEqual(prepared_train.shape, (4, 1))
        self.assertEqual(ids.tolist(), [0])
        expected = (50 - train[:, 0].mean()) / train[:, 0].std()
        self.assertAlmostEqual(float(prepared_val[0, 0]), float(expected), places=5)

    def test_new_ledger_requires_new_baseline_and_cannot_mix_results(self):
        provenance = {"experiment_id": "flip-test", "combined_metadata": {
            "sources": {name: {"extraction_seconds": 1.0} for name in features.MODELS}}}
        historical = {"ledger_sha256": "historical-fingerprint", "correct": 9475}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flip"
            experiment = fusion.FlipFusionExperiment(path, provenance, historical)
            self.assertEqual(experiment.state["initial"][0], fusion.PREVIOUS_WINNER)
            with self.assertRaisesRegex(RuntimeError, "previous winner"):
                experiment.run_search(lambda _: self.fail("Should not fit"))
            with patch("builtins.print"):
                report = experiment.reproduce_baseline(lambda _: mock_metrics(9400))
            self.assertEqual(report["completed_candidates"], 1)
            self.assertEqual(report["best"]["correct"], 9400)  # No 9475 equality gate.
            self.assertEqual(report["preferred_representation"], "original")
            # Attempting to open this flip ledger using the original class is refused.
            with self.assertRaisesRegex(ValueError, "changed"):
                original.FusionExperiment(path, provenance)
            changed = dict(provenance, experiment_id="other")
            with self.assertRaisesRegex(ValueError, "changed"):
                fusion.FlipFusionExperiment(path, changed, historical)
            resumed = fusion.FlipFusionExperiment(path, provenance, historical)
            resumed.reproduce_baseline(lambda _: self.fail("Completed candidate repeated"))
            with patch("builtins.print"):
                summary = resumed.run_search(lambda _: mock_metrics(9550))
            self.assertEqual(summary["stop_reason"], "preferred_readiness_reached")
            self.assertEqual(summary["completed_candidates"], 2)
            self.assertEqual(summary["preferred_representation"], "flip-averaged")

    def test_notebooks_disable_optimization_execution(self):
        names = ["Feature Engineering Flip Extraction.ipynb",
                 "Feature Engineering Flip Combination.ipynb",
                 "Feature Engineering Flip Optimization.ipynb"]
        extraction_flags = {"RUN_RGB_EXTRACTION", "RUN_AVG_EXTRACTION", "RUN_NTSC_EXTRACTION"}
        disabled_flags = {"RUN_COMBINATION", "RUN_PREVIOUS_WINNER", "RUN_SEARCH",
                          "CONTINUE_AFTER_REVIEW"}
        flags = extraction_flags | disabled_flags
        seen = set()
        for name in names:
            notebook = json.loads((ROOT / name).read_text())
            self.assertEqual(notebook["nbformat"], 4)
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                tree = ast.parse(source)
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in flags:
                                seen.add(target.id)
                                self.assertIsInstance(node.value.value, bool)
                                if target.id in disabled_flags:
                                    self.assertIs(node.value.value, False)
        self.assertEqual(seen, flags)

    def test_original_experiment_and_v3_archive_are_read_only(self):
        original_ledger = ROOT / fusion.HISTORICAL_LEDGER
        before = original.file_digest(original_ledger)
        original_archives = {
            name: original.file_digest(features.original_path(ROOT, name))
            for name in features.MODELS
        }
        historical = fusion.read_historical_result(ROOT)
        self.assertEqual(historical["correct"], 9475)
        self.assertEqual(historical["parameters"], fusion.PREVIOUS_WINNER)
        self.assertEqual(original.file_digest(original_ledger), before)
        self.assertEqual(original_archives, {
            name: original.file_digest(features.original_path(ROOT, name))
            for name in features.MODELS
        })


if __name__ == "__main__":
    unittest.main()
