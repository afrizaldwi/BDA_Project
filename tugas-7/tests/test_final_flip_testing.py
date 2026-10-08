"""Lightweight final-evaluation safety and pipeline checks."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_flip_testing as final


class FinalFlipTestingTests(unittest.TestCase):
    def synthetic_features(self, rows=8):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(rows, 1536)).astype(np.float32)
        features[:, 0] = 3.0
        return features

    def test_ledger_freezes_the_requested_winner_before_test_access(self):
        selection = final.load_frozen_selection(ROOT)
        self.assertEqual(selection["completed_candidates"], 45)
        self.assertEqual(selection["parameters"], final.EXPECTED_SELECTION)

    def test_preprocessing_is_fit_on_development_rows_and_keeps_block_alignment(self):
        development = self.synthetic_features()
        selector, scaler, block_ids, prepared = final.fit_preprocessing(
            development, expected_rows=len(development)
        )
        self.assertEqual(prepared.dtype, np.float32)
        self.assertEqual(prepared.shape, (8, 1535))
        self.assertEqual(np.bincount(block_ids).tolist(), [511, 512, 512])
        held_out = np.full((2, 1536), 100.0, dtype=np.float32)
        transformed = final.transform_features(
            held_out, selector, scaler, block_ids, [1.0, 0.5, 0.5]
        )
        self.assertEqual(transformed.shape, (2, 1535))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertFalse(np.allclose(transformed, 0.0))

    def test_transform_rejects_wrong_output_dimensions_and_nonfinite_values(self):
        development = self.synthetic_features()
        selector, scaler, block_ids, _ = final.fit_preprocessing(
            development, expected_rows=len(development)
        )
        with self.assertRaisesRegex(ValueError, "1,536"):
            final.transform_features(
                np.zeros((2, 1535), dtype=np.float32),
                selector,
                scaler,
                block_ids,
                [1.0, 0.5, 0.5],
            )
        bad = np.zeros((2, 1536), dtype=np.float32)
        bad[0, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            final.transform_features(bad, selector, scaler, block_ids, [1.0, 0.5, 0.5])

    def test_artifacts_persist_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = final.create_final_directory(directory, "final")
            files = final.save_final_artifacts(
                output,
                "model",
                "selector",
                "scaler",
                {"C": 1.0, "gamma": 0.1, "weights": [1.0, 0.5, 0.5]},
                {"ledger_sha256": "ledger"},
                {"accuracy": 0.95},
            )
            self.assertEqual(
                set(files),
                {
                    "final_svm.joblib",
                    "preprocessing.joblib",
                    "parameters.json",
                    "provenance.json",
                    "evaluation.json",
                },
            )
            with self.assertRaisesRegex(ValueError, "existing"):
                final.create_final_directory(directory, "final")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                final.save_final_artifacts(
                    output, "model", "selector", "scaler", {}, {}, {}
                )


if __name__ == "__main__":
    unittest.main()
