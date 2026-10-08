"""Control-flow tests only: no CIFAR files, CNN loading, or SVM fits."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fusion_search as search


def metrics(correct=9467):
    return {"correct": correct, "validation_rows": 10000, "prediction_seconds": 0.01,
            "fit_seconds": 0.02, "effective_gamma": 0.0005, "support_vectors": 42}


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "experiment"
        self.provenance = {"development": "synthetic-test-fingerprint"}

    def experiment(self, **kwargs):
        return search.FusionExperiment(self.directory, self.provenance, **kwargs)

    def baseline(self, experiment):
        with patch("builtins.print"):
            experiment.reproduce_baseline(lambda _: metrics())

    def test_initial_list_is_deterministic_diverse_and_contains_baseline(self):
        candidates = search.initial_candidates()
        self.assertEqual(candidates, search.initial_candidates())
        self.assertEqual(len(candidates), 25)
        self.assertEqual(len({search.candidate_id(p) for p in candidates}), 25)
        self.assertEqual(candidates[0], search.BASELINE)
        self.assertTrue(any(p["C"] < 1 for p in candidates))
        self.assertTrue(any(isinstance(p["gamma"], float) and p["gamma"] < 0.0005 for p in candidates))
        self.assertGreaterEqual(len({tuple(p["weights"]) for p in candidates}), 8)

    def test_search_requires_exact_baseline_and_blocks_discrepancies(self):
        experiment = self.experiment()
        with self.assertRaisesRegex(RuntimeError, "baseline reproduction"):
            experiment.run_search(lambda _: self.fail("Should not evaluate"))
        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "Baseline mismatch"):
            experiment.reproduce_baseline(lambda _: metrics(9468))
        with self.assertRaisesRegex(RuntimeError, "Baseline mismatch"):
            experiment.run_search(lambda _: self.fail("Should not evaluate"))
        self.assertEqual(experiment.state["stop_reason"], "baseline_discrepancy")
        self.assertEqual(json.loads(experiment.path.read_text())["stop_reason"], "baseline_discrepancy")

    def test_baseline_resume_skips_fit_and_manifest_changes_are_rejected(self):
        self.baseline(self.experiment())
        resumed = self.experiment()
        resumed.reproduce_baseline(lambda _: self.fail("Baseline must be cached"))
        self.assertEqual(len(resumed.state["results"]), 1)
        with self.assertRaisesRegex(ValueError, "changed"):
            search.FusionExperiment(self.directory, {"development": "different"})
        policy = dict(search.DEFAULT_POLICY, budget_seconds=8000)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.experiment(policy=policy)

    def test_pause_above_95_resume_after_review_and_stop_at_preferred(self):
        experiment = self.experiment()
        self.baseline(experiment)
        with patch("builtins.print"):
            report = experiment.run_search(lambda _: metrics(9501))
        self.assertEqual(report["stop_reason"], "review_above_95_before_more_search")
        self.assertEqual(report["completed_candidates"], 2)
        resumed = self.experiment()
        report = resumed.run_search(lambda _: self.fail("Must wait for review"))
        self.assertEqual(report["completed_candidates"], 2)
        with self.assertRaisesRegex(ValueError, "rationale"):
            resumed.run_search(lambda _: metrics(), review_note=" ")
        with patch("builtins.print"):
            report = resumed.run_search(lambda _: metrics(9550), review_note="Remaining budget checked; bounded refinement justified.")
        self.assertEqual(report["stop_reason"], "preferred_readiness_reached")
        self.assertEqual(report["completed_candidates"], 3)
        resumed.run_search(lambda _: self.fail("Preferred readiness must stop further fits"))

    def test_review_cannot_preapprove_search_before_above_95_result(self):
        experiment = self.experiment()
        self.baseline(experiment)
        with self.assertRaisesRegex(ValueError, "measured score"):
            experiment.run_search(lambda _: self.fail("Should not evaluate"), review_note="Premature review")

    def test_exact_95_does_not_trigger_readiness_and_search_is_bounded(self):
        experiment = self.experiment()
        self.baseline(experiment)
        with patch("builtins.print"):
            report = experiment.run_search(lambda _: metrics(9500))
        self.assertEqual(report["stop_reason"], "bounded_search_complete_review_results")
        self.assertEqual(len(experiment.state["initial"]), 25)
        self.assertTrue(10 <= len(experiment.state["refinement"]) <= 20)
        self.assertEqual(report["completed_candidates"], 25 + len(experiment.state["refinement"]))
        frozen = copy.deepcopy(experiment.state["refinement"])
        resumed = self.experiment()
        resumed.run_search(lambda _: self.fail("Completed fits must be skipped"))
        self.assertEqual(frozen, resumed.state["refinement"])
        self.assertFalse(list(self.directory.glob("*.joblib")))

    def test_budget_persists_across_restarts_and_no_next_fit_starts(self):
        experiment = self.experiment()
        self.baseline(experiment)
        row = experiment.state["results"][search.candidate_id(search.BASELINE)]
        row["attempts"][0]["duration_seconds"] = 7201
        experiment.save()
        resumed = self.experiment()
        report = resumed.run_search(lambda _: self.fail("Budget already exhausted"))
        self.assertEqual(report["stop_reason"], "compute_budget_exhausted")
        self.assertEqual(report["remaining_seconds"], 0)
        self.assertEqual(report["compute_seconds"], 7201)

    def test_keyboard_interrupt_records_attempt_then_retries_only_unfinished(self):
        experiment = self.experiment()
        self.baseline(experiment)
        def interrupted(_):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            experiment.run_search(interrupted)
        saved = json.loads(experiment.path.read_text())
        failed = [row for row in saved["results"].values() if row["status"] == "interrupted"]
        self.assertEqual(len(failed), 1)
        self.assertIn("KeyboardInterrupt", failed[0]["attempts"][0]["error"])
        self.assertGreaterEqual(failed[0]["attempts"][0]["duration_seconds"], 0)
        resumed = self.experiment()
        with patch("builtins.print"):
            report = resumed.run_search(lambda _: metrics(9550))
        retried = resumed.state["results"][failed[0]["candidate_id"]]
        self.assertEqual(len(retried["attempts"]), 2)
        self.assertEqual(report["completed_candidates"], 2)

    def test_failed_candidate_saved_and_stops_search(self):
        experiment = self.experiment()
        self.baseline(experiment)
        def failure(_):
            raise ValueError("synthetic fit failure")
        with self.assertRaisesRegex(ValueError, "synthetic fit failure"):
            experiment.run_search(failure)
        state = json.loads(experiment.path.read_text())
        rows = list(state["results"].values())
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["status"] == "failed" for row in rows), 1)
        self.assertEqual(state["stop_reason"], "candidate_failed")

    def test_fit_crossing_budget_finishes_but_next_fit_does_not_start(self):
        experiment = self.experiment()
        self.baseline(experiment)
        clock = [100.0]
        def slow_evaluator(_):
            clock[0] += 7201
            return metrics(9470)
        with patch.object(search.time, "monotonic", side_effect=lambda: clock[0]), patch("builtins.print"):
            report = experiment.run_search(slow_evaluator)
        self.assertEqual(report["completed_candidates"], 2)
        self.assertEqual(report["stop_reason"], "compute_budget_exhausted")

    def test_svm_callback_uses_training_rows_and_weights_after_scaling(self):
        train = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        validation = np.array([[7, 8, 9]], dtype=np.float32)
        y_train, y_val = np.array([0, 1]), np.array([1])
        with patch.object(search, "SVC") as constructor:
            model = constructor.return_value
            model.predict.return_value = np.array([1])
            model._gamma = 0.0005
            model.support_ = np.array([0, 1])
            evaluate = search.svm_evaluator(train, validation, y_train, y_val, np.array([0, 1, 2]))
            constructor.assert_not_called()
            result = evaluate(search.BASELINE)
            constructor.assert_called_once_with(kernel="rbf", C=1.0, gamma=0.0005, cache_size=2048)
            np.testing.assert_array_equal(model.fit.call_args.args[0], train * [1, 0.5, 1])
            np.testing.assert_array_equal(model.fit.call_args.args[1], y_train)
            np.testing.assert_array_equal(model.predict.call_args.args[0], validation * [1, 0.5, 1])
            self.assertEqual(result["correct"], 1)

    def test_abrupt_exit_uses_heartbeat_without_charging_downtime(self):
        experiment = self.experiment()
        self.baseline(experiment)
        params = experiment.state["initial"][1]
        experiment.state["results"][search.candidate_id(params)] = {
            "candidate_id": search.candidate_id(params), "parameters": params,
            "status": "running", "attempts": [{"status": "running", "duration_seconds": 12.0}],
        }
        experiment.save()
        resumed = self.experiment()
        row = resumed.state["results"][search.candidate_id(params)]
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(row["attempts"][0]["duration_seconds"], 17.0)
        again = self.experiment()
        self.assertEqual(again.state["results"][search.candidate_id(params)]["attempts"][0]["duration_seconds"], 17.0)

    def test_incremental_ledger_exists_before_next_evaluation(self):
        experiment = self.experiment()
        self.baseline(experiment)
        completed = []
        def evaluator(params):
            saved = json.loads(experiment.path.read_text())
            for key in completed:
                self.assertEqual(saved["results"][key]["status"], "completed")
            key = search.candidate_id(params)
            self.assertEqual(saved["results"][key]["status"], "running")
            completed.append(key)
            return metrics(9550 if len(completed) == 3 else 9470)
        with patch("builtins.print"):
            experiment.run_search(evaluator)
        self.assertEqual(len(completed), 3)

    def test_refinement_size_uses_measured_runtime_and_remaining_budget(self):
        experiment = self.experiment()
        self.baseline(experiment)
        # Synthetic completed initial results; no model fits.
        for index, params in enumerate(experiment.state["initial"]):
            key = search.candidate_id(params)
            experiment.state["results"][key] = {
                "candidate_id": key, "parameters": params, "status": "completed",
                "validation_accuracy": 0.94 + index / 100000,
                "prediction_seconds": 0.01, "effective_gamma": 0.0005,
                "duration_seconds": 200, "attempts": [{"duration_seconds": 200}],
            }
        self.assertEqual(len(search.refinement_candidates(experiment.state)), 10)
        experiment.state["policy"]["budget_seconds"] = 9000
        self.assertEqual(len(search.refinement_candidates(experiment.state)), 15)
        experiment.state["policy"]["budget_seconds"] = 20000
        self.assertEqual(len(search.refinement_candidates(experiment.state)), 20)

    def test_preprocessing_does_not_fit_validation_values(self):
        rng = np.random.default_rng(1)
        train = rng.normal(size=(8, 1536)).astype(np.float32)
        train[:, 0] = 1  # Only a train-constant feature should be removed.
        validation = rng.normal(size=(3, 1536)).astype(np.float32) + 100
        X_train, X_val, ids = search.prepare_features({"X_train_combined": train, "X_val_combined": validation})
        self.assertEqual(X_train.shape, (8, 1535))
        self.assertEqual(np.bincount(ids).tolist(), [511, 512, 512])
        self.assertTrue(np.allclose(X_train.mean(axis=0), 0, atol=1e-6))
        expected = (validation[:, 1:] - train[:, 1:].mean(axis=0)) / train[:, 1:].std(axis=0)
        self.assertTrue(np.allclose(X_val, expected, rtol=1e-5))

    def test_concurrent_writer_is_rejected(self):
        experiment = self.experiment()
        with experiment.locked():
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                self.experiment()


if __name__ == "__main__":
    unittest.main()
