"""Independent budgeted SVM experiment for averaged development embeddings."""

import copy
import json
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np

import fusion_search as original
from flip_features import (METHOD, MODELS, _read_feature, combined_path,
                           experiment_dir, feature_path, original_path, verify_combined)


PREVIOUS_WINNER = {"weights": [1.0, 0.5, 1.0],
                   "C": 0.5477225575051661, "gamma": 0.00025}
HISTORICAL_CORRECT = 9475
HISTORICAL_LEDGER = "feature-engineering/experiments/fusion_original_v4_001/search_state.json"


def read_historical_result(root):
    path = Path(root) / HISTORICAL_LEDGER
    state = json.loads(path.read_text())
    original.require(state["stop_reason"] == "bounded_search_complete_review_results",
                     "Original feature search is not recorded as complete")
    rows = original.ranked(state)
    original.require(len(rows) == 45 and rows[0]["correct"] == HISTORICAL_CORRECT
                     and rows[0]["parameters"] == PREVIOUS_WINNER,
                     "Historical winner differs from reviewed 94.75% result")
    return {"ledger_path": str(path.resolve()), "ledger_sha256": original.file_digest(path),
            "correct": rows[0]["correct"], "accuracy": rows[0]["validation_accuracy"],
            "parameters": rows[0]["parameters"], "search_seconds": original.used_seconds(state),
            "extraction_seconds": None, "extraction_method": "original v3 embeddings",
            "source": "Saved historical ledger; not newly measured"}


def load_flip_development(root, experiment_id):
    """Verify the new archive and every source block without loading test data."""
    root = Path(root)
    path = combined_path(root, experiment_id)
    data, metadata = verify_combined(path, experiment_id)
    source_hashes = {}
    for block, name in enumerate(MODELS):
        source = feature_path(root, experiment_id, name)
        arrays, source_metadata = _read_feature(source, name, experiment_id)
        original.require(source_metadata == metadata["sources"][name],
                         f"{name} metadata differs from combined archive")
        source_hashes[name] = original.file_digest(source)
        original.require(source_hashes[name] == metadata["source_archive_sha256"][name],
                         f"{name} source changed after combination")
        original.require(source_metadata["cnn_sha256"] == original.file_digest(root / MODELS[name]),
                         f"{name} CNN checkpoint changed")
        original.require(source_metadata["original_archive_sha256"]
                         == original.file_digest(original_path(root, name)),
                         f"{name} original v3 archive changed")
        for split in ("train", "val"):
            original.require(np.array_equal(arrays[f"{split}_indices"], data[f"{split}_indices"])
                             and np.array_equal(arrays[f"y_{split}"], data[f"y_{split}"]),
                             f"{name} {split} alignment mismatch")
            original.require(np.array_equal(arrays[f"X_{split}_features"],
                                            data[f"X_{split}_combined"][:, block * 512:(block + 1) * 512]),
                             f"{name} {split} block differs from combined archive")
    indices = np.concatenate((data["train_indices"], data["val_indices"]))
    original.require(np.array_equal(np.sort(indices), np.arange(50000)), "Invalid development split")
    historical = read_historical_result(root)
    provenance = {
        "combined_path": str(path.resolve()), "combined_sha256": original.file_digest(path),
        "combined_metadata": metadata, "source_archive_sha256": source_hashes,
        "historical_ledger_sha256": historical["ledger_sha256"],
        "development_arrays": {key: original.array_digest(value) for key, value in data.items()},
        "extraction": METHOD, "experiment_id": experiment_id,
        "feature_order": list(MODELS), "feature_dims": [512, 512, 512], "seed": original.SEED,
    }
    return data, provenance, historical


def flip_initial_candidates():
    candidates = [copy.deepcopy(PREVIOUS_WINNER), *original.initial_candidates()[1:]]
    original.require(len(candidates) == 25
                     and len({original.candidate_id(p) for p in candidates}) == 25,
                     "Expected 25 unique candidates including previous winner")
    return candidates


class FlipFusionExperiment(original.FusionExperiment):
    """Reuses the proven ledger, timeout, resume, and refinement machinery.

    The original module and its implementation fingerprint are unchanged.
    This subclass owns a distinct directory, baseline rule, and manifest.
    """

    def __init__(self, directory, provenance, historical):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "search_state.json"
        self.historical = historical
        self.manifest = {
            "format": "fusion-flip-development-v1", "provenance": provenance,
            "policy": copy.deepcopy(original.DEFAULT_POLICY),
            "implementation_sha256": {
                "flip_fusion_search": original.file_digest(__file__),
                "fusion_search": original.file_digest(original.__file__),
            },
            "environment": {"python": platform.python_version(),
                            **{name: version(name) for name in ("numpy", "scikit-learn", "scipy")}},
            "preprocessing": "40K fit: VarianceThreshold(0), StandardScaler, float32, then block weights",
            "baseline": {"parameters": PREVIOUS_WINNER, "expected_correct": None,
                         "purpose": "Measure previous winner on new embeddings; 9475 is original-only comparison"},
            "initial": flip_initial_candidates(),
            "historical_ledger_sha256": historical["ledger_sha256"],
        }
        with self.locked():
            if self.path.exists():
                self.read()
            else:
                self.state = {"manifest": self.manifest, "fingerprint": original.digest(self.manifest),
                              "policy": self.manifest["policy"], "initial": self.manifest["initial"],
                              "refinement": None, "results": {}, "review": None,
                              "stop_reason": "not_started"}
                self.save()

    def _require_baseline(self):
        row = self.state["results"].get(original.candidate_id(PREVIOUS_WINNER))
        if not row or row["status"] != "completed":
            raise RuntimeError("Evaluate the previous winner on flip embeddings before optimization")

    def reproduce_baseline(self, evaluate):
        with self.locked():
            self.read()
            row = self.state["results"].get(original.candidate_id(PREVIOUS_WINNER))
            if not row or row["status"] != "completed":
                original.require(original.used_seconds(self.state) < self.state["policy"]["budget_seconds"],
                                 "Budget exhausted")
                self._evaluate(PREVIOUS_WINNER, evaluate)
            self._require_baseline()
            self.state["stop_reason"] = "previous_winner_measured_on_flip"
            self.save()
        return self.summary()

    def summary(self):
        summary = super().summary()
        best = summary["best"]
        summary["historical_original"] = self.historical
        summary["flip_best_status"] = "measured" if best else "pending"
        if best:
            summary["difference_in_correct_predictions"] = best["correct"] - HISTORICAL_CORRECT
            summary["preferred_representation"] = (
                "flip-averaged" if best["correct"] > HISTORICAL_CORRECT else "original")
            summary["flip_extraction_seconds"] = sum(
                self.manifest["provenance"]["combined_metadata"]["sources"][name]["extraction_seconds"]
                for name in MODELS)
        else:
            summary["difference_in_correct_predictions"] = None
            summary["preferred_representation"] = "pending"
            summary["flip_extraction_seconds"] = None
        return summary


def new_search_directory(root, experiment_id):
    return experiment_dir(root, experiment_id) / "svm_search"
