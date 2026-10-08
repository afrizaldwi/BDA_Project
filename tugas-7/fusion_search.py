"""Development-only, budgeted fusion experiments. Importing this module fits nothing."""

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version

import fcntl
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 30092026
BASELINE_CORRECT = 9467
BASELINE = {"weights": [1.0, 0.5, 1.0], "C": 1.0, "gamma": 0.0005}
DEFAULT_POLICY = {
    "budget_seconds": 7200,
    "initial_candidates": 25,
    "refinement_min": 10,
    "refinement_target": 15,
    "refinement_max": 20,
    "preferred_accuracy": 0.955,
    "review_accuracy": 0.95,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_development(root):
    """Validate original development blocks; never access official test arrays."""
    root = Path(root)
    path = root / "feature-engineering/combined_features_v3.npz"
    keys = ("X_train_combined", "X_val_combined", "y_train", "y_val",
            "train_indices", "val_indices")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
        require(int(archive["seed"].item()) == SEED, "Combined seed changed")
        require(archive["feature_order"].tolist() == ["RGB", "AVG", "NTSC"], "Block order changed")
        require(archive["feature_dims"].tolist() == [512, 512, 512], "Block dimensions changed")
        sources = archive["source_archives"].tolist()
        run_id = str(archive["run_id"].item())
    expected_sources = [f"features/{name}_features_v3.npz" for name in ("rgb", "avg", "ntsc")]
    require(sources == expected_sources, "Expected original v3 source archives")
    for split, count in (("train", 40000), ("val", 10000)):
        X, y, indices = (data[f"X_{split}_combined"], data[f"y_{split}"], data[f"{split}_indices"])
        require(X.shape == (count, 1536) and y.shape == (count,), f"Invalid {split} shape")
        require(np.isfinite(X).all(), f"Nonfinite {split} features")
        require(indices.shape == (count,) and np.issubdtype(indices.dtype, np.integer), "Invalid indices")
        require(np.issubdtype(y.dtype, np.integer), "Labels must be integers")
        require(np.array_equal(np.unique(y), np.arange(10)), "Expected ten classes")
        require(np.array_equal(np.bincount(y), np.full(10, count // 10)), "Class balance changed")
    indices = np.concatenate((data["train_indices"], data["val_indices"]))
    require(np.array_equal(np.sort(indices), np.arange(50000)), "Splits overlap or omit development rows")
    labels = np.empty(50000, dtype=np.int64)
    for split in ("train", "val"):
        labels[data[f"{split}_indices"]] = data[f"y_{split}"]
    expected = train_test_split(np.arange(50000), test_size=0.2, random_state=SEED, stratify=labels)
    for split, expected_indices in zip(("train", "val"), expected):
        require(np.array_equal(data[f"{split}_indices"], expected_indices), "Original seeded split changed")
    source_fingerprints = {}
    for block, (name, source) in enumerate(zip(("RGB", "AVG", "NTSC"), sources)):
        with np.load(root / source, allow_pickle=False) as archive:
            require(int(archive["seed"].item()) == SEED, f"{name} seed changed")
            require(archive["representation"].item() == name, "Representation mismatch")
            require(archive["feature_layer"].item() == "svm_features", "Feature layer changed")
            require(int(archive["feature_dim"].item()) == 512, "Feature dimension changed")
            fingerprints = {}
            for split in ("train", "val"):
                for key in (f"{split}_indices", f"y_{split}"):
                    value = archive[key]
                    require(np.array_equal(value, data[key]), f"{source}: {key} alignment mismatch")
                    fingerprints[key] = array_digest(value)
                value = archive[f"X_{split}_features"]
                require(np.array_equal(value, data[f"X_{split}_combined"][:, block * 512:(block + 1) * 512]),
                        f"{source}: combined {split} block is stale")
                fingerprints[f"X_{split}_features"] = array_digest(value)
            source_fingerprints[source] = fingerprints
    models = ["cifar10_custom_cnn_v3.keras", "cifar10_avg_cnn_v3.keras", "cifar10_ntsc_cnn_v3.keras"]
    provenance = {
        "combined_path": str(path.resolve()), "combined_run_id": run_id,
        "development_arrays": {key: array_digest(value) for key, value in data.items()},
        "sources": source_fingerprints,
        "cnn_sha256": {name: file_digest(root / "models" / name) for name in models},
        "feature_order": ["RGB", "AVG", "NTSC"], "feature_dims": [512, 512, 512],
        "extraction": "original single-view svm_features; no CNN changes",
        "seed": SEED,
        "provenance_limit": "Legacy archives have no CNN hashes; recorded hashes identify current files, not historical extraction proof.",
    }
    return data, provenance


def prepare_features(data):
    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(data["X_train_combined"].astype(np.float32))
    X_val = selector.transform(data["X_val_combined"].astype(np.float32))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    block_ids = np.repeat(np.arange(3), 512)[selector.get_support()]
    return X_train, X_val, block_ids


def candidate_id(params):
    return digest(params)[:20]


def initial_candidates():
    """Persisted order: baseline, explicit lower neighbors, then seeded diversity."""
    candidates = [copy.deepcopy(BASELINE)]
    for C, gamma in ((0.1, 0.0005), (0.3, 0.0005), (1.0, 0.0001), (1.0, 0.00025), (0.3, 0.00025)):
        candidates.append({"weights": [1.0, 0.5, 1.0], "C": C, "gamma": gamma})
    pool = [{"weights": [1.0, a, b], "C": C, "gamma": gamma}
            for a, b, C, gamma in itertools.product(
                (0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75, 1.0),
                (0.1, 0.3, 1.0, 3.0, 10.0), (0.0001, 0.00025, 0.0005, 0.001, "scale"))]
    rng = np.random.default_rng(SEED)
    seen = {candidate_id(p) for p in candidates}
    for index in rng.permutation(len(pool)):
        params = pool[index]
        if candidate_id(params) not in seen:
            candidates.append(params)
            seen.add(candidate_id(params))
        if len(candidates) == 25:
            return candidates


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ranked(state):
    return sorted((row for row in state["results"].values() if row["status"] == "completed"),
                  key=lambda row: (-row["validation_accuracy"], row["prediction_seconds"], row["candidate_id"]))


def used_seconds(state):
    return sum(attempt["duration_seconds"] for row in state["results"].values() for attempt in row["attempts"])


def stopping_reason(state):
    rows = ranked(state)
    if rows and rows[0]["validation_accuracy"] >= state["policy"]["preferred_accuracy"]:
        return "preferred_readiness_reached"
    if rows and rows[0]["validation_accuracy"] > state["policy"]["review_accuracy"] and not state["review"]:
        return "review_above_95_before_more_search"
    if used_seconds(state) >= state["policy"]["budget_seconds"]:
        return "compute_budget_exhausted"
    return None


def refinement_candidates(state):
    """One frozen local list; no recursively generated rounds."""
    seen = {candidate_id(p) for p in state["initial"]}
    neighborhoods = []
    for row in ranked(state)[:3]:
        base = row["parameters"]
        neighbors = []
        for dimension, grid in (("C", [0.1, 0.3, 1.0, 3.0, 10.0]),
                                ("gamma", [0.0001, 0.00025, 0.0005, 0.001])):
            value = row["effective_gamma"] if base[dimension] == "scale" else base[dimension]
            lower = [x for x in grid if x < value]
            upper = [x for x in grid if x > value]
            adjacent = [max(lower) if lower else value / 3, min(upper) if upper else value * 3]
            for other in adjacent:
                params = copy.deepcopy(base)
                params[dimension] = float(np.sqrt(value * other))
                neighbors.append(params)
        for block in (1, 2):
            for delta in (-0.125, 0.125):
                params = copy.deepcopy(base)
                params["weights"][block] = max(0.125, params["weights"][block] + delta)
                neighbors.append(params)
        neighborhoods.append(neighbors)
    pool = []
    for group in zip(*neighborhoods):
        for params in group:
            key = candidate_id(params)
            if key not in seen:
                seen.add(key)
                pool.append(params)
    durations = [row["duration_seconds"] for row in ranked(state)]
    estimate = max(float(np.median(durations)) * 1.25, 0.001)
    affordable = int(max(0, state["policy"]["budget_seconds"] - used_seconds(state)) / estimate)
    policy = state["policy"]
    target = policy["refinement_target"]
    # More than 15 only when ample time remains; otherwise reduce toward 10.
    if affordable >= 2 * policy["refinement_max"]:
        target = policy["refinement_max"]
    elif affordable < target:
        target = max(policy["refinement_min"], affordable)
    return pool[:target]


class FusionExperiment:
    """One process at a time; cumulative budget includes failed/interrupted fits."""

    def __init__(self, directory, provenance, policy=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "search_state.json"
        self.manifest = {
            "format": "fusion-development-v1", "provenance": provenance,
            "policy": copy.deepcopy(DEFAULT_POLICY if policy is None else policy),
            "implementation_sha256": file_digest(__file__),
            "environment": {"python": platform.python_version(),
                            **{name: version(name) for name in ("numpy", "scikit-learn", "scipy")}},
            "preprocessing": "40K fit: VarianceThreshold(0), StandardScaler, float32, then block weights",
            "baseline": {"parameters": BASELINE, "expected_correct": BASELINE_CORRECT, "validation_rows": 10000},
            "initial": initial_candidates(),
        }
        with self.locked():
            if self.path.exists():
                self.read()
            else:
                self.state = {"manifest": self.manifest, "fingerprint": digest(self.manifest),
                              "policy": self.manifest["policy"], "initial": self.manifest["initial"],
                              "refinement": None, "results": {}, "review": None, "stop_reason": "not_started"}
                self.save()

    @contextmanager
    def locked(self):
        with (self.directory / ".search.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another process is using this experiment") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self):
        self.state = json.loads(self.path.read_text())
        require(self.state["fingerprint"] == digest(self.manifest)
                and self.state["manifest"] == self.manifest,
                "Inputs, code, environment, or policy changed. Use a new experiment ID; do not reuse results.")
        for row in self.state["results"].values():
            if row["status"] == "running":
                row["status"] = "interrupted"
                attempt = row["attempts"][-1]
                attempt["status"] = "interrupted"
                # Last heartbeat + one interval; downtime is not compute time.
                attempt["duration_seconds"] += 5.0
                attempt["error"] = "Process ended without final record; duration conservatively estimated from heartbeat."
                self.state["stop_reason"] = "candidate_interrupted"
        self.save()

    def save(self):
        atomic_json(self.path, self.state)

    def _evaluate(self, params, evaluate):
        key = candidate_id(params)
        existing = self.state["results"].get(key)
        if existing and existing["status"] == "completed":
            return existing
        row = existing or {"candidate_id": key, "parameters": copy.deepcopy(params), "attempts": []}
        attempt = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": 0.0}
        row["attempts"].append(attempt)
        row["status"] = "running"
        self.state["results"][key] = row
        self.save()
        started = time.monotonic()
        finished = threading.Event()
        heartbeat_errors = []

        def heartbeat():
            while not finished.wait(5):
                attempt["duration_seconds"] = time.monotonic() - started
                try:
                    self.save()
                except Exception as error:
                    heartbeat_errors.append(error)
                    return

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            metrics = evaluate(params)
            finished.set()
            worker.join()
            if heartbeat_errors:
                raise RuntimeError("Incremental checkpoint failed") from heartbeat_errors[0]
            require(metrics["validation_rows"] == 10000, "Expected 10K validation predictions")
            require(0 <= metrics["correct"] <= 10000, "Invalid correct count")
            row.update(metrics)
            row["validation_accuracy"] = metrics["correct"] / metrics["validation_rows"]
            row["status"] = attempt["status"] = "completed"
        except BaseException as error:
            finished.set()
            worker.join()
            row["status"] = attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            attempt["error"] = f"{type(error).__name__}: {error}"
            self.state["stop_reason"] = f"candidate_{row['status']}"
            raise
        finally:
            finished.set()
            worker.join()
            attempt["duration_seconds"] = time.monotonic() - started
            row["duration_seconds"] = attempt["duration_seconds"]
            self.save()
        print(f"{key}: {row['validation_accuracy']:.4%}, {row['duration_seconds']:.1f}s, {params}")
        return row

    def _require_baseline(self):
        row = self.state["results"].get(candidate_id(BASELINE))
        if not row or row["status"] != "completed":
            raise RuntimeError("Run baseline reproduction before optimization")
        if row["correct"] != BASELINE_CORRECT:
            self.state["stop_reason"] = "baseline_discrepancy"
            self.save()
            raise RuntimeError(
                f"Baseline mismatch: {row['correct']}/10000 vs 9467/10000. STOP: inspect source alignment, "
                "feature/CNN provenance, preprocessing, and recorded package versions. "
                "Do not lower the expected score or continue optimization before investigation."
            )

    def reproduce_baseline(self, evaluate):
        with self.locked():
            self.read()
            row = self.state["results"].get(candidate_id(BASELINE))
            if not row or row["status"] != "completed":
                require(used_seconds(self.state) < self.state["policy"]["budget_seconds"], "Budget exhausted")
                self._evaluate(BASELINE, evaluate)
            self._require_baseline()
            self.state["stop_reason"] = "baseline_reproduced"
            self.save()
        return self.summary()

    def run_search(self, evaluate, review_note=None):
        with self.locked():
            self.read()
            self._require_baseline()
            if review_note is not None:
                require(bool(review_note.strip()), "Provide the compute/generalization review rationale")
                best_accuracy = ranked(self.state)[0]["validation_accuracy"]
                require(self.state["policy"]["review_accuracy"] < best_accuracy
                        < self.state["policy"]["preferred_accuracy"],
                        "Continuation review applies only to a measured score above 95% and below 95.5%")
                self.state["review"] = {"note": review_note.strip(), "used_seconds": used_seconds(self.state),
                                        "reviewed_at": datetime.now(timezone.utc).isoformat(),
                                        "best_validation_accuracy": ranked(self.state)[0]["validation_accuracy"]}
                self.save()
            for stage in ("initial", "refinement"):
                reason = stopping_reason(self.state)
                if reason:
                    self.state["stop_reason"] = reason
                    self.save()
                    return self.summary()
                if self.state[stage] is None:
                    self.state[stage] = refinement_candidates(self.state)
                    self.save()  # Freeze candidates BEFORE any local fits.
                for params in self.state[stage]:
                    reason = stopping_reason(self.state)
                    if reason:
                        self.state["stop_reason"] = reason
                        self.save()
                        return self.summary()
                    self._evaluate(params, evaluate)
            self.state["stop_reason"] = stopping_reason(self.state) or "bounded_search_complete_review_results"
            self.save()
        return self.summary()

    def summary(self):
        rows = ranked(self.state)
        elapsed = used_seconds(self.state)
        return {"stop_reason": self.state["stop_reason"], "completed_candidates": len(rows),
                "compute_seconds": elapsed,
                "remaining_seconds": max(0, self.state["policy"]["budget_seconds"] - elapsed),
                "best": rows[0] if rows else None,
                "official_test": "Not evaluated. Prior individual-model test results were already observed.",
                "scope": "Development selection only; no final 50K fit or test-ready model is saved."}


def svm_evaluator(X_train, X_val, y_train, y_val, block_ids):
    """Construct a callback; no fit occurs until baseline/search is explicitly run."""
    def evaluate(params):
        columns = np.asarray(params["weights"], dtype=np.float32)[block_ids]
        train = np.ascontiguousarray(X_train * columns)
        validation = np.ascontiguousarray(X_val * columns)
        model = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], cache_size=2048)
        started = time.monotonic()
        model.fit(train, y_train)
        fit_seconds = time.monotonic() - started
        started = time.monotonic()
        predicted = model.predict(validation)
        prediction_seconds = time.monotonic() - started
        return {"correct": int(np.count_nonzero(predicted == y_val)), "validation_rows": len(y_val),
                "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                "effective_gamma": float(model._gamma), "support_vectors": int(len(model.support_))}
    return evaluate
