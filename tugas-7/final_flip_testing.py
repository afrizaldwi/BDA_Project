"""Final CIFAR-10 evaluation helpers for the frozen flip-fusion experiment.

Importing this module does not load TensorFlow, CIFAR-10 test data, or fit a model.
"""

import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import flip_features
import fusion_search

EXPERIMENT_ID = "fusion_flip_v1_001"
FINAL_DIRECTORY = "feature-engineering/final-testing/flip_v1_001"
LEDGER_RELATIVE_PATH = (
    "feature-engineering/experiments/fusion_flip_v1_001/svm_search/search_state.json"
)
EXPECTED_SELECTION = {
    "weights": [1.0, 0.5, 0.5],
    "C": 1.0,
    "gamma": 0.00015811388300841897,
}
EXPECTED_COMPLETED_CANDIDATES = 45
FEATURE_ORDER = ["RGB", "AVG", "NTSC"]
FEATURE_DIMENSIONS = [512, 512, 512]
BATCH_SIZE = 128


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_frozen_selection(root, expected=EXPECTED_SELECTION):
    """Load and verify the reviewed winner before any official test access."""
    root = Path(root)
    ledger_path = root / LEDGER_RELATIVE_PATH
    _require(ledger_path.is_file(), f"Missing completed search ledger: {ledger_path}")
    state = json.loads(ledger_path.read_text())
    _require(
        state.get("stop_reason") == "bounded_search_complete_review_results",
        "Flip search ledger is not recorded as complete",
    )
    rows = fusion_search.ranked(state)
    _require(
        len(rows) == EXPECTED_COMPLETED_CANDIDATES,
        "Completed flip ledger does not contain exactly 45 candidates",
    )
    winner = rows[0]
    _require(
        winner["parameters"] == expected,
        "Ledger winner does not match the requested frozen hyperparameters: "
        f"{winner['parameters']!r} != {expected!r}",
    )
    return {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": fusion_search.file_digest(ledger_path),
        "completed_candidates": len(rows),
        "validation_accuracy": winner["validation_accuracy"],
        "validation_correct": winner["correct"],
        "parameters": winner["parameters"],
        "kernel": "rbf",
    }


def load_development_features(root, experiment_id=EXPERIMENT_ID):
    """Load the two development splits as one ordered 50K training dataset."""
    path = flip_features.combined_path(root, experiment_id)
    data, metadata = flip_features.verify_combined(path, experiment_id)
    X = np.concatenate((data["X_train_combined"], data["X_val_combined"]), axis=0)
    y = np.concatenate((data["y_train"], data["y_val"]), axis=0)
    indices = np.concatenate((data["train_indices"], data["val_indices"]), axis=0)
    _require(
        X.shape == (50000, 1536) and y.shape == (50000,),
        "Development features must contain 50,000 rows and 1,536 columns",
    )
    _require(
        np.array_equal(np.sort(indices), np.arange(50000)),
        "Development split indices overlap or omit rows",
    )
    _require(np.isfinite(X).all(), "Development features contain NaN or Inf")
    _require(np.array_equal(np.unique(y), np.arange(10)), "Expected CIFAR-10 labels")
    return (
        X.astype(np.float32, copy=False),
        y.astype(np.int64, copy=False),
        {
            "combined_path": str(path.resolve()),
            "combined_sha256": fusion_search.file_digest(path),
            "combined_metadata": metadata,
            "feature_order": FEATURE_ORDER,
            "feature_dims": FEATURE_DIMENSIONS,
            "development_indices_sha256": fusion_search.array_digest(indices),
            "development_features_sha256": fusion_search.array_digest(X),
            "development_labels_sha256": fusion_search.array_digest(y),
        },
    )


def fit_preprocessing(X_development, *, expected_rows=50000):
    """Fit the optimization pipeline afresh on all 50K development rows."""
    X_development = np.asarray(X_development, dtype=np.float32)
    _require(
        X_development.shape == (expected_rows, 1536),
        f"Expected {expected_rows} x 1536 development features",
    )
    selector = VarianceThreshold(threshold=0.0)
    selected = selector.fit_transform(X_development)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected).astype(np.float32)
    block_ids = np.repeat(np.arange(3), 512)[selector.get_support()]
    _require(
        scaled.dtype == np.float32, "Preprocessed development features must be float32"
    )
    return selector, scaler, block_ids, scaled


def transform_features(features, selector, scaler, block_ids, weights):
    features = np.asarray(features, dtype=np.float32)
    _require(
        features.ndim == 2 and features.shape[1] == 1536,
        "Expected feature rows with 1,536 columns",
    )
    transformed = scaler.transform(selector.transform(features)).astype(np.float32)
    weighted = np.ascontiguousarray(
        transformed * np.asarray(weights, dtype=np.float32)[block_ids]
    )
    _require(np.isfinite(weighted).all(), "Preprocessed features contain NaN or Inf")
    return weighted


def fit_final_model(X_development, y_development, selection):
    selector, scaler, block_ids, _ = fit_preprocessing(X_development)
    weighted = transform_features(
        X_development, selector, scaler, block_ids, selection["parameters"]["weights"]
    )
    model = SVC(
        kernel="rbf",
        C=selection["parameters"]["C"],
        gamma=selection["parameters"]["gamma"],
        cache_size=2048,
    )
    started = time.monotonic()
    model.fit(weighted, y_development)
    return (
        model,
        selector,
        scaler,
        block_ids,
        {"fit_seconds": time.monotonic() - started},
    )


def load_official_test_data():
    """Load official test images and labels only after selection is frozen."""
    import tensorflow as tf

    _, test = tf.keras.datasets.cifar10.load_data()
    images, labels = test
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    _require(
        images.shape == (10000, 32, 32, 3) and labels.shape == (10000,),
        "Unexpected official CIFAR-10 test shape",
    )
    return images, labels


def extract_official_test_features(root, progress=print):
    """Extract RGB, AVG, and NTSC test blocks sequentially from frozen CNNs."""
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model

    images, labels = load_official_test_data()
    blocks = []
    checkpoint_hashes = {}
    for representation, model_relative in flip_features.MODELS.items():
        model_path = Path(root) / model_relative
        _require(model_path.is_file(), f"Missing CNN checkpoint: {model_path}")
        checkpoint_hashes[representation] = fusion_search.file_digest(model_path)
        tf.keras.backend.clear_session()
        model = load_model(model_path, compile=False)
        extractor = Model(model.input, model.get_layer("svm_features").output)
        processed = flip_features.preprocess(images, representation)
        original = np.asarray(
            extractor.predict(processed, batch_size=BATCH_SIZE, verbose=1),
            dtype=np.float32,
        )
        flipped = np.asarray(
            extractor.predict(
                flip_features.horizontal_flip(processed),
                batch_size=BATCH_SIZE,
                verbose=1,
            ),
            dtype=np.float32,
        )
        block = ((original + flipped) / np.float32(2.0)).astype(np.float32, copy=False)
        _require(
            block.shape == (10000, 512) and np.isfinite(block).all(),
            f"Invalid {representation} test block",
        )
        blocks.append(block)
        if progress:
            progress(
                f"{representation}: extracted {block.shape[0]} rows x {block.shape[1]} features"
            )
        del extractor, model
    features = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    _require(
        features.shape == (10000, 1536) and np.isfinite(features).all(),
        "Official test features must be finite with shape (10000, 1536)",
    )
    return (
        features,
        labels,
        {
            "checkpoint_sha256": checkpoint_hashes,
            "test_features_sha256": fusion_search.array_digest(features),
            "test_labels_sha256": fusion_search.array_digest(labels),
            "feature_order": FEATURE_ORDER,
            "feature_dims": FEATURE_DIMENSIONS,
            "batch_size": BATCH_SIZE,
            "extraction": flip_features.METHOD,
        },
    )


def evaluate_official(
    model, test_features, test_labels, selector, scaler, block_ids, weights
):
    prepared = transform_features(test_features, selector, scaler, block_ids, weights)
    predicted = model.predict(prepared)
    matrix = confusion_matrix(test_labels, predicted, labels=np.arange(10))
    report = classification_report(
        test_labels,
        predicted,
        labels=np.arange(10),
        target_names=[str(index) for index in range(10)],
        output_dict=True,
        zero_division=0,
    )
    correct = int(np.count_nonzero(predicted == test_labels))
    accuracy = correct / len(test_labels)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "rows": len(test_labels),
        "strictly_greater_than_95_percent": bool(accuracy > 0.95),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "predictions_sha256": fusion_search.array_digest(predicted),
    }


def create_final_directory(root, relative_path=FINAL_DIRECTORY):
    path = Path(root) / relative_path
    _require(
        not path.exists(),
        f"Refusing to overwrite existing final-testing directory: {path}",
    )
    path.mkdir(parents=True)
    return path


def save_final_artifacts(
    directory, model, selector, scaler, parameters, provenance, evaluation
):
    directory = Path(directory)
    _require(directory.is_dir(), "Final-testing directory must already exist")
    targets = [
        directory / name
        for name in (
            "final_svm.joblib",
            "preprocessing.joblib",
            "parameters.json",
            "provenance.json",
            "evaluation.json",
        )
    ]
    _require(
        not any(path.exists() for path in targets),
        "Refusing to overwrite final artifacts",
    )
    try:
        joblib.dump(model, targets[0])
        joblib.dump(
            {"variance_threshold": selector, "standard_scaler": scaler}, targets[1]
        )
        targets[2].write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n")
        targets[3].write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        targets[4].write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    except Exception:
        for path in targets:
            path.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return {path.name: fusion_search.file_digest(path) for path in targets}
