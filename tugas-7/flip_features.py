"""Frozen-CNN horizontal-flip embeddings for CIFAR-10 development images only.

No TensorFlow import, dataset load, or extraction occurs on module import.
"""

import json
import os
from pathlib import Path
import pickle
import tempfile
import time
import uuid

import numpy as np
from sklearn.model_selection import train_test_split

from fusion_search import SEED, array_digest, file_digest, require


MODELS = {
    "RGB": "models/cifar10_custom_cnn_v3.keras",
    "AVG": "models/cifar10_avg_cnn_v3.keras",
    "NTSC": "models/cifar10_ntsc_cnn_v3.keras",
}
PREPROCESSING = {
    "RGB": "RGB float32 / 255.0",
    "AVG": "mean(R,G,B) in float32 / 255.0",
    "NTSC": "(0.299R + 0.587G + 0.114B) in float32 / 255.0",
}
METHOD = "mean of original and width-axis horizontal flip svm_features embeddings"
ATOL = 5e-4
ABSOLUTE_TOLERANCES = {"RGB": 5e-4, "AVG": 5e-4, "NTSC": 6e-4}
RTOL = 1e-4
PREDICT_BATCH_SIZE = 128
DIAGNOSTIC_PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.9, 100.0)
DIAGNOSTIC_EXAMPLE_LIMIT = 20
VERIFICATION_PROTOCOL = "same-session-exact-repeatability-v1"
REPEATABILITY_CRITERION = "two float32 predict outputs must be exactly equal for every feature value"
HISTORICAL_LEDGER = "feature-engineering/experiments/fusion_original_v4_001/search_state.json"


def experiment_dir(root, experiment_id):
    require(bool(experiment_id) and Path(experiment_id).name == experiment_id
            and experiment_id not in (".", ".."), "Experiment ID must be one directory name")
    return Path(root) / "feature-engineering" / "experiments" / experiment_id


def original_path(root, representation):
    require(representation in MODELS, "Unknown representation")
    return Path(root) / "features" / f"{representation.lower()}_features_v3.npz"


def feature_path(root, experiment_id, representation):
    return experiment_dir(root, experiment_id) / f"{representation.lower()}_flip_features.npz"


def combined_path(root, experiment_id):
    return experiment_dir(root, experiment_id) / "combined_flip_features.npz"


def preprocess(images, representation):
    """Match the three v3 extraction notebooks, including operation order."""
    require(representation in MODELS, "Unknown representation")
    images = np.asarray(images).astype(np.float32)
    require(images.ndim == 4 and images.shape[1:] == (32, 32, 3), "Expected RGB CIFAR-10 images")
    if representation == "RGB":
        return images / 255.0
    if representation == "AVG":
        return np.mean(images, axis=-1, keepdims=True) / 255.0
    gray = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
    return gray[..., np.newaxis] / 255.0


def horizontal_flip(images):
    require(images.ndim == 4, "Expected [batch, height, width, channel] images")
    return np.ascontiguousarray(np.flip(images, axis=2))


def final_batch_rows(total_rows, batch_size=PREDICT_BATCH_SIZE):
    require(isinstance(total_rows, int) and total_rows > 0, "Total rows must be positive")
    require(batch_size == PREDICT_BATCH_SIZE,
            f"Compatibility extraction requires batch_size={PREDICT_BATCH_SIZE}")
    return total_rows % batch_size or batch_size


def absolute_tolerance(representation):
    require(representation in ABSOLUTE_TOLERANCES, "Unknown representation")
    return ABSOLUTE_TOLERANCES[representation]


def numerical_diagnostics(predicted, reference, *, atol=ATOL, rtol=RTOL,
                          example_limit=DIAGNOSTIC_EXAMPLE_LIMIT):
    """Summarize the complete original-embedding comparison."""
    predicted = np.asarray(predicted, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    require(predicted.shape == reference.shape and predicted.ndim == 2,
            "Predicted and reference embeddings must have the same 2-D shape")
    require(np.isfinite(predicted).all() and np.isfinite(reference).all(),
            "Embedding comparison contains NaN or Inf")
    absolute = np.abs(predicted - reference)
    tolerance = atol + rtol * np.abs(reference)
    violations = absolute > tolerance
    affected = np.flatnonzero(np.any(violations, axis=1))
    violating_rows, violating_columns = np.nonzero(violations)
    if violating_rows.size:
        order = np.argsort(absolute[violating_rows, violating_columns])[::-1][:example_limit]
        examples = [
            {
                "image_offset": int(violating_rows[index]),
                "feature_index": int(violating_columns[index]),
                "absolute_difference": float(absolute[violating_rows[index], violating_columns[index]]),
                "predicted": float(predicted[violating_rows[index], violating_columns[index]]),
                "reference": float(reference[violating_rows[index], violating_columns[index]]),
                "allowed_difference": float(tolerance[violating_rows[index], violating_columns[index]]),
            }
            for index in order
        ]
    else:
        examples = []
    return {
        "atol": float(atol),
        "rtol": float(rtol),
        "feature_values": int(absolute.size),
        "images": int(absolute.shape[0]),
        "max_absolute_difference": float(absolute.max(initial=0.0)),
        "mean_absolute_difference": float(absolute.mean()) if absolute.size else 0.0,
        "percentile_absolute_difference": {
            str(percentile): float(np.percentile(absolute, percentile))
            for percentile in DIAGNOSTIC_PERCENTILES
        },
        "violating_feature_values": int(violations.sum()),
        "affected_images": int(affected.size),
        "affected_image_offsets": affected[:example_limit].astype(int).tolist(),
        "affected_image_offsets_truncated": bool(affected.size > example_limit),
        "largest_violation_examples": examples,
        "within_provisional_tolerance": bool(not violations.any()),
    }


def repeatability_diagnostics(first, second):
    """Require exact equality between two same-session float32 predictions."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    require(first.shape == second.shape and first.ndim == 2,
            "Repeated embeddings must have the same 2-D shape")
    require(np.isfinite(first).all() and np.isfinite(second).all(),
            "Repeated embedding comparison contains NaN or Inf")
    difference = np.abs(first - second)
    changed = first != second
    return {
        "criterion": REPEATABILITY_CRITERION,
        "images": int(first.shape[0]),
        "feature_values": int(first.size),
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "differing_feature_values": int(changed.sum()),
        "affected_images": int(np.any(changed, axis=1).sum()),
        "exact_match": bool(np.array_equal(first, second)),
    }


def _predict(extractor, images, batch_size):
    """Match v3 extraction: one predict call for a complete ordered split."""
    require(batch_size == PREDICT_BATCH_SIZE,
            f"Compatibility extraction requires batch_size={PREDICT_BATCH_SIZE}")
    output = extractor.predict(images, batch_size=batch_size, verbose=1)
    return np.asarray(output, dtype=np.float32)


def infer_averaged(extractor, images, saved_original, batch_size=PREDICT_BATCH_SIZE, *,
                   representation, progress=None, split="train"):
    """Verify repeatability and report historical differences before averaging.

    Keras ``predict`` supplies inference behavior and internally creates the
    same 128-row batches as v3 extraction. It receives the complete ordered
    split in one call, so the final partial batch is also handled by the same
    data adapter as the original notebooks. Two original-view calls must be
    exactly equal. Historical v3 differences are reported but are not an
    acceptance criterion.
    """
    require(isinstance(batch_size, int) and batch_size == PREDICT_BATCH_SIZE,
            f"Compatibility extraction requires batch_size={PREDICT_BATCH_SIZE}")
    require(images.ndim == 4 and images.shape[1:3] == (32, 32), "Invalid image shape")
    require(saved_original.shape == (len(images), 512), "Invalid saved original feature shape")
    atol = absolute_tolerance(representation)
    contiguous_images = np.ascontiguousarray(images, dtype=np.float32)
    original = _predict(extractor, contiguous_images, batch_size)
    require(original.shape == (len(images), 512), "Original CNN embedding must have 512 channels")
    repeated = _predict(extractor, contiguous_images, batch_size)
    require(repeated.shape == original.shape, "Repeated CNN embedding shape changed")
    repeatability = repeatability_diagnostics(original, repeated)
    if progress is not None:
        progress(f"{split} repeatability: {json.dumps(repeatability, sort_keys=True)}")
    if not repeatability["exact_match"]:
        raise RuntimeError(
            f"{split} original embeddings are not exactly repeatable in the same session; "
            f"diagnostics={json.dumps(repeatability, sort_keys=True)}. STOP: investigate the "
            "predict environment before creating flip features."
        )
    historical = numerical_diagnostics(original, saved_original, atol=atol, rtol=RTOL)
    if progress is not None:
        progress(f"{split} historical v3 comparison (reporting only): "
                 f"{json.dumps(historical, sort_keys=True)}")
    flipped = _predict(extractor, horizontal_flip(contiguous_images), batch_size)
    require(flipped.shape == original.shape and np.isfinite(flipped).all(), "Invalid flipped embeddings")
    result = ((original + flipped) / np.float32(2.0)).astype(np.float32, copy=False)
    if progress is not None:
        progress(f"{split}: {len(images)}/{len(images)} original embeddings verified and flip averages computed")
    require(np.isfinite(result).all(), "Nonfinite averaged embeddings")
    return result, {"repeatability": repeatability, "historical_comparison": historical}


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Archive already exists; refusing to overwrite: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".flip-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        # Same-filesystem hard link publishes the complete archive and refuses
        # an existing destination, including one from a concurrent notebook.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cached_training_batches():
    """Read Keras's five cached CIFAR-10 training batches, never test_batch."""
    keras_home = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
    base = keras_home / "datasets/cifar-10-batches-py-target/cifar-10-batches-py"
    images, labels, hashes = [], [], {}
    for number in range(1, 6):
        path = base / f"data_batch_{number}"
        require(path.is_file(), f"Missing cached CIFAR-10 training batch {path}")
        hashes[path.name] = file_digest(path)
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        pixels = batch[b"data"]
        batch_labels = np.asarray(batch[b"labels"], dtype=np.int64)
        require(pixels.shape == (10000, 3072) and batch_labels.shape == (10000,),
                "Cached CIFAR-10 training batch has unexpected shape")
        images.append(pixels.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.append(batch_labels)
    return np.concatenate(images), np.concatenate(labels), hashes


def _verify_legacy_compatibility_metadata(metadata, representation):
    comparison = metadata["original_comparison"]
    expected_atol = absolute_tolerance(representation)
    # Archives created before representation-specific tolerances recorded the
    # selected value only inside each split diagnostic. Keep those completed
    # RGB/AVG archives valid without rewriting them.
    require(comparison.get("absolute_tolerance", expected_atol) == expected_atol
            and comparison.get("relative_tolerance", RTOL) == RTOL,
            "Original embedding tolerance policy is incompatible")
    require(metadata["inference_api"] == "keras.Model.predict"
            and metadata["predict_call_scope"] == "one call per complete ordered split and view"
            and metadata["batch_size"] == PREDICT_BATCH_SIZE
            and comparison["all_development_values_verified"] is True,
            "Original embedding compatibility was not verified using the approved predict path")
    for split, expected_images in (("train", 40000), ("val", 10000)):
        diagnostics = comparison["splits"][split]
        require(diagnostics["images"] == expected_images
                and diagnostics["feature_values"] == expected_images * 512
                and diagnostics["atol"] == expected_atol and diagnostics["rtol"] == RTOL
                and diagnostics["within_provisional_tolerance"] is True,
                f"{split} compatibility diagnostics are incomplete or incompatible")


def _verify_compatibility_metadata(metadata, representation):
    if "verification_protocol" not in metadata:
        require(representation in ("RGB", "AVG"),
                "Only completed legacy RGB/AVG archives are accepted without repeatability metadata")
        _verify_legacy_compatibility_metadata(metadata, representation)
        return "legacy-historical-threshold-v1"

    require(metadata["verification_protocol"] == VERIFICATION_PROTOCOL
            and metadata["inference_api"] == "keras.Model.predict"
            and metadata["predict_call_scope"]
            == "two original calls and one flipped call per complete ordered split"
            and metadata["batch_size"] == PREDICT_BATCH_SIZE,
            "Feature archive does not use the approved repeatability protocol")
    repeatability = metadata["repeatability_verification"]
    require(repeatability["criterion"] == REPEATABILITY_CRITERION
            and repeatability["all_development_splits_repeatable"] is True,
            "Feature archive did not pass strict repeatability verification")
    comparison = metadata["original_comparison"]
    require(comparison["role"] == "reporting_only"
            and comparison["absolute_tolerance"] == absolute_tolerance(representation)
            and comparison["relative_tolerance"] == RTOL,
            "Historical comparison reporting metadata is incompatible")
    identity = metadata["checkpoint_identity"]
    require(identity["current_checkpoint_sha256"] == metadata["cnn_sha256"]
            and identity["historical_extraction_checkpoint_identity_verified"] is False,
            "Checkpoint identity provenance is inconsistent")
    if identity["available_recorded_sha256"] is not None:
        require(identity["matches_available_record"] is True
                and identity["available_recorded_sha256"] == metadata["cnn_sha256"],
                "CNN checkpoint differs from available provenance")
    for split, expected_images in (("train", 40000), ("val", 10000)):
        repeated = repeatability["splits"][split]
        historical = comparison["splits"][split]
        require(repeated["images"] == expected_images
                and repeated["feature_values"] == expected_images * 512
                and repeated["criterion"] == REPEATABILITY_CRITERION
                and repeated["exact_match"] is True
                and repeated["differing_feature_values"] == 0
                and repeated["max_absolute_difference"] == 0.0,
                f"{split} repeatability diagnostics are incomplete or incompatible")
        require(historical["images"] == expected_images
                and historical["feature_values"] == expected_images * 512
                and historical["atol"] == absolute_tolerance(representation)
                and historical["rtol"] == RTOL,
                f"{split} historical comparison report is incomplete or incompatible")
    return VERIFICATION_PROTOCOL


def checkpoint_identity_report(root, representation, current_hash):
    """Compare with available provenance without overstating historical proof."""
    ledger_path = Path(root) / HISTORICAL_LEDGER
    recorded_hash = None
    limitation = "The v3 feature archive contains no CNN checkpoint hash."
    if ledger_path.is_file():
        state = json.loads(ledger_path.read_text())
        provenance = state.get("manifest", {}).get("provenance", {})
        recorded_hash = provenance.get("cnn_sha256", {}).get(Path(MODELS[representation]).name)
        limitation = provenance.get("provenance_limit", limitation)
    matches = recorded_hash == current_hash if recorded_hash is not None else None
    require(matches is not False, "Current CNN checkpoint differs from available provenance")
    return {
        "current_checkpoint_sha256": current_hash,
        "comparison_source": str(ledger_path.relative_to(root)) if ledger_path.is_file() else None,
        "available_recorded_sha256": recorded_hash,
        "matches_available_record": matches,
        "historical_extraction_checkpoint_identity_verified": False,
        "limitation": limitation,
    }


def _read_feature(path, representation, experiment_id):
    with np.load(path, allow_pickle=False) as archive:
        require("X_test_features" not in archive.files and "y_test" not in archive.files,
                "Official test arrays must not appear in flip archives")
        require(archive["representation"].item() == representation, "Representation mismatch")
        require(archive["experiment_id"].item() == experiment_id, "Experiment ID mismatch")
        require(archive["extraction_method"].item() == METHOD, "Extraction method mismatch")
        require(int(archive["seed"].item()) == SEED, "Seed mismatch")
        require(int(archive["feature_dim"].item()) == 512, "Feature dimension mismatch")
        result = {key: archive[key] for key in (
            "X_train_features", "X_val_features", "y_train", "y_val",
            "train_indices", "val_indices")}
        metadata = json.loads(str(archive["provenance_json"].item()))
    require(metadata["experiment_id"] == experiment_id
            and metadata["representation"] == representation
            and metadata["extraction_method"] == METHOD
            and metadata["feature_dim"] == 512 and metadata["seed"] == SEED,
            "Feature provenance disagrees with archive fields")
    _verify_compatibility_metadata(metadata, representation)
    for split, size in (("train", 40000), ("val", 10000)):
        require(result[f"X_{split}_features"].shape == (size, 512), "Feature shape mismatch")
        require(result[f"X_{split}_features"].dtype == np.float32, "Expected float32 features")
        require(result[f"y_{split}"].shape == (size,) and result[f"{split}_indices"].shape == (size,),
                "Label/index shape mismatch")
        require(np.isfinite(result[f"X_{split}_features"]).all(), "Nonfinite features")
        require(array_digest(result[f"{split}_indices"]) == metadata["split_sha256"][split],
                "Split fingerprint mismatch")
        for key in (f"X_{split}_features", f"y_{split}", f"{split}_indices"):
            require(array_digest(result[key]) == metadata["array_sha256"][key],
                    f"Saved/reloaded {key} differs from provenance")
    return result, metadata


def concatenate_aligned(archives, *, expected_rows=(40000, 10000)):
    """Validate sample identity, then keep all RGB, AVG, NTSC channels."""
    require(set(archives) == set(MODELS), "Expected RGB, AVG, and NTSC archives")
    reference = archives["RGB"]
    output = {}
    for split, count in zip(("train", "val"), expected_rows):
        for key in (f"{split}_indices", f"y_{split}"):
            require(reference[key].shape == (count,), f"RGB {key} shape mismatch")
            for name in ("AVG", "NTSC"):
                require(np.array_equal(archives[name][key], reference[key]),
                        f"{split} {name} {key} differs from RGB")
        blocks = [archives[name][f"X_{split}_features"] for name in MODELS]
        require(all(block.shape == (count, 512) and np.isfinite(block).all() for block in blocks),
                f"Invalid {split} feature blocks")
        output[f"X_{split}_combined"] = np.concatenate(blocks, axis=1).astype(np.float32)
    output.update({key: reference[key] for key in ("y_train", "y_val", "train_indices", "val_indices")})
    return output


def extract_representation(root, experiment_id, representation, *,
                           batch_size=PREDICT_BATCH_SIZE, progress=print):
    """Explicit expensive entrypoint; validates both development splits before saving."""
    root = Path(root)
    require(representation in MODELS, "Unknown representation")
    destination = feature_path(root, experiment_id, representation)
    reference = original_path(root, representation)
    model_path = root / MODELS[representation]
    require(reference.is_file() and model_path.is_file(), "Missing original v3 archive or CNN")
    model_hash = file_digest(model_path)
    source_hash = file_digest(reference)
    checkpoint_identity = checkpoint_identity_report(root, representation, model_hash)
    if destination.exists():
        _, metadata = _read_feature(destination, representation, experiment_id)
        require(metadata["cnn_sha256"] == model_hash and metadata["original_archive_sha256"] == source_hash,
                "Existing flip archive belongs to different source files")
        if progress:
            progress(f"Verified existing {destination}; skipping extraction")
        return destination

    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.random.set_seed(SEED)
    raw_images, labels, training_batch_hashes = load_cached_training_batches()
    require(raw_images.shape == (50000, 32, 32, 3) and labels.shape == (50000,),
            "Expected 50K official development images")
    train_indices, val_indices = train_test_split(
        np.arange(50000), test_size=0.20, random_state=SEED, stratify=labels)
    with np.load(reference, allow_pickle=False) as archive:
        require(archive["representation"].item() == representation, "Original representation mismatch")
        require(archive["feature_layer"].item() == "svm_features", "Original feature layer mismatch")
        require(int(archive["seed"].item()) == SEED, "Original seed mismatch")
        require(int(archive["feature_dim"].item()) == 512, "Original feature dimension mismatch")
        for split, indices in (("train", train_indices), ("val", val_indices)):
            require(np.array_equal(archive[f"{split}_indices"], indices), f"{split} indices differ from v3")
            require(np.array_equal(archive[f"y_{split}"], labels[indices]), f"{split} labels differ from v3")
        tf.keras.backend.clear_session()
        model = load_model(model_path, compile=False)
        require(model.input_shape[1:] == (32, 32, 3 if representation == "RGB" else 1),
                "CNN input shape mismatch")
        extractor = Model(model.input, model.get_layer("svm_features").output)
        require(extractor.output_shape[-1] == 512, "CNN feature layer is not 512-dimensional")
        started = time.monotonic()
        arrays = {}
        verification_diagnostics = {}
        for split, indices in (("train", train_indices), ("val", val_indices)):
            images = preprocess(raw_images[indices], representation)
            features, verification = infer_averaged(
                extractor, images, archive[f"X_{split}_features"], batch_size,
                representation=representation, progress=progress,
                split=f"{representation} {split}")
            arrays[f"X_{split}_features"] = features
            verification_diagnostics[split] = verification
        duration = time.monotonic() - started

    arrays.update(y_train=labels[train_indices], y_val=labels[val_indices],
                  train_indices=train_indices, val_indices=val_indices)
    metadata = {
        "experiment_id": experiment_id, "representation": representation,
        "cnn_filename": MODELS[representation], "cnn_sha256": model_hash,
        "checkpoint_identity": checkpoint_identity,
        "original_archive": str(reference.relative_to(root)), "original_archive_sha256": source_hash,
        "extraction_method": METHOD, "feature_layer": "svm_features",
        "preprocessing": PREPROCESSING[representation], "normalization": "divide by 255.0",
        "flip_axis": 2, "feature_dim": 512, "seed": SEED,
        "split_sha256": {split: array_digest(indices) for split, indices in
                         (("train", train_indices), ("val", val_indices))},
        "array_sha256": {key: array_digest(value) for key, value in arrays.items()},
        "original_comparison": {
            "role": "reporting_only",
            "threshold_status": "historical diagnostic only; not an acceptance criterion",
            "absolute_tolerance": absolute_tolerance(representation),
            "relative_tolerance": RTOL,
            "splits": {
                split: diagnostics["historical_comparison"]
                for split, diagnostics in verification_diagnostics.items()
            },
        },
        "verification_protocol": VERIFICATION_PROTOCOL,
        "repeatability_verification": {
            "criterion": REPEATABILITY_CRITERION,
            "all_development_splits_repeatable": all(
                diagnostics["repeatability"]["exact_match"]
                for diagnostics in verification_diagnostics.values()),
            "splits": {
                split: diagnostics["repeatability"]
                for split, diagnostics in verification_diagnostics.items()
            },
        },
        "inference_api": "keras.Model.predict",
        "predict_call_scope": "two original calls and one flipped call per complete ordered split",
        "batch_size": batch_size,
        "expected_final_batch_rows": {
            "train": final_batch_rows(40000, batch_size),
            "val": final_batch_rows(10000, batch_size),
        },
        "extraction_seconds": duration,
        "training_batch_sha256": training_batch_hashes,
    }
    _atomic_npz(destination, {**arrays, "representation": np.array(representation),
                              "experiment_id": np.array(experiment_id),
                              "extraction_method": np.array(METHOD),
                              "seed": np.array(SEED), "feature_dim": np.array(512),
                              "provenance_json": np.array(json.dumps(metadata, sort_keys=True))})
    saved, _ = _read_feature(destination, representation, experiment_id)
    for key, value in arrays.items():
        require(np.array_equal(saved[key], value), f"Reloaded {key} changed")
    if progress:
        progress(f"Saved and reloaded {destination} ({duration:.1f}s)")
    return destination


def combine_features(root, experiment_id):
    """Build the development-only 1536-channel archive after strict alignment checks."""
    root = Path(root)
    destination = combined_path(root, experiment_id)
    source_paths = {name: feature_path(root, experiment_id, name) for name in MODELS}
    archives = {}
    provenance = {}
    for name, path in source_paths.items():
        require(path.is_file(), f"Missing {name} flip archive: {path}")
        arrays, metadata = _read_feature(path, name, experiment_id)
        require(metadata["cnn_filename"] == MODELS[name], "CNN filename mismatch")
        require(metadata["cnn_sha256"] == file_digest(root / MODELS[name]), "CNN checkpoint changed")
        require(metadata["original_archive_sha256"] == file_digest(original_path(root, name)),
                "Original feature archive changed")
        require(metadata["preprocessing"] == PREPROCESSING[name] and metadata["flip_axis"] == 2,
                "Preprocessing or flip method changed")
        archives[name] = arrays
        provenance[name] = metadata
    reference = archives["RGB"]
    for split in ("train", "val"):
        fingerprints = {provenance[name]["split_sha256"][split] for name in MODELS}
        require(len(fingerprints) == 1 and fingerprints.pop() == array_digest(reference[f"{split}_indices"]),
                "Split fingerprints differ")
    indices = np.concatenate((reference["train_indices"], reference["val_indices"]))
    require(np.array_equal(np.sort(indices), np.arange(50000)), "Split indices overlap or omit rows")
    arrays = concatenate_aligned(archives)
    for split, count in (("train", 40000), ("val", 10000)):
        require(arrays[f"X_{split}_combined"].shape == (count, 1536), "Combined shape mismatch")
        require(np.isfinite(arrays[f"X_{split}_combined"]).all(), "Nonfinite combined features")
    source_hashes = {name: file_digest(path) for name, path in source_paths.items()}
    if destination.exists():
        with np.load(destination, allow_pickle=False) as existing:
            existing_meta = json.loads(str(existing["provenance_json"].item()))
        require(existing_meta["source_archive_sha256"] == source_hashes and
                existing_meta["experiment_id"] == experiment_id,
                "Existing combined archive has different sources")
        verify_combined(destination, experiment_id)
        return destination
    run_id = str(uuid.uuid4())
    metadata = {"experiment_id": experiment_id, "extraction_method": METHOD,
                "feature_order": list(MODELS), "feature_dims": [512, 512, 512],
                "seed": SEED, "run_id": run_id, "source_archive_sha256": source_hashes,
                "sources": provenance,
                "array_sha256": {key: array_digest(value) for key, value in arrays.items()}}
    _atomic_npz(destination, {**arrays, "experiment_id": np.array(experiment_id),
                              "extraction_method": np.array(METHOD),
                              "feature_order": np.array(list(MODELS)),
                              "feature_dims": np.array([512, 512, 512]),
                              "total_feature_dim": np.array(1536),
                              "seed": np.array(SEED), "run_id": np.array(run_id),
                              "provenance_json": np.array(json.dumps(metadata, sort_keys=True))})
    verify_combined(destination, experiment_id)
    return destination


def verify_combined(path, experiment_id):
    with np.load(path, allow_pickle=False) as archive:
        require("X_test_combined" not in archive.files and "y_test" not in archive.files,
                "Official test arrays must not appear in combined archive")
        require(archive["experiment_id"].item() == experiment_id, "Experiment ID mismatch")
        require(archive["extraction_method"].item() == METHOD, "Extraction method mismatch")
        require(archive["feature_order"].tolist() == list(MODELS), "Feature order mismatch")
        require(archive["feature_dims"].tolist() == [512, 512, 512], "Feature dims mismatch")
        require(int(archive["seed"].item()) == SEED, "Seed mismatch")
        metadata = json.loads(str(archive["provenance_json"].item()))
        require(metadata["experiment_id"] == experiment_id
                and metadata["extraction_method"] == METHOD
                and metadata["feature_order"] == list(MODELS)
                and metadata["feature_dims"] == [512, 512, 512]
                and metadata["seed"] == SEED
                and metadata["run_id"] == archive["run_id"].item(),
                "Combined provenance disagrees with archive fields")
        arrays = {key: archive[key] for key in (
            "X_train_combined", "X_val_combined", "y_train", "y_val", "train_indices", "val_indices")}
    for split, count in (("train", 40000), ("val", 10000)):
        require(arrays[f"X_{split}_combined"].shape == (count, 1536), "Combined shape mismatch")
        require(arrays[f"y_{split}"].shape == (count,) and arrays[f"{split}_indices"].shape == (count,),
                "Combined label/index shape mismatch")
        require(np.isfinite(arrays[f"X_{split}_combined"]).all(), "Nonfinite combined features")
    for key, value in arrays.items():
        require(array_digest(value) == metadata["array_sha256"][key], f"Reloaded {key} differs")
    return arrays, metadata
