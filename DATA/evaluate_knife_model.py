#!/usr/bin/env python
"""Evaluate the deployed knife TFLite model on the frozen repository test split."""

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "app" / "src" / "main" / "assets" / "knife_classifier.tflite"
TEST = HERE / "knife_dataset" / "test"
RESULTS = HERE / "knife_evaluation.json"
PREDICTIONS = HERE / "knife_test_predictions.csv"
THRESHOLD = 0.50


def preprocess(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[None, ...] / 255.0


def average_precision(y_true, scores):
    order = np.argsort(-scores, kind="stable")
    ranked = y_true[order]
    positives = int(ranked.sum())
    if positives == 0:
        return 0.0
    cumulative_tp = np.cumsum(ranked)
    precision_at_rank = cumulative_tp / (np.arange(len(ranked)) + 1)
    return float((precision_at_rank * ranked).sum() / positives)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL), num_threads=1)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    samples = []
    for label, truth in (("knife", 1), ("non_knife", 0)):
        for path in sorted((TEST / label).iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, truth))

    # Warm up the interpreter before timing.
    warmup = preprocess(samples[0][0])
    for _ in range(20):
        interpreter.set_tensor(input_info["index"], warmup)
        interpreter.invoke()

    records = []
    latencies_ms = []
    for path, truth in samples:
        tensor = preprocess(path)
        interpreter.set_tensor(input_info["index"], tensor)
        start = time.perf_counter_ns()
        interpreter.invoke()
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        raw_non_knife = float(interpreter.get_tensor(output_info["index"]).reshape(-1)[0])
        knife_score = 1.0 - min(1.0, max(0.0, raw_non_knife))
        prediction = int(knife_score >= THRESHOLD)
        latencies_ms.append(latency_ms)
        records.append({
            "file": str(path.relative_to(HERE)),
            "truth": "knife" if truth else "non_knife",
            "knife_score": knife_score,
            "prediction": "knife" if prediction else "non_knife",
            "inference_ms": latency_ms,
        })

    truth = np.asarray([1 if r["truth"] == "knife" else 0 for r in records], dtype=np.int32)
    pred = np.asarray([1 if r["prediction"] == "knife" else 0 for r in records], dtype=np.int32)
    scores = np.asarray([r["knife_score"] for r in records], dtype=np.float64)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    results = {
        "model": str(MODEL.relative_to(HERE.parent)),
        "test_set": str(TEST.relative_to(HERE.parent)),
        "threshold": THRESHOLD,
        "positive_class": "knife",
        "sample_count": len(records),
        "class_counts": {"knife": int(truth.sum()), "non_knife": int((truth == 0).sum())},
        "metrics": {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "f1_score": f1,
            "pr_auc_average_precision": average_precision(truth, scores),
            "accuracy": float((truth == pred).mean()),
            "confusion_matrix": {
                "layout": "rows=actual [knife, non_knife], columns=predicted [knife, non_knife]",
                "values": [[tp, fn], [fp, tn]],
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            },
        },
        "host_cpu_tflite_latency_ms": {
            "note": "Interpreter.invoke only; excludes image decode and resize; one CPU thread; not Android-device latency",
            "mean": float(np.mean(latencies_ms)),
            "median": percentile(latencies_ms, 50),
            "p90": percentile(latencies_ms, 90),
            "p95": percentile(latencies_ms, 95),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
    }

    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
