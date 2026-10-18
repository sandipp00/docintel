"""Train the document classifier and report metrics.

Steps: load synthetic train.jsonl -> stratified 80/20 validation -> fit on all training data ->
evaluate on the held-out files in evaluation/test_docs (run through the REAL ingestion/OCR path).

Usage::

    python -m training.train_classifier
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.classifier import DOC_TYPES, DocumentClassifier, build_pipeline
from src.config import get_settings
from src.ingestion import extract_document

logger = logging.getLogger("train_classifier")
ROOT = Path(__file__).resolve().parent.parent


def load_train(path: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def evaluate_on_files(clf: DocumentClassifier, test_dir: Path, truncate: int | None = None) -> dict:
    """Classify held-out files via real text extraction and compute metrics.

    ``truncate`` keeps only the first N characters (robustness probe for short snippets).
    """
    labels = json.loads((test_dir / "labels.json").read_text(encoding="utf-8"))
    names = sorted(labels)
    texts = [extract_document(test_dir / n).full_text[:truncate] for n in names]
    y_true = [labels[n]["doc_type"] for n in names]
    preds = clf.predict_many(texts)
    y_pred = [p.label for p in preds]
    return metrics(y_true, y_pred) | {
        "errors": [{"file": n, "true": t, "pred": p, "confidence": c.confidence,
                    "layout": labels[n]["layout"], "format": labels[n]["format"]}
                   for n, t, p, c in zip(names, y_true, y_pred, preds, strict=True) if t != p],
        "mean_confidence": round(float(np.mean([p.confidence for p in preds])), 4),
    }


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 and a confusion matrix (rows=true, cols=pred)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=DOC_TYPES, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=DOC_TYPES, average="macro", zero_division=0), 4),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "support": int(s[i])}
                      for i, c in enumerate(DOC_TYPES)},
        "labels": DOC_TYPES,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=DOC_TYPES).tolist(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src.ingestion").setLevel(logging.WARNING)
    texts, labels = load_train(ROOT / "training" / "data" / "train.jsonl")
    x_tr, x_val, y_tr, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels,
                                                random_state=0)
    val_clf = DocumentClassifier(build_pipeline().fit(x_tr, y_tr))
    val = metrics(y_val, [p.label for p in val_clf.predict_many(x_val)])
    logger.info("Validation (20%% of synthetic train): acc=%.4f macro-F1=%.4f", val["accuracy"], val["macro_f1"])

    clf = DocumentClassifier(build_pipeline().fit(texts, labels))
    path = clf.save()
    logger.info("Saved model to %s (%.0f KB)", path, path.stat().st_size / 1024)

    test = evaluate_on_files(clf, get_settings().evaluation_dir / "test_docs")
    logger.info("Held-out test (%d files): acc=%.4f macro-F1=%.4f", test["n"], test["accuracy"], test["macro_f1"])
    logger.info("Confusion matrix (rows=true, cols=pred) %s:\n%s", DOC_TYPES, np.array(test["confusion_matrix"]))
    for e in test["errors"]:
        logger.info("  misclassified: %s", e)
    short = evaluate_on_files(clf, get_settings().evaluation_dir / "test_docs", truncate=200)
    logger.info("Robustness probe, first 200 chars only: acc=%.4f macro-F1=%.4f", short["accuracy"],
                short["macro_f1"])
    for e in short["errors"]:
        logger.info("  misclassified: %s", e)
    out = ROOT / "models" / "classifier_metrics.json"
    out.write_text(json.dumps({"validation": val, "held_out": test, "held_out_first_200_chars": short}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
