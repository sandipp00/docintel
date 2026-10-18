"""Field-extraction scoring shared by run_eval.py and ad-hoc checks.

A document's fields are flattened into a set of *facts* ``(field, canonical_value)``: scalars give one
fact, list items one each (dict items compare as a whole). Precision/recall/F1 are then computed
micro-averaged over facts, overall and per field. Wrong values count as both FP and FN.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.extraction import _empty, canonical
from src.ingestion import extract_document

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "evaluation" / "test_docs"
CACHE = ROOT / "data" / "eval_text_cache.json"


def facts(fields: dict) -> set[tuple]:
    """Flatten extracted fields into comparable (field, canonical value) facts."""
    out: set[tuple] = set()
    for k, v in fields.items():
        if _empty(v):
            continue
        if isinstance(v, list):
            out |= {(k, canonical(x)) for x in v if not _empty(x)}
        else:
            out.add((k, canonical(v)))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score_fields(preds: list[dict], golds: list[dict]) -> dict:
    """Micro P/R/F1 over all facts, plus a per-field breakdown."""
    tot = defaultdict(lambda: [0, 0, 0])  # field -> tp, fp, fn
    for pred, gold in zip(preds, golds, strict=True):
        pf, gf = facts(pred), facts(gold)
        for field in {f for f, _ in pf | gf}:
            p = {x for x in pf if x[0] == field}
            g = {x for x in gf if x[0] == field}
            tot[field][0] += len(p & g)
            tot[field][1] += len(p - g)
            tot[field][2] += len(g - p)
    overall = [sum(v[i] for v in tot.values()) for i in range(3)]
    return {"overall": prf(*overall), "per_field": {k: prf(*v) for k, v in sorted(tot.items())}}


def load_test_texts() -> dict[str, str]:
    """Extract (and cache) text for every held-out file via the real ingestion/OCR path."""
    labels = json.loads((TEST_DIR / "labels.json").read_text(encoding="utf-8"))
    cache: dict = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    changed = False
    for name in labels:
        stamp = (TEST_DIR / name).stat().st_mtime_ns
        if cache.get(name, {}).get("stamp") != stamp:
            cache[name] = {"stamp": stamp, "text": extract_document(TEST_DIR / name).full_text}
            changed = True
    if changed:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return {n: cache[n]["text"] for n in labels}
