import json
from pathlib import Path

from training.generate_data import CLASSES, generate


def test_generate_counts_and_no_leakage(tmp_path: Path):
    out, test, samples = tmp_path / "d", tmp_path / "t", tmp_path / "s"
    summary = generate(out, test, samples, train_per_class=20, test_per_class=4, seed=5)
    assert summary["train"] == 20 * len(CLASSES) and summary["test"] == 4 * len(CLASSES)

    rows = [json.loads(line) for line in (out / "train.jsonl").read_text("utf-8").splitlines()]
    assert {r["label"] for r in rows} == set(CLASSES)
    assert all(r["layout"] != 3 for r in rows), "held-out layout must not appear in training"

    labels = json.loads((test / "labels.json").read_text("utf-8"))
    assert len(labels) == 4 * len(CLASSES)
    assert all((test / name).exists() for name in labels)
    inv = next(v for v in labels.values() if v["doc_type"] == "invoice")
    expected = {"vendor", "invoice_number", "date", "line_items", "total", "tax"}
    assert expected <= inv["fields"].keys()
    qa = json.loads((test / "qa.json").read_text("utf-8"))
    assert any(not q["answerable"] for q in qa) and any(q["answerable"] for q in qa)
