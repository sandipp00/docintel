"""Evaluate DocIntel end to end on the held-out documents in evaluation/test_docs.

    python -m evaluation.run_eval            # offline: classifier, rule-based extraction, retrieval
    python -m evaluation.run_eval --llm      # also LLM extraction and generated answers (needs a key)

Three stages, each scored against labels.json / qa.json:
  1. classification - accuracy, macro-F1, confusion matrix, accuracy per file format
  2. extraction     - micro P/R/F1 per field (rule-based; with --llm also LLM and merged "final")
  3. Q&A            - retrieval hit rate, answer accuracy on answerable questions, refusal rate on
                      unanswerable ones
Results are printed and written to data/eval_report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict

from evaluation.metrics import ROOT, TEST_DIR, load_test_texts, prf, score_fields
from src.classifier import DOC_TYPES, DocumentClassifier
from src.config import get_settings
from src.extraction import _variants, extract, extract_baseline, parse_date
from src.llm import NullProvider, get_provider
from src.rag import VectorStore, answer_question
from src.schemas import FIELD_MODELS

REPORT = ROOT / "data" / "eval_report.json"


# ----------------------------------------------------------------------------- classification
def eval_classification(texts: dict[str, str], labels: dict) -> dict:
    names = list(texts)
    preds = DocumentClassifier.load().predict_many([texts[n] for n in names])
    gold = [labels[n]["doc_type"] for n in names]
    guess = [p.label for p in preds]
    classes = [c for c in DOC_TYPES if c in set(gold) | set(guess)]
    per_class, f1s = {}, []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, guess, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, guess, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, guess, strict=True))
        per_class[c] = prf(tp, fp, fn)
        f1s.append(per_class[c]["f1"])
    by_format: dict[str, list[bool]] = defaultdict(list)
    for n, g, p in zip(names, gold, guess, strict=True):
        by_format[labels[n]["format"]].append(g == p)
    confusion = Counter((g, p) for g, p in zip(gold, guess, strict=True))
    return {
        "n": len(names),
        "accuracy": round(sum(g == p for g, p in zip(gold, guess, strict=True)) / len(names), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "accuracy_by_format": {f: round(sum(v) / len(v), 4) for f, v in sorted(by_format.items())},
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
        "errors": [{"file": n, "gold": g, "predicted": p}
                   for n, g, p in zip(names, gold, guess, strict=True) if g != p],
    }


# ----------------------------------------------------------------------------- extraction
def eval_extraction(texts: dict[str, str], labels: dict, use_llm: bool) -> dict:
    files = [n for n in texts if labels[n]["doc_type"] in FIELD_MODELS]
    golds = [labels[n]["fields"] for n in files]
    out: dict = {"n": len(files), "by_type": {}}
    types = sorted({labels[n]["doc_type"] for n in files})
    runs: dict[str, list[dict]] = {"baseline": []}
    status: Counter = Counter()
    if use_llm:
        runs |= {"llm": [], "final": []}
    for n in files:
        t = labels[n]["doc_type"]
        if use_llm:
            r = extract(texts[n], t, use_llm=True, provider=get_provider())
            status[r.llm_status] += 1
            runs["baseline"].append(r.baseline)
            runs["llm"].append(r.llm or {})
            runs["final"].append(r.final)
        else:
            runs["baseline"].append(extract_baseline(texts[n], t))
    for name, preds in runs.items():
        out[name] = score_fields(preds, golds)
        out["by_type"][name] = {
            t: score_fields([p for p, n in zip(preds, files, strict=True) if labels[n]["doc_type"] == t],
                            [g for g, n in zip(golds, files, strict=True) if labels[n]["doc_type"] == t])["overall"]
            for t in types}
    if use_llm:
        out["llm_status"] = dict(status)
    return out


# ----------------------------------------------------------------------------- Q&A
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@]+", "", s.casefold())


def answer_matches(gold: str, text: str) -> bool:
    """Gold answer appears in ``text`` (ignoring case/punctuation; dates match across formats)."""
    if _norm(gold) and _norm(gold) in _norm(text):
        return True
    iso = parse_date(gold)
    return bool(iso) and any(_norm(v) in _norm(text) for v in _variants("", iso))


def eval_qa(texts: dict[str, str], qa: list[dict], use_llm: bool) -> dict:
    store = VectorStore()
    for name, text in texts.items():
        store.add_document(name, name, [(1, text)])
    k = get_settings().top_k
    # Offline runs must never hit the network, even if .env holds a key; NullProvider forces the
    # extractive fallback, so only the retrieval-score cutoff can refuse a question.
    provider = get_provider() if use_llm else NullProvider()
    stats = {"answerable": [], "retrieval": [], "unanswerable": []}
    statuses: Counter = Counter()
    failures = []
    for item in qa:
        hits = store.search(item["question"], k, [item["file"]])
        if item["answerable"]:
            stats["retrieval"].append(any(answer_matches(item["answer"], h.chunk.text) for h in hits))
        ans = answer_question(store, item["question"], [item["file"]], provider=provider)
        statuses[ans.status] += 1
        if item["answerable"]:
            ok = ans.found and answer_matches(item["answer"], ans.answer)
            stats["answerable"].append(ok)
            if not ok:
                failures.append({"file": item["file"], "question": item["question"], "gold": item["answer"],
                                 "got": ans.answer[:160], "status": ans.status})
        else:
            stats["unanswerable"].append(not ans.found)

    def rate(v: list[bool]) -> float | None:
        return round(sum(v) / len(v), 4) if v else None

    return {
        "mode": "llm" if use_llm else "extractive (no LLM)",
        "n_answerable": len(stats["answerable"]), "n_unanswerable": len(stats["unanswerable"]),
        "retrieval_hit_rate": rate(stats["retrieval"]),
        "answer_accuracy": rate(stats["answerable"]),
        "correct_refusal_rate": rate(stats["unanswerable"]),
        "status_counts": dict(statuses),
        "failures": failures[:25],
    }


# ----------------------------------------------------------------------------- reporting
def print_report(rep: dict) -> None:
    c = rep["classification"]
    print(f"\n== Classification ({c['n']} docs) ==")
    print(f"accuracy {c['accuracy']:.1%}   macro-F1 {c['macro_f1']:.3f}")
    print("by format:", ", ".join(f"{f} {a:.0%}" for f, a in c["accuracy_by_format"].items()))
    for e in c["errors"]:
        print(f"  miss: {e['file']} gold={e['gold']} predicted={e['predicted']}")

    e = rep["extraction"]
    print(f"\n== Extraction ({e['n']} docs) ==")
    for name in [k for k in ("baseline", "llm", "final") if k in e]:
        o = e[name]["overall"]
        print(f"{name:9s} P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}")
        for t, m in e["by_type"][name].items():
            print(f"   {t:9s} F1 {m['f1']:.3f}")
    worst = sorted(e["baseline"]["per_field"].items(), key=lambda kv: kv[1]["f1"])[:5]
    print("weakest baseline fields:", ", ".join(f"{k} {v['f1']:.2f}" for k, v in worst))
    if "llm_status" in e:
        print("LLM status:", e["llm_status"])

    q = rep["qa"]
    print(f"\n== Q&A ({q['mode']}) ==")
    print(f"retrieval hit rate    {q['retrieval_hit_rate']:.1%}  ({q['n_answerable']} answerable questions)")
    print(f"answer accuracy       {q['answer_accuracy']:.1%}")
    print(f"correct refusal rate  {q['correct_refusal_rate']:.1%}  ({q['n_unanswerable']} unanswerable questions)")
    print("statuses:", q["status_counts"])
    print(f"\nFull report: {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="also evaluate LLM extraction and generated answers")
    ap.add_argument("--qa-limit", type=int, default=0, help="only the first N questions (saves API calls)")
    args = ap.parse_args()
    if args.llm and not get_settings().llm_enabled:
        ap.error("--llm needs LLM_PROVIDER and its API key set in .env")

    t0 = time.perf_counter()
    labels = json.loads((TEST_DIR / "labels.json").read_text(encoding="utf-8"))
    qa = json.loads((TEST_DIR / "qa.json").read_text(encoding="utf-8"))
    if args.qa_limit:
        qa = qa[: args.qa_limit]
    print("Reading test documents (cached after the first run)...")
    texts = load_test_texts()
    rep = {
        "classification": eval_classification(texts, labels),
        "extraction": eval_extraction(texts, labels, args.llm),
        "qa": eval_qa(texts, qa, args.llm),
        "llm": get_settings().llm_provider if args.llm else None,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print_report(rep)


if __name__ == "__main__":
    main()
