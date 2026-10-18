import json
import random

import pytest
from faker import Faker

from src.extraction import (
    canonical,
    compare_fields,
    extract,
    extract_baseline,
    parse_amount,
    parse_date,
)
from src.llm import LLMProvider, LLMRateLimited
from training.generate_data import make_doc


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, reply: str | Exception):
        self.reply, self.calls = reply, []

    def _request(self, system, user, json_mode, max_tokens):
        self.calls.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def generate(self, system, user, **kw):  # skip retry/backoff in tests
        return self._request(system, user, kw.get("json_mode", False), 0)


def _docs(doc_type: str, n: int = 12, seed: int = 21):
    fake = Faker("en_US")
    fake.seed_instance(seed)
    rng = random.Random(seed)
    return [make_doc(rng, fake, doc_type, f"d{i}", allow_held_out=False) for i in range(n)]


def test_parse_helpers():
    assert parse_date("March 05, 2024") == "2024-03-05"
    assert parse_date("31.03.2025") == "2025-03-31"
    assert parse_date("2025-13-45") is None
    assert parse_amount("1,234.50") == 1234.5
    assert parse_amount("1.234,50") == 1234.5
    assert parse_amount("1234.50") == 1234.5


def test_canonical_ignores_case_order_format():
    assert canonical(["B b", "a"]) == canonical(["A", "bB"])
    assert canonical(10) == canonical(10.0)


@pytest.mark.parametrize("doc_type,keys", [
    ("invoice", ["invoice_number", "total", "vendor", "date"]),
    ("resume", ["email", "name"]),
    ("contract", ["effective_date", "term"]),
])
def test_baseline_gets_most_core_fields_on_seen_layouts(doc_type, keys):
    docs = _docs(doc_type)
    hits = total = 0
    for d in docs:
        got = extract_baseline(d.text, doc_type)
        for k in keys:
            total += 1
            hits += canonical(got[k]) == canonical(d.fields[k])
    assert hits / total >= 0.9, (doc_type, hits, total)


def test_other_docs_skip_extraction():
    r = extract("hello", "other")
    assert r.llm_status == "skipped" and r.baseline == {}


def test_llm_rate_limit_falls_back_to_baseline():
    doc = _docs("invoice", 1)[0]
    r = extract(doc.text, "invoice", provider=FakeProvider(LLMRateLimited("slow down")))
    assert r.llm_status == "rate_limited" and r.llm is None
    assert r.baseline["invoice_number"] == doc.fields["invoice_number"]
    assert r.final["invoice_number"] == doc.fields["invoice_number"]
    assert r.confidence <= 0.7


def test_invalid_llm_output_is_rejected():
    doc = _docs("invoice", 1)[0]
    for bad in ["I cannot help with that", '{"total": "not a number"}', "[1, 2]"]:
        r = extract(doc.text, "invoice", provider=FakeProvider(bad))
        assert r.llm_status == "invalid_output" and r.llm is None


def test_llm_ok_comparison_and_highlights():
    doc = _docs("invoice", 1)[0]
    f = doc.fields
    reply = json.dumps({"vendor": f["vendor"], "invoice_number": f["invoice_number"] + "X", "total": f["total"],
                        "date": f["date"], "line_items": [], "tax": f["tax"], "subtotal": f["subtotal"],
                        "currency": f["currency"]})
    r = extract(doc.text, "invoice", provider=FakeProvider(reply))
    assert r.llm_status == "ok"
    assert r.comparison["total"] == "match"
    assert r.comparison["invoice_number"] in ("mismatch", "baseline_only")
    assert any(h["field"] == "total" for h in r.highlights)
    for h in r.highlights:
        assert doc.text[h["start"]:h["end"]].lower() == doc.text[h["start"]:h["end"]].lower()


def test_compare_fields_statuses():
    c = compare_fields({"a": 1, "b": None, "c": "x", "d": None}, {"a": 1.0, "b": "y", "c": None, "d": None})
    assert c == {"a": "match", "b": "llm_only", "c": "baseline_only", "d": "both_missing"}
