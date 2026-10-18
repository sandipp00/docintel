from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def invoice(client):
    with (SAMPLES / "sample_invoice.pdf").open("rb") as fh:
        r = client.post("/api/upload", files=[("files", ("sample_invoice.pdf", fh, "application/pdf"))])
    assert r.status_code == 200, r.text
    return r.json()[0]


def test_upload_classifies_invoice(invoice):
    assert invoice["classification"]["label"] == "invoice"
    assert invoice["n_pages"] >= 1 and invoice["suggested_questions"]


def test_list_get_and_missing(client, invoice):
    assert any(d["doc_id"] == invoice["doc_id"] for d in client.get("/api/documents").json())
    detail = client.get(f"/api/documents/{invoice['doc_id']}").json()
    assert detail["pages"][0]["text"].strip()
    assert client.get("/api/documents/nope").status_code == 404


def test_extract_without_llm(client, invoice):
    r = client.post(f"/api/documents/{invoice['doc_id']}/extract", json={"use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert body["doc_type"] == "invoice" and body["final"].get("total") is not None


def test_summary_falls_back_to_extractive(client, invoice):
    r = client.post(f"/api/documents/{invoice['doc_id']}/summary")
    assert r.status_code == 200 and r.json()["summary"]


def test_ask_validation_and_answer(client, invoice):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": "What is the total?",
                                          "doc_ids": ["nope"]}).status_code == 404
    r = client.post("/api/ask", json={"question": "What is the total amount due?",
                                      "doc_ids": [invoice["doc_id"]]})
    assert r.status_code == 200 and r.json()["status"] in {"ok", "extractive", "rate_limited_extractive",
                                                           "not_found"}


def test_rejects_bad_uploads(client):
    r = client.post("/api/upload", files=[("files", ("x.exe", b"MZ....", "application/octet-stream"))])
    assert r.status_code == 415
    r = client.post("/api/upload", files=[("files", ("fake.pdf", b"not really a pdf", "application/pdf"))])
    assert r.status_code == 422
    r = client.post("/api/upload", files=[("files", ("empty.pdf", b"", "application/pdf"))])
    assert r.status_code == 400


def test_delete(client):
    with (SAMPLES / "sample_resume.pdf").open("rb") as fh:
        doc = client.post("/api/upload", files=[("files", ("r.pdf", fh, "application/pdf"))]).json()[0]
    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 204
    assert client.get(f"/api/documents/{doc['doc_id']}").status_code == 404


def test_web_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocIntel" in r.text
    assert client.get("/app.js").status_code == 200 and client.get("/style.css").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"  # API routes win over the static mount


def test_samples_listed_and_loadable(client):
    names = client.get("/api/samples").json()
    assert "invoice" in names
    r = client.post("/api/samples/invoice")
    assert r.status_code == 200 and r.json()["classification"]["label"] == "invoice"
    assert client.post("/api/samples/..%2F..%2F.env").status_code in (404, 405)  # never served
    assert client.post("/api/samples/nope").status_code == 404


def test_rate_limit_key_uses_last_forwarded_hop():
    from starlette.requests import Request

    from api.routes import client_key

    def req(headers):
        return Request({"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers],
                        "client": ("10.0.0.1", 1234)})

    assert client_key(req([("x-forwarded-for", "6.6.6.6, 203.0.113.9")])) == "203.0.113.9"  # spoofed prefix ignored
    assert client_key(req([])) == "10.0.0.1"
