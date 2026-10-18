from fastapi.testclient import TestClient

from main import app
from src.config import Settings
from src.utils import redact_secrets, sanitize_filename


def test_defaults_and_limits():
    s = Settings(_env_file=None)
    assert s.max_upload_bytes == 10 * 1024 * 1024
    assert s.chunk_overlap < s.chunk_size


def test_llm_disabled_without_key():
    assert Settings(_env_file=None, llm_provider="groq", groq_api_key="").llm_enabled is False
    assert Settings(_env_file=None, llm_provider="groq", groq_api_key="x").llm_enabled is True


def test_secret_not_in_repr():
    s = Settings(_env_file=None, groq_api_key="gsk_supersecretvalue123")
    assert "supersecretvalue" not in repr(s)


def test_redaction():
    assert "gsk_abcdefghijklmnop" not in redact_secrets("key=gsk_abcdefghijklmnop")


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("a b<c>.pdf") == "a_b_c_.pdf"
    assert sanitize_filename("...") .startswith("file_")


def test_health():
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
