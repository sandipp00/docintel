"""Shared helpers: logging setup, filename sanitising, secret redaction."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import PurePath

_SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),  # Groq
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google
    re.compile(r"(?i)(api[_-]?key|authorization)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",]+)"),
]


def redact_secrets(text: str) -> str:
    """Mask anything that looks like an API key."""
    text = _SECRET_PATTERNS[0].sub("[REDACTED]", text)
    text = _SECRET_PATTERNS[1].sub("[REDACTED]", text)
    return _SECRET_PATTERNS[2].sub(r"\1\2[REDACTED]", text)


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(record.getMessage())
        record.args = None
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with secret redaction on every record."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_RedactingFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Strip any path, keep a safe character set, and never return an empty name."""
    base = PurePath(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base[:max_len] or f"file_{uuid.uuid4().hex[:8]}"


_STOPWORDS = frozenset(
    "the a an and or of to in on for with by from is are was were be been this that these those it its as at "
    "which who whom what when where how not no yes has have had will would can could should may might also "
    "than then their there they them into about over under between during per each other such more most "
    "some any all one two".split())


def _numbers(text: str) -> set[str]:
    """Numbers as bare digit strings, so 1,234.50 / 1.234,50 / 123450 all compare equal."""
    return {re.sub(r"\D", "", n) for n in re.findall(r"\d[\d.,]*", text)} - {""}


def support_score(claim: str, sources: list[str]) -> float:
    """Lexical support of ``claim`` by ``sources`` in [0, 1].

    Every number in the claim must occur in the sources (otherwise 0.0); the score is then the share of
    the claim's content words (>3 letters, not stopwords) found in the sources. A cheap, deterministic
    faithfulness proxy: it catches invented figures and off-topic text but not subtle misreadings.
    """
    src = " ".join(sources).casefold()
    nums = _numbers(claim)
    src_nums = _numbers(src)
    if nums and not nums <= src_nums:
        return 0.0
    words = {w for w in re.findall(r"[a-z]{4,}", claim.casefold()) if w not in _STOPWORDS}
    if not words:
        return 1.0
    src_words = set(re.findall(r"[a-z]{4,}", src))
    return len(words & src_words) / len(words)
