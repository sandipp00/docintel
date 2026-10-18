"""In-memory registry of uploaded documents, with time-based retention.

Uploaded files are never kept on disk: only the extracted text, classification and (lazily computed)
results live here, and expire after ``retention_minutes``.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from src.classifier import Classification, DocumentClassifier
from src.config import get_settings
from src.rag import VectorStore
from src.schemas import ExtractedDocument, ExtractionResult, Summary


@dataclass
class DocRecord:
    doc_id: str
    extracted: ExtractedDocument
    classification: Classification
    n_chunks: int
    created: float = field(default_factory=time.time)
    extraction: ExtractionResult | None = None
    summary: Summary | None = None


class DocumentStore:
    """Thread-safe documents + the shared vector index; expired documents are purged on access."""

    def __init__(self) -> None:
        self._docs: dict[str, DocRecord] = {}
        self._lock = threading.RLock()
        self.vectors = VectorStore()
        self._classifier: DocumentClassifier | None = None

    @property
    def classifier(self) -> DocumentClassifier:
        if self._classifier is None:
            self._classifier = DocumentClassifier.load()
        return self._classifier

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def add(self, record: DocRecord) -> None:
        with self._lock:
            self._docs[record.doc_id] = record

    def get(self, doc_id: str) -> DocRecord | None:
        self.purge_expired()
        with self._lock:
            return self._docs.get(doc_id)

    def list(self) -> list[DocRecord]:
        self.purge_expired()
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.created)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            existed = self._docs.pop(doc_id, None) is not None
        if existed:
            self.vectors.remove_document(doc_id)
        return existed

    def purge_expired(self) -> None:
        cutoff = time.time() - get_settings().retention_minutes * 60
        with self._lock:
            stale = [d for d, r in self._docs.items() if r.created < cutoff]
        for doc_id in stale:
            self.delete(doc_id)


_store: DocumentStore | None = None


def get_store() -> DocumentStore:
    """Process-wide store (created lazily so importing the app stays cheap)."""
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
