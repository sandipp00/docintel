"""Pydantic models shared across the core library (extraction schemas are added in Phase 4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FileType = Literal["pdf", "png", "jpg", "docx"]
ExtractionMethod = Literal["digital", "ocr", "mixed", "docx"]


class PageText(BaseModel):
    """Text of one page and how it was obtained."""

    number: int = Field(ge=1)
    text: str
    method: Literal["digital", "ocr", "docx"]
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)


class ExtractedDocument(BaseModel):
    """Result of text extraction for a whole file."""

    filename: str
    file_type: FileType
    pages: list[PageText]
    method: ExtractionMethod
    ocr_confidence: float | None = None
    processing_seconds: float = 0.0

    @property
    def full_text(self) -> str:
        """All pages joined, with page breaks as form feeds."""
        return "\f".join(p.text for p in self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


# ----------------------------------------------------------------------------- extraction schemas
class LineItem(BaseModel):
    """One invoice line."""

    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class InvoiceFields(BaseModel):
    """Structured invoice data; dates are ISO (YYYY-MM-DD)."""

    vendor: str | None = None
    invoice_number: str | None = None
    date: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None


class Experience(BaseModel):
    title: str | None = None
    company: str | None = None
    start: str | None = None
    end: str | None = None


class Education(BaseModel):
    degree: str | None = None
    institution: str | None = None
    year: str | None = None


class ResumeFields(BaseModel):
    """Structured resume data."""

    name: str | None = None
    email: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)


class ContractFields(BaseModel):
    """Structured contract data; effective_date is ISO."""

    parties: list[str] = Field(default_factory=list)
    effective_date: str | None = None
    term: str | None = None
    key_clauses: list[str] = Field(default_factory=list)


FIELD_MODELS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceFields, "resume": ResumeFields, "contract": ContractFields,
}


class ExtractionResult(BaseModel):
    """Baseline vs LLM extraction for one document, plus per-field agreement."""

    doc_type: str
    baseline: dict
    llm: dict | None = None
    final: dict = Field(default_factory=dict)
    llm_status: Literal["ok", "unavailable", "rate_limited", "invalid_output", "skipped"] = "skipped"
    llm_note: str | None = None
    comparison: dict[str, Literal["match", "mismatch", "baseline_only", "llm_only", "both_missing"]] = {}
    confidence: float = 0.0
    field_success_rate: float = 0.0
    highlights: list[dict] = Field(default_factory=list)
    security_flags: list[str] = Field(
        default_factory=list, description="Lines removed from the LLM input because they looked like prompt injection.")


# ----------------------------------------------------------------------------- RAG / summary
class Citation(BaseModel):
    """A source passage backing an answer. ``start``/``end`` are offsets into that page's text."""

    source_id: int
    file: str
    page: int
    text: str
    start: int
    end: int
    score: float


class Answer(BaseModel):
    """A grounded answer, or an explicit refusal (found=False) when the documents do not contain it."""

    question: str
    answer: str
    found: bool
    status: Literal["ok", "not_found", "extractive", "rate_limited_extractive"]
    citations: list[Citation] = Field(default_factory=list)
    support: float | None = None
    note: str | None = None


class Summary(BaseModel):
    """Short summary plus key bullet points."""

    summary: str
    key_points: list[str]
    method: Literal["llm", "extractive"]
