"""HTTP API: upload, classify, extract, summarise and ask questions about documents."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.store import DocRecord, DocumentStore, get_store
from src.classifier import Classification
from src.config import get_settings
from src.extraction import extract
from src.ingestion import IngestionError, extract_document
from src.llm import get_provider
from src.rag import answer_question, suggested_questions
from src.schemas import Answer, ExtractionResult, Summary
from src.summarize import summarize
from src.utils import sanitize_filename

logger = logging.getLogger(__name__)

settings = get_settings()


def client_key(request: Request) -> str:
    """Rate-limit key: the client address as seen by the reverse proxy, if there is one.

    Behind a proxy (Hugging Face Spaces, Fly, Render) every request arrives from the proxy's own IP, so
    keying on it would make all visitors share one limit. The proxy appends the real client address to
    X-Forwarded-For; earlier entries are client-supplied and spoofable, so only the last one is trusted.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    last = forwarded.rsplit(",", 1)[-1].strip()
    return last or get_remote_address(request)


limiter = Limiter(key_func=client_key, default_limits=[settings.rate_limit_default])
router = APIRouter(prefix="/api")

_CHUNK = 1024 * 1024


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    n_pages: int
    n_chars: int
    method: str
    ocr_confidence: float | None
    processing_seconds: float
    classification: Classification
    suggested_questions: list[str]


class DocumentDetail(DocumentInfo):
    pages: list[dict]


class ExtractRequest(BaseModel):
    use_llm: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    doc_ids: list[str] | None = None


def _info(rec: DocRecord) -> DocumentInfo:
    ex = rec.extracted
    return DocumentInfo(
        doc_id=rec.doc_id, filename=ex.filename, file_type=ex.file_type, n_pages=len(ex.pages),
        n_chars=ex.n_chars, method=ex.method, ocr_confidence=ex.ocr_confidence,
        processing_seconds=ex.processing_seconds, classification=rec.classification,
        suggested_questions=suggested_questions(rec.classification.label))


def _get_or_404(store: DocumentStore, doc_id: str) -> DocRecord:
    rec = store.get(doc_id)
    if rec is None:
        raise HTTPException(404, "Document not found. It may have expired; please upload it again.")
    return rec


async def _read_limited(upload: UploadFile) -> bytes:
    """Read an upload, rejecting it as soon as it exceeds the size limit."""
    limit = settings.max_upload_bytes
    buf = bytearray()
    while chunk := await upload.read(_CHUNK):
        buf += chunk
        if len(buf) > limit:
            raise HTTPException(413, f"{upload.filename or 'File'} is larger than {settings.max_upload_mb} MB.")
    return bytes(buf)


def _process(store: DocumentStore, data: bytes, filename: str) -> DocRecord:
    """Extract text, classify and index one file. The bytes are written to a temp file and deleted."""
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(415, f"{filename}: unsupported file type. Upload a PDF, PNG, JPG or DOCX.")
    with tempfile.TemporaryDirectory(prefix="docintel_") as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            extracted = extract_document(path, display_name=filename)
        except IngestionError as exc:
            raise HTTPException(422, f"{filename}: {exc}") from exc
    if not extracted.full_text.strip():
        raise HTTPException(422, f"{filename}: no readable text was found in this file.")
    doc_id = store.new_id()
    classification = store.classifier.predict(extracted.full_text)
    n_chunks = store.vectors.add_document(doc_id, filename, [(p.number, p.text) for p in extracted.pages])
    rec = DocRecord(doc_id=doc_id, extracted=extracted, classification=classification, n_chunks=n_chunks)
    store.add(rec)
    return rec


@router.post("/upload", response_model=list[DocumentInfo])
@limiter.limit(settings.rate_limit_upload)
async def upload(request: Request, files: list[UploadFile] = File(...),
                 store: DocumentStore = Depends(get_store)) -> list[DocumentInfo]:
    """Upload one or more documents; each is read, classified and indexed for Q&A."""
    if not files:
        raise HTTPException(400, "No files were uploaded.")
    if len(files) > 5:
        raise HTTPException(400, "Please upload at most 5 files at a time.")
    out: list[DocumentInfo] = []
    for f in files:
        name = sanitize_filename(f.filename or "upload")
        data = await _read_limited(f)
        if not data:
            raise HTTPException(400, f"{name} is empty.")
        out.append(_info(_process(store, data, name)))
    return out


def _samples() -> dict[str, Path]:
    """Bundled demo files by display name, e.g. {"invoice": .../sample_invoice.pdf}."""
    folder = settings.sample_dir
    if not folder.is_dir():
        return {}
    return {p.stem.removeprefix("sample_").replace("_", " "): p for p in sorted(folder.iterdir())
            if p.suffix.lower() in settings.allowed_extensions}


@router.get("/samples", response_model=list[str])
def list_samples() -> list[str]:
    """Names of the bundled sample documents the UI offers as one-click demos."""
    return list(_samples())


@router.post("/samples/{name}", response_model=DocumentInfo)
@limiter.limit(settings.rate_limit_upload)
def load_sample(request: Request, name: str, store: DocumentStore = Depends(get_store)) -> DocumentInfo:
    """Load a bundled sample as if it had been uploaded (name must be one listed by GET /samples)."""
    path = _samples().get(name)
    if path is None:
        raise HTTPException(404, "Unknown sample.")
    return _info(_process(store, path.read_bytes(), path.name))


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(store: DocumentStore = Depends(get_store)) -> list[DocumentInfo]:
    return [_info(r) for r in store.list()]


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str, store: DocumentStore = Depends(get_store)) -> DocumentDetail:
    rec = _get_or_404(store, doc_id)
    pages = [p.model_dump() for p in rec.extracted.pages]
    return DocumentDetail(**_info(rec).model_dump(), pages=pages)


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, store: DocumentStore = Depends(get_store)) -> None:
    if not store.delete(doc_id):
        raise HTTPException(404, "Document not found.")


@router.post("/documents/{doc_id}/extract", response_model=ExtractionResult)
def extract_fields(doc_id: str, body: ExtractRequest | None = None,
                   store: DocumentStore = Depends(get_store)) -> ExtractionResult:
    """Structured field extraction (baseline + LLM cross-check) for invoices, resumes and contracts."""
    rec = _get_or_404(store, doc_id)
    use_llm = body.use_llm if body else True
    if rec.extraction is None or (use_llm and rec.extraction.llm_status == "skipped"):
        rec.extraction = extract(rec.extracted.full_text, rec.classification.label, use_llm=use_llm,
                                 provider=get_provider())
    return rec.extraction


@router.post("/documents/{doc_id}/summary", response_model=Summary)
def summarise_document(doc_id: str, store: DocumentStore = Depends(get_store)) -> Summary:
    rec = _get_or_404(store, doc_id)
    if rec.summary is None:
        rec.summary = summarize(rec.extracted.full_text, provider=get_provider())
    return rec.summary


@router.post("/ask", response_model=Answer)
@limiter.limit(settings.rate_limit_ask)
def ask(request: Request, body: AskRequest, store: DocumentStore = Depends(get_store)) -> Answer:
    """Answer a question from the uploaded documents, with citations or an explicit refusal."""
    if body.doc_ids:
        for d in body.doc_ids:
            _get_or_404(store, d)
    elif not store.list():
        raise HTTPException(400, "Upload a document before asking questions.")
    return answer_question(store.vectors, body.question.strip(), body.doc_ids, provider=get_provider())
