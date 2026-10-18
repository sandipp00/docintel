"""Text extraction from PDF / image / DOCX, preserving page numbers.

Digital PDFs use PyMuPDF; pages with no usable text layer (scans) and images fall back to OCR.
File type is decided from magic bytes, never from the (untrusted) filename.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pymupdf  # PyMuPDF
from PIL import Image

from src.config import get_settings
from src.ocr import ocr_image
from src.schemas import ExtractedDocument, FileType, PageText

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised with a user-friendly message when a file cannot be processed."""


def detect_file_type(data: bytes) -> FileType | None:
    """Identify pdf/png/jpg/docx from magic bytes; None if unsupported."""
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            return None
    return None


def _open_image(source: Path | bytes) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = get_settings().max_image_pixels
    try:
        img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
        img.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise IngestionError("That image is too large to process.") from exc
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("That image could not be read; it may be corrupt.") from exc
    if img.width * img.height > get_settings().max_image_pixels:
        raise IngestionError("That image is too large to process.")
    return img


def _extract_pdf(path: Path) -> tuple[list[PageText], str]:
    s = get_settings()
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("That PDF could not be opened; it may be corrupt.") from exc
    with doc:
        if doc.needs_pass:
            raise IngestionError("Password-protected PDFs are not supported.")
        if doc.page_count > s.max_pages:
            raise IngestionError(f"PDF has {doc.page_count} pages; the limit is {s.max_pages}.")
        pages: list[PageText] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) >= s.ocr_min_chars_per_page:
                pages.append(PageText(number=i, text=text, method="digital"))
                continue
            pix = page.get_pixmap(dpi=s.ocr_dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            res = ocr_image(img)
            pages.append(PageText(number=i, text=res.text or text, method="ocr",
                                  ocr_confidence=res.confidence))
    methods = {p.method for p in pages}
    return pages, "digital" if methods == {"digital"} else "ocr" if methods == {"ocr"} else "mixed"


def _extract_image(path: Path) -> tuple[list[PageText], str]:
    res = ocr_image(_open_image(path))
    return [PageText(number=1, text=res.text, method="ocr", ocr_confidence=res.confidence)], "ocr"


def _extract_docx(path: Path) -> tuple[list[PageText], str]:
    from docx import Document

    try:
        doc = Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("That DOCX could not be opened; it may be corrupt.") from exc
    pages: list[list[str]] = [[]]
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            pages[-1].append("".join(t.text or "" for t in child.iter() if t.tag.endswith("}t")))
            if 'w:type="page"' in child.xml:
                pages.append([])
        elif tag == "tbl":
            from docx.table import Table

            for row in Table(child, doc).rows:
                pages[-1].append("  ".join(c.text.strip() for c in row.cells))
    out = [PageText(number=i, text="\n".join(lines).strip(), method="docx")
           for i, lines in enumerate(pages, start=1) if "".join(lines).strip()]
    return out or [PageText(number=1, text="", method="docx")], "docx"


def extract_document(path: Path, display_name: str | None = None) -> ExtractedDocument:
    """Extract per-page text from a stored upload.

    Raises:
        IngestionError: unsupported/corrupt file, too many pages, oversized image.
    """
    start = time.perf_counter()
    with path.open("rb") as fh:
        ftype = detect_file_type(fh.read())
    if ftype is None:
        raise IngestionError("Unsupported file. Please upload a PDF, PNG, JPG or DOCX.")
    pages, method = {"pdf": _extract_pdf, "png": _extract_image, "jpg": _extract_image,
                     "docx": _extract_docx}[ftype](path)
    confs = [p.ocr_confidence for p in pages if p.ocr_confidence is not None]
    result = ExtractedDocument(
        filename=display_name or path.name, file_type=ftype, pages=pages, method=method,  # type: ignore[arg-type]
        ocr_confidence=sum(confs) / len(confs) if confs else None,
        processing_seconds=round(time.perf_counter() - start, 3))
    logger.info("Extracted %s: type=%s pages=%d method=%s chars=%d in %.2fs",
                result.filename, ftype, len(pages), method, result.n_chars,
                result.processing_seconds)
    return result
