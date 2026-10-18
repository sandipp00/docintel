import random
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf
import pytest
from faker import Faker

from src.ingestion import IngestionError, detect_file_type, extract_document
from src.ocr import is_tesseract_available
from training.generate_data import make_doc, render

needs_ocr = pytest.mark.skipif(not is_tesseract_available(), reason="tesseract not installed")


def _doc(doc_type: str, seed: int = 1):
    fake = Faker("en_US")
    fake.seed_instance(seed)
    return make_doc(random.Random(seed), fake, doc_type, f"t{seed}", allow_held_out=False)


def _sim(a: str, b: str) -> float:
    norm = lambda s: " ".join(s.split()).lower()  # noqa: E731
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def test_digital_pdf_text_and_pages(tmp_path: Path):
    doc = _doc("contract", 3)
    p = render(doc, tmp_path / "c", "pdf", random.Random(1))
    out = extract_document(p)
    assert out.method == "digital" and out.file_type == "pdf"
    assert [pg.number for pg in out.pages] == list(range(1, len(out.pages) + 1))
    assert doc.fields["parties"][0].split()[0] in out.full_text
    assert _sim(doc.text.replace("\f", "\n"), out.full_text) > 0.9


def test_docx_pages(tmp_path: Path):
    doc = _doc("contract", 4)
    p = render(doc, tmp_path / "c", "docx", random.Random(1))
    out = extract_document(p)
    assert out.method == "docx"
    assert len(out.pages) == len(doc.text.split("\f"))


@needs_ocr
def test_png_ocr_accuracy(tmp_path: Path):
    doc = _doc("invoice", 5)
    p = render(doc, tmp_path / "i", "png", random.Random(2))
    out = extract_document(p)
    assert out.method == "ocr" and out.ocr_confidence and out.ocr_confidence > 0.6
    assert _sim(doc.text, out.full_text) > 0.8


@needs_ocr
def test_scanned_pdf_falls_back_to_ocr(tmp_path: Path):
    doc = _doc("invoice", 6)
    png = render(doc, tmp_path / "i", "png", random.Random(2))
    pdf = tmp_path / "scan.pdf"
    d = pymupdf.open()
    page = d.new_page()
    page.insert_image(page.rect, filename=str(png))
    d.save(pdf)
    out = extract_document(pdf)
    assert out.method == "ocr" and out.pages[0].method == "ocr"
    assert doc.fields["invoice_number"] in out.full_text.replace(" ", "")


def test_detect_by_magic_bytes_not_extension(tmp_path: Path):
    assert detect_file_type(b"%PDF-1.7 ...") == "pdf"
    assert detect_file_type(b"MZ\x90\x00") is None
    fake_pdf = tmp_path / "evil.pdf"
    fake_pdf.write_bytes(b"MZ\x90\x00 not a pdf")
    with pytest.raises(IngestionError):
        extract_document(fake_pdf)


def test_corrupt_pdf_is_friendly_error(tmp_path: Path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\ngarbage")
    with pytest.raises(IngestionError):
        extract_document(bad)
