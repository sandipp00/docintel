"""Tesseract OCR with light preprocessing (grayscale, upscale, deskew) and confidence."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from src.config import get_settings

logger = logging.getLogger(__name__)

_configured = False

# Default install locations, tried when tesseract isn't on PATH and TESSERACT_CMD is unset.
_WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


@dataclass
class OcrResult:
    """OCR text and mean word confidence in [0, 1] (None if no words were found)."""

    text: str
    confidence: float | None


def configure_tesseract() -> None:
    """Point pytesseract at the configured binary (once)."""
    global _configured
    if _configured:
        return
    cmd = get_settings().tesseract_cmd
    if not cmd and shutil.which("tesseract") is None:
        cmd = next((p for p in _WINDOWS_TESSERACT_PATHS if os.path.isfile(p)), None)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    _configured = True


def is_tesseract_available() -> bool:
    """True if a tesseract binary can be found/executed."""
    configure_tesseract()
    cmd = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(cmd) is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - any failure means unavailable
        return False


def estimate_skew(img: Image.Image, max_angle: float = 4.0, step: float = 0.5) -> float:
    """Estimate page skew in degrees via projection-profile variance on a small binary copy."""
    small = ImageOps.contain(img.convert("L"), (600, 800))
    arr = np.asarray(small)
    binary = Image.fromarray(((arr < arr.mean() - 25) * 255).astype("uint8"))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-6, step):
        rows = np.asarray(binary.rotate(float(angle), fillcolor=0)).sum(axis=1).astype("float64")
        score = float(np.var(rows))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def preprocess(img: Image.Image) -> Image.Image:
    """Grayscale, autocontrast, upscale small images, and correct small skews."""
    img = ImageOps.exif_transpose(img).convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((1000, int(img.height * scale)), Image.LANCZOS)
    angle = estimate_skew(img)
    if abs(angle) >= 0.5:
        img = img.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
    return img


def ocr_image(img: Image.Image) -> OcrResult:
    """Run Tesseract on a PIL image and return text (line structure kept) plus confidence."""
    configure_tesseract()
    data = pytesseract.image_to_data(preprocess(img), lang="eng", config="--psm 6",
                                     output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return OcrResult(text=text, confidence=(sum(confs) / len(confs) / 100) if confs else None)
