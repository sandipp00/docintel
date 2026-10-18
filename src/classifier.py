"""Document-type classifier: TF-IDF + Logistic Regression (scikit-learn). No LLM involved."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import get_settings

logger = logging.getLogger(__name__)

DOC_TYPES = ["invoice", "resume", "contract", "report", "other"]
MAX_CHARS = 6000  # only the start of a document is needed to tell its type


class Classification(BaseModel):
    """Predicted document type with confidence and the full probability distribution."""

    label: str
    confidence: float
    probabilities: dict[str, float]


def build_pipeline() -> Pipeline:
    """TF-IDF (1-2 grams) feeding a multinomial logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  max_features=40_000)),
        ("clf", LogisticRegression(C=10.0, max_iter=1000)),
    ])


def model_path() -> Path:
    return get_settings().models_dir / "classifier.joblib"


class DocumentClassifier:
    """Thin wrapper around a fitted sklearn pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: Path | None = None) -> DocumentClassifier:
        """Load the trained model; raises FileNotFoundError with a hint if it is missing."""
        path = path or model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m training.generate_data && "
                "python -m training.train_classifier")
        return cls(joblib.load(path))

    def save(self, path: Path | None = None) -> Path:
        path = path or model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path, compress=3)
        return path

    def predict(self, text: str) -> Classification:
        """Classify one document's text."""
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[Classification]:
        """Classify several documents at once."""
        probs = self.pipeline.predict_proba([t[:MAX_CHARS] for t in texts])
        classes = list(self.pipeline.classes_)
        out = []
        for row in probs:
            best = int(row.argmax())
            out.append(Classification(
                label=classes[best], confidence=round(float(row[best]), 4),
                probabilities={c: round(float(p), 4) for c, p in zip(classes, row, strict=True)}))
        return out
