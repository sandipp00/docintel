import random

import pytest
from faker import Faker

from src.classifier import DOC_TYPES, DocumentClassifier, model_path
from training.generate_data import make_doc

pytestmark = pytest.mark.skipif(not model_path().exists(), reason="model not trained")


@pytest.fixture(scope="module")
def clf() -> DocumentClassifier:
    return DocumentClassifier.load()


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_fresh_synthetic_docs_classified(clf, doc_type):
    fake = Faker("en_US")
    fake.seed_instance(777)
    rng = random.Random(777)
    docs = [make_doc(rng, fake, doc_type, f"x{i}", allow_held_out=True) for i in range(10)]
    preds = clf.predict_many([d.text for d in docs])
    assert sum(p.label == doc_type for p in preds) >= 9


def test_probabilities_are_a_distribution(clf):
    p = clf.predict("Invoice No: 123 Total Due: $40.00 Subtotal Tax")
    assert p.label == "invoice"
    assert abs(sum(p.probabilities.values()) - 1) < 1e-3
    assert p.confidence == max(p.probabilities.values())


def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_classifier"):
        DocumentClassifier.load(tmp_path / "nope.joblib")
