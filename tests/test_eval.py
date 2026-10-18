from evaluation.run_eval import answer_matches


def test_answer_matches_ignores_case_and_punctuation():
    assert answer_matches("€22,158.30", "Grand Total: 22158.30 EUR")
    assert answer_matches("Prime Thompson Consulting", "issued by prime thompson consulting.")
    assert not answer_matches("INV-54399", "Invoice INV-54398")


def test_answer_matches_dates_across_formats():
    assert answer_matches("Oct 22, 2022", "effective October 22, 2022")
    assert answer_matches("29 March 2025", "Effective Date: 2025-03-29")
