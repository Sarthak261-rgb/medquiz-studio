import pytest

from medquiz.metrics import cohen_kappa, confusion, human_summary, wilson


def test_wilson_interval():
    lo, hi = wilson(18, 100)
    assert lo == pytest.approx(0.1170, abs=1e-3)
    assert hi == pytest.approx(0.2664, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(0, 10)[0] == 0.0


def test_cohen_kappa_matches_hand_computed_value():
    # 2x2 table: a=20 both yes, b=5, c=10, d=15  -> po=0.70, pe=0.50, kappa=0.40
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.40)
    assert cohen_kappa([1, 1], [1, 1]) is None  # chance agreement is 1


def test_confusion():
    c = confusion([True, True, False, False], [True, False, True, False])
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == 0.5 and c["recall"] == 0.5 and c["accuracy"] == 0.5


def test_human_summary_counts_primary_reason_once():
    qs = [{"id": f"q{i}", "chunk_id": "19.1-000" if i < 2 else "22.1-000", "target_type": "recall"} for i in range(4)]
    reviews = {
        "q0": {"decision": "accept", "reasons": [], "seconds": 10},
        "q1": {"decision": "reject", "reasons": ["multiple_correct", "weak_distractors"], "seconds": 30},
        "q2": {"decision": "reject", "reasons": ["weak_distractors"], "seconds": 20},
    }
    h = human_summary(qs, reviews)
    assert h["n_reviewed"] == 3 and h["n_total"] == 4
    assert h["rejection"]["k"] == 2
    assert h["primary_reason"]["multiple_correct"]["k"] == 1
    assert h["any_reason"]["weak_distractors"]["k"] == 2
    assert h["by_chapter"]["19"]["k"] == 1 and h["by_chapter"]["22"]["n"] == 1
    assert h["review_seconds_median"] == 20
