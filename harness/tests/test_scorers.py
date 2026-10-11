from pricing_reversal_harness.scorers import get_scorer, registered_scorer_names
from pricing_reversal_harness.scorers.boxed import score_boxed_integer
from pricing_reversal_harness.scorers.mcq import score_mcq_letter
from pricing_reversal_harness.scorers.none_scorer import score_none


def test_boxed_integer_exact_match():
    score, pred = score_boxed_integer("Final answer: \\boxed{33}", "33")
    assert score == 1.0
    assert pred == "33"


def test_boxed_integer_mismatch():
    score, pred = score_boxed_integer("\\boxed{24}", "33")
    assert score == 0.0
    assert pred == "24"


def test_boxed_nested_fraction_reduces():
    score, pred = score_boxed_integer("Answer: \\boxed{\\tfrac{25}{8}}", "25/8")
    assert score == 1.0
    assert pred == "25/8"


def test_boxed_last_wins_when_multiple():
    text = "first: \\boxed{1}\nfinal: \\boxed{7}"
    score, pred = score_boxed_integer(text, "7")
    assert score == 1.0
    assert pred == "7"


def test_boxed_fallback_last_number():
    score, pred = score_boxed_integer("The answer is 42", "42")
    assert pred == "42"
    assert score == 1.0


def test_boxed_no_prediction():
    score, pred = score_boxed_integer("", "42")
    assert score is None
    assert pred is None


def test_mcq_last_answer_wins():
    text = "Working out shows Answer: A\n... Answer: C"
    score, pred = score_mcq_letter(text, "C")
    assert score == 1.0
    assert pred == "C"


def test_mcq_dollar_prefix():
    score, pred = score_mcq_letter("Answer: $D", "D")
    assert score == 1.0
    assert pred == "D"


def test_mcq_ground_truth_case_insensitive():
    score, pred = score_mcq_letter("Answer: b", "B")
    assert score == 1.0
    assert pred == "B"


def test_mcq_no_letter_found():
    score, pred = score_mcq_letter("no answer here", "A")
    assert score is None
    assert pred is None


def test_mcq_wrong_letter():
    score, pred = score_mcq_letter("Answer: A", "B")
    assert score == 0.0
    assert pred == "A"


# ---- MMLU-Pro (A-J) coverage --------------------------------------------


def test_mcq_ten_option_correct():
    """MMLU-Pro's ground truth range is A-J. The scorer accepts it."""
    for letter in "EFGHIJ":
        score, pred = score_mcq_letter(f"Working out...\nAnswer: {letter}", letter)
        assert score == 1.0, f"letter {letter} should match"
        assert pred == letter


def test_mcq_ten_option_wrong():
    score, pred = score_mcq_letter("Answer: H", "J")
    assert score == 0.0
    assert pred == "H"


def test_mcq_ten_option_case_insensitive():
    score, pred = score_mcq_letter("Answer: g", "G")
    assert score == 1.0
    assert pred == "G"


def test_mcq_rejects_letters_past_j():
    """K is outside the A-J MMLU-Pro range. Ground truth K → None."""
    score, pred = score_mcq_letter("Answer: A", "K")
    # We still extract the prediction but refuse to score against an
    # out-of-range ground truth.
    assert score is None
    assert pred == "A"


# ---- `none` scorer + registry -------------------------------------------


def test_none_scorer_returns_null_score_and_prediction():
    score, pred = score_none("anything", "anything")
    assert score is None
    assert pred is None


def test_registry_exposes_all_registered_names():
    # Includes `dispatch_by_source_dataset` — the meta scorer used by
    # the totrandom dataset, which resolves the real scorer per row.
    # config-lint validates dataset.scorer values against this set.
    names = registered_scorer_names()
    assert names == {
        "boxed_integer",
        "mcq_letter",
        "none",
        "dispatch_by_source_dataset",
    }


def test_registry_lookup_by_name():
    assert get_scorer("boxed_integer") is score_boxed_integer
    assert get_scorer("mcq_letter") is score_mcq_letter
    assert get_scorer("none") is score_none


def test_registry_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        get_scorer("does-not-exist")
