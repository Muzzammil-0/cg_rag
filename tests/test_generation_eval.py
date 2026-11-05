# tests/test_generation_eval.py
from src.evaluation.evaluate_generation import (
    keyword_coverage,
    is_real_answer,
    cited_chunks,
    aggregate_generation_metrics,
)


def test_keyword_coverage_full():
    assert keyword_coverage("death and imprisonment for life", ["death", "imprisonment for life"]) == 1.0


def test_keyword_coverage_partial():
    assert keyword_coverage("death only", ["death", "fine"]) == 0.5


def test_keyword_coverage_no_keywords():
    assert keyword_coverage("anything", []) == 0.0


def test_is_real_answer_true():
    assert is_real_answer("The punishment is death.") is True


def test_is_real_answer_not_found():
    assert is_real_answer("Not found in provided context.") is False


def test_is_real_answer_error():
    assert is_real_answer("Error.") is False


def test_is_real_answer_empty():
    assert is_real_answer("") is False


def test_cited_chunks_filters_out_of_range():
    results = [{"text": "a"}, {"text": "b"}]
    assert cited_chunks(results, [1, 2, 5]) == [{"text": "a"}, {"text": "b"}]


def test_aggregate_generation_metrics_basic():
    rows = [
        {"expected_source": "IPC", "answered": True, "citation_hit": True, "keyword_coverage": 1.0, "confidence": 0.9},
        {"expected_source": "IPC", "answered": True, "citation_hit": False, "keyword_coverage": 0.5, "confidence": 0.5},
    ]
    s = aggregate_generation_metrics(rows)
    assert s["total_evaluated"] == 2
    assert s["answer_rate"] == 1.0
    assert s["citation_grounding"] == 0.5
    assert s["keyword_coverage"] == 0.75
    assert s["mean_confidence"] == 0.7


def test_aggregate_generation_metrics_skips_errors_and_notfound():
    rows = [
        {"expected_source": "IPC", "answered": True, "citation_hit": True, "keyword_coverage": 1.0, "confidence": 0.8},
        {"expected_source": "IPC", "error": "boom"},
        {"expected_source": None, "answered": False, "citation_hit": False, "keyword_coverage": 0.0, "confidence": 0.0},
    ]
    s = aggregate_generation_metrics(rows)
    assert s["total_evaluated"] == 1
    assert s["answer_rate"] == 1.0


def test_aggregate_generation_metrics_empty():
    assert aggregate_generation_metrics([]) == {"total_evaluated": 0}