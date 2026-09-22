# tests/test_eval_metrics.py
from src.evaluation.evaluate import (
    recall_at_k,
    reciprocal_rank,
    section_matches,
    source_matches,
)


def _make_result(source, section, doc_type="act"):
    return {
        "text": "test",
        "metadata": {"source": source, "section": section, "doc_type": doc_type},
    }


# ------------------------------------------------------------
# section_matches
# ------------------------------------------------------------
def test_section_matches_exact():
    assert section_matches("302", "302") is True


def test_section_matches_with_trailing_dot():
    assert section_matches("302.", "302") is True


def test_section_matches_none_expected():
    assert section_matches("anything", None) is True


def test_section_matches_mismatch():
    assert section_matches("302", "303") is False


# ------------------------------------------------------------
# source_matches
# ------------------------------------------------------------
def test_source_matches_normalizes_underscores():
    assert source_matches("rent_control_act.pdf", "rent control") is True


def test_source_matches_normalizes_hyphens():
    assert source_matches("Lok-Sewa-Guarantee.pdf", "Lok_Sewa") is True


def test_source_matches_case_insensitive():
    assert source_matches("IPC_eng.pdf", "ipc") is True


def test_source_matches_none_expected():
    assert source_matches("anything.pdf", None) is True


# ------------------------------------------------------------
# recall_at_k
# ------------------------------------------------------------
def test_recall_at_k_hit_at_first():
    results = [_make_result("IPC.pdf", "302")]
    assert recall_at_k(results, "IPC", "302", k=5) == 1


def test_recall_at_k_hit_later():
    results = [
        _make_result("IPC.pdf", "100"),
        _make_result("IPC.pdf", "200"),
        _make_result("IPC.pdf", "302"),
    ]
    assert recall_at_k(results, "IPC", "302", k=5) == 1


def test_recall_at_k_miss():
    results = [_make_result("IPC.pdf", "100")]
    assert recall_at_k(results, "IPC", "302", k=5) == 0


def test_recall_at_k_out_of_window():
    results = [_make_result("IPC.pdf", "100")] * 10 + [_make_result("IPC.pdf", "302")]
    assert recall_at_k(results, "IPC", "302", k=5) == 0


# ------------------------------------------------------------
# reciprocal_rank
# ------------------------------------------------------------
def test_reciprocal_rank_first_position():
    results = [_make_result("IPC.pdf", "302")]
    assert reciprocal_rank(results, "IPC", "302") == 1.0


def test_reciprocal_rank_third_position():
    results = [
        _make_result("IPC.pdf", "100"),
        _make_result("IPC.pdf", "200"),
        _make_result("IPC.pdf", "302"),
    ]
    assert reciprocal_rank(results, "IPC", "302") == 1.0 / 3


def test_reciprocal_rank_no_hit():
    results = [_make_result("IPC.pdf", "100")]
    assert reciprocal_rank(results, "IPC", "302") == 0.0