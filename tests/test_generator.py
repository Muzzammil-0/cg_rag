# tests/test_generator.py
from src.generation.generator import extract_json, verify_provision_quotes


# ------------------------------------------------------------
# extract_json
# ------------------------------------------------------------
def test_extract_json_array_returns_list():
    # A bare JSON array parses to a list; generate_answer guards against this
    # with an isinstance(parsed, dict) check before using it.
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


# ------------------------------------------------------------
# verify_provision_quotes
# ------------------------------------------------------------
def test_verify_keeps_genuine_quote_from_cited_act():
    results = [
        {"metadata": {"source": "BNS.pdf", "section": "103"},
         "text": "103. Whoever commits murder shall be punished with death or imprisonment for life and shall also be liable to fine."},
    ]
    parsed = {"provisions": [
        {"source": "BNS", "section": "103",
         "quote": "Whoever commits murder shall be punished with death or imprisonment for life"},
    ]}
    out, dropped = verify_provision_quotes(parsed, results, strict=True)
    assert dropped == 0
    assert len(out["provisions"]) == 1
    assert out["provisions"][0]["quote_verified"] is True


def test_verify_drops_quote_found_only_in_a_different_act():
    # Both acts have a "section 138", but the quote only exists in OtherAct.
    # Citing Negotiable must NOT validate against OtherAct's text.
    results = [
        {"metadata": {"source": "Negotiable.pdf", "section": "138"},
         "text": "138. Dishonour of cheque for insufficiency of funds in the account."},
        {"metadata": {"source": "OtherAct.pdf", "section": "138"},
         "text": "138. The quick brown fox jumps over the lazy dog repeatedly every day."},
    ]
    parsed = {"provisions": [
        {"source": "Negotiable Instruments Act", "section": "138",
         "quote": "The quick brown fox jumps over the lazy dog repeatedly"},
    ]}
    out, dropped = verify_provision_quotes(parsed, results, strict=True)
    assert dropped == 1
    assert out["provisions"] == []


def test_verify_falls_back_when_source_name_matches_nothing():
    # Inconsistent source naming should not drop a genuinely present quote.
    results = [
        {"metadata": {"source": "BNS.pdf", "section": "103"},
         "text": "103. Whoever commits murder shall be punished with death or imprisonment for life."},
    ]
    parsed = {"provisions": [
        {"source": "Totally Unknown Name", "section": "103",
         "quote": "Whoever commits murder shall be punished with death"},
    ]}
    out, dropped = verify_provision_quotes(parsed, results, strict=True)
    assert dropped == 0
    assert out["provisions"][0]["quote_verified"] is True


def test_verify_drops_provision_missing_section():
    results = [
        {"metadata": {"source": "BNS.pdf", "section": "103"}, "text": "103. Some text."},
    ]
    parsed = {"provisions": [{"source": "BNS", "section": "", "quote": "Some text."}]}
    out, dropped = verify_provision_quotes(parsed, results, strict=True)
    assert dropped == 1
    assert out["provisions"] == []


def test_verify_non_strict_marks_unverified_but_keeps():
    results = [
        {"metadata": {"source": "BNS.pdf", "section": "103"},
         "text": "103. Whoever commits murder shall be punished with death."},
    ]
    parsed = {"provisions": [
        {"source": "BNS", "section": "103", "quote": "An entirely fabricated quotation that appears nowhere at all"},
    ]}
    out, dropped = verify_provision_quotes(parsed, results, strict=False)
    assert dropped == 0
    assert len(out["provisions"]) == 1
    assert out["provisions"][0]["quote_verified"] is False