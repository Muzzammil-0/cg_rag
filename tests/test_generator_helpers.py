# tests/test_generator_helpers.py
from src.generation.generator import (
    extract_json,
    validate_citations,
    validate_reasoning_provisions,
)


# ------------------------------------------------------------
# extract_json
# ------------------------------------------------------------
def test_extract_json_direct():
    text = '{"answer": "test", "citations": [1]}'
    parsed = extract_json(text)
    assert parsed == {"answer": "test", "citations": [1]}


def test_extract_json_strips_thinking_block():
    text = '<think>I should answer carefully.</think>{"answer": "test", "citations": [1]}'
    parsed = extract_json(text)
    assert parsed == {"answer": "test", "citations": [1]}


def test_extract_json_strips_code_fence():
    text = '```json\n{"answer": "test", "citations": [1]}\n```'
    parsed = extract_json(text)
    assert parsed == {"answer": "test", "citations": [1]}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("this is not json at all") is None


def test_extract_json_handles_text_before_json():
    text = 'Some preamble text. {"answer": "test", "citations": [1]}'
    parsed = extract_json(text)
    assert parsed is not None
    assert parsed["answer"] == "test"


# ------------------------------------------------------------
# validate_citations
# ------------------------------------------------------------
def test_validate_citations_drops_out_of_range():
    parsed = {"citations": [1, 2, 5, 10]}  # only 3 results
    results = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    parsed = validate_citations(parsed, results)
    assert parsed["citations"] == [1, 2]


def test_validate_citations_converts_strings():
    parsed = {"citations": ["1", "2"]}
    results = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    parsed = validate_citations(parsed, results)
    assert parsed["citations"] == [1, 2]


def test_validate_citations_handles_missing_key():
    parsed = {"answer": "test"}
    results = [{"text": "a"}]
    parsed = validate_citations(parsed, results)
    assert parsed["citations"] == []


# ------------------------------------------------------------
# validate_reasoning_provisions
# ------------------------------------------------------------
def test_validate_reasoning_provisions_drops_hallucinated():
    parsed = {
        "provisions": [
            {"source": "IPC.pdf", "section": "376"},
            {"source": "BNS.pdf", "section": "999"},  # not in results
        ]
    }
    results = [
        {"metadata": {"section": "376", "article": "", "para": ""}, "text": "..."},
    ]
    parsed = validate_reasoning_provisions(parsed, results)
    assert len(parsed["provisions"]) == 1
    assert parsed["provisions"][0]["section"] == "376"


def test_validate_reasoning_provisions_keeps_all_when_all_present():
    parsed = {
        "provisions": [
            {"source": "IPC.pdf", "section": "302"},
            {"source": "IPC.pdf", "section": "34"},
        ]
    }
    results = [
        {"metadata": {"section": "302", "article": "", "para": ""}, "text": "..."},
        {"metadata": {"section": "34", "article": "", "para": ""}, "text": "..."},
    ]
    parsed = validate_reasoning_provisions(parsed, results)
    assert len(parsed["provisions"]) == 2