# tests/test_chunker.py
from src.chunking.chunker import (
    chunk_generic_act,
    chunk_judgment,
    clean_text,
    is_noise_chunk,
    is_toc_chunk,
    split_constitution_subsections,
)


SAMPLE_ACT = """1. Short title. This Act may be called the Test Act, 2024.
2. Definition. In this Act, "test" means a sample.
3. Application. This Act applies to all tests.
"""

SAMPLE_JUDGMENT = """1. This appeal is filed by the appellant challenging the judgment and decree dated 14.11.2005 passed by the learned First Additional District Judge, Mahasamund in Civil Suit No. 18-A/2003, whereby the learned trial Court dismissed the suit filed by the appellant/plaintiff.
2. Facts relevant for disposal of this appeal are that the appellant/plaintiff filed a suit for declaring the electricity bill dated 07.08.2003 amounting to Rs.1,60,610/- to be illegal, and also sought grant of permanent injunction restraining the defendants from disconnecting the power supply line of the plaintiff's rice mill.
3. The respondents/defendants filed their written statement denying the pleadings made by the plaintiff, and further pleaded that as the plaintiff's rice mill is situated more than 100 metres from the National Highway, there was no possibility of the alleged theft having occurred in the manner described.
4. Learned counsel for the appellant submits that the trial Court erred in dismissing the suit on the ground of limitation, and that the finding recorded by the trial Court is perverse and contrary to the material on record.
"""

def test_chunk_generic_act_splits_by_section():
    chunks = chunk_generic_act(SAMPLE_ACT, "test_act.pdf")
    assert len(chunks) == 3
    assert chunks[0]["metadata"]["section"] == "1"
    assert chunks[0]["metadata"]["doc_type"] == "act"
    assert "Short title" in chunks[0]["text"]


def test_chunk_judgment_splits_by_paragraph():
    chunks = chunk_judgment(SAMPLE_JUDGMENT, "test_judgment.pdf")
    assert len(chunks) >= 3
    assert chunks[0]["metadata"]["doc_type"] == "judgment"
    assert chunks[0]["metadata"]["para"] == "1"
    assert "challenging the judgment" in chunks[0]["text"]


def test_chunk_judgment_falls_back_when_no_paragraphs():
    # Text with no numbered paragraphs → single chunk fallback
    text = "This is a short order without paragraph numbers."
    chunks = chunk_judgment(text, "short_order.pdf")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["para"] == "unknown"


def test_clean_text_collapses_excess_newlines():
    text = "Line 1\n\n\n\n\nLine 2"
    cleaned = clean_text(text)
    assert "\n\n\n" not in cleaned


def test_is_noise_chunk_rejects_short_text():
    chunk = {"text": "1. Short.", "metadata": {}}
    assert is_noise_chunk(chunk) is True


def test_is_noise_chunk_accepts_real_content():
    chunk = {
        "text": "1. This is a substantive legal paragraph with enough content to pass the noise filter.",
        "metadata": {},
    }
    assert is_noise_chunk(chunk) is False


def test_split_constitution_subsections_does_not_split_non_constitution():
    chunks = [
        {"text": "21. Protection of life.", "metadata": {"doc_type": "act", "section": "21"}}
    ]
    result = split_constitution_subsections(chunks)
    assert len(result) == 1