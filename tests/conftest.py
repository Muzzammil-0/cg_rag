# tests/conftest.py
import io
import sys
import builtins
from pathlib import Path
from unittest.mock import MagicMock

# ------------------------------------------------------------
# 1. Make repo root importable
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------
# 2. Mock heavy modules BEFORE anything imports them
# ------------------------------------------------------------
def _mock_module(name, attrs):
    mod = MagicMock()
    for attr, val in attrs.items():
        setattr(mod, attr, val)
    sys.modules[name] = mod


# sentence-transformers: no model loading
_mock_module("sentence_transformers", {
    "SentenceTransformer": MagicMock(return_value=MagicMock()),
    "CrossEncoder": MagicMock(return_value=MagicMock()),
})

# chromadb: no persistent client
_mock_module("chromadb", {
    "PersistentClient": MagicMock(return_value=MagicMock()),
})

# groq: no API client
_mock_module("groq", {
    "Groq": MagicMock(return_value=MagicMock()),
})


# ------------------------------------------------------------
# 3. Patch open() so chunks.jsonl reads return a tiny fake file
# ------------------------------------------------------------
_original_open = builtins.open

_FAKE_CHUNKS = (
    '{"text": "Section 1. Short title. This Act may be called the Test Act.", '
    '"metadata": {"source": "test_act.pdf", "doc_type": "act", "section": "1"}}\n'
    '{"text": "Section 2. Definition. In this Act, test means a sample.", '
    '"metadata": {"source": "test_act.pdf", "doc_type": "act", "section": "2"}}\n'
    '{"text": "Paragraph 5. The appellant challenges the order.", '
    '"metadata": {"source": "test_judgment.pdf", "doc_type": "judgment", "para": "5"}}\n'
)


def _patched_open(file, *args, **kwargs):
    if str(file).endswith("chunks.jsonl"):
        return io.StringIO(_FAKE_CHUNKS)
    return _original_open(file, *args, **kwargs)


builtins.open = _patched_open