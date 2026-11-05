# tests/conftest.py
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ------------------------------------------------------------
# Make repo root importable
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------
# Mock heavy modules BEFORE anything imports them, so tests never
# load models, open a Chroma client, or call the Groq API.
# ------------------------------------------------------------
def _mock_module(name, attrs):
    mod = MagicMock()
    for attr, val in attrs.items():
        setattr(mod, attr, val)
    sys.modules[name] = mod


_mock_module("sentence_transformers", {
    "SentenceTransformer": MagicMock(return_value=MagicMock()),
    "CrossEncoder": MagicMock(return_value=MagicMock()),
})

_mock_module("chromadb", {
    "PersistentClient": MagicMock(return_value=MagicMock()),
})

_mock_module("groq", {
    "Groq": MagicMock(return_value=MagicMock()),
})