
try:
    import spaces
    _SPACES_AVAILABLE = True
except ImportError:
    _SPACES_AVAILABLE = False


# Register a dummy GPU function to satisfy ZeroGPU startup scan
if _SPACES_AVAILABLE:
    @spaces.GPU
    def _dummy_gpu_marker():
        return "ok"


import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from src.generation.generator import generate_answer
from src.embeddings.startup_embed import build_if_needed


# --- Build ChromaDB before the server starts ---
print("Pre-startup: building ChromaDB...")
build_if_needed()
print("Pre-startup: complete.")

print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))


def chat_fn(message, history):
    if not message or len(message.strip()) < 3:
        return "Please enter a legal question."

    try:
        answer_data, results = generate_answer(message, top_k=5, verbose=False)
    except Exception as e:
        return f"Error: {e}"

    answer = answer_data.get("answer", "")
    citations = answer_data.get("citations", [])

    citation_lines = []
    for c in citations:
        idx = c - 1
        if 0 <= idx < len(results):
            m = results[idx]["metadata"]
            src = m.get("source", "")
            sec = m.get("section", m.get("article", ""))
            citation_lines.append(f"[{c}] {src} — Section {sec}")

    if citation_lines:
        answer += "\n\n**Sources:**\n" + "\n".join(citation_lines)

    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="⚖️ CG Legal RAG",
    description="Ask legal questions about Indian and Chhattisgarh law. Answers are cited from statutes.",
    examples=[
        "What is the punishment for murder under BNS?",
        "What is the procedure for registering an FIR under CrPC?",
        "What are the grounds for eviction under the Chhattisgarh Rent Control Act?",
    ],
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
)