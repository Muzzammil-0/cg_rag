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
from src.generation.generator import generate_answer, generate_reasoning
from src.embeddings.startup_embed import build_if_needed


# --- Build ChromaDB before the server starts ---
print("Pre-startup: building ChromaDB...")
build_if_needed()
print("Pre-startup: complete.")

print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))


# ------------------------------------------------------------
# Tab 1: Ask a legal question
# ------------------------------------------------------------
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
            sec = m.get("section", m.get("article", m.get("para", "")))
            citation_lines.append(f"[{c}] {src} — {sec}")

    if citation_lines:
        answer += "\n\n**Sources:**\n" + "\n".join(citation_lines)

    return answer


# ------------------------------------------------------------
# Tab 2: Simulate case reasoning
# ------------------------------------------------------------
def reasoning_fn(fact_pattern):
    if not fact_pattern or len(fact_pattern.strip()) < 20:
        return (
            "Please enter a fact pattern (at least 20 characters).",
            "", "", "", "", "", "",
        )

    # Reject question-style input — this tab expects a scenario, not a question
    text = fact_pattern.strip().lower()
    question_starters = (
        "what ", "how ", "when ", "why ", "who ", "is ", "are ",
        "can ", "does ", "should ", "which ", "where ",
    )
    if text.startswith(question_starters):
        msg = (
            "**This tab expects a fact pattern, not a question.**\n\n"
            "**Wrong:** *What are the conditions for rape?*\n\n"
            "**Right:** *A 25-year-old man forced himself on a 20-year-old woman at a party. "
            "She reported the next day. The defence claims the act was consensual.*\n\n"
            "If you want to ask a question, use the **Ask a legal question** tab."
        )
        return (msg, "", "", "", "", "", "")
    try:
        result, results = generate_reasoning(fact_pattern, top_k=10, verbose=False)
    except Exception as e:
        return (f"Error: {e}", "", "", "", "", "", "")

    if result.get("error"):
        return (
            f"Failed to parse model output. Try again.\n\nRaw: {result.get('raw', '')[:500]}",
            "", "", "", "", "", "",
        )

    # Format each field as markdown
    issues = "\n".join(f"- {x}" for x in result.get("issues", [])) or "—"

    provisions_lines = []
    for p in result.get("provisions", []):
        provisions_lines.append(
            f"- **{p.get('source', '')} {p.get('section', '')}** — {p.get('relevance', '')}"
        )
    provisions = "\n".join(provisions_lines) or "—"

    pet = "\n".join(f"- {x}" for x in result.get("petitioner_arguments", [])) or "—"
    resp = "\n".join(f"- {x}" for x in result.get("respondent_arguments", [])) or "—"

    reasoning = result.get("likely_reasoning", "") or "—"
    outcome = result.get("possible_outcome", "") or "—"

    # Citations + sources
    citations = result.get("citations", [])
    source_lines = []
    for c in citations:
        idx = c - 1
        if 0 <= idx < len(results):
            m = results[idx]["metadata"]
            src = m.get("source", "")
            sec = m.get("section", m.get("article", m.get("para", "")))
            source_lines.append(f"[{c}] {src} — {sec}")

    confidence = result.get("confidence", 0.0)
    sources_md = ""
    if source_lines:
        sources_md = "\n\n**Sources:**\n" + "\n".join(source_lines)
    sources_md += f"\n\n**Confidence:** {confidence}"

    return (
        issues,
        provisions,
        pet,
        resp,
        reasoning,
        outcome,
        sources_md,
    )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
with gr.Blocks(title="CG Legal RAG") as demo:
    gr.Markdown("# ⚖️ CG Legal RAG")
    gr.Markdown(
        "Legal research assistant for Indian and Chhattisgarh law. "
        "Two modes: **Ask** (quick cited answers) and **Reason** (simulate judicial reasoning on a fact pattern)."
    )

    with gr.Tab("💬 Ask a legal question"):
        gr.ChatInterface(
            fn=chat_fn,
            title="",
            description="Ask a legal question in plain English. Answers are cited from statutes and judgments.",
            examples=[
                "What is the punishment for murder under BNS?",
                "What is the procedure for registering an FIR under CrPC?",
                "What are the grounds for eviction under the Chhattisgarh Rent Control Act?",
                "Show me case law on electricity theft in Chhattisgarh.",
            ],
        )

    with gr.Tab("⚖️ Simulate case reasoning"):
        gr.Markdown(
            "Enter a fact pattern. The system retrieves relevant provisions and judgments, "
            "then simulates how a Chhattisgarh court might reason through it.\n\n"
            "**Educational use only. Not legal advice. Not a prediction of any actual case.**"
        )

        fact_input = gr.Textbox(
            label="Fact pattern",
            placeholder=(
                "Example: A tenant has not paid rent for 8 months. The landlord issued "
                "a notice under Section 12 of the CG Rent Control Act demanding payment "
                "and threatening eviction. The tenant claims financial hardship."
            ),
            lines=6,
        )
        reason_btn = gr.Button("Simulate reasoning", variant="primary")

        gr.Markdown("### Issues identified")
        issues_out = gr.Markdown()

        gr.Markdown("### Applicable provisions")
        provisions_out = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Petitioner arguments")
                pet_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Respondent arguments")
                resp_out = gr.Markdown()

        gr.Markdown("### Likely reasoning")
        reasoning_out = gr.Markdown()

        gr.Markdown("### Possible outcome (with caveats)")
        outcome_out = gr.Markdown()

        gr.Markdown("### Sources & confidence")
        sources_out = gr.Markdown()

        reason_btn.click(
            fn=reasoning_fn,
            inputs=[fact_input],
            outputs=[
                issues_out,
                provisions_out,
                pet_out,
                resp_out,
                reasoning_out,
                outcome_out,
                sources_out,
            ],
        )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
)