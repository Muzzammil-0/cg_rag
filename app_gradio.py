try:
    import spaces
    _SPACES_AVAILABLE = True
except ImportError:
    _SPACES_AVAILABLE = False

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

# --- Configuration ---
DEFAULT_TOP_K_CHAT = 20
DEFAULT_TOP_K_REASONING = 15

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
        answer_data, results = generate_answer(message, top_k=DEFAULT_TOP_K_CHAT, verbose=False)
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
            citation_lines.append(f"- **[Citation {c}]** {src} — Section {sec}")

    if citation_lines:
        answer += "\n\n### Sources\n" + "\n".join(citation_lines)

    return answer


# ------------------------------------------------------------
# Tab 2: Simulate case reasoning
# ------------------------------------------------------------
def reasoning_fn(fact_pattern):
    if not fact_pattern or len(fact_pattern.strip()) < 20:
        yield ("Please enter a fact pattern (at least 20 characters).", "", "", "", "", "", "")
        return

    text = fact_pattern.strip().lower()
    question_starters = ("what ", "how ", "when ", "why ", "who ", "is ", "are ", "can ", "does ", "should ", "which ", "where ")
    if text.startswith(question_starters):
        msg = ("**This tab expects a fact pattern, not a question.**\n\n"
               "**Wrong:** *What are the conditions for rape?*\n\n"
               "**Right:** *A 25-year-old man forced himself on a 20-year-old woman at a party. She reported the next day.*")
        yield (msg, "", "", "", "", "", "")
        return

    # --- YIELD 1: IMMEDIATE FEEDBACK ---
    # This fires the exact millisecond the button is clicked
    yield (
        "*Analyzing fact pattern...*", 
        "*Extracting legal issues and running multi-query retrieval...*", 
        "*Waiting for LLM...*", 
        "", "", "", ""
    )

    try:
        result, results = generate_reasoning(fact_pattern, top_k=DEFAULT_TOP_K_REASONING, verbose=False)
    except Exception as e:
        yield (f"Error: {e}", "", "", "", "", "", "")
        return

    if not result or result.get("error"):
        yield (f"Failed to parse model output. Try again.\n\nRaw: {result.get('raw', '')[:500]}", "", "", "", "", "", "")
        return

    # 1. Issues
    issues = "\n".join(f"- {x}" for x in result.get("issues", [])) or "—"

    # 2. Provisions
    provisions_lines = []
    for p in result.get("provisions", []):
        quote = p.get('quote', '')
        quote_str = f"\n  > *\"{quote}\"*" if quote else ""
        provisions_lines.append(
            f"- **{p.get('source', '')} {p.get('section', '')}**\n  **Relevance:** {p.get('relevance', '')}{quote_str}"
        )
    provisions = "\n\n".join(provisions_lines) or "—"

    # 3. Arguments
    pet = "\n".join(f"- {x}" for x in result.get("petitioner_arguments", [])) or "—"
    resp = "\n".join(f"- {x}" for x in result.get("respondent_arguments", [])) or "—"

    # 4. Reasoning
    reasoning = result.get("likely_reasoning", "") or "—"

    # 5. Multi-Outcome
    outcomes_list = result.get("possible_outcomes", [])
    if outcomes_list:
        outcome_parts = []
        for o in outcomes_list:
            likelihood = o.get('likelihood', 'Outcome')
            scenario = o.get('scenario', '')
            rationale = o.get('rationale', '')
            outcome_parts.append(f"### {likelihood}\n**Scenario:** {scenario}\n\n*Rationale:* {rationale}")
        outcome = "\n\n---\n\n".join(outcome_parts)
    else:
        outcome = "—"

    # 6. Sources
    citations = result.get("citations", [])
    source_lines = []
    for c in citations:
        idx = c - 1
        if 0 <= idx < len(results):
            m = results[idx]["metadata"]
            src = m.get("source", "")
            sec = m.get("section", m.get("article", m.get("para", "")))
            source_lines.append(f"- **[Citation {c}]** {src} — Section {sec}")

    confidence = result.get("confidence", 0.0)
    sources_md = ""
    if source_lines:
        sources_md = "\n".join(source_lines)
    sources_md += f"\n\n**Model Confidence:** {confidence}"

    # --- YIELD 2: FINAL OUTPUT ---
    yield (issues, provisions, pet, resp, reasoning, outcome, sources_md)


# ------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------
with gr.Blocks(title="CG Legal RAG") as demo:
    gr.Markdown("# CG Legal RAG")
    gr.Markdown("Legal research assistant for Indian and Chhattisgarh law.")
    
    with gr.Tab("Ask a legal question"):
        gr.ChatInterface(
            fn=chat_fn, 
            title="", 
            description="Ask a legal question in plain English. Answers are cited from statutes and judgments.",
            examples=[
                "What is the punishment for murder under BNS?",
                "What is the penalty for a data fiduciary failing to prevent a data breach under the DPDP Act?",
                "What are the grounds for eviction under the Chhattisgarh Rent Control Act?"
            ]
        )

    with gr.Tab("Simulate case reasoning"):
        gr.Markdown(
            "Enter a fact pattern. The system retrieves relevant provisions and judgments, "
            "then simulates how a Chhattisgarh court might reason through it.\n\n"
            "**Educational use only. Not legal advice. Not a prediction of any actual case.**"
        )

        fact_input = gr.Textbox(
            label="Fact pattern",
            placeholder=(
                "Example: Anjali discovers her company is secretly harvesting user data in violation of the DPDP Act. "
                "She reports it, and the CEO fires her without notice, refuses to pay her pending salary, and threatens to arrest her for defamation."
            ),
            lines=6,
        )
        reason_btn = gr.Button("Simulate reasoning", variant="primary")

        gr.Markdown("### Issues Identified")
        issues_out = gr.Markdown()

        gr.Markdown("### Applicable Provisions & Verified Quotes")
        provisions_out = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Petitioner Arguments")
                pet_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Respondent Arguments")
                resp_out = gr.Markdown()

        gr.Markdown("### Likely Reasoning")
        reasoning_out = gr.Markdown()

        gr.Markdown("### Possible Outcomes (Ranked)")
        outcome_out = gr.Markdown()

        gr.Markdown("### Sources & Confidence")
        sources_out = gr.Markdown()

        reason_btn.click(
            fn=reasoning_fn, 
            inputs=[fact_input], 
            outputs=[
                issues_out, provisions_out, pet_out, resp_out, 
                reasoning_out, outcome_out, sources_out
            ],
            show_progress="full"
        )


demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)