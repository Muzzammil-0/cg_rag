---
title: CG Legal RAG
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app_gradio.py
pinned: false
---

# ⚖️ CG Legal RAG (V2)

**Live demo:** [huggingface.co/spaces/Muzzammil55/cg-legal-rag](https://huggingface.co/spaces/Muzzammil55/cg-legal-rag)

A legal research assistant for Indian central law and Chhattisgarh state law. Ask a question in plain English to get cited answers, or enter a complex fact pattern to simulate a courtroom's legal reasoning.

Built end-to-end: data ingestion -> chunking -> embeddings -> **Multi-Query Hybrid Retrieval** -> LLM generation -> **Quote Verification** -> Dockerized deployment.

---

## 🚀 V2 Architecture Upgrades

* **Contextual Chunk Embedding:** Each chunk is prepended with its Act name and Section (e.g., `[Chhattisgarh Rent Control Act] [Section 11]`) before embedding, so the vector represents the Act and section context rather than isolated text.
* **Stratified Reranking (50/50 Split):** At typical `top_k`, the final context balances Bare Acts (statutes) and Judgments (case law) roughly 50/50, so long High Court judgments are less likely to crowd out statutes.
* **Multi-Query Reasoning:** For complex fact patterns, an LLM extracts 3–6 distinct legal issues, runs a separate search for each, and merges the results, which reduces the "query dilution" problem.
* **Quote Verification (anti-hallucination):** The LLM is asked to return the exact text it used. A Python check (`verify_provision_quotes`) drops provisions whose quote cannot be matched to the cited act's retrieved text and regenerates the narrative. This catches many fabrications, but it is an n-gram heuristic, not a proof of correctness.
* **Neural Retrieval:** Manual `SECTION_HINTS` were removed in favor of neural embeddings and the Cross-Encoder.

---

## 📊 Corpus

| Category | Count |
|----------|-------|
| Central Acts | 48 |
| Chhattisgarh Acts | 16 |
| CG High Court Judgments | 2,576 |
| **Total documents** | **2,640** |
| **Total indexed chunks** | **36,019** |

Includes criminal law, civil procedure, labour codes (Wages, Social Security), data privacy (DPDP), RERA, consumer protection, and Chhattisgarh state laws.

---

## 🏗️ System Architecture

User Query
    │
    ▼
[ Query Analysis & Multi-Query Expansion ]
    │ • Issue extraction via Groq LLM
    ▼
[ Hybrid Retrieval (Exhaustive) ]
    │ • BM25 (sparse, Porter-stemmed)
    │ • MiniLM-L6 dense embeddings (ChromaDB)
    │ • Reciprocal Rank Fusion (RRF)
    ▼
[ Stratified Reranking ]
    │ • Split candidates: Acts vs Judgments
    │ • Cross-Encoder (ms-marco) rerank
    │ • Enforce 50/50 context split
    ▼
[ LLM Generation & Guardrails ]
    │ • Strict "use only context" prompt
    │ • N-Gram Quote Verification (anti-hallu)
    │ • Ranked Multi-Outcome generation
    ▼
Cited answer + retrieved chunks

---

## 📈 Evaluation Metrics

### Retrieval (Strict Bare-Act Matching)

> **Note:** V1 achieved a higher Recall@1 by using hardcoded section hints. V2 uses pure neural retrieval and intentionally surfaces High Court judgments alongside Bare Acts. This lowers strict Recall@1; the trade-off is intended to give the LLM more case-law context for reasoning.

| Metric | Value |
|--------|-------|
| recall@1 | 0.414 |
| recall@3 | 0.500 |
| recall@5 | 0.621 |
| recall@10 | 0.621 |
| MRR | 0.481 |

### Generation (20-question sample)

To evaluate the actual answers (not just the retrieval), `evaluate_generation.py` was run on a 20-question sample.

| Metric | Value |
|--------|-------|
| answer_rate | 0.800 |
| citation_grounding | 0.600 |
| keyword_coverage | 0.642 |
| mean_confidence | 0.972 |
| not_found_accuracy | 3/3 PASS |

**What this means:**

* **Answer Rate (80%):** The system successfully provided a grounded answer for 16 out of 20 questions.
* **Not-Found (100%):** On the out-of-domain tests (crypto tax, space mining, jaywalking), the guardrail returned "Not found in provided context."
* **Citation Grounding (60%):** In 12 out of 20 cases, the system cited the exact bare act section we expected. The other 8 cases fell into two buckets:
    1. **Refusal (Safe Fallback):** For `ipc_theft_def`, `bns_theft`, `crpc_fir`, and `bnss_fir`, the retriever pulled semantically related provisions but missed the exact definition section. The LLM correctly refused to answer rather than hallucinating.
    2. **Judgment Citations:** For `const_article21` and `crpc_bail`, the LLM cited a High Court judgment discussing the law rather than the bare act itself, which is a natural byproduct of the 50/50 Stratified Reranking.

---

## 🛠️ Stack

| Layer | Tool |
|-------|------|
| PDF extraction | PyMuPDF |
| Chunking | Custom regex (acts by section, judgments by paragraph) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (cosine) |
| Sparse retrieval | rank_bm25 with Porter stemming |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (openai/gpt-oss-120b) |
| API | FastAPI + Pydantic |
| UI | Gradio |
| Storage | SQLite (query logs) |
| Deployment | HuggingFace Spaces + Docker |

---

## 🚦 Running Locally

    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Set up environment variables
    cp .env.example .env
    # Add your own GROQ_API_KEY to the .env file

    # 3. Build the database (optional, if data is not already cached)
    python src/ingestion/extract_pdf.py
    python src/chunking/chunker.py
    python src/embeddings/embed.py

    # 4. Run the UI
    python app_gradio.py

Optional data-pipeline / training dependencies (classifier training, judgment fetching) are listed in `requirements-dev.txt`.

### Docker

    docker compose build
    docker compose up

---

## 🧪 Testing & Evaluation

    # Unit tests — heavy modules are mocked and no network calls are made.
    # Requires GROQ_API_KEY in .env (the generator builds a client at import time).
    pip install -r requirements-dev.txt
    python -m pytest

    # Retrieval metrics: recall@k, MRR, not-found guardrail
    python src/evaluation/evaluate.py

    # Generation metrics: answer_rate, citation_grounding, keyword_coverage.
    # Calls the live LLM and uses API quota — cap it with --limit.
    python src/evaluation/evaluate_generation.py --limit 20

---

## Engineering highlights

1. **Contextual Embeddings:** Prepending metadata to chunk text before vectorization.
2. **Multi-Query Decomposition:** Splitting a fact pattern into sub-queries, running deep searches, and merging the results.
3. **Deterministic Guardrails:** Python N-gram matching checks LLM JSON output to catch fabricated legal quotes.
4. **Dynamic UI Generation:** Gradio generators (`yield`) stream status updates and render complex JSON into Markdown.
5. **Dockerization:** A CPU-optimized image (torch CPU wheels, model caching) reduces image size from ~5 GB to ~1.8 GB.

---

## ⚠️ Limitations

* **Corpus Coverage:** The judgment corpus is a 2,576-document sample, not the complete Indian Kanoon archive.
* **Section Matching:** Exhaustive search prioritizes semantic similarity. For highly specific section lookups, it may retrieve a judgment discussing the section rather than the bare section itself.
* **Not Legal Advice:** This is an educational tool, not a substitute for a licensed attorney.

---

## License

MIT — see [LICENSE](LICENSE).