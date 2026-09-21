```markdown
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

# ⚖️ CG Legal RAG

**Live demo:** [huggingface.co/spaces/Muzzammil55/cg-legal-rag](https://huggingface.co/spaces/Muzzammil55/cg-legal-rag)

A legal research assistant for Indian central law and Chhattisgarh state law. Ask a question in plain English, get a cited answer from indexed statutes and High Court judgments. Or enter a fact pattern and see a structured simulation of how a Chhattisgarh court might reason through it.

Built end-to-end: data collection → extraction → chunking → embeddings → hybrid retrieval → LLM generation → citation validation → deployment.

---

## What It Does

Two modes, one interface:

**Ask a legal question** — Retrieval-augmented question answering with citations.
> *"What are the grounds for eviction under the Chhattisgarh Rent Control Act?"*
> → Returns the grounds list from Section 11 with a source citation.

**Simulate case reasoning** — Educational tool for law students.
> *"A tenant has not paid rent for 8 months. The landlord issued a notice under Section 12..."*
> → Produces issues, applicable provisions, arguments for both sides, likely reasoning, and a hedged outcome.

Out-of-scope questions return `"Not found in provided context."` — the system refuses rather than hallucinates.

---

## Corpus

| Category | Count |
|----------|-------|
| Central Acts | 10 |
| Chhattisgarh Acts | 11 |
| CG High Court Judgments | 1,576 |
| **Total documents** | **1,597** |
| **Total indexed chunks** | **19,034** |

**Central acts:** Constitution of India, IPC, CrPC, Indian Evidence Act, CPC, BNS, BNSS, BSA, PESA Act 1996, Forest Rights Act Rules

**Chhattisgarh acts:** Land Revenue Code 1959, Excise Act 1915, Rent Control Act 2011, Rent Control Adaptation Rules, Municipalities Act 1961, Panchayat Raj Adhiniyam 1993, Tendu Patta Adhiniyam 1964, Vanopaj Adhiniyam 1969, Lok Sewa Guarantee Adhiniyam 2011, Special Courts Act 2015, Agricultural Cattle Preservation Rules 2014

**Judgments:** Chhattisgarh High Court, 2010–2025, sampled via [bharat-courts](https://pypi.org/project/bharat-courts/) from the [Indian High Court Judgments archive](https://registry.opendata.aws/indian-high-court-judgments/).

---

## Architecture

```

User query
│
▼
┌─────────────────────────────────────────┐
│  Query analysis                          │
│  • Act-source detection (21 patterns)    │
│  • Judgment-mode detection (26 triggers) │
│  • Section extraction (explicit + hints) │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│  Hybrid retrieval                        │
│  • BM25 (sparse, Porter-stemmed)         │
│  • MiniLM-L6 dense embeddings (ChromaDB) │
│  • Reciprocal Rank Fusion (k=60)         │
│  • Section injection for exact matches   │
│  • Cross-encoder rerank (ms-marco)       │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│  LLM generation (Groq gpt-oss-120b)      │
│  • Strict "use only context" prompt      │
│  • Citation validation                   │
│  • Provision validation (reasoning mode) │
│  • Not-found guardrail                   │
└─────────────────────────────────────────┘
│
▼
Cited answer + retrieved chunks

```

### Key Design Decisions

**Act-aware retrieval.** IPC and BNS have nearly identical section wording (e.g., theft, murder). Without filtering, a BNS query retrieves IPC chunks and vice versa. The source filter detects the act from query keywords and restricts the search accordingly.

**Judgment filtering.** Queries like "what have courts said about X" or "case law on Y" are routed to judgments only, ignoring acts. Implemented via a pattern list in `retriever.py`.

**Section hints.** A curated mapping of ~200 common legal phrases to their exact section numbers (e.g., "FIR" → CrPC 154, "grounds for eviction" → CG Rent Control Act 11). When a hint fires, the matching chunk is pinned to rank 1 and bypasses the reranker.

**Citation validation.** Every citation number the LLM returns is checked against the retrieved chunks. Invalid numbers are dropped. The LLM cannot cite a chunk that wasn't retrieved.

**Provision validation (reasoning mode).** Section numbers mentioned inside the reasoning output are checked against retrieved metadata. Hallucinated provisions are dropped with a warning logged.

**Not-found guardrail.** The prompt instructs the LLM to return exactly `"Not found in provided context."` when the answer isn't in the retrieved chunks. Tested on three out-of-scope queries, all passing.

**Query expansion for reasoning.** Plain-English fact patterns are converted to legal terms by the LLM before retrieval. This bridges the vocabulary gap (e.g., "forced himself" → "rape, consent, sexual assault") and improves retrieval from the wrong act (BNSS arrest procedure) to the right one (IPC 376).

---

## Evaluation

Two test sets, both in `data/evaluation/`.

### Act retrieval (47 questions)

| Metric | Value |
|--------|-------|
| recall@1 | 0.841 |
| recall@3 | 0.955 |
| **recall@5** | **0.977** |
| recall@10 | 0.977 |
| MRR | 0.896 |
| Not-found tests | 3/3 PASS |

### Case-law retrieval (20 questions)

| Metric | Value |
|--------|-------|
| Judgment filter trigger rate | 100% |
| All top-5 are judgments | 100% |
| Keyword coverage | 95% |
| **Pass rate** | **95%** |

Run both:

```bash
python src/evaluation/evaluate.py --skip-notfound
python src/evaluation/evaluate_case_law.py
```

Results are written to data/evaluation/results.json and data/evaluation/case_law_results.json.

---

Stack

Layer Tool
PDF extraction PyMuPDF
Chunking Custom regex (acts by section, judgments by paragraph)
Embeddings sentence-transformers/all-MiniLM-L6-v2
Vector store ChromaDB (cosine)
Sparse retrieval rank_bm25 with Porter stemming
Reranking cross-encoder/ms-marco-MiniLM-L-6-v2
LLM Groq (openai/gpt-oss-120b)
API FastAPI + Pydantic
UI Gradio
Storage SQLite (query logs)
Deployment HuggingFace Spaces (ZeroGPU) + Git LFS

---

Place source documents in:

```
data/raw/
├── central_acts/       # Constitution, IPC, BNS, etc.
├── cg_acts/            # Chhattisgarh state legislation
└── judgments/          # CG High Court PDFs
```

Then run the pipeline:

```bash
python src/ingestion/extract_pdf.py      # PDFs → extracted_text.json
python src/chunking/chunker.py           # → chunks.jsonl
python src/embeddings/embed.py           # → ChromaDB

python app_gradio.py                     # Gradio UI
# or
python -m uvicorn src.api.app:app --reload   # FastAPI + /docs
```

Get a free Groq API key at console.groq.com.

---

CLI Usage

```bash
python src/generation/generator.py
```

Choose query for retrieval-augmented answers or reason for the reasoning simulation. Prints structured output to the terminal.

---

Limitations

Corpus coverage. The judgment corpus is a 1,576-document sample from 2010–2025, not the full archive. It covers common topics (motor accidents, criminal appeals, land disputes) but misses niche areas (rent control eviction at HC level, tribal land disputes). Queries on thin topics return honest "not found" rather than poor answers.

English only. The embedding model is English-trained. Hindi documents (which exist for some CG excise rules) are not indexed.

Section hints are manual. Every new act requires hand-curated phrase → section mappings. This doesn't scale past a few dozen acts. A production system would use a semantic router.

Reasoning mode is nondeterministic. Temperature 0.2 means the same fact pattern can produce slightly different structures across runs. Acceptable for educational use, not for legal-adjacent production.

Not legal advice. The reasoning tab is an educational simulation. It is not a prediction of any actual case outcome.

---

Project Structure

```
cg_rag/
├── app_gradio.py                 # Gradio UI (HF Spaces entry point)
├── data/
│   ├── raw/                      # Source PDFs (gitignored)
│   ├── processed/                # chunks.jsonl (tracked via LFS)
│   ├── vector_db/                # ChromaDB (gitignored)
│   └── evaluation/               # Eval sets + scripts
├── src/
│   ├── api/                      # FastAPI backend
│   ├── chunking/                 # Section + paragraph chunkers
│   ├── embeddings/               # Embedding + startup build
│   ├── evaluation/               # Retrieval metrics
│   ├── generation/               # LLM prompts + validation
│   ├── ingestion/                # PDF extraction + judgment fetch
│   ├── retrieval/                # Hybrid search
│   └── storage/                  # SQLite logging
└── requirements.txt
```

Notes on Deployment

The Space runs on HuggingFace's free ZeroGPU tier. To satisfy the ZeroGPU startup scan (which expects a @spaces.GPU decorated function), a dummy no-op function is registered at the top of app_gradio.py. All computation runs on CPU.

The chunks.jsonl file is ~18 MB and tracked via Git LFS — plain Git rejects files over 10 MB.

Cold start after 48 hours of inactivity takes ~45 seconds (model download + ChromaDB rebuild from chunks.jsonl).

---

License

MIT

```