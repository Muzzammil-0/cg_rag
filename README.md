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

# CG Legal RAG

A legal research assistant for Indian central law and Chhattisgarh state law.

## API

- POST /query
- GET /health
- GET /docs

## Stack

Hybrid retrieval (BM25 + dense), cross-encoder reranking, Groq LLM with citation validation.
