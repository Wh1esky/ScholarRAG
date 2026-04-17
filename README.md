# ScholarRAG

> **A Multi-Granularity RAG System for Academic Paper Question Answering**

ScholarRAG is a Retrieval-Augmented Generation (RAG) system designed for answering questions about academic papers. It combines **multi-granularity chunking**, a **Mix-of-Granularity (MoG) router**, **three-way hybrid retrieval** (Dense + Sparse + BM25), **cross-encoder reranking**, **context expansion**, and **LLM-based answer generation** into a complete end-to-end pipeline. A Perplexity-style web interface is included for interactive exploration.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command-Line Interface](#command-line-interface)
  - [Evaluation Pipeline](#evaluation-pipeline)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Evaluation Results](#evaluation-results)
  - [Retrieval Performance](#retrieval-performance)
  - [Generation Quality (LLM Evaluation)](#generation-quality-llm-evaluation)
  - [Hit Rate by Difficulty](#hit-rate-by-difficulty)
  - [Hit Rate by Question Category](#hit-rate-by-question-category)
- [Technical Details](#technical-details)
  - [Multi-Granularity Chunking](#multi-granularity-chunking)
  - [Mix-of-Granularity Router](#mix-of-granularity-router)
  - [Three-Way Hybrid Retrieval](#three-way-hybrid-retrieval)
  - [Cross-Encoder Reranking](#cross-encoder-reranking)
  - [Context Expansion](#context-expansion)
  - [LLM Answer Generation](#llm-answer-generation)
- [Models Used](#models-used)
- [Contributions & Innovations](#contributions--innovations)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgments](#acknowledgments)

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Granularity Chunking** | Papers are chunked at 4 levels — sentence (~100 tokens), paragraph (~300-500 tokens), section (~1000-2000 tokens), and document (~4000+ tokens) — with configurable overlap |
| **MoG Router** | A trained MLP neural network (backed by `stsb-roberta-large`) predicts the optimal granularity distribution for each query, with an adaptive rule-based fallback |
| **Three-Way Hybrid Retrieval** | Combines Dense (FAISS + BGE-M3), Learned Sparse (BGE-M3 lexical weights), and BM25 via Reciprocal Rank Fusion (RRF) |
| **Cross-Encoder Reranking** | `BAAI/bge-reranker-base` reranks top candidates for precision |
| **Context Expansion** | Expands top-ranked chunks with neighboring context within the same paper section |
| **Streaming LLM Generation** | DeepSeek-Reasoner generates answers with real-time token streaming and reasoning chain display |
| **Perplexity-Style Web UI** | Interactive web interface with pipeline progress tracking, source citations, paper browser, and dark-green theme |
| **Automated Evaluation** | 4-step evaluation pipeline with ROUGE-L, Token-F1, and 4-dimension LLM scoring (Context Recall / Precision / Faithfulness / Answer Relevancy) |
| **Incremental & Resumable** | Batch evaluation supports auto-save every 5 questions and checkpoint resumption |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │    MoG Router (MLP / Rules)   │
                │  → predict granularity weights │
                └──────────────┬───────────────┘
                               │  granularity: [sentence, paragraph, section, document]
                               ▼
         ┌─────────────────────────────────────────────┐
         │         Three-Way Hybrid Retrieval           │
         │                                              │
         │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
         │  │  Dense    │ │  Sparse  │ │    BM25      │ │
         │  │ (FAISS)  │ │ (Learned)│ │ (Token-based)│ │
         │  │ w = 0.4  │ │ w = 0.3  │ │   w = 0.3   │ │
         │  └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
         │       └────────────┼───────────────┘         │
         │                    ▼                         │
         │         Reciprocal Rank Fusion (k=60)        │
         └─────────────────────┬───────────────────────┘
                               │  top-50 candidates per granularity
                               ▼
                ┌──────────────────────────────┐
                │   Cross-Encoder Reranker     │
                │   (bge-reranker-base)        │
                │   50 → top-10                │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │     Context Expander         │
                │  window_size = 2 neighbors   │
                │  (same paper + same section) │
                └──────────────┬───────────────┘
                               │  10 enriched chunks
                               ▼
                ┌──────────────────────────────┐
                │   Evidence-Grounded Prompt   │
                │   + DeepSeek-Reasoner LLM    │
                │   (streaming response)       │
                └──────────────────────────────┘
                               │
                               ▼
                       Final Answer
              (with [Source N] citations)
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-compatible GPU (recommended; CPU-only is supported but slower)
- [Conda](https://docs.conda.io/) (recommended)

### 1. Clone & Create Environment

```bash
git clone <repo-url>
cd RAG_base

conda create -n scholarrag python=3.10 -y
conda activate scholarrag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Deep learning backend |
| `sentence-transformers` | Router encoding & cross-encoder reranking |
| `faiss-cpu` | Dense vector search |
| `flask` | Web interface |
| `openai` | LLM API client (DeepSeek / MiniMax) |
| `python-dotenv` | Environment variable management |
| `scikit-learn` | Metrics & feature processing |
| `onnxruntime` | Optimized inference |
| `tqdm` | Progress bars |

### 3. Download Models

```bash
python download_models.py
```

This downloads three models from HuggingFace:

| Model | HuggingFace Repo | Local Path | Purpose |
|-------|-------------------|------------|---------|
| BGE-M3 | `BAAI/bge-m3` | `models/embedding/bge-m3` | Embedding (1024-dim) |
| BGE-Reranker | `BAAI/bge-reranker-base` | `models/reranker/bge-reranker-base` | Cross-encoder reranking |
| stsb-RoBERTa-large | `sentence-transformers/stsb-roberta-large` | `models/router/stsb-roberta-large` | Router query encoding |

> **Note**: Model download requires internet access to HuggingFace and needs about 3.5 GB of disk space. If your network is restricted, you may need a proxy or mirror.

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Required for evaluation with MiniMax
OPENAI_API_KEY=your_minimax_api_key_here
OPENAI_BASE_URL=https://api.minimaxi.com/v1
```

> **Note**:
> - `DEEPSEEK_API_KEY` is required for web/CLI question answering.
> - If you want to run the evaluation pipeline, you also need to configure the MiniMax API credentials above.

### 5. Build Index (if starting from parsed PDFs)

```bash
python rebuild_index.py
```

This processes `parsed_pdf/*.json` → multi-granularity chunking → BGE-M3 embedding → FAISS index.

> **Note**: Building the FAISS index usually takes about 5-10 minutes depending on your hardware.

---

## Quick Start

Before running the system, make sure you have:
- Python 3.10+
- Internet access for model download
- A DeepSeek API key in `.env`
- A CUDA-compatible GPU for better performance (CPU-only is supported but slower)

```bash
# Start the web interface
python web_app.py

# Open http://127.0.0.1:5000 in your browser
```

> **Note**: The first run may take longer because models and indexes need to be loaded into memory.

---

## Usage

### Web Interface

```bash
python web_app.py              # Default: http://127.0.0.1:5000
python web_app.py --port 8080  # Custom port
```

**Web UI Features:**
- Perplexity-style dark-green theme with centered search interface
- Real-time pipeline progress indicators (Route → Retrieve → Rerank → Generate)
- Streaming answer display with DeepSeek reasoning chain toggle
- Source citation panel with paper IDs, section types, and relevance scores
- Paper browser: click "100 Papers" in sidebar to browse all indexed papers with arXiv links
- Example question cards for quick exploration

### Command-Line Interface

```bash
python final_pipeline.py
```

Interactive CLI with step-by-step pipeline logging. Type your question and press Enter. Type `q` to exit.

### Evaluation Pipeline

If you want to run evaluation, make sure your `.env` file includes the MiniMax API configuration:

```env
OPENAI_API_KEY=your_minimax_api_key_here
OPENAI_BASE_URL=https://api.minimaxi.com/v1
```

Run the full 4-step evaluation:

```bash
python run_full_eval.py                # Full evaluation (all 4 steps)
python run_full_eval.py --test         # Quick test (2 questions per step)
python run_full_eval.py --step 1       # Run only step 1
python run_full_eval.py --step 2 4     # Run steps 2 and 4
python run_full_eval.py --from-step 3  # Resume from step 3
```

| Step | Dataset | Task |
|------|---------|------|
| 1 | `train_set_100papers.json` (1935 Qs) | Retrieval metrics (no LLM) |
| 2 | `train_set_100papers_sample50.json` (50 Qs) | Generation metrics + LLM scoring |
| 3 | `evaluation_set_100papers.json` (204 Qs) | Retrieval metrics (no LLM) |
| 4 | `evaluation_set_100papers.json` (sampled 44 Qs) | Generation metrics + LLM scoring |

---

## Project Structure

```
RAG_base/
├── .env                         # API keys (DEEPSEEK_API_KEY, etc.)
├── .gitignore
├── requirements.txt             # Python dependencies
├── README.md
│
├── final_pipeline.py            # CLI interactive QA entry point
├── web_app.py                   # Flask web interface (SSE streaming)
├── templates/
│   └── index.html               # Perplexity-style frontend
│
├── run_full_eval.py             # Automated 4-step evaluation runner
├── batch_evaluate.py            # Core evaluation engine (retrieval + generation metrics)
├── llm_evaluator.py             # LLM-based quality scoring (MiniMax-M2.7)
├── auto_train.py                # Auto-trigger router training after data generation
├── download_models.py           # Download HuggingFace models
├── rebuild_index.py             # Full index rebuild from parsed PDFs
│
├── train_set_100papers.json     # Training question set (1935 questions)
├── train_set_100papers_sample50.json   # Sampled 50 for generation eval
├── evaluation_set_100papers.json       # Evaluation question set (204 questions)
├── evaluation_set_100papers_sample44.json  # Sampled 44 for generation eval
│
├── eval_results/                # Evaluation output JSONs and summary CSVs
│   ├── summary_retrieval.csv
│   ├── summary_generation_llm.csv
│   ├── summary_hit_rate_by_difficulty.csv
│   └── summary_hit_rate_by_label.csv
│
├── models/                      # Downloaded model weights (git-ignored)
│   ├── embedding/bge-m3/
│   ├── reranker/bge-reranker-base/
│   └── router/stsb-roberta-large/
│
├── parsed_pdf/                  # Parsed PDF content (JSON per paper)
│
└── src/                         # Core source modules
    ├── chunking/                # Multi-granularity chunking & routing
    │   ├── adaptive_router.py       # Rule-based query router (8 query types)
    │   ├── mlp_router.py            # MLP neural router (MoG implementation)
    │   ├── train_router.py          # Router training pipeline
    │   ├── granularity_chunker.py   # 4-level chunking (sentence/paragraph/section/document)
    │   ├── structure_recognizer.py  # Paper structure recognition
    │   ├── unified_format.py        # Unified chunk format converter
    │   └── batch_process.py         # Batch processing for multiple papers
    │
    ├── embedding/               # Vector embedding
    │   ├── bge_embedder.py          # BGE-M3 embedder (1024-dim, FP16)
    │   ├── batch_embedder.py        # Batch embedding pipeline
    │   ├── index_builder.py         # FAISS FlatIP index builder
    │   ├── config.py                # Embedding configuration
    │   └── run_pipeline.py          # End-to-end embedding pipeline
    │
    ├── retrieval/               # Search & retrieval
    │   ├── dense_retriever.py       # FAISS-based dense retrieval
    │   ├── sparse_retriever.py      # Learned sparse retrieval (BGE-M3 lexical weights)
    │   ├── bm25_retriever.py        # BM25 keyword retrieval
    │   ├── hybrid_retriever.py      # Three-way RRF fusion
    │   ├── reranker.py              # Cross-encoder reranker (bge-reranker-base)
    │   └── context_expander.py      # Neighboring chunk expansion
    │
    ├── rag/                     # RAG answer generation
    │   ├── prompt_template.py       # Prompt templates (evidence extraction + grounded answer)
    │   ├── answer_generator.py      # Answer generation logic
    │   ├── llm_client.py            # LLM API wrapper
    │   └── rag_pipeline.py          # Full RAG pipeline orchestration
    │
    └── utils/                   # Utility functions
```

---

## Dataset

The system indexes **100 academic papers** from arXiv, covering diverse AI/ML research areas.

| Dataset | Questions | Purpose |
|---------|-----------|---------|
| `train_set_100papers.json` | 1,935 | Training & retrieval evaluation |
| `train_set_100papers_sample50.json` | 50 | Sampled for generation quality evaluation |
| `evaluation_set_100papers.json` | 204 | Held-out evaluation set |
| `evaluation_set_100papers_sample44.json` | 44 | Sampled for generation quality evaluation |

**Question categories** (8 types): experimental results, findings/assumptions, previous methods, methods, motivation, research domain, experimental settings, existing challenges.

**Difficulty levels**: Easy, Medium, Hard — determined by the complexity of reasoning required.

**Index statistics**: 26,329 total chunks across 3 granularity levels (sentence, paragraph, section), stored in a FAISS FlatIP index with 1024-dimensional BGE-M3 embeddings.

---

## Evaluation Results

### Retrieval Performance

| Dataset | Questions | Hit Rate | MRR | R@1 | R@3 | R@5 | R@10 |
|---------|-----------|----------|-----|-----|-----|-----|------|
| Train | 1,935 | **96.5%** | 0.920 | 89.3% | 94.4% | 95.7% | 96.5% |
| Evaluation | 204 | **93.1%** | 0.871 | 82.8% | 91.2% | 93.1% | 93.1% |

### Generation Quality (LLM Evaluation)

| Dataset | Questions | ROUGE-L | Token-F1 | Context Recall | Context Precision | Faithfulness | Answer Relevancy |
|---------|-----------|---------|----------|----------------|-------------------|--------------|------------------|
| Train | 50 | 0.183 | 0.307 | 3.50 / 5 | 3.90 / 5 | **4.44 / 5** | 4.09 / 5 |
| Evaluation | 44 | 0.194 | 0.314 | 3.40 / 5 | **4.11 / 5** | 4.25 / 5 | 3.95 / 5 |

> **Note**: ROUGE-L and Token-F1 are computed against reference answers; LLM scores (1-5 scale) are assessed by MiniMax-M2.7 across 4 RAGAS-inspired dimensions.

### Hit Rate by Difficulty

| Dataset | Easy | Medium | Hard |
|---------|------|--------|------|
| Train | 91.1% (235) | 96.6% (1,001) | **98.3%** (699) |
| Evaluation | 75.0% (24) | **98.0%** (102) | 92.3% (78) |

### Hit Rate by Question Category

| Category | Train | Evaluation |
|----------|-------|------------|
| Experimental Results | 97.3% | 97.6% |
| Findings / Assumptions | 97.3% | **100.0%** |
| Previous Methods | 97.4% | **100.0%** |
| Methods | **99.6%** | 91.7% |
| Motivation | 95.8% | 90.0% |
| Research Domain | 99.4% | 92.3% |
| Experimental Settings | 92.0% | 82.5% |
| Existing Challenges | 95.0% | 93.3% |

---

## Technical Details

### Multi-Granularity Chunking

Papers are processed at 4 granularity levels to capture information at different scales:

| Granularity | Target Tokens | Use Case |
|-------------|---------------|----------|
| **Sentence** | ~100 | Factual lookups, definitions |
| **Paragraph** | 300–500 | General QA, method descriptions |
| **Section** | 1,000–2,000 | Comparisons, summaries |
| **Document** | 4,000+ | Whole-paper overviews |

**Overlap mechanism**: Each chunk includes an overlap region (default 100 tokens for paragraph, 200 for section) from the tail of the previous chunk, with sentence-boundary-aware truncation.

**Academic-aware sentence splitting**: Protects 30+ abbreviations (e.g., "i.e.", "et al.", "Fig.", "Eq.") and decimal numbers from incorrect splitting.

### Mix-of-Granularity Router

Inspired by [Mix-of-Granularity (COLING 2025)](https://aclanthology.org/), the router predicts query-specific granularity weights:

**Architecture (MLP, 5-layer):**

```
Input: stsb-roberta-large embedding
  → Linear(embed_dim, 1024) → LayerNorm → ReLU → Dropout(0.2)
  → Linear(1024, 512) → LayerNorm → ReLU → Dropout(0.2)
  → Linear(512, 256) → LayerNorm → ReLU → Dropout(0.1)
  → Linear(256, 64) → LayerNorm → ReLU
  → Linear(64, 4) → Softmax
Output: [sentence, paragraph, section, document] weights
```

**Training**: KL-Divergence loss with soft labels (constructed via stsb-roberta-large semantic similarity between top-retrieved chunks and reference answers), AdamW optimizer (lr=1e-4, weight_decay=1e-4), StepLR scheduler (decay 0.5 every 10 epochs), early stopping after epoch 20.

**Fallback**: When no trained model is available, the system uses a rule-based `AdaptiveRouter` that classifies queries into 8 types (FACTUAL, METHOD, COMPARISON, SUMMARY, EXPERIMENTAL, DEFINITION, REASONING, LIST) via regex patterns and keyword matching.

### Three-Way Hybrid Retrieval

Three retrieval strategies are fused using **Reciprocal Rank Fusion (RRF)**:

$$\text{score}(d) = \sum_{i \in \{dense, sparse, bm25\}} \frac{w_i}{k + \text{rank}_i(d)}$$

| Retriever | Model | Weight | Method |
|-----------|-------|--------|--------|
| **Dense** | BGE-M3 (1024-dim) + FAISS FlatIP | 0.4 | Cosine similarity via inner product (L2-normalized vectors) |
| **Sparse** | BGE-M3 lexical weights | 0.3 | Inverted index with learned token weights, dot-product scoring |
| **BM25** | Token-based | 0.3 | Classical term-frequency keyword matching |

Fusion constant: $k = 60$. Each retriever returns top-50 candidates per granularity before fusion.

### Cross-Encoder Reranking

After hybrid retrieval, a **cross-encoder** provides fine-grained relevance scoring:

- **Model**: `BAAI/bge-reranker-base` (XLM-RoBERTa backbone)
- **Input**: (query, chunk_text) pairs, `max_length = 512`
- **Process**: 50 candidates → cross-encoder scoring → top-10 selected
- **Post-rerank boosting**: Additional keyword-match and query-type-specific score adjustments for experimental settings, limitations, and comparison queries

### Context Expansion

After reranking, each result chunk is expanded with its neighboring chunks:

- **Window**: ±2 chunks within the **same paper and same section type**
- **Deduplication**: Chunks already covered by a previous expansion are skipped to avoid redundancy
- **Result**: Up to 10 context-enriched chunks are passed to the LLM

### LLM Answer Generation

- **Model**: DeepSeek-Reasoner (via OpenAI-compatible API)
- **Prompt Strategy**: Two-stage evidence-grounded approach:
  1. **Evidence Extraction**: Extract up to 4 key evidence statements from sources
  2. **Grounded Answer**: Generate a structured answer based on extracted evidence with `[Source N]` citations
- **Streaming**: Server-Sent Events (SSE) for real-time token delivery
- **Reasoning Chain**: DeepSeek-Reasoner's intermediate reasoning is captured and optionally displayed

---

## Models Used

| Component | Model | Source | Dimensions |
|-----------|-------|--------|-----------|
| Embedding | `BAAI/bge-m3` | [HuggingFace](https://huggingface.co/BAAI/bge-m3) | 1024 |
| Reranker | `BAAI/bge-reranker-base` | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-base) | — |
| Router Encoder | `stsb-roberta-large` | [HuggingFace](https://huggingface.co/sentence-transformers/stsb-roberta-large) | 1024 |
| LLM (Generation) | DeepSeek-Reasoner | [DeepSeek API](https://api.deepseek.com) | — |
| LLM (Evaluation) | MiniMax-M2.7 | [MiniMax API](https://api.minimaxi.com/v1) | — |

---

## Contributions & Innovations

Compared to a baseline RAG system (naive chunking + single-vector retrieval + direct LLM prompting), ScholarRAG introduces the following **new contributions**:

| # | Innovation | What's New | Impact |
|---|-----------|------------|--------|
| 1 | **Multi-Granularity Chunking** | Papers are split at 4 structural levels (sentence / paragraph / section / document) with overlap and academic-aware sentence splitting (30+ abbreviation protections) | Captures information at different scales — fine-grained facts and coarse-grained summaries — in a single index |
| 2 | **Mix-of-Granularity (MoG) Router** | A 5-layer MLP (backed by stsb-roberta-large) trained with KL-Divergence on soft labels (constructed via RoBERTa semantic similarity) to predict per-query granularity distributions | Dynamically routes each query to the most informative chunk level instead of one-size-fits-all chunking |
| 3 | **Three-Way Hybrid Retrieval + RRF** | Combines Dense (FAISS + BGE-M3), Learned Sparse (BGE-M3 lexical weights), and BM25 via Reciprocal Rank Fusion | Combines semantic understanding with exact keyword matching, achieving 96.5% Hit Rate on the training set |
| 4 | **Cross-Encoder Reranking with Post-Rerank Boosting** | bge-reranker-base reranks 50 → 10 candidates, with additional keyword-match and query-type-specific score adjustments | Significantly improves precision for specific query types (experimental settings, limitations, comparisons) |
| 5 | **Context Expansion** | Top reranked chunks are expanded with ±2 neighboring chunks within the same paper and section | Recovers surrounding context that chunking may have split, reducing information fragmentation |
| 6 | **Comprehensive Evaluation Framework** | 4-step automated pipeline with both retrieval metrics (Hit Rate, MRR, R@k) and RAGAS-inspired LLM-based generation quality scoring (Context Recall/Precision, Faithfulness, Answer Relevancy) | Enables systematic, reproducible evaluation beyond simple accuracy |
| 7 | **Ablation Study Support** | Modular architecture allows toggling each component (router, sparse retrieval, BM25, reranker, context expansion) independently | Quantifies the contribution of each module to overall system performance |

### Key Findings from Evaluation

- The MoG Router improves retrieval hit rate by routing factual queries to sentence-level chunks and comparison queries to section-level chunks.
- Three-way hybrid retrieval outperforms any single retrieval method: Dense-only achieves ~90% hit rate, adding Sparse and BM25 lifts it to 96.5%.
- Context expansion is especially beneficial for non-factual queries requiring broader understanding, improving LLM answer relevancy scores.
- The system maintains strong Faithfulness (4.44/5 on train, 4.25/5 on eval), indicating answers are well-grounded in retrieved evidence.

---

## Limitations & Future Work

### Current Limitations

1. **PDF Parsing Quality**: The system relies on pre-parsed PDF JSONs. Tables, figures, and mathematical equations may lose formatting during extraction, affecting QA quality on visually-rich content.
2. **MoG Router Data Dependency**: The router's soft labels are derived from RoBERTa semantic similarity between retrieved chunks and reference answers (a proxy for true optimal granularity), which may not always perfectly reflect the best retrieval granularity.
3. **Fixed Retrieval Weights**: The hybrid retrieval weights (Dense 0.4, Sparse 0.3, BM25 0.3) and RRF constant (k=60) are manually tuned rather than learned.
4. **Single-Hop Retrieval Only**: The system retrieves in a single pass; multi-hop questions requiring iterative retrieval are not explicitly handled.
5. **LLM API Dependency**: Answer generation relies on external API calls (DeepSeek), introducing latency and cost constraints.

### Future Work

- **Learned Fusion Weights**: Train the retrieval weight combination end-to-end using relevance feedback.
- **Multi-Hop Retrieval**: Implement iterative retrieval strategies (e.g., IRCoT) for complex reasoning chains.
- **Table & Figure QA**: Integrate multimodal parsing to handle tables and figures in academic papers.
- **Local LLM Deployment**: Support local models (e.g., Qwen, LLaMA) to remove API dependency.
- **Larger Paper Corpus**: Scale beyond 100 papers to test system robustness and index efficiency.

---

## Acknowledgments

- **Mix-of-Granularity (MoG)**: The granularity routing approach is inspired by the [Mix-of-Granularity paper (COLING 2025)](https://aclanthology.org/).
- **BGE-M3**: Multi-lingual, multi-granularity embedding model by [BAAI](https://huggingface.co/BAAI/bge-m3).
- **RAGAS**: Evaluation dimensions (Context Recall, Context Precision, Faithfulness, Answer Relevancy) are adapted from the [RAGAS framework](https://docs.ragas.io/).
- **DeepSeek**: LLM generation powered by [DeepSeek-Reasoner](https://www.deepseek.com/).

---

*Built for DSAI5201 — AI and Big Data Computing in Practice @ The Hong Kong Polytechnic University, 2025–2026 Semester 2.*
