"""
ScholarRAG Web Interface
========================
Flask-based web UI for the RAG academic paper QA system.

Usage:
    python web_app.py                 # Start on http://localhost:5000
    python web_app.py --port 8080     # Custom port
"""

import os
import re
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from openai import OpenAI

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.adaptive_router import AdaptiveRouter
from src.chunking.train_router import RouterTrainingPipeline
from src.retrieval import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.context_expander import ContextExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.rag.prompt_template import PromptTemplate, RetrievalContext

# ── Constants ──
EMBEDDING_OUTPUT_DIR = PROJECT_ROOT / "src" / "embedding" / "output"
ROUTER_MODEL_DIR = PROJECT_ROOT / "src" / "chunking" / "models"
ROUTER_MODEL_CANDIDATES = ["router_best.pt", "router_final.pt"]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))


# ══════════════════════════════════════════════════════════════
# Router (same as final_pipeline.py)
# ══════════════════════════════════════════════════════════════

@dataclass
class RouterDecision:
    router_name: str
    suggested_granularity: str
    confidence: float
    raw_scores: Optional[Dict[str, float]] = None


class RoutedGranularitySelector:
    def __init__(self):
        self.rule_router = AdaptiveRouter()
        self.trained_router_pipeline: Optional[RouterTrainingPipeline] = None
        self.router_name = "rule_based"
        self._try_load_trained_router()

    def _try_load_trained_router(self):
        for model_name in ROUTER_MODEL_CANDIDATES:
            model_path = ROUTER_MODEL_DIR / model_name
            if not model_path.exists():
                continue
            try:
                pipeline = RouterTrainingPipeline(model_dir=str(ROUTER_MODEL_DIR))
                pipeline.load(model_name)
                self.trained_router_pipeline = pipeline
                self.router_name = f"trained_mlp ({model_name})"
                print(f"[Router] Loaded trained model: {model_path}")
                return
            except Exception as exc:
                print(f"[Router] Failed to load trained model, fallback to rule-based: {exc}")
        print("[Router] No trained model found, using rule-based router.")

    def classify_query(self, query: str) -> RouterDecision:
        if self.trained_router_pipeline is not None:
            scores = self.trained_router_pipeline.predict(query)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = [g for g, w in ranked[:2] if w >= 0.15]
            if not selected:
                selected = [ranked[0][0]]
            suggested = ", ".join(selected)
            confidence = max(scores.values()) if scores else 0.0
            return RouterDecision(self.router_name, suggested, confidence, scores)

        classification = self.rule_router.classify_query(query)
        return RouterDecision(
            classification.query_type.value,
            classification.suggested_granularity,
            classification.confidence,
        )


def score_evidence(query: str, result) -> float:
    """Re-score retrieved results for better ranking."""
    text = (result.text or "").lower()
    query_lower = query.lower()
    keywords = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", query_lower) if len(t) > 2]
    score = float(result.score)
    score += min(sum(1 for t in keywords if t in text), 4) * 0.08

    if re.search(r"\b(table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration)\b", query_lower):
        if any(t in text for t in ["optimizer", "learning rate", "batch", "epoch", "configuration", "setting", "table"]):
            score += 0.20
    if re.search(r"\b(limitations?|challenges?|drawbacks?|issues?|bottleneck)\b", query_lower):
        if any(t in text for t in ["challenge", "limitation", "issue", "difficulty", "because", "reason"]):
            score += 0.18
    if re.search(r"\b(compared\s+to|compared\s+with|versus|vs\.?|baseline|outperform)\b", query_lower):
        if any(t in text for t in ["baseline", "compared", "outperform", "versus", "better", "worse"]):
            score += 0.18

    citation_count = len(re.findall(r"\[[0-9,\-\s]+\]", result.text or ""))
    if citation_count >= 6:
        score -= 0.20
    return score


# ══════════════════════════════════════════════════════════════
# Global Pipeline (loaded once at startup)
# ══════════════════════════════════════════════════════════════

class RAGPipeline:
    """Singleton wrapper for the entire RAG pipeline."""

    def __init__(self):
        self.ready = False
        self.error = None

    def load(self):
        print("=" * 60)
        print("  ScholarRAG: Loading pipeline components...")
        print("=" * 60)
        try:
            metadata_path = EMBEDDING_OUTPUT_DIR / "unified_metadata.json"
            sparse_path = EMBEDDING_OUTPUT_DIR / "unified_sparse.pkl"

            self.dense_retriever = DenseRetriever.from_output_dir(str(EMBEDDING_OUTPUT_DIR))
            self.bm25_retriever = BM25Retriever(str(metadata_path))

            self.sparse_retriever = None
            if sparse_path.exists():
                self.sparse_retriever = SparseRetriever(
                    sparse_path=str(sparse_path),
                    metadata_path=str(metadata_path),
                    embedder=self.dense_retriever.embedder,
                )

            self.hybrid_retriever = HybridRetriever(
                self.dense_retriever, self.bm25_retriever,
                sparse_retriever=self.sparse_retriever,
                dense_weight=0.4, sparse_weight=0.3, bm25_weight=0.3,
            )

            self.reranker = CrossEncoderReranker()
            self.context_expander = ContextExpander(
                metadata=self.dense_retriever.metadata,
                window_size=2,
            )
            self.router = RoutedGranularitySelector()
            self.prompt_template = PromptTemplate()

            # LLM client (DeepSeek)
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                print("[LLM] DeepSeek API connected")
            else:
                self.client = None
                print("[LLM] WARNING: DEEPSEEK_API_KEY not set")

            self.ready = True
            print("=" * 60)
            print("  ScholarRAG: Pipeline ready!")
            print("=" * 60)
        except Exception as e:
            self.error = str(e)
            traceback.print_exc()

    def retrieve_and_answer(self, query: str):
        """
        Full RAG pipeline: route -> retrieve -> rerank -> expand -> generate.
        Returns a generator that yields SSE events.
        """
        if not self.ready:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Pipeline not ready'})}\n\n"
            return

        t0 = time.time()

        # Step 1: Route
        decision = self.router.classify_query(query)
        preferred = [g.strip() for g in decision.suggested_granularity.split(",") if g.strip()]
        route_info = {
            "type": "status",
            "step": "route",
            "router": decision.router_name,
            "confidence": round(decision.confidence, 3),
            "granularity": decision.suggested_granularity,
        }
        yield f"data: {json.dumps(route_info)}\n\n"

        # Step 2: Hybrid retrieval
        all_results = []
        for gran in preferred:
            gran_results = self.hybrid_retriever.retrieve(query, top_k=50, granularity_filter=gran)
            all_results.extend(gran_results)

        if len(all_results) < 10:
            fallback = self.hybrid_retriever.retrieve(query, top_k=50)
            existing_ids = {r.chunk_id for r in all_results}
            for r in fallback:
                if r.chunk_id not in existing_ids:
                    all_results.append(r)

        yield f"data: {json.dumps({'type': 'status', 'step': 'retrieve', 'count': len(all_results)})}\n\n"

        # Step 3: Rerank
        final_results = self.reranker.rerank(query, all_results, top_k=10)
        final_results = sorted(final_results, key=lambda r: score_evidence(query, r), reverse=True)

        # Step 4: Context expansion
        query_type_name = decision.router_name
        should_expand = True
        if query_type_name in ("factual", "definition"):
            should_expand = False
        if re.search(r"\b(table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration)\b", query.lower()):
            should_expand = False
        if should_expand:
            anchor_results = final_results[:2]
            expanded = self.context_expander.expand_results(anchor_results, top_k=len(anchor_results))
            expanded_ids = set((r.paper_id, r.chunk_id) for r in expanded)
            final_results = expanded + [r for r in final_results if (r.paper_id, r.chunk_id) not in expanded_ids]
        final_results = final_results[:10]

        # Send sources to frontend
        sources = []
        for i, res in enumerate(final_results):
            sources.append({
                "index": i + 1,
                "paper_id": res.paper_id,
                "section_type": res.section_type,
                "granularity": res.granularity,
                "score": round(float(res.score), 4),
                "text": res.text[:500] if res.text else "",
            })
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Step 5: Generate answer with LLM
        if not self.client:
            yield f"data: {json.dumps({'type': 'error', 'content': 'LLM API key not configured'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'step': 'generate'})}\n\n"

        contexts = [
            RetrievalContext(
                text=res.text,
                paper_id=res.paper_id,
                chunk_id=res.chunk_id,
                section_type=res.section_type,
                score=res.score,
                granularity=res.granularity,
            )
            for res in final_results
        ]
        messages = self.prompt_template.build_qa_messages(query, contexts)

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                rc = getattr(delta, 'reasoning_content', None)
                ac = getattr(delta, 'content', None)
                if rc:
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': rc})}\n\n"
                if ac:
                    yield f"data: {json.dumps({'type': 'answer', 'content': ac})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        elapsed = time.time() - t0
        yield f"data: {json.dumps({'type': 'done', 'elapsed': round(elapsed, 1)})}\n\n"


# Global pipeline instance
pipeline = RAGPipeline()


# ══════════════════════════════════════════════════════════════
# Flask Routes
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({"ready": pipeline.ready, "error": pipeline.error})


@app.route("/api/papers")
def papers():
    """Return list of all indexed papers with titles extracted from parsed PDFs."""
    parsed_dir = PROJECT_ROOT / "parsed_pdf"
    papers_list = []
    for f in sorted(parsed_dir.glob("*_content_list_v2.json")):
        paper_id = f.name.replace("_content_list_v2.json", "")
        title = paper_id  # fallback
        try:
            with open(f, "r", encoding="utf-8") as fh:
                content = json.load(fh)
            for item in content:
                if item.get("type") == "title":
                    tc = item.get("content", {}).get("title_content", [])
                    parts = [p.get("content", "") for p in tc if isinstance(p, dict)]
                    t = "".join(parts).strip()
                    if t:
                        title = t
                    break
        except Exception:
            pass
        # count chunks for this paper
        chunk_count = sum(1 for m in pipeline.dense_retriever.metadata if m.get("paper_id") == paper_id) if pipeline.ready else 0
        papers_list.append({
            "paper_id": paper_id,
            "title": title,
            "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
            "chunks": chunk_count,
        })
    return jsonify(papers_list)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        yield from pipeline.retrieve_and_answer(query)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScholarRAG Web Interface")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    pipeline.load()

    if not pipeline.ready:
        print(f"[FATAL] Pipeline failed to load: {pipeline.error}")
        sys.exit(1)

    print(f"\n  Open http://{args.host}:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
