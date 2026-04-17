import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from src.chunking.adaptive_router import AdaptiveRouter
from src.chunking.train_router import RouterTrainingPipeline
from src.retrieval import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.context_expander import ContextExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.rag.prompt_template import PromptTemplate, RetrievalContext


BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_OUTPUT_DIR = BASE_DIR / "src" / "embedding" / "output"
ROUTER_MODEL_DIR = BASE_DIR / "src" / "chunking" / "models"
ROUTER_MODEL_CANDIDATES = ["router_best.pt", "router_final.pt"]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


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
                print(f"[Router] 已加载训练好的模型: {model_path}")
                return
            except Exception as exc:
                print(f"[Router] 加载训练模型失败，回退到规则路由: {exc}")

        print("[Router] 未发现训练好的模型，使用规则路由。")

    def classify_query(self, query: str) -> RouterDecision:
        if self.trained_router_pipeline is not None:
            scores = self.trained_router_pipeline.predict(query)
            suggested = self._weights_to_granularity(scores)
            confidence = max(scores.values()) if scores else 0.0
            return RouterDecision(
                router_name=self.router_name,
                suggested_granularity=suggested,
                confidence=confidence,
                raw_scores=scores,
            )

        classification = self.rule_router.classify_query(query)
        return RouterDecision(
            router_name=classification.query_type.value,
            suggested_granularity=classification.suggested_granularity,
            confidence=classification.confidence,
        )

    @staticmethod
    def _weights_to_granularity(scores: Dict[str, float]) -> str:
        if not scores:
            return "paragraph"

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        selected = [gran for gran, weight in ranked[:2] if weight >= 0.15]
        if not selected:
            selected = [ranked[0][0]]
        return ", ".join(selected)


def score_interactive_evidence(query: str, result) -> float:
    text = (result.text or "").lower()
    query_lower = query.lower()
    keywords = [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", query_lower) if len(token) > 2]
    score = float(result.score)
    score += min(sum(1 for token in keywords if token in text), 4) * 0.08

    if re.search(r"\b(table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration|specific\s+conditions?)\b", query_lower):
        if any(token in text for token in ["optimizer", "learning rate", "batch", "epoch", "configuration", "setting", "table"]):
            score += 0.20
    if re.search(r"\b(limitations?|challenges?|drawbacks?|issues?|bottleneck|difficulty|scaling)\b", query_lower):
        if any(token in text for token in ["challenge", "limitation", "issue", "difficulty", "because", "reason"]):
            score += 0.18
    if re.search(r"\b(compared\s+to|compared\s+with|versus|vs\.?|baseline|outperform|difference\s+between)\b", query_lower):
        if any(token in text for token in ["baseline", "compared", "outperform", "versus", "better", "worse"]):
            score += 0.18

    citation_count = len(re.findall(r"\[[0-9,\-\s]+\]", result.text or ""))
    if citation_count >= 6:
        score -= 0.20
    return score


def build_llm_client() -> Optional[OpenAI]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print(f"[Warning] 未设置 DEEPSEEK_API_KEY。可在环境变量中设置，或写入 {ENV_PATH}")
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def main():
    print("=" * 60)
    print("ScholarRAG 启动中...")
    print("=" * 60)

    client = build_llm_client()

    metadata_path = EMBEDDING_OUTPUT_DIR / "unified_metadata.json"
    sparse_path = EMBEDDING_OUTPUT_DIR / "unified_sparse.pkl"
    dense_retriever = DenseRetriever.from_output_dir(str(EMBEDDING_OUTPUT_DIR))
    bm25_retriever = BM25Retriever(str(metadata_path))

    # 加载 Sparse 检索器（如果存在）
    sparse_retriever = None
    if sparse_path.exists():
        sparse_retriever = SparseRetriever(
            sparse_path=str(sparse_path),
            metadata_path=str(metadata_path),
            embedder=dense_retriever.embedder,
        )

    hybrid_retriever = HybridRetriever(
        dense_retriever, bm25_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=0.4, sparse_weight=0.3, bm25_weight=0.3,
    )
    reranker = CrossEncoderReranker()
    context_expander = ContextExpander(
        metadata=dense_retriever.metadata,
        window_size=2,
    )
    router = RoutedGranularitySelector()

    print("\n[Ready] 系统就绪，可开始提问。(输入 'q' 退出)")
    print("-" * 60)

    while True:
        query = input("\n你: ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            break
        if not query:
            continue

        decision = router.classify_query(query)
        query_type_name = decision.router_name  # rule_based 时就是 query_type.value
        preferred_granularities = [g.strip() for g in decision.suggested_granularity.split(",") if g.strip()]
        print(
            f"\n[0/3] 路由器: {decision.router_name} | 置信度: {decision.confidence:.2f} | "
            f"推荐粒度: {decision.suggested_granularity}"
        )
        if decision.raw_scores:
            print(f"      粒度权重: {decision.raw_scores}")

        print("[1/3] 正在执行混合检索 (Dense + BM25)...")
        all_results = []
        for gran in preferred_granularities:
            gran_results = hybrid_retriever.retrieve(query, top_k=50, granularity_filter=gran)
            all_results.extend(gran_results)
            print(f"      粒度 [{gran}] 检索到 {len(gran_results)} 个片段")

        if len(all_results) < 10:
            fallback = hybrid_retriever.retrieve(query, top_k=50)
            existing_ids = {r.chunk_id for r in all_results}
            for result in fallback:
                if result.chunk_id not in existing_ids:
                    all_results.append(result)
            print(f"      兜底检索后共有 {len(all_results)} 个片段")

        print(f"[2/3] 正在重排序 ({len(all_results)} -> 10)...")
        final_results = reranker.rerank(query, all_results, top_k=10)
        final_results = sorted(final_results, key=lambda item: score_interactive_evidence(query, item), reverse=True)

        # 上下文窗口扩展：FACTUAL/DEFINITION/table-config 类查询不扩展
        should_expand = True
        if query_type_name in ("factual", "definition"):
            should_expand = False
        if re.search(r"\b(table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration|specific\s+conditions?)\b", query.lower()):
            should_expand = False
        if should_expand:
            anchor_results = final_results[:2]
            expanded = context_expander.expand_results(anchor_results, top_k=len(anchor_results))
            expanded_ids = set((r.paper_id, r.chunk_id) for r in expanded)
            final_results = expanded + [r for r in final_results if (r.paper_id, r.chunk_id) not in expanded_ids]
        final_results = final_results[:10]

        # 使用统一的 prompt template
        prompt_template = PromptTemplate()
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
        messages = prompt_template.build_qa_messages(query, contexts)

        if client is None:
            print("[3/3] 未设置 API Key，已跳过答案生成。请先设置 DEEPSEEK_API_KEY 后重试。")
            print("-" * 60)
            continue

        print("[3/3] 正在生成最终回答 (deepseek-reasoner)...")

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True,
        )

        reasoning_buf = []
        answer_buf = []
        in_answer = False
        for chunk in response:
            delta = chunk.choices[0].delta
            rc = getattr(delta, 'reasoning_content', None)
            ac = getattr(delta, 'content', None)
            if rc:
                reasoning_buf.append(rc)
            if ac:
                if not in_answer:
                    print("\n最终回答: ", end="", flush=True)
                    in_answer = True
                print(ac, end="", flush=True)
                answer_buf.append(ac)
        if reasoning_buf:
            print(f"\n[思考过程({len(''.join(reasoning_buf))}字符已省略)]")
        if not in_answer:
            print("\n最终回答: (无输出)")
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()