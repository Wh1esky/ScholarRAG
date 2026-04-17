"""
RAGPipeline - 核心 RAG 流程

完整的检索增强生成 Pipeline:
1. 接收用户 Query
2. 检索相关 Chunk
3. 生成答案
4. 返回答案和来源

使用方法:
    from src.rag import RAGPipeline
    from src.retrieval import DenseRetriever
    from src.rag import LLMClient, PromptTemplate, AnswerGenerator

    # 初始化组件
    retriever = DenseRetriever.from_output_dir("src/embedding/output")
    llm = LLMClient(model_name="gpt-4")
    template = PromptTemplate()
    generator = AnswerGenerator(llm, template)

    # 创建 Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    # 问答
    result = pipeline.answer("What is the main contribution?")
    print(result.answer)
    print(result.sources)
"""

import os
import json
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..retrieval.dense_retriever import RetrievalResult
from ..retrieval.context_expander import ContextExpander
from .answer_generator import AnswerGenerator, GenerationResult, GenerationMode
from .prompt_template import RetrievalContext


@dataclass
class RAGAnswer:
    """RAG 答案"""
    answer: str
    query: str
    sources: List[Dict[str, Any]]
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": self.sources,
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "timestamp": self.timestamp
        }

    def to_json(self, **kwargs) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    def print_summary(self, max_source_chars: int = 100):
        """打印答案摘要"""
        print("=" * 60)
        print(f"问题: {self.query}")
        print("=" * 60)
        print(f"\n答案:\n{self.answer}")
        print(f"\n{'=' * 60}")
        print(f"来源 ({len(self.sources)} 个):")
        print("-" * 60)
        for i, src in enumerate(self.sources, 1):
            paper = src.get('paper_id', 'Unknown')
            section = src.get('section_type', 'Unknown')
            text = src.get('text', '')[:max_source_chars]
            score = src.get('score', 0)
            print(f"  [{i}] {paper} | {section} | Score: {score:.4f}")
            print(f"      {text}...")
        print("=" * 60)

        if self.retrieval_metrics:
            print(f"\n检索指标: {self.retrieval_metrics}")
        if self.generation_metrics:
            print(f"生成指标: {self.generation_metrics}")


class RAGPipeline:
    """
    完整的 RAG Pipeline

    流程:
    Query → 检索 → 上下文构建 → LLM 生成 → 答案输出
    """

    def __init__(
        self,
        retriever,
        generator: AnswerGenerator,
        reranker=None,
        context_expander: Optional[ContextExpander] = None,
        default_top_k: int = 10,
        top_k_retrieve: int = 50,
        min_relevance_score: float = 0.0,
        return_sources: bool = True,
        track_metrics: bool = True
    ):
        """
        初始化 RAG Pipeline

        Args:
            retriever: 检索器 (DenseRetriever / BM25Retriever / HybridRetriever，
                       只需实现 retrieve(query, top_k) -> List[RetrievalResult])
            generator: 答案生成器
            reranker: 可选的重排序器 (需实现 rerank(query, results, top_k))
            context_expander: 可选的上下文扩展器
            default_top_k: 最终返回的 Top-K 结果数 (rerank 后)
            top_k_retrieve: 初始检索 Top-K 数 (rerank 前)
            min_relevance_score: 最小相关度分数
            return_sources: 是否返回来源
            track_metrics: 是否追踪指标
        """
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.context_expander = context_expander
        self.default_top_k = default_top_k
        self.top_k_retrieve = top_k_retrieve
        self.min_relevance_score = min_relevance_score
        self.return_sources = return_sources
        self.track_metrics = track_metrics

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "total_retrieval_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
            "total_tokens": 0
        }

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_sources: Optional[bool] = None,
        mode: Union[str, GenerationMode] = GenerationMode.DEFAULT,
        verbose: bool = False
    ) -> RAGAnswer:
        """
        回答问题

        Args:
            query: 问题
            top_k: Top-K 检索数
            min_score: 最小相关度分数
            return_sources: 是否返回来源
            mode: 生成模式
            verbose: 是否打印详细信息

        Returns:
            RAGAnswer: 答案对象
        """
        k = top_k or self.default_top_k
        min_s = min_score if min_score is not None else self.min_relevance_score
        ret_src = return_sources if return_sources is not None else self.return_sources

        if isinstance(mode, str):
            mode = GenerationMode(mode)

        # 更新统计
        self.stats["total_queries"] += 1

        if verbose:
            print(f"\n[Query] {query}")

        # =====================
        # 1. 检索阶段
        # =====================
        retrieval_start = time.time()
        retrieval_results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k_retrieve
        )

        # 1.5 可选重排序
        if self.reranker and retrieval_results:
            try:
                retrieval_results = self.reranker.rerank(
                    query, retrieval_results, top_k=k
                )
            except Exception as e:
                if verbose:
                    print(f"[Rerank] 失败, 用原始排序: {e}")
                retrieval_results = retrieval_results[:k]
        else:
            retrieval_results = retrieval_results[:k]

        if self.context_expander and retrieval_results:
            retrieval_results = self.context_expander.expand_results(
                retrieval_results,
                top_k=k
            )

        # 过滤低分结果
        if min_s > 0:
            retrieval_results = [r for r in retrieval_results if r.score >= min_s]

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        if self.track_metrics:
            self.stats["total_retrieval_time_ms"] += retrieval_time_ms

        if verbose:
            print(f"[检索] 找到 {len(retrieval_results)} 个结果, 耗时 {retrieval_time_ms:.2f}ms")

        # =====================
        # 2. 构建上下文
        # =====================
        contexts = self._build_contexts(retrieval_results)

        if verbose:
            if contexts:
                print(f"[上下文] 使用 {len(contexts)} 个 Chunk 构建上下文")
            else:
                print("[上下文] 无相关结果")

        # =====================
        # 3. 生成答案
        # =====================
        generation_start = time.time()
        gen_result = self.generator.generate(
            query=query,
            contexts=contexts,
            mode=mode
        )
        generation_time_ms = (time.time() - generation_start) * 1000

        if self.track_metrics:
            self.stats["total_generation_time_ms"] += generation_time_ms
            self.stats["total_tokens"] += gen_result.total_tokens

        if verbose:
            if gen_result.error:
                print(f"[生成] 错误: {gen_result.error}")
            else:
                print(f"[生成] 耗时 {generation_time_ms:.2f}ms, 使用 {gen_result.total_tokens} tokens")

        # =====================
        # 4. 构建结果
        # =====================
        sources = []
        if ret_src:
            sources = self._build_sources(retrieval_results)

        retrieval_metrics = self._compute_retrieval_metrics(retrieval_results) if self.track_metrics else {}

        generation_metrics = {
            "model": gen_result.model,
            "total_tokens": gen_result.total_tokens,
            "prompt_tokens": gen_result.prompt_tokens,
            "completion_tokens": gen_result.completion_tokens,
            "latency_ms": gen_result.latency_ms,
            "error": gen_result.error
        }

        answer = RAGAnswer(
            answer=gen_result.answer if not gen_result.error else f"生成失败: {gen_result.error}",
            query=query,
            sources=sources,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics
        )

        if verbose:
            print("\n" + "=" * 60)
            print("最终答案:")
            print("-" * 60)
            print(answer.answer)

        return answer

    def answer_with_citation(
        self,
        query: str,
        top_k: Optional[int] = None,
        citation_format: str = "numeric",
        **kwargs
    ) -> RAGAnswer:
        """
        回答问题并生成引用

        Args:
            query: 问题
            top_k: Top-K 数
            citation_format: 引用格式
            **kwargs: 其他参数

        Returns:
            RAGAnswer: 带引用的答案
        """
        k = top_k or self.default_top_k

        # 检索 + 重排序
        retrieval_results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k_retrieve
        )
        if self.reranker and retrieval_results:
            retrieval_results = self.reranker.rerank(
                query, retrieval_results, top_k=k
            )
        else:
            retrieval_results = retrieval_results[:k]

        if self.context_expander and retrieval_results:
            retrieval_results = self.context_expander.expand_results(
                retrieval_results,
                top_k=k
            )

        # 构建上下文
        contexts = self._build_contexts(retrieval_results)

        # 生成带引用的答案
        gen_result = self.generator.generate_with_citation(
            query=query,
            contexts=contexts,
            citation_format=citation_format
        )

        # 构建结果
        sources = self._build_sources(retrieval_results)

        return RAGAnswer(
            answer=gen_result.answer,
            query=query,
            sources=sources,
            retrieval_metrics=self._compute_retrieval_metrics(retrieval_results),
            generation_metrics={
                "model": gen_result.model,
                "total_tokens": gen_result.total_tokens,
                "citation_format": citation_format
            }
        )

    def batch_answer(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[RAGAnswer]:
        """
        批量回答问题

        Args:
            queries: 问题列表
            top_k: Top-K 数
            show_progress: 是否显示进度
            **kwargs: 其他参数

        Returns:
            List[RAGAnswer]: 答案列表
        """
        results = []

        iterator = queries
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(queries, desc="RAG 问答中")
            except ImportError:
                pass

        for query in iterator:
            result = self.answer(query, top_k=top_k, **kwargs)
            results.append(result)

        return results

    def _build_contexts(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[RetrievalContext]:
        """
        将检索结果转换为生成器需要的上下文格式

        Args:
            retrieval_results: 检索结果

        Returns:
            List[RetrievalContext]: 上下文列表
        """
        contexts = []

        for result in retrieval_results:
            if not result.text or not result.text.strip():
                continue

            ctx = RetrievalContext(
                text=result.text,
                paper_id=result.paper_id,
                chunk_id=result.chunk_id,
                section_type=result.section_type,
                score=result.score,
                granularity=result.granularity
            )
            contexts.append(ctx)

        return contexts

    def _build_sources(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """
        构建来源列表

        Args:
            retrieval_results: 检索结果

        Returns:
            List[Dict]: 来源列表
        """
        sources = []

        for result in retrieval_results:
            source = {
                "rank": result.rank,
                "paper_id": result.paper_id,
                "chunk_id": result.chunk_id,
                "section_type": result.section_type,
                "granularity": result.granularity,
                "text": result.text,
                "score": result.score
            }
            sources.append(source)

        return sources

    def _compute_retrieval_metrics(
        self,
        results: List[RetrievalResult]
    ) -> Dict[str, float]:
        """
        计算检索指标

        Args:
            results: 检索结果

        Returns:
            Dict: 指标字典
        """
        if not results:
            return {
                "num_results": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0
            }

        scores = [r.score for r in results]

        return {
            "num_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores)
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()

        if stats["total_queries"] > 0:
            stats["avg_retrieval_time_ms"] = (
                stats["total_retrieval_time_ms"] / stats["total_queries"]
            )
            stats["avg_generation_time_ms"] = (
                stats["total_generation_time_ms"] / stats["total_queries"]
            )

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_queries": 0,
            "total_retrieval_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
            "total_tokens": 0
        }

    def save_results(
        self,
        results: List[RAGAnswer],
        output_path: str
    ):
        """
        保存结果到文件

        Args:
            results: 答案列表
            output_path: 输出路径
        """
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "results": [r.to_dict() for r in results],
            "stats": self.get_stats()
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_path}")


def create_pipeline(
    embedding_output_dir: str = "src/embedding/output",
    model_name: str = "deepseek-chat",
    provider: str = "openai_compatible",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    dense_weight: float = 0.4,
    sparse_weight: float = 0.3,
    bm25_weight: float = 0.3,
    top_k: int = 10,
    top_k_retrieve: int = 50,
    **kwargs
) -> RAGPipeline:
    """
    创建 RAG Pipeline 的工厂函数

    Args:
        embedding_output_dir: Embedding 输出目录
        model_name: LLM 模型名称
        provider: LLM Provider ("openai" / "openai_compatible" / "anthropic")
        api_key: API Key
        base_url: API base URL
        use_hybrid: 是否使用混合检索 (Dense + Sparse + BM25)
        use_reranker: 是否使用 Reranker
        dense_weight: Dense 检索权重
        sparse_weight: BGE-M3 Sparse 检索权重
        bm25_weight: BM25 检索权重
        top_k: 最终返回 Top-K
        top_k_retrieve: 初始检索 Top-K
        **kwargs: 其他参数

    Returns:
        RAGPipeline 实例
    """
    from ..retrieval.dense_retriever import DenseRetriever
    from ..retrieval.bm25_retriever import BM25Retriever
    from ..retrieval.sparse_retriever import SparseRetriever
    from ..retrieval.hybrid_retriever import HybridRetriever
    from ..retrieval.context_expander import ContextExpander
    from ..retrieval.reranker import CrossEncoderReranker
    from .llm_client import create_llm_client
    from .prompt_template import PromptTemplate

    # 初始化检索器
    dense_retriever = DenseRetriever.from_output_dir(
        output_dir=embedding_output_dir
    )
    context_expander = ContextExpander(
        metadata=dense_retriever.metadata,
        window_size=2
    )

    if use_hybrid:
        metadata_path = os.path.join(embedding_output_dir, "unified_metadata.json")
        sparse_path = os.path.join(embedding_output_dir, "unified_sparse.pkl")

        bm25_retriever = BM25Retriever(metadata_path)

        # 尝试加载 Sparse 检索器
        sparse_retriever = None
        if os.path.exists(sparse_path):
            sparse_retriever = SparseRetriever(
                sparse_path=sparse_path,
                metadata_path=metadata_path,
                embedder=dense_retriever.embedder,
            )
        else:
            print(f"⚠ Sparse 向量文件不存在 ({sparse_path})，跳过 Sparse 检索路")

        retriever = HybridRetriever(
            dense_retriever, bm25_retriever,
            sparse_retriever=sparse_retriever,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            bm25_weight=bm25_weight,
        )
    else:
        retriever = dense_retriever

    # 初始化 Reranker
    reranker = None
    if use_reranker:
        try:
            reranker = CrossEncoderReranker()
        except Exception as e:
            print(f"Reranker 加载失败: {e}")

    # 初始化 LLM + 生成器
    llm_kwargs = {}
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = create_llm_client(
        model_name=model_name,
        provider=provider,
        **llm_kwargs
    )
    template = PromptTemplate()
    generator = AnswerGenerator(llm, template)

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        reranker=reranker,
        context_expander=context_expander,
        default_top_k=top_k,
        top_k_retrieve=top_k_retrieve
    )


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline 测试")
    print("=" * 60)

    # 检查索引是否存在
    output_dir = "src/embedding/output"
    index_path = os.path.join(output_dir, "unified_index.faiss")

    if not os.path.exists(index_path):
        print(f"\n警告: 索引文件不存在: {index_path}")
        print("请先运行 embedding pipeline:")
        print("  python src/embedding/run_pipeline.py")
        print("\n测试将使用模拟模式...")
    else:
        # 导入必要的模块
        from ..retrieval.dense_retriever import DenseRetriever

        print("\n初始化组件...")
        retriever = DenseRetriever.from_output_dir(output_dir)

        # 测试问答
        test_queries = [
            "What is the main contribution of the paper?",
            "How does the proposed method work?",
            "What are the experimental results?"
        ]

        print("\n测试问答:")
        print("-" * 40)

        for query in test_queries:
            print(f"\n问题: {query}")

            # 检索测试
            results = retriever.retrieve(query, top_k=3)
            print(f"检索到 {len(results)} 个结果")

            if results:
                print("Top-1 结果:")
                print(f"  {results[0].text[:100]}...")
