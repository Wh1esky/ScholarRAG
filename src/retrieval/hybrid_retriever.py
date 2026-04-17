"""
Hybrid Retriever - 混合检索器（三路召回）

功能：
- 融合 Dense (FAISS) + BGE-M3 Sparse + BM25 检索结果
- 使用 Reciprocal Rank Fusion (RRF) 合并排名
- 兼顾语义相似、学习型词法匹配和经典词频匹配
"""

from typing import List, Optional, Dict
from .dense_retriever import DenseRetriever, RetrievalResult
from .bm25_retriever import BM25Retriever
from .sparse_retriever import SparseRetriever


class HybridRetriever:
    """
    混合检索器（支持 2 路或 3 路召回）

    使用 RRF (Reciprocal Rank Fusion) 融合多路结果。
    RRF 公式: score = sum(weight_i / (k + rank_i))，k=60 为经典默认值。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        sparse_retriever: Optional[SparseRetriever] = None,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.3,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        """
        Args:
            dense_retriever:  Dense (FAISS) 检索器
            bm25_retriever:   BM25 检索器
            sparse_retriever: BGE-M3 学习型稀疏检索器 (可选)
            dense_weight:     Dense 检索的权重
            sparse_weight:    BGE-M3 Sparse 检索的权重
            bm25_weight:      BM25 检索的权重
            rrf_k:            RRF 中的 k 参数
        """
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight if sparse_retriever else 0.0
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        routes = ["Dense", "BM25"]
        if sparse_retriever:
            routes.insert(1, "BGE-M3 Sparse")
        print(f"HybridRetriever 初始化: {' + '.join(routes)}")
        print(f"  权重: Dense={self.dense_weight}, Sparse={self.sparse_weight}, BM25={self.bm25_weight}, RRF_k={rrf_k}")

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        granularity_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """混合检索"""
        # 分别检索，各取 top_k 个
        dense_results = self.dense.retrieve(
            query, top_k=top_k, granularity_filter=granularity_filter
        )
        bm25_results = self.bm25.retrieve(
            query, top_k=top_k, granularity_filter=granularity_filter
        )

        # RRF 融合
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Dense
        for rank, r in enumerate(dense_results, start=1):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + \
                self.dense_weight / (self.rrf_k + rank)
            result_map[r.chunk_id] = r

        # BGE-M3 Sparse
        if self.sparse:
            sparse_results = self.sparse.retrieve(
                query, top_k=top_k, granularity_filter=granularity_filter
            )
            for rank, r in enumerate(sparse_results, start=1):
                rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + \
                    self.sparse_weight / (self.rrf_k + rank)
                if r.chunk_id not in result_map:
                    result_map[r.chunk_id] = r

        # BM25
        for rank, r in enumerate(bm25_results, start=1):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + \
                self.bm25_weight / (self.rrf_k + rank)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 过滤参考文献/致谢 chunks（占索引 15%, 是 CP 噪声主要来源）
        _noise_sections = {"reference", "references", "bibliography",
                           "acknowledgment", "acknowledgement",
                           "acknowledgments", "acknowledgements"}

        results = []
        for i, chunk_id in enumerate(sorted_ids):
            r = result_map[chunk_id]
            if (r.section_type or "").lower() in _noise_sections:
                continue
            results.append(RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                paper_id=r.paper_id,
                granularity=r.granularity,
                section_type=r.section_type,
                score=rrf_scores[chunk_id],
                rank=len(results) + 1,
            ))

            if len(results) >= top_k:
                break

        return results
