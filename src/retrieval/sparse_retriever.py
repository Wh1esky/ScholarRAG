"""
BGE-M3 Sparse Retrieval 模块

功能：
- 使用 BGE-M3 模型输出的学习型稀疏向量 (lexical weights) 进行检索
- 构建 token-level 倒排索引，高效计算 dot-product 分数
- 与 Dense / BM25 配合组成多路召回
"""

import json
import pickle
from typing import List, Dict, Optional
from collections import defaultdict

from .dense_retriever import RetrievalResult


class SparseRetriever:
    """
    BGE-M3 学习型稀疏检索器

    原理：
    - BGE-M3 为每个文本输出 token_id → weight 的稀疏表示
    - 检索时，query 与 doc 的稀疏向量做 dot-product 作为相关性分数
    - 使用倒排索引加速，只计算有交集 token 的文档
    """

    def __init__(self, sparse_path: str, metadata_path: str, embedder=None):
        """
        Args:
            sparse_path:   unified_sparse.pkl 路径
            metadata_path: unified_metadata.json 路径
            embedder:      BGEM3Embedder 实例（用于实时编码 query）
        """
        self.embedder = embedder

        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # 加载 sparse 向量
        with open(sparse_path, 'rb') as f:
            self.sparse_weights: List[Dict[int, float]] = pickle.load(f)

        assert len(self.sparse_weights) == len(self.metadata), \
            f"sparse ({len(self.sparse_weights)}) 与 metadata ({len(self.metadata)}) 数量不匹配"

        # 构建倒排索引：token_id → [(doc_idx, weight), ...]
        print(f"构建 Sparse 倒排索引 ({len(self.metadata)} docs)...")
        self.inverted_index: Dict[int, List[tuple]] = defaultdict(list)
        for doc_idx, sw in enumerate(self.sparse_weights):
            for token_id, weight in sw.items():
                self.inverted_index[token_id].append((doc_idx, weight))

        total_postings = sum(len(v) for v in self.inverted_index.values())
        print(f"  词条数: {len(self.inverted_index)}, 倒排记录数: {total_postings}")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        granularity_filter: Optional[str] = None,
        query_sparse: Optional[Dict[int, float]] = None,
    ) -> List[RetrievalResult]:
        """
        稀疏检索

        Args:
            query:              查询文本（如果提供 query_sparse 可忽略）
            top_k:              返回结果数
            granularity_filter: 按粒度筛选
            query_sparse:       预计算的 query sparse 向量（可选）

        Returns:
            List[RetrievalResult]
        """
        # 获取 query 的 sparse 向量
        if query_sparse is None:
            if self.embedder is None:
                raise ValueError("需要 embedder 或提供 query_sparse")
            out = self.embedder.encode_multi([query], batch_size=1)
            query_sparse = out['sparse_weights'][0]

        # Dot-product 打分（利用倒排索引）
        doc_scores: Dict[int, float] = {}
        for token_id, q_weight in query_sparse.items():
            if token_id not in self.inverted_index:
                continue
            for doc_idx, d_weight in self.inverted_index[token_id]:
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + q_weight * d_weight

        # 排序 & 过滤
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs:
            meta = self.metadata[doc_idx]
            if granularity_filter and meta.get('granularity') != granularity_filter:
                continue
            results.append(RetrievalResult(
                chunk_id=meta.get('chunk_id', f'chunk_{doc_idx}'),
                text=meta.get('text', ''),
                paper_id=meta.get('paper_id', 'unknown'),
                granularity=meta.get('granularity', 'unknown'),
                section_type=meta.get('section_type', 'unknown'),
                score=score,
                rank=len(results) + 1,
            ))
            if len(results) >= top_k:
                break

        return results
