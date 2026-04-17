"""
Dense Retrieval 模块

功能：
- 使用 BGE-M3 进行向量检索
- FAISS 索引搜索
- 支持按粒度筛选
- 返回检索结果及元信息
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import faiss

from ..embedding.bge_embedder import BGEM3Embedder
from ..embedding.config import EMBEDDING_CONFIG, RETRIEVAL_CONFIG


@dataclass
class RetrievalResult:
    """
    检索结果
    
    Attributes:
        chunk_id: Chunk 唯一标识
        text: Chunk 原文
        paper_id: 所属论文 ID
        granularity: 粒度 (sentence/paragraph/section)
        section_type: 章节类型
        score: 相似度分数
        rank: 排名
    """
    chunk_id: str
    text: str
    paper_id: str
    granularity: str
    section_type: str
    score: float
    rank: int


class DenseRetriever:
    """
    Dense Retriever - 密集向量检索器
    
    检索流程：
    1. 用 BGE-M3 将 Query 编码为向量
    2. 在 FAISS 索引中搜索 Top-K
    3. 从元数据中获取 Chunk 详情
    4. 返回结构化检索结果
    """
    
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedder: Optional[BGEM3Embedder] = None,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None
    ):
        """
        初始化 Dense Retriever
        
        Args:
            index_path: FAISS 索引文件路径
            metadata_path: 元数据 JSON 文件路径
            embedder: 已初始化的向量化器（可选）
            model_name: 模型名称
            device: 设备 ('cuda', 'cpu')
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # 加载 FAISS 索引（FAISS C++ 层不支持非 ASCII 路径，用临时文件绕过）
        print(f"加载 FAISS 索引: {index_path}")
        try:
            self.index = faiss.read_index(index_path)
        except RuntimeError:
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                tmp_path = tmp.name
            shutil.copy2(index_path, tmp_path)
            self.index = faiss.read_index(tmp_path)
            os.remove(tmp_path)
        print(f"  索引向量数: {self.index.ntotal}")
        
        # 加载元数据
        print(f"加载元数据: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"  元数据条数: {len(self.metadata)}")
        
        # 初始化向量化器
        if embedder is None:
            self.embedder = BGEM3Embedder(
                model_name=model_name,
                device=device,
                normalize_embeddings=True
            )
        else:
            self.embedder = embedder
        
        # 获取向量维度
        self.dimension = self.embedder.get_embedding_dim()
        print(f"  向量维度: {self.dimension}")
    
    @classmethod
    def from_output_dir(
        cls,
        output_dir: str,
        granularity: str = "unified",
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None
    ):
        """
        从输出目录加载 Retriever
        
        Args:
            output_dir: embedding/output 目录
            granularity: 粒度 ('sentence', 'paragraph', 'section', 'unified')
            model_name: 模型名称
            device: 设备
        """
        index_path = os.path.join(output_dir, f"{granularity}_index.faiss")
        metadata_path = os.path.join(output_dir, f"{granularity}_metadata.json")
        
        return cls(
            index_path=index_path,
            metadata_path=metadata_path,
            model_name=model_name,
            device=device
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        将查询文本编码为向量
        
        Args:
            query: 查询文本
            
        Returns:
            numpy.ndarray: 查询向量
        """
        embedding = self.embedder.encode_single(query)
        return embedding
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        granularity_filter: Optional[str] = None,
        paper_id_filter: Optional[str] = None,
        return_texts: bool = True
    ) -> List[RetrievalResult]:
        """
        检索相关 Chunk
        
        Args:
            query: 查询文本
            top_k: 返回的 Top-K 结果数
            min_score: 最小相似度分数阈值
            granularity_filter: 按粒度筛选（如 'sentence'）
            paper_id_filter: 按论文 ID 筛选
            return_texts: 是否返回完整文本
            
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        # 编码查询
        query_vector = self.encode_query(query)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 搜索
        distances, indices = self.index.search(query_vector, top_k * 10)  # 多搜一些用于过滤
        
        results = []
        
        for rank, idx in enumerate(indices[0]):
            if idx == -1:  # 无效索引
                continue
            
            # 获取元数据
            if idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            # 粒度筛选
            if granularity_filter and meta.get('granularity') != granularity_filter:
                continue
            
            # 论文 ID 筛选
            if paper_id_filter and meta.get('paper_id') != paper_id_filter:
                continue
            
            # 计算分数（内积即相似度，已经是归一化后的）
            score = float(distances[0][rank])
            
            # 分数阈值筛选
            if min_score and score < min_score:
                continue
            
            # 构建结果
            result = RetrievalResult(
                chunk_id=meta.get('chunk_id', f'chunk_{idx}'),
                text=meta.get('text', '') if return_texts else '',
                paper_id=meta.get('paper_id', 'unknown'),
                granularity=meta.get('granularity', 'unknown'),
                section_type=meta.get('section_type', 'unknown'),
                score=score,
                rank=len(results) + 1
            )
            
            results.append(result)
            
            # 达到 top_k
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_by_granularity(
        self,
        query: str,
        top_k_per_gran: int = 3,
        granularities: Optional[List[str]] = None
    ) -> Dict[str, List[RetrievalResult]]:
        """
        按粒度分别检索
        
        Args:
            query: 查询文本
            top_k_per_gran: 每个粒度返回的结果数
            granularities: 要检索的粒度列表
            
        Returns:
            Dict[str, List[RetrievalResult]]: 按粒度分组的检索结果
        """
        if granularities is None:
            granularities = ['sentence', 'paragraph', 'section']
        
        results = {}
        
        for gran in granularities:
            results[gran] = self.retrieve(
                query,
                top_k=top_k_per_gran,
                granularity_filter=gran
            )
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        show_progress: bool = True
    ) -> List[List[RetrievalResult]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: Top-K
            show_progress: 是否显示进度
            
        Returns:
            List[List[RetrievalResult]]: 每个查询的检索结果
        """
        results = []
        
        from tqdm import tqdm
        iterator = tqdm(queries, desc="检索中") if show_progress else queries
        
        for query in iterator:
            result = self.retrieve(query, top_k=top_k)
            results.append(result)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        根据 Chunk ID 获取详情
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Dict: Chunk 元数据
        """
        for meta in self.metadata:
            if meta.get('chunk_id') == chunk_id:
                return meta
        return None
    
    def get_chunks_by_paper(
        self, 
        paper_id: str,
        granularity: Optional[str] = None
    ) -> List[Dict]:
        """
        获取指定论文的所有 Chunks
        
        Args:
            paper_id: 论文 ID
            granularity: 粒度筛选
            
        Returns:
            List[Dict]: Chunk 列表
        """
        chunks = []
        
        for meta in self.metadata:
            if meta.get('paper_id') == paper_id:
                if granularity is None or meta.get('granularity') == granularity:
                    chunks.append(meta)
        
        return chunks
    
    def evaluate_query(
        self,
        query: str,
        ground_truth: str,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        评估单个查询的检索效果
        
        Args:
            query: 查询
            ground_truth: 标准答案
            top_k: Top-K
            
        Returns:
            Dict: 评估指标
        """
        results = self.retrieve(query, top_k=top_k)
        
        # 简单评估：检索结果是否包含答案相关词汇
        retrieved_texts = [r.text.lower() for r in results]
        ground_truth_lower = ground_truth.lower()
        
        # 计算命中词数
        gt_words = set(ground_truth_lower.split())
        hit_count = 0
        for text in retrieved_texts:
            text_words = set(text.split())
            hits = len(gt_words & text_words)
            hit_count += hits
        
        # Precision@K
        precision = hit_count / (top_k * len(gt_words)) if gt_words else 0
        
        return {
            'top_k': top_k,
            'num_results': len(results),
            'avg_score': np.mean([r.score for r in results]) if results else 0,
            'max_score': results[0].score if results else 0,
            'precision': precision,
            'hit_count': hit_count
        }
    
    def print_results(self, results: List[RetrievalResult], max_text_len: int = 150):
        """
        打印检索结果
        
        Args:
            results: 检索结果
            max_text_len: 最大文本显示长度
        """
        print(f"\n检索到 {len(results)} 个结果:")
        print("=" * 80)
        
        for r in results:
            text_preview = r.text[:max_text_len] + '...' if len(r.text) > max_text_len else r.text
            
            print(f"\n[Rank {r.rank}] Score: {r.score:.4f}")
            print(f"  Paper: {r.paper_id}")
            print(f"  Granularity: {r.granularity} | Section: {r.section_type}")
            print(f"  Text: {text_preview}")
            print("-" * 80)


def main():
    """测试代码"""
    print("=" * 50)
    print("Dense Retriever 测试")
    print("=" * 50)
    
    # 检查是否有已构建的索引
    output_dir = "src/embedding/output"
    index_path = os.path.join(output_dir, "unified_index.faiss")
    metadata_path = os.path.join(output_dir, "unified_metadata.json")
    
    if not os.path.exists(index_path):
        print("错误: 索引文件不存在，请先运行批量向量化")
        print(f"预期路径: {index_path}")
        return
    
    # 初始化 Retriever
    retriever = DenseRetriever(
        index_path=index_path,
        metadata_path=metadata_path
    )
    
    # 测试检索
    test_queries = [
        "What is the main contribution of this paper?",
        "How does the proposed method work?",
        "What are the experimental results?"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.retrieve(query, top_k=3)
        retriever.print_results(results)


if __name__ == "__main__":
    main()
