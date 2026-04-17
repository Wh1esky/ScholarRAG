"""
批量向量化模块（多向量版）

功能：
- 读取分块结果 JSON 文件
- 按粒度分组处理
- 批量生成 Dense + Sparse 向量
- 保存向量和元数据映射
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from .bge_embedder import BGEM3Embedder
from .config import OUTPUT_DIR, VECTOR_DIR


@dataclass
class ChunkInfo:
    """Chunk 元信息"""
    chunk_id: str
    text: str
    granularity: str
    paper_id: str
    section_type: str
    index_in_granularity: int  # 在该粒度中的索引


class BatchEmbedder:
    """
    批量向量化器
    
    处理流程：
    1. 读取分块结果 JSON
    2. 按粒度（sentence/paragraph/section）分组
    3. 批量编码生成向量
    4. 保存向量文件 + 元数据映射
    """
    
    def __init__(
        self,
        chunking_results_path: str,
        output_dir: str = VECTOR_DIR,
        batch_size: int = 32,
        model_name: str = "BAAI/bge-m3"
    ):
        """
        初始化批量向量化器
        
        Args:
            chunking_results_path: 分块结果 JSON 文件路径
            output_dir: 输出目录
            batch_size: 批处理大小
            model_name: embedding 模型名称
        """
        self.chunking_results_path = chunking_results_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化向量化器
        self.embedder = BGEM3Embedder(model_name=model_name)
        
        # 数据容器
        self.chunks_by_granularity: Dict[str, List[ChunkInfo]] = {
            'sentence': [],
            'paragraph': [],
            'section': []
        }
        self.all_chunks: List[ChunkInfo] = []
        
    def load_chunking_results(self) -> Dict[str, int]:
        """
        加载分块结果
        
        Returns:
            Dict[str, int]: 各粒度的 chunk 数量统计
        """
        print(f"正在加载分块结果: {self.chunking_results_path}")
        
        with open(self.chunking_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 🌟 核心修复区：智能识别数据格式 🌟
        if isinstance(data, dict):
            # 如果字典里直接有 paper_id，说明这是单篇论文！包上列表。
            if 'paper_id' in data:
                paper_list = [data]
            # 否则才是组员设想的以论文名为 key 的多篇论文字典。
            else:
                paper_list = list(data.values())
        else:
            paper_list = data
        
        counts = {'sentence': 0, 'paragraph': 0, 'section': 0}
        
        # 遍历每篇论文
        for paper in tqdm(paper_list, desc="处理论文"):
            
            # 获取论文 ID
            paper_id = paper.get('paper_id', 'unknown')
            
            # 🌟 核心修复区 2：智能寻找 chunks 在哪一层 🌟
            # (有些 JSON 结构会把数据包在 granularity_results 里面)
            chunk_container = paper.get('granularity_results', paper)
            
            # 遍历各粒度的 chunks
            for granularity in ['sentence', 'paragraph', 'section']:
                chunks = chunk_container.get(granularity, [])
                
                for idx, chunk in enumerate(chunks):
                    chunk_info = ChunkInfo(
                        chunk_id=f"{paper_id}_{granularity}_{idx}",
                        text=chunk.get('text', ''),
                        granularity=granularity,
                        paper_id=paper_id,
                        section_type=chunk.get('section_type', 'unknown'),
                        index_in_granularity=counts[granularity]
                    )
                    
                    self.chunks_by_granularity[granularity].append(chunk_info)
                    self.all_chunks.append(chunk_info)
                    counts[granularity] += 1
        
        print(f"\n加载完成！统计信息:")
        for gran, count in counts.items():
            print(f"  - {gran}: {count} chunks")
        
        return counts
    
    def embed_by_granularity(self, granularity: str) -> Tuple[np.ndarray, List[ChunkInfo], List[Dict[int, float]]]:
        """
        对指定粒度的所有 chunks 进行向量化（Dense + Sparse）
        
        Returns:
            Tuple: (dense向量矩阵, ChunkInfo列表, sparse权重列表)
        """
        chunks = self.chunks_by_granularity[granularity]
        
        if not chunks:
            print(f"警告: {granularity} 没有数据")
            return np.array([]), [], []
        
        print(f"\n正在向量化 {granularity} 粒度，共 {len(chunks)} 条数据...")
        
        texts = [chunk.text for chunk in chunks]
        
        # 使用 encode_multi 同时获取 Dense + Sparse
        result = self.embedder.encode_multi(texts, batch_size=self.batch_size)
        embeddings = result['dense_vecs']
        sparse_weights = result['sparse_weights']
        
        print(f"  Dense 矩阵形状: {embeddings.shape}")
        print(f"  Sparse 向量数: {len(sparse_weights)}")
        avg_nnz = np.mean([len(sw) for sw in sparse_weights]) if sparse_weights else 0
        print(f"  Sparse 平均非零项: {avg_nnz:.1f}")
        
        return embeddings, chunks, sparse_weights
    
    def embed_all(self) -> Dict[str, Tuple[np.ndarray, List[ChunkInfo], List[Dict[int, float]]]]:
        """
        对所有粒度进行向量化
        
        Returns:
            Dict[str, Tuple]: 各粒度的 (dense向量, ChunkInfo列表, sparse权重列表)
        """
        results = {}
        
        for granularity in ['sentence', 'paragraph', 'section']:
            embeddings, chunks, sparse_weights = self.embed_by_granularity(granularity)
            results[granularity] = (embeddings, chunks, sparse_weights)
        
        return results
    
    def save_vectors(
        self,
        granularity: str,
        embeddings: np.ndarray,
        chunks: List[ChunkInfo],
        sparse_weights: Optional[List[Dict[int, float]]] = None,
    ):
        """
        保存向量和元数据
        """
        # 保存 Dense 向量
        vector_path = os.path.join(self.output_dir, f"{granularity}_vectors.npy")
        np.save(vector_path, embeddings)
        print(f"  Dense 向量已保存: {vector_path}")
        
        # 保存 Sparse 向量
        if sparse_weights:
            sparse_path = os.path.join(self.output_dir, f"{granularity}_sparse.pkl")
            with open(sparse_path, 'wb') as f:
                pickle.dump(sparse_weights, f)
            print(f"  Sparse 向量已保存: {sparse_path}")
        
        # 保存元数据映射
        metadata_path = os.path.join(self.output_dir, f"{granularity}_metadata.json")
        metadata = []
        for chunk in chunks:
            metadata.append({
                'chunk_id': chunk.chunk_id,
                'paper_id': chunk.paper_id,
                'text': chunk.text[:200] + '...' if len(chunk.text) > 200 else chunk.text,
                'full_text': chunk.text,
                'granularity': chunk.granularity,
                'section_type': chunk.section_type,
                'index': chunk.index_in_granularity
            })
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  元数据已保存: {metadata_path}")
    
    def save_all_vectors(self, results: Dict[str, Tuple[np.ndarray, List[ChunkInfo], List[Dict[int, float]]]]):
        """
        保存所有粒度的向量
        """
        for granularity, (embeddings, chunks, sparse_weights) in results.items():
            if len(embeddings) > 0:
                self.save_vectors(granularity, embeddings, chunks, sparse_weights)
    
    def create_unified_index(self, results: Dict[str, Tuple[np.ndarray, List[ChunkInfo], List[Dict[int, float]]]]):
        """
        创建统一索引（所有粒度合并，包含 Dense + Sparse）
        """
        all_embeddings = []
        all_chunks = []
        all_sparse = []
        
        for granularity, (embeddings, chunks, sparse_weights) in results.items():
            if len(embeddings) > 0:
                all_embeddings.append(embeddings)
                all_chunks.extend(chunks)
                all_sparse.extend(sparse_weights)
        
        if not all_embeddings:
            print("警告: 没有可用的向量")
            return
        
        # 合并
        unified_vectors = np.vstack(all_embeddings)
        
        print(f"\n统一索引统计:")
        print(f"  总向量数: {len(unified_vectors)}")
        print(f"  向量维度: {unified_vectors.shape[1]}")
        
        # 保存 Dense 向量
        vector_path = os.path.join(self.output_dir, "unified_vectors.npy")
        np.save(vector_path, unified_vectors)
        print(f"  Dense 向量已保存: {vector_path}")
        
        # 保存 Sparse 向量
        if all_sparse:
            sparse_path = os.path.join(self.output_dir, "unified_sparse.pkl")
            with open(sparse_path, 'wb') as f:
                pickle.dump(all_sparse, f)
            avg_nnz = np.mean([len(sw) for sw in all_sparse])
            print(f"  Sparse 向量已保存: {sparse_path} (平均非零项: {avg_nnz:.1f})")
        
        # 保存元数据
        metadata_path = os.path.join(self.output_dir, "unified_metadata.json")
        metadata = []
        for chunk in all_chunks:
            metadata.append({
                'chunk_id': chunk.chunk_id,
                'paper_id': chunk.paper_id,
                'text': chunk.text,
                'granularity': chunk.granularity,
                'section_type': chunk.section_type,
            })
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  元数据已保存: {metadata_path}")
        
        return unified_vectors, all_chunks
    
    def run(self, save_unified: bool = True) -> Dict[str, Tuple[np.ndarray, List[ChunkInfo], List[Dict[int, float]]]]:
        """
        执行完整的批量向量化流程（Dense + Sparse）
        
        Args:
            save_unified: 是否保存统一索引
            
        Returns:
            Dict: 各粒度的 (dense向量, chunks, sparse权重)
        """
        # 1. 加载分块结果
        self.load_chunking_results()
        
        # 2. 向量化
        results = self.embed_all()
        
        # 3. 保存各粒度向量
        self.save_all_vectors(results)
        
        # 4. 保存统一索引
        if save_unified:
            self.create_unified_index(results)
        
        print("\n批量向量化完成！")
        
        return results


def main():
    """测试代码"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量向量化脚本')
    parser.add_argument('--input', type=str, 
                        default='src/chunking/output/batch_processing_results.json',
                        help='分块结果 JSON 文件路径')
    parser.add_argument('--output', type=str, 
                        default='src/embedding/output',
                        help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小')
    
    args = parser.parse_args()
    
    # 执行
    embedder = BatchEmbedder(
        chunking_results_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    
    embedder.run()


if __name__ == "__main__":
    main()
