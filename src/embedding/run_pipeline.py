"""
Embedding Pipeline - 一键运行脚本

功能：
1. 批量向量化分块结果
2. 构建 FAISS 索引
3. 保存索引和元数据

使用方法：
    python src/embedding/run_pipeline.py
    python src/embedding/run_pipeline.py --input src/chunking/output/batch_processing_results.json --batch-size 32
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embedding.batch_embedder import BatchEmbedder
from src.embedding.index_builder import FAISSIndexBuilder
from src.embedding.config import VECTOR_DIR, INDEX_DIR


def run_embedding_pipeline(
    input_path: str = "src/chunking/output/batch_processing_results.json",
    output_dir: str = "src/embedding/output",
    batch_size: int = 32,
    index_type: str = "FlatIP",
    build_unified_index: bool = True,
    build_granularity_indices: bool = True
):
    """
    运行完整的 embedding pipeline
    
    Args:
        input_path: 分块结果 JSON 路径
        output_dir: 输出目录
        batch_size: 批处理大小
        index_type: 索引类型
        build_unified_index: 是否构建统一索引
        build_granularity_indices: 是否按粒度构建索引
    """
    print("=" * 60)
    print("ScholarRAG Embedding Pipeline")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # =====================
    # 步骤 1: 批量向量化
    # =====================
    print("\n" + "=" * 60)
    print("步骤 1: 批量向量化")
    print("=" * 60)
    
    embedder = BatchEmbedder(
        chunking_results_path=input_path,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    # 执行向量化
    results = embedder.run(save_unified=build_unified_index)
    
    # =====================
    # 步骤 2: 构建 FAISS 索引
    # =====================
    print("\n" + "=" * 60)
    print("步骤 2: 构建 FAISS 索引")
    print("=" * 60)
    
    index_builder = FAISSIndexBuilder(
        dimension=1024,  # BGE-M3 维度
        index_type=index_type,
        output_dir=output_dir
    )
    
    # 构建统一索引
    if build_unified_index:
        unified_vector_path = os.path.join(output_dir, "unified_vectors.npy")
        
        if os.path.exists(unified_vector_path):
            print(f"\n加载向量文件: {unified_vector_path}")
            import numpy as np
            unified_vectors = np.load(unified_vector_path)
            
            print(f"向量形状: {unified_vectors.shape}")
            
            # 构建索引
            index_path = os.path.join(output_dir, "unified_index.faiss")
            index_builder.build_from_vectors(
                vectors=unified_vectors,
                save_path=index_path,
                train=False  # FlatIP 不需要训练
            )
    
    # 按粒度构建索引
    if build_granularity_indices:
        for granularity in ['sentence', 'paragraph', 'section']:
            vector_path = os.path.join(output_dir, f"{granularity}_vectors.npy")
            
            if os.path.exists(vector_path):
                print(f"\n构建 {granularity} 索引...")
                import numpy as np
                vectors = np.load(vector_path)
                
                index_path = os.path.join(output_dir, f"{granularity}_index.faiss")
                index_builder.build_from_vectors(
                    vectors=vectors,
                    save_path=index_path,
                    train=False
                )
    
    # =====================
    # 完成
    # =====================
    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 向量文件: {output_dir}/*.npy")
    print(f"  - 索引文件: {output_dir}/*.faiss")
    print(f"  - 元数据: {output_dir}/*_metadata.json")
    print("\n下一步: 使用 DenseRetriever 进行检索测试")
    print("  from src.retrieval import DenseRetriever")
    print("  retriever = DenseRetriever.from_output_dir('src/embedding/output')")
    print("  results = retriever.retrieve('your query here', top_k=5)")


def main():
    parser = argparse.ArgumentParser(description='Embedding Pipeline')
    parser.add_argument('--input', type=str,
                        default='src/chunking/output/batch_processing_results.json',
                        help='分块结果 JSON 文件路径')
    parser.add_argument('--output', type=str,
                        default='src/embedding/output',
                        help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--index-type', type=str, default='FlatIP',
                        choices=['FlatIP', 'IVFFlat', 'HNSW'],
                        help='FAISS 索引类型')
    parser.add_argument('--no-unified', action='store_true',
                        help='不构建统一索引')
    parser.add_argument('--no-granularity', action='store_true',
                        help='不按粒度构建索引')
    
    args = parser.parse_args()
    
    run_embedding_pipeline(
        input_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        index_type=args.index_type,
        build_unified_index=not args.no_unified,
        build_granularity_indices=not args.no_granularity
    )


if __name__ == "__main__":
    main()
