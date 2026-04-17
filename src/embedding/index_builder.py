"""
FAISS 索引构建器

功能：
- 构建多种类型的 FAISS 索引
- 支持 IndexFlatIP、IndexIVFFlat、IndexHNSW
- 保存和加载索引
- 索引管理和搜索
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import faiss
from .config import FAISS_CONFIG, INDEX_DIR


@dataclass
class IndexStats:
    """索引统计信息"""
    index_type: str
    num_vectors: int
    dimension: int
    total_memory_mb: float
    
    
class FAISSIndexBuilder:
    """
    FAISS 索引构建器
    
    支持的索引类型：
    - FlatIP: 精确内积检索（适合小规模数据）
    - IVFFlat: 倒排文件索引（适合中等规模）
    - HNSW: 分层导航小世界图（适合大规模，高速）
    """
    
    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "FlatIP",
        output_dir: str = INDEX_DIR
    ):
        """
        初始化索引构建器
        
        Args:
            dimension: 向量维度（BGE-M3 为 1024）
            index_type: 索引类型 ('FlatIP', 'IVFFlat', 'HNSW')
            output_dir: 索引输出目录
        """
        self.dimension = dimension
        self.index_type = index_type.upper()
        self.output_dir = output_dir
        self.index: Optional[faiss.Index] = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def create_flat_ip_index(self) -> faiss.IndexFlatIP:
        """
        创建内积检索的 Flat 索引
        
        特点：
        - 精确检索，无召回率损失
        - 速度快，适合 <100k 向量
        - 内存占用：d * n * 4 bytes
        
        Returns:
            faiss.IndexFlatIP: 索引对象
        """
        index = faiss.IndexFlatIP(self.dimension)
        print(f"创建 FlatIP 索引，维度: {self.dimension}")
        return index
    
    def create_ivf_flat_index(self, nlist: int = 100) -> faiss.IndexIVFFlat:
        """
        创建倒排文件索引
        
        特点：
        - 近似检索，有一定召回率损失
        - 速度快，适合 100k-1M 向量
        - 需要训练
        
        Args:
            nlist: 聚类中心数量
            
        Returns:
            faiss.IndexIVFFlat: 索引对象
        """
        # 创建量化器
        quantizer = faiss.IndexFlatIP(self.dimension)
        
        # 创建 IVF 索引
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        print(f"创建 IVFFlat 索引，维度: {self.dimension}, nlist: {nlist}")
        return index
    
    def create_hnsw_index(
        self,
        M: int = 32,
        efConstruction: int = 200
    ) -> faiss.IndexHNSW:
        """
        创建 HNSW 图索引
        
        特点：
        - 近似检索，高召回率
        - 构建慢，检索极快
        - 适合 >1M 向量
        
        Args:
            M: 每个节点的连接数
            efConstruction: 构建时的搜索范围
            
        Returns:
            faiss.IndexHNSW: 索引对象
        """
        index = faiss.IndexHNSWFlat(self.dimension, M)
        index.hnsw.efConstruction = efConstruction
        
        print(f"创建 HNSW 索引，维度: {self.dimension}, M: {M}, efConstruction: {efConstruction}")
        return index
    
    def create_index(self, **kwargs) -> faiss.Index:
        """
        根据配置的索引类型创建索引
        
        Args:
            **kwargs: 传递给具体索引创建方法的参数
            
        Returns:
            faiss.Index: 索引对象
        """
        if self.index_type == "FLATIP":
            self.index = self.create_flat_ip_index()
        elif self.index_type == "IVFFLAT":
            nlist = kwargs.get('nlist', FAISS_CONFIG.get('nlist', 100))
            self.index = self.create_ivf_flat_index(nlist)
        elif self.index_type == "HNSW":
            M = kwargs.get('M', FAISS_CONFIG.get('M', 32))
            efConstruction = kwargs.get('efConstruction', FAISS_CONFIG.get('efConstruction', 200))
            self.index = self.create_hnsw_index(M, efConstruction)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        return self.index
    
    def train(self, vectors: np.ndarray):
        """
        训练索引（如需要）
        
        IVF 和某些索引需要训练数据来建立量化器
        
        Args:
            vectors: 训练向量，shape = (n, dimension)
        """
        if self.index is None:
            raise ValueError("索引未创建，请先调用 create_index()")
        
        if not self.index.is_trained:
            print(f"正在训练索引，训练数据形状: {vectors.shape}")
            
            # 确保是 float32
            vectors = vectors.astype(np.float32)
            
            # 训练
            self.index.train(vectors)
            print("索引训练完成")
    
    def add_vectors(self, vectors: np.ndarray):
        """
        添加向量到索引
        
        Args:
            vectors: 向量矩阵，shape = (n, dimension)
        """
        if self.index is None:
            raise ValueError("索引未创建，请先调用 create_index()")
        
        # 确保是 float32
        vectors = vectors.astype(np.float32)
        
        # 添加
        self.index.add(vectors)
        print(f"已添加 {len(vectors)} 个向量，索引总数: {self.index.ntotal}")
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最近邻
        
        Args:
            query_vectors: 查询向量，shape = (n, dimension) 或 (dimension,)
            k: 返回的近邻数量
            
        Returns:
            Tuple: (distances, indices)
                - distances: 距离/相似度矩阵，shape = (n, k)
                - indices: 索引矩阵，shape = (n, k)
        """
        if self.index is None:
            raise ValueError("索引未创建")
        
        # 处理单个查询向量
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # 确保是 float32
        query_vectors = query_vectors.astype(np.float32)
        
        # 搜索
        distances, indices = self.index.search(query_vectors, k)
        
        return distances, indices
    
    def save_index(self, save_path: str):
        """
        保存索引到文件
        
        Args:
            save_path: 保存路径
        """
        if self.index is None:
            raise ValueError("索引未创建")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # FAISS C++ 层不支持非 ASCII 路径（如中文），用临时文件绕过
        import tempfile, shutil
        try:
            faiss.write_index(self.index, save_path)
        except RuntimeError:
            with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                tmp_path = tmp.name
            faiss.write_index(self.index, tmp_path)
            shutil.move(tmp_path, save_path)
        print(f"索引已保存到: {save_path}")
    
    def load_index(self, load_path: str):
        """
        从文件加载索引
        
        Args:
            load_path: 加载路径
        """
        import tempfile, shutil
        try:
            self.index = faiss.read_index(load_path)
        except RuntimeError:
            with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                tmp_path = tmp.name
            shutil.copy2(load_path, tmp_path)
            self.index = faiss.read_index(tmp_path)
            os.remove(tmp_path)
        print(f"索引已从 {load_path} 加载，向量数: {self.index.ntotal}")
    
    def get_stats(self) -> IndexStats:
        """
        获取索引统计信息
        
        Returns:
            IndexStats: 统计信息
        """
        if self.index is None:
            raise ValueError("索引未创建")
        
        # 计算内存占用（估算）
        memory_bytes = self.dimension * self.index.ntotal * 4  # float32 = 4 bytes
        memory_mb = memory_bytes / (1024 * 1024)
        
        return IndexStats(
            index_type=self.index_type,
            num_vectors=self.index.ntotal,
            dimension=self.dimension,
            total_memory_mb=memory_mb
        )
    
    def build_from_vectors(
        self,
        vectors: np.ndarray,
        save_path: str,
        train: bool = True,
        **kwargs
    ) -> faiss.Index:
        """
        从向量构建并保存索引
        
        Args:
            vectors: 向量矩阵
            save_path: 保存路径
            train: 是否训练
            **kwargs: 传递给 create_index 的参数
            
        Returns:
            faiss.Index: 构建的索引
        """
        print(f"开始构建索引...")
        print(f"  向量形状: {vectors.shape}")
        print(f"  索引类型: {self.index_type}")
        
        # 创建索引
        self.create_index(**kwargs)
        
        # 训练（如需要）
        if train and not self.index.is_trained:
            self.train(vectors)
        
        # 添加向量
        self.add_vectors(vectors)
        
        # 保存
        self.save_index(save_path)
        
        # 打印统计
        stats = self.get_stats()
        print(f"\n索引构建完成！")
        print(f"  类型: {stats.index_type}")
        print(f"  向量数: {stats.num_vectors}")
        print(f"  维度: {stats.dimension}")
        print(f"  内存占用: {stats.total_memory_mb:.2f} MB")
        
        return self.index
    
    def set_ef_search(self, ef: int):
        """
        设置 HNSW 的搜索参数
        
        Args:
            ef: 搜索范围，值越大越准确但越慢
        """
        if self.index is None:
            raise ValueError("索引未创建")
        
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef
            print(f"HNSW efSearch 设置为: {ef}")
        else:
            print("警告: 当前索引不是 HNSW 类型")


def main():
    """测试代码"""
    print("=" * 50)
    print("测试 FAISS 索引构建器")
    print("=" * 50)
    
    # 创建测试数据
    dimension = 1024
    num_vectors = 1000
    
    print(f"\n生成测试数据: {num_vectors} x {dimension}")
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    # 归一化（用于内积检索）
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # 测试 FlatIP 索引
    print("\n1. 测试 FlatIP 索引:")
    builder = FAISSIndexBuilder(dimension=dimension, index_type="FlatIP")
    index = builder.build_from_vectors(
        vectors, 
        save_path="src/embedding/output/test_index.faiss",
        train=False
    )
    
    # 测试搜索
    query = vectors[0].reshape(1, -1)
    distances, indices = builder.search(query, k=5)
    print(f"\n搜索结果 (Top-5):")
    print(f"  索引: {indices[0]}")
    print(f"  距离: {distances[0]}")
    
    # 测试 HNSW 索引
    print("\n2. 测试 HNSW 索引:")
    builder_hnsw = FAISSIndexBuilder(dimension=dimension, index_type="HNSW")
    index_hnsw = builder_hnsw.build_from_vectors(
        vectors[:500],  # 用较少的向量测试
        save_path="src/embedding/output/test_hnsw_index.faiss",
        train=False,
        M=16,
        efConstruction=100
    )
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
