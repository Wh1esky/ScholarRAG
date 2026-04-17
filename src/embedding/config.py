"""
Embedding 模块配置文件
"""

# BGE-M3 模型配置
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-m3",
    "embedding_dim": 1024,
    "normalize_embeddings": True,
    "batch_size": 32,
}

# FAISS 索引配置
FAISS_CONFIG = {
    "index_type": "FlatIP",  # FlatIP, IVFFlat, HNSW
    "nlist": 100,  # IVF 聚类数
    "nprobe": 10,  # IVF 搜索范围
    "M": 32,  # HNSW M 参数
    "efConstruction": 200,  # HNSW 构建参数
    "efSearch": 100,  # HNSW 搜索参数
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "min_score": 0.5,
    "rerank": False,
}

# 输出目录
OUTPUT_DIR = "src/embedding/output"
VECTOR_DIR = f"{OUTPUT_DIR}/vectors"
INDEX_DIR = f"{OUTPUT_DIR}/indices"
