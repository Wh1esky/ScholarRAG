"""
BGE-M3 向量化器（多向量版）

功能：
- 加载 BAAI/bge-m3 模型（通过 FlagEmbedding）
- 输出 Dense (1024维) + Sparse (学习词法权重) 向量
- 支持批量编码

参考：https://huggingface.co/BAAI/bge-m3
"""

import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# ── FlagEmbedding 兼容性补丁 ──────────────────────────────
# FlagEmbedding 1.3.x 引用了 transformers 已移除的 is_torch_fx_available
import transformers.utils.import_utils as _tiu
if not hasattr(_tiu, 'is_torch_fx_available'):
    _tiu.is_torch_fx_available = lambda: True
# ──────────────────────────────────────────────────────────

from FlagEmbedding import BGEM3FlagModel
from .config import EMBEDDING_CONFIG


class BGEM3Embedder:
    """
    BGE-M3 多向量化器

    输出：
    - Dense embedding (1024 维, L2 归一化)
    - Sparse lexical weights (token_id → weight 字典)

    使用方式：
    1. 初始化时自动加载模型
    2. encode() → dense向量 (numpy)
    3. encode_multi() → dict{dense_vecs, sparse_weights}
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_folder: Optional[str] = None,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 模型路径解析
        if cache_folder is None:
            cache_folder = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'models', 'embedding'
            )
        preferred_local_dir = Path(cache_folder) / 'bge-m3'

        # 如果本地目录存在则直接用，否则用 repo_id 让 FlagEmbedding 下载
        model_path = str(preferred_local_dir) if preferred_local_dir.exists() else model_name

        print(f"正在加载 BGE-M3 (FlagEmbedding): {model_path}")
        print(f"设备: {self.device}, FP16: {use_fp16}")

        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16 and self.device == 'cuda',
        )
        self.embedding_dim = 1024
        print(f"向量维度: {self.embedding_dim}")

    # ── Dense-only 接口（兼容原有调用） ──────────────────

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Dense 编码，返回 (n, 1024) numpy 矩阵"""
        if isinstance(texts, str):
            texts = [texts]
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out['dense_vecs']

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1, show_progress=False)[0]

    # ── 多向量接口 ─────────────────────────────────────

    def encode_multi(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> Dict[str, object]:
        """
        同时返回 Dense + Sparse 向量。

        Returns:
            {
                'dense_vecs': np.ndarray (n, 1024),
                'sparse_weights': List[Dict[int, float]]  # token_id → weight
            }
        """
        if isinstance(texts, str):
            texts = [texts]
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # 将 defaultdict 转为普通 dict
        sparse_weights = [dict(w) for w in out['lexical_weights']]
        return {
            'dense_vecs': out['dense_vecs'],
            'sparse_weights': sparse_weights,
        }

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def save_index(self, embeddings: np.ndarray, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embeddings)
        print(f"向量已保存到: {save_path}")

    def load_index(self, load_path: str) -> np.ndarray:
        embeddings = np.load(load_path)
        print(f"向量已从 {load_path} 加载，shape: {embeddings.shape}")
        return embeddings



if __name__ == "__main__":
    emb = BGEM3Embedder()
    out = emb.encode_multi(["Hello world", "Test sentence"], batch_size=2)
    print("Dense:", out['dense_vecs'].shape)
    print("Sparse count:", len(out['sparse_weights']))
