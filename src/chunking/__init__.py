"""
ScholarRAG: ScholarRAG Chunking Module
自适应分块策略实现 - 第二阶段

参考论文: Mix-of-Granularity (COLING 2025)
GitHub: https://github.com/ZGChung/Mix-of-Granularity

模块结构:
- unified_format.py: 数据格式转换
- structure_recognizer.py: 论文结构识别
- granularity_chunker.py: 多粒度分块
- adaptive_router.py: 自适应路由（简化版 - 规则匹配）
- mlp_router.py: MLP Router（完整版 - 神经网络）
- evaluator.py: 质量评估

训练流程:
1. prepare_training_data.py: 准备训练数据
2. train_router.py: 训练 Router 模型
"""

from .unified_format import MinerUToUnifiedConverter
from .structure_recognizer import StructureRecognizer
from .granularity_chunker import GranularityChunker, ChunkGranularity
from .adaptive_router import AdaptiveRouter  # 简化版
from .evaluator import ChunkEvaluator

# MLP Router（需要额外安装 sentence-transformers）
try:
    from .mlp_router import MoGRouter, MoGRouterTrainer, SoftLabelBuilder, RetrievalResult
    HAS_MLP_ROUTER = True
except ImportError:
    HAS_MLP_ROUTER = False
    MoGRouter = None
    MoGRouterTrainer = None

__all__ = [
    "MinerUToUnifiedConverter",
    "StructureRecognizer",
    "GranularityChunker",
    "ChunkGranularity",
    "AdaptiveRouter",
    "ChunkEvaluator",
    "HAS_MLP_ROUTER",
    # MLP Router (optional)
    "MoGRouter",
    "MoGRouterTrainer",
    "SoftLabelBuilder",
    "RetrievalResult",
]
