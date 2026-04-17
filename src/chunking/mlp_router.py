"""
MLP Router 模块 - 参考 MoG 论文实现

原论文: Mix-of-Granularity (COLING 2025)
GitHub: https://github.com/ZGChung/Mix-of-Granularity

核心思想：
- 使用 MLP 网络根据 query 预测各粒度的权重
- 使用 Soft Labels 训练（基于 stsb-roberta-large 语义相似度）
- 输出：[sentence_weight, paragraph_weight, section_weight, document_weight]

与简化版对比：
- 简化版：用正则匹配关键词（快速但不准确）
- MLP版：用神经网络预测权重（更准确但需要训练）

使用流程：
1. 准备训练数据（不同粒度的检索结果 + QA对）
2. 构建 Soft Labels（计算检索结果与答案的相似度）
3. 训练 Router
4. 推理时用 Router 预测权重
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

from .granularity_chunker import ChunkGranularity
from ..utils.hf_utils import configure_hf_environment, resolve_model_source


DEFAULT_ROUTER_MODEL = os.environ.get('ROUTER_EMBEDDING_MODEL', 'sentence-transformers/stsb-roberta-large')


def _resolve_router_embedding_model(model_name: str = DEFAULT_ROUTER_MODEL):
    configure_hf_environment()
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return resolve_model_source(
        repo_id=model_name,
        preferred_local_dir=os.path.join(base_dir, 'models', 'router', 'stsb-roberta-large')
    )


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    question: str
    ground_truth: str  # 标准答案
    retrieved_snippets: Dict[str, List[str]]  # {granularity: [chunks]}
    soft_labels: List[float]  # 各粒度的软标签
    scores: Dict[str, float]  # 各粒度的分数


class MoGRouter(nn.Module):
    """
    MoG Router - 基于 MLP 的粒度权重预测器
    
    原理：
    1. 用 Sentence-Transformer 把 query 编码成向量
    2. MLP 网络预测各粒度的权重
    3. 输出 softmax 后的权重向量（和为1）
    
    输出维度：4（对应 sentence, paragraph, section, document）
    """
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_ROUTER_MODEL,
        output_dim: int = 4,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        super(MoGRouter, self).__init__()
        
        self.output_dim = output_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_folder = cache_folder
        resolved_model = _resolve_router_embedding_model(embedding_model)
        
        # Query 编码器
        print(f"Loading embedding model: {resolved_model}")
        self.query_encoder = SentenceTransformer(
            resolved_model,
            cache_folder=cache_folder
        )
        
        # 获取 embedding 维度
        sample_embedding = self.query_encoder.encode("test")
        self.embedding_dim = sample_embedding.shape[0]

        # MLP 网络 (工业级升级版：加宽、加深、加 LayerNorm 防止死记硬背)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2), # 提高 Dropout，强迫模型提取共有特征
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        print(f"Router initialized. Embedding dim: {self.embedding_dim}, Device: {self.device}")
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        编码查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            torch.Tensor: query 向量
        """
        embedding = self.query_encoder.encode(query)
        return torch.tensor(embedding, dtype=torch.float32).to(self.device)
    
    def forward(self, query: str) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询文本
            
        Returns:
            torch.Tensor: 各粒度的权重 [sentence, paragraph, section, document]
        """
        query_embedding = self.encode_query(query)
        weights = self.mlp(query_embedding)
        return weights
    
    def predict(self, query: str) -> Dict[str, float]:
        """
        预测各粒度的权重
        
        Args:
            query: 查询文本
            
        Returns:
            Dict[str, float]: 各粒度的权重
        """
        with torch.no_grad():
            weights = self.forward(query)
            weights = weights.cpu().numpy()
        
        granularities = ['sentence', 'paragraph', 'section', 'document']
        return {g: float(w) for g, w in zip(granularities, weights)}
    
    def predict_granularity(self, query: str) -> Tuple[str, float]:
        """
        预测最优粒度
        
        Args:
            query: 查询文本
            
        Returns:
            Tuple[str, float]: (最优粒度, 置信度)
        """
        weights = self.predict(query)
        best_gran = max(weights, key=weights.get)
        return best_gran, weights[best_gran]


class SimLoss(nn.Module):
    """
    相似度损失函数 - 用于训练 Router
    
    原理：
    - 使用 stsb-roberta-large 计算检索片段与答案的语义相似度
    - 相似度高的粒度应该有更高的权重
    """
    
    def __init__(self):
        super(SimLoss, self).__init__()
        self.query_encoder = SentenceTransformer(
            _resolve_router_embedding_model(),
            cache_folder='./models'
        )
    
    def compute_similarity(self, snippet: str, context: str) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            snippet: 检索片段
            context: 上下文（通常是答案）
            
        Returns:
            float: 相似度分数 [0, 1]
        """
        snippet_emb = self.query_encoder.encode(snippet).reshape(1, -1)
        context_emb = self.query_encoder.encode(context).reshape(1, -1)
        return cosine_similarity(snippet_emb, context_emb)[0][0]
    
    def forward(self, snippet: str, context: str) -> torch.Tensor:
        """
        计算损失
        
        原理：loss = 1 - similarity，越相似损失越低
        """
        similarity = self.compute_similarity(snippet, context)
        loss = 1 - similarity
        return torch.tensor(loss, dtype=torch.float32, requires_grad=True)


class SoftLabelBuilder:
    """
    Soft Labels 构建器 - 核心创新点
    
    原理：
    1. 对每个 QA 对，用不同粒度检索
    2. 计算每个粒度的检索结果与答案的相似度
    3. 相似度作为该粒度的 soft label
    
    例如：
    - QA: "什么是Transformer?" → 答案: "Transformer是一种..."
    - Sentence 检索结果: "Transformer是一种..." → 相似度 0.9 → label = 0.9
    - Section 检索结果: "3. Model Architecture..." → 相似度 0.3 → label = 0.3
    """
    
    def __init__(self):
        self.query_encoder = SentenceTransformer(_resolve_router_embedding_model(), cache_folder='./models')
    
    def build_soft_labels(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        为检索结果构建 soft labels（批量编码加速版）
        
        Args:
            retrieval_results: 包含检索结果的列表
            
        Returns:
            List[RetrievalResult]: 添加了 soft_labels 的结果
        """
        granularities = ['sentence', 'paragraph', 'section', 'document']
        total = len(retrieval_results)

        # === 批量预编码：收集所有需要编码的文本，一次性编码 ===
        print(f"  Pre-encoding all texts with RoBERTa (batch mode)...")
        all_texts = set()
        for result in retrieval_results:
            all_texts.add(result.ground_truth)
            for gran in granularities:
                snippets = result.retrieved_snippets.get(gran, [])
                if snippets and snippets[0] and snippets[0] != "NO_TEXT_RETRIEVED":
                    all_texts.add(snippets[0])

        all_texts_list = list(all_texts)
        print(f"  Total unique texts to encode: {len(all_texts_list)}")
        # 批量编码，比逐条快 10-50 倍
        all_embeddings = self.query_encoder.encode(
            all_texts_list, batch_size=64, show_progress_bar=True
        )
        # 建立文本到向量的映射
        self._embedding_cache = {
            text: emb.reshape(1, -1) for text, emb in zip(all_texts_list, all_embeddings)
        }
        print(f"  Pre-encoding done!")

        for i, result in enumerate(retrieval_results):
            if (i + 1) % 200 == 0 or i == 0:
                print(f"  Building soft labels: {i+1}/{total}")

            sim_scores = []

            for gran in granularities:
                snippets = result.retrieved_snippets.get(gran, [])

                if not snippets:
                    sim_scores.append(0.0)
                    continue

                top_snippet = snippets[0] if snippets else ""

                if not top_snippet or top_snippet == "NO_TEXT_RETRIEVED":
                    sim_scores.append(0.0)
                    continue

                similarity = self._compute_similarity(top_snippet, result.ground_truth)
                sim_scores.append(similarity)

            result.soft_labels = self._transform_to_soft_labels(sim_scores)

        # 清理缓存释放内存
        if hasattr(self, '_embedding_cache'):
            del self._embedding_cache

        return retrieval_results
    
    def _compute_similarity(self, snippet: str, context: str) -> float:
        """计算 RoBERTa 语义相似度（优先从批量预编码缓存中取向量）"""
        try:
            # 优先从批量预编码缓存中取
            if hasattr(self, '_embedding_cache'):
                snippet_emb = self._embedding_cache.get(snippet)
                context_emb = self._embedding_cache.get(context)
                if snippet_emb is not None and context_emb is not None:
                    return float(cosine_similarity(snippet_emb, context_emb)[0][0])
            # 回退到逐条编码
            snippet_emb = self.query_encoder.encode(snippet).reshape(1, -1)
            context_emb = self.query_encoder.encode(context).reshape(1, -1)
            return float(cosine_similarity(snippet_emb, context_emb)[0][0])
        except:
            return 0.0
    
    def _transform_to_soft_labels(self, sim_scores: List[float]) -> List[float]:
        """
        将相似度分数转换为 soft labels
        
        原理：
        - 找出 top-2 相似度
        - 最高的设为 0.8
        - 第二高的设为 0.2
        - 其余为 0
        """
        scores = sim_scores.copy()
        
        # 如果所有分数都是0，随机选两个
        if all(s == 0 for s in scores):
            indices = list(range(len(scores)))
            random_indices = random.sample(indices, min(2, len(indices)))
            max_idx = random_indices[0]
            second_max_idx = random_indices[1] if len(random_indices) > 1 else max_idx
        else:
            # 找最大的
            max_idx = scores.index(max(scores))
            
            # 找第二大的（排除最大）
            scores_copy = scores.copy()
            scores_copy[max_idx] = -1
            second_max_idx = scores_copy.index(max(scores_copy))
        
        # 创建 soft labels
        soft_labels = [0.0] * len(scores)
        soft_labels[max_idx] = 0.8
        soft_labels[second_max_idx] = 0.2
        
        return soft_labels


class RouterDataset(Dataset):
    """
    Router 训练数据集
    
    数据格式：
    {
        "question": "What is X?",
        "ground_truth": "X is a...",
        "retrieved_snippets": {
            "sentence": ["snippet1", "snippet2"],
            "paragraph": ["snippet1", "snippet2"],
            ...
        },
        "soft_labels": [0.8, 0.2, 0.0, 0.0]
    }
    """
    
    def __init__(self, data_path: str, embedding_model: str = DEFAULT_ROUTER_MODEL):
        self.data = self._load_data(data_path)
        self.embedding_model = embedding_model
        
        if self.data:
            self.encoder = SentenceTransformer(_resolve_router_embedding_model(embedding_model), cache_folder='./models')
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            Tuple: (query_embedding, soft_labels)
        """
        item = self.data[idx]
        
        # 编码 query
        query_embedding = self.encoder.encode(item['question'])
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        
        # 获取 soft labels
        soft_labels = torch.tensor(item['soft_labels'], dtype=torch.float32)
        
        return query_tensor, soft_labels


class MoGRouterTrainer:
    """
    MoG Router 训练器 (升级版：加入学习率调度器)
    """
    
    def __init__(
        self,
        router: MoGRouter,
        lr: float = 2e-4, # 初始学习率稍微调大一点点
        weight_decay: float = 1e-4 # 增加正则化惩罚，防止死记硬背
    ):
        self.router = router
        self.optimizer = torch.optim.AdamW( # 改用 AdamW，对正则化支持更好
            router.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        # 添加学习率衰减器 (每过 10 轮，学习率减半)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        show_progress: bool = True
    ) -> float:
        """训练一个 epoch"""
        self.router.train()
        total_loss = 0.0
        num_batches = 0
        
        iterator = enumerate(dataloader)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(dataloader), desc="Training")
        
        for batch_idx, (query_embeddings, soft_labels) in iterator:
            query_embeddings = query_embeddings.to(self.router.device)
            soft_labels = soft_labels.to(self.router.device)
            
            predictions = self.router.mlp(query_embeddings)
            log_predictions = torch.log(predictions + 1e-8)
            loss = self.criterion(log_predictions, soft_labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        # 每个 epoch 结束后，让学习率衰减
        self.scheduler.step()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            Dict: 评估指标
        """
        self.router.eval()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for query_embeddings, soft_labels in dataloader:
                query_embeddings = query_embeddings.to(self.router.device)
                soft_labels = soft_labels.to(self.router.device)
                
                predictions = self.router.mlp(query_embeddings)
                
                log_predictions = torch.log(predictions + 1e-8)
                loss = self.criterion(log_predictions, soft_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 计算 top-1 准确率
                pred_indices = predictions.argmax(dim=1)
                true_indices = soft_labels.argmax(dim=1)
                correct += (pred_indices == true_indices).sum().item()
                total += predictions.size(0)
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'router_state_dict': self.router.state_dict(),
            'embedding_dim': self.router.embedding_dim,
            'output_dim': self.router.output_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.router.device)
        self.router.load_state_dict(checkpoint['router_state_dict'])
        print(f"Model loaded from {path}")


def main():
    """
    测试代码
    """
    print("=" * 50)
    print("Testing MoG Router")
    print("=" * 50)
    
    # 测试 Router
    print("\n1. 初始化 Router...")
    router = MoGRouter(output_dim=4)
    
    # 测试预测
    print("\n2. 测试预测...")
    test_queries = [
        "What is the accuracy of BERT?",
        "How does the attention mechanism work?",
        "Compare GPT-3 and GPT-4",
    ]
    
    for query in test_queries:
        weights = router.predict(query)
        best_gran, confidence = router.predict_granularity(query)
        print(f"\nQuery: {query}")
        print(f"  Weights: {weights}")
        print(f"  Best: {best_gran} ({confidence:.2f})")
    
    # 测试 Soft Label Builder
    print("\n" + "=" * 50)
    print("Testing Soft Label Builder")
    print("=" * 50)
    
    builder = SoftLabelBuilder()
    
    # 模拟检索结果
    test_result = RetrievalResult(
        query="What is X?",
        question="What is X?",
        ground_truth="X is a method for solving problems efficiently.",
        retrieved_snippets={
            'sentence': ["X is a method.", "It solves problems."],
            'paragraph': ["X is a method for solving problems efficiently with high accuracy."],
            'section': ["1. Introduction\nX is a method...\n2. Related Work\n..."],
            'document': ["Full document content..."]
        },
        soft_labels=[],
        scores={}
    )
    
    results = builder.build_soft_labels([test_result])
    print(f"\nSoft Labels: {results[0].soft_labels}")
    print("([sentence, paragraph, section, document])")


if __name__ == "__main__":
    main()
