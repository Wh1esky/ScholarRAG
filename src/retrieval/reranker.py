import torch
from sentence_transformers import CrossEncoder
from pathlib import Path

from ..utils.hf_utils import configure_hf_environment, resolve_model_source

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device=None):
        """
        初始化重排序模型 (Cross-Encoder)
        """
        print(f"正在加载重排序模型: {model_name} ...")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        configure_hf_environment()
        base_dir = Path(__file__).resolve().parents[2]
        model_source = resolve_model_source(
            repo_id=model_name,
            preferred_local_dir=base_dir / "models" / "reranker" / "bge-reranker-base"
        )
        self.available = True
            
        # 加载 BGE 重排序模型
        # 注意: bge-reranker-base 基于 XLM-RoBERTa，max_position_embeddings=514
        # 设置 max_length=512 避免超出位置编码范围导致 CUDA assert error
        try:
            self.model = CrossEncoder(model_source, max_length=512, device=self.device)
            print(f"重排序模型加载完毕 (设备: {self.device})")
        except Exception as exc:
            self.available = False
            self.model = None
            print(f"[Warning] 重排序模型加载失败，已跳过 rerank: {exc}")

    def rerank(self, query: str, retrieved_results: list, top_k: int = 3):
        """
        对 FAISS 初排的结果进行交叉编码重排序
        """
        if not retrieved_results:
            return []
        if not self.available:
            return retrieved_results[:top_k]

        # 把 query 和每一个检索到的文本拼接成对 (pair)
        # 注意：这里假设 retrieved_results 里面的对象有 text 和 score 属性
        pairs = [[query, res.text] for res in retrieved_results]
        
        # 让大模型给这些对重新打分
        rerank_scores = self.model.predict(pairs)
        
        # 将新的分数赋给结果对象，并排序
        for i, res in enumerate(retrieved_results):
            # 把原来 FAISS 的分数保存起来备份，用 rerank_score 覆盖
            res.faiss_score = res.score 
            res.score = float(rerank_scores[i])
            
        # 根据重排序的新分数从高到低排序
        retrieved_results.sort(key=lambda x: x.score, reverse=True)
        
        
        # 只返回前 top_k 个最精准的结果
        return retrieved_results[:top_k]