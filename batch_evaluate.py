"""
批量评估脚本 (v2)

用 train_set_100papers.json 的问题跑完整 RAG pipeline，
与标准答案对比，计算评估指标。

特性:
  - 增量保存 (每 5 题自动保存, 断点续跑)
  - CUDA 错误自动恢复 (reranker 跑 CPU)
  - 检索准确率 + ROUGE-L + Token F1

使用方法:
    python batch_evaluate.py --dataset train_set_100papers.json
    python batch_evaluate.py --dataset evaluation_set_100papers.json
    python batch_evaluate.py --resume eval_results/train_set_100papers_XXXXXXXX.json

输出:
    eval_results/ 目录下的 JSON 结果文件
"""

import json
import os
import re
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from openai import OpenAI

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import DenseRetriever, BM25Retriever, SparseRetriever, HybridRetriever, ContextExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.rag.prompt_template import PromptTemplate, RetrievalContext
from src.chunking.adaptive_router import AdaptiveRouter, QueryType


# ============================================================
# 评估指标 (纯 Python, 无外部依赖)
# ============================================================

def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-L (基于最长公共子序列)"""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / m if m else 0.0
    recall = lcs_len / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 (类似 SQuAD F1)"""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def check_retrieval_hit(results: list, target_paper_id: str) -> Dict:
    """检查检索结果中是否命中了目标论文"""
    if not target_paper_id:
        return {"hit": False, "best_rank": -1, "total_results": len(results), "unique_papers": []}

    paper_ids_in_results = [r.paper_id for r in results]
    hit = target_paper_id in paper_ids_in_results
    best_rank = -1
    if hit:
        for idx, pid in enumerate(paper_ids_in_results):
            if pid == target_paper_id:
                best_rank = idx + 1
                break

    return {
        "hit": hit,
        "best_rank": best_rank,
        "total_results": len(results),
        "unique_papers": list(set(paper_ids_in_results)),
    }


# ============================================================
# RAG Pipeline (批量模式)
# ============================================================

class RAGEvaluator:
    """批量 RAG 评估器"""

    QUERY_POLICIES = {
        QueryType.EXPERIMENTAL: {
            "granularities": ["sentence", "paragraph", "section"],
            "dense_weight": 0.25,
            "sparse_weight": 0.20,
            "bm25_weight": 0.55,
            "preferred_sections": ["experiment", "evaluation", "result", "dataset", "setup", "implementation", "training", "ablation"],
            "max_evidence": 5,
        },
        QueryType.METHOD: {
            "granularities": ["paragraph", "section"],
            "dense_weight": 0.50,
            "sparse_weight": 0.30,
            "bm25_weight": 0.20,
            "preferred_sections": ["method", "approach", "model", "architecture", "framework"],
            "max_evidence": 6,
        },
        QueryType.REASONING: {
            "granularities": ["paragraph", "section"],
            "dense_weight": 0.45,
            "sparse_weight": 0.30,
            "bm25_weight": 0.25,
            "preferred_sections": ["introduction", "motivation", "discussion", "analysis", "challenge", "limitation", "error", "failure"],
            "max_evidence": 6,
        },
        QueryType.COMPARISON: {
            "granularities": ["paragraph", "section"],
            "dense_weight": 0.35,
            "sparse_weight": 0.20,
            "bm25_weight": 0.45,
            "preferred_sections": ["related work", "comparison", "baseline", "result", "evaluation", "discussion"],
            "max_evidence": 5,
        },
        QueryType.SUMMARY: {
            "granularities": ["section", "paragraph"],
            "dense_weight": 0.50,
            "sparse_weight": 0.30,
            "bm25_weight": 0.20,
            "preferred_sections": ["abstract", "introduction", "conclusion", "summary"],
            "max_evidence": 5,
        },
        QueryType.FACTUAL: {
            "granularities": ["sentence", "paragraph"],
            "dense_weight": 0.25,
            "sparse_weight": 0.15,
            "bm25_weight": 0.60,
            "preferred_sections": ["result", "experiment", "dataset", "implementation", "method", "setup", "appendix"],
            "max_evidence": 4,
        },
        QueryType.DEFINITION: {
            "granularities": ["paragraph", "sentence"],
            "dense_weight": 0.35,
            "sparse_weight": 0.30,
            "bm25_weight": 0.35,
            "preferred_sections": ["abstract", "introduction", "background", "preliminaries", "method"],
            "max_evidence": 4,
        },
        QueryType.LIST: {
            "granularities": ["paragraph", "section"],
            "dense_weight": 0.40,
            "sparse_weight": 0.25,
            "bm25_weight": 0.35,
            "preferred_sections": ["method", "pipeline", "component", "framework", "algorithm"],
            "max_evidence": 6,
        },
    }

    def __init__(self, use_reranker: bool = True, use_llm: bool = True,
                 use_sparse: bool = True):
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        self.use_llm = use_llm

        embed_dir = str(PROJECT_ROOT / "src" / "embedding" / "output")
        metadata_path = str(PROJECT_ROOT / "src" / "embedding" / "output" / "unified_metadata.json")

        print("=" * 60)
        print("加载 RAG Pipeline 组件...")
        if not use_llm:
            print("[轻量模式] 仅评估检索指标, 跳过 LLM / ContextExpander / ROUGE / F1")
        print("=" * 60)

        # Dense Retriever
        self.dense_retriever = DenseRetriever.from_output_dir(embed_dir)

        # BM25 Retriever
        self.bm25_retriever = BM25Retriever(metadata_path)

        # Sparse Retriever (BGE-M3 学习型稀疏) —— 轻量模式或 --no-sparse 时跳过
        sparse_path = str(PROJECT_ROOT / "src" / "embedding" / "output" / "unified_sparse.pkl")
        self.sparse_retriever = None
        if use_sparse and os.path.exists(sparse_path):
            self.sparse_retriever = SparseRetriever(
                sparse_path=sparse_path,
                metadata_path=metadata_path,
                embedder=self.dense_retriever.embedder,
            )
        elif not use_sparse:
            print("已跳过 Sparse Retriever (--no-sparse)")

        # Hybrid Retriever
        if self.sparse_retriever:
            # 三路: Dense + Sparse + BM25
            self.hybrid_retriever = HybridRetriever(
                self.dense_retriever, self.bm25_retriever,
                sparse_retriever=self.sparse_retriever,
                dense_weight=0.4, sparse_weight=0.3, bm25_weight=0.3,
            )
        else:
            # 二路: Dense + BM25 (更快)
            self.hybrid_retriever = HybridRetriever(
                self.dense_retriever, self.bm25_retriever,
                sparse_retriever=None,
                dense_weight=0.5, sparse_weight=0.0, bm25_weight=0.5,
            )

        # Reranker - max_length已修为512，可安全使用GPU
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = CrossEncoderReranker()
            except Exception as e:
                print(f"Reranker 加载失败: {e}")
                self.reranker = None

        # LLM Client
        self.client = None
        if use_llm:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                print("DeepSeek API 已连接")
            else:
                print("未设置 DEEPSEEK_API_KEY，跳过答案生成")

        # Context Expander —— 仅在需要生成答案时才启用
        self.context_expander = None
        if use_llm:
            self.context_expander = ContextExpander(
                metadata=self.dense_retriever.metadata,
                window_size=2,
            )
        else:
            print("已跳过 ContextExpander (--no-llm 轻量模式)")

        self.query_router = AdaptiveRouter()

        print("\nPipeline 加载完毕！\n")

    def extract_paper_id(self, url: str) -> str:
        """从 arxiv URL 提取论文 ID"""
        match = re.search(r"(\d{4}\.\d{4,5})", url)
        return match.group(1) if match else ""

    def _expand_query(self, query: str) -> List[str]:
        """多角度查询扩展，提高 Context Recall"""
        queries = [query]
        q_lower = query.lower().strip()

        # 1) 去掉疑问词，提取核心短语
        core = re.sub(
            r'^(what|how|why|which|where|when|who|did|does|do|is|are|was|were)\s+',
            '', q_lower, flags=re.IGNORECASE
        ).strip()
        if core and core != q_lower:
            queries.append(core)

        # 2) 提取关键实体短语 (连续大写开头的词 / 引号内容)
        # 例: "What dataset does BERT use?" → "BERT dataset"
        _stop_words = {
            'What', 'How', 'Why', 'Which', 'Where', 'When', 'Who', 'Did', 'Does',
            'Do', 'Is', 'Are', 'Was', 'Were', 'The', 'This', 'That', 'These',
            'Those', 'Can', 'Could', 'Would', 'Should', 'May', 'Might', 'Will',
            'Has', 'Have', 'Had', 'For', 'And', 'But', 'Not', 'All', 'Any',
            'Each', 'Every', 'Some', 'No', 'Yes', 'Its', 'Our', 'Their',
        }
        entities = [e for e in re.findall(r'[A-Z][A-Za-z0-9_-]{2,}', query) if e not in _stop_words]
        if entities:
            entity_query = ' '.join(entities[:4])
            if entity_query.lower() != q_lower:
                queries.append(entity_query)

        # 3) 去掉 "in this paper / in the paper / proposed" 等冗余
        cleaned = re.sub(
            r'\b(in this paper|in the paper|proposed in|the authors|this study|this work|the paper)\b',
            '', q_lower, flags=re.IGNORECASE
        ).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if cleaned and cleaned != q_lower and cleaned not in [q.lower() for q in queries]:
            queries.append(cleaned)

        return queries[:4]  # 最多 4 个查询角度

    def _get_query_policy(self, query: str) -> Dict:
        """根据 query 类型选择检索/证据策略"""
        classification = self.query_router.classify_query(query)
        # 硬覆盖逻辑已在 AdaptiveRouter.classify_query() 中统一处理，此处不再重复

        policy = {
            "query_type": classification.query_type.value,
            "confidence": classification.confidence,
            "granularities": [g.strip() for g in classification.suggested_granularity.split(",") if g.strip()],
            "dense_weight": self.hybrid_retriever.dense_weight,
            "sparse_weight": self.hybrid_retriever.sparse_weight,
            "bm25_weight": self.hybrid_retriever.bm25_weight,
            "preferred_sections": [],
            "max_evidence": 6,
        }
        policy.update(self.QUERY_POLICIES.get(classification.query_type, {}))
        return policy

    def _temporary_set_hybrid_weights(self, dense_weight: float, sparse_weight: float, bm25_weight: float):
        """临时调整 HybridRetriever 权重"""
        old = (
            self.hybrid_retriever.dense_weight,
            self.hybrid_retriever.sparse_weight,
            self.hybrid_retriever.bm25_weight,
        )
        self.hybrid_retriever.dense_weight = dense_weight
        self.hybrid_retriever.sparse_weight = sparse_weight if self.sparse_retriever else 0.0
        self.hybrid_retriever.bm25_weight = bm25_weight
        return old

    @staticmethod
    def _query_keywords(query: str) -> List[str]:
        stopwords = {
            "what", "which", "where", "when", "who", "why", "how", "does", "did", "do",
            "is", "are", "was", "were", "the", "a", "an", "of", "for", "to", "in", "on",
            "and", "or", "with", "from", "by", "this", "that", "these", "those", "their",
            "authors", "paper", "study", "research", "proposed",
        }
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", query.lower())
        return [token for token in tokens if len(token) > 2 and token not in stopwords]

    @staticmethod
    def _query_phrases(query: str) -> List[str]:
        """提取 2~3 gram 查询短语，用于提升 Top1 排序的精确匹配能力"""
        stopwords = {
            "what", "which", "where", "when", "who", "why", "how", "does", "did", "do",
            "is", "are", "was", "were", "the", "a", "an", "of", "for", "to", "in", "on",
            "and", "or", "with", "from", "by", "this", "that", "these", "those", "their",
            "authors", "paper", "study", "research", "proposed",
        }
        tokens = [
            token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", query.lower())
            if token not in stopwords
        ]

        phrases = []
        seen = set()
        for n in (3, 2):
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i + n]
                if any(len(token) <= 2 for token in phrase_tokens):
                    continue
                phrase = " ".join(phrase_tokens).strip()
                if len(phrase) < 10 or phrase in seen:
                    continue
                seen.add(phrase)
                phrases.append(phrase)
        return phrases[:6]

    @staticmethod
    def _section_matches(section_type: str, preferred_sections: List[str]) -> bool:
        section = (section_type or "").lower()
        return any(keyword in section for keyword in preferred_sections)

    def _retrieve_candidates(self, query: str, top_k_retrieve: int, policy: Dict) -> list:
        """按 query policy 做多粒度召回"""
        all_queries = self._expand_query(query)
        all_results = []
        seen_ids = set()

        old_weights = self._temporary_set_hybrid_weights(
            policy["dense_weight"],
            policy["sparse_weight"],
            policy["bm25_weight"],
        )

        try:
            for q in all_queries:
                for granularity in policy["granularities"]:
                    results = self.hybrid_retriever.retrieve(
                        q,
                        top_k=top_k_retrieve,
                        granularity_filter=granularity,
                    )
                    for result in results:
                        rid = (result.paper_id, result.chunk_id)
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            all_results.append(result)

                if len(all_results) < top_k_retrieve:
                    fallback = self.hybrid_retriever.retrieve(q, top_k=top_k_retrieve)
                    for result in fallback:
                        rid = (result.paper_id, result.chunk_id)
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            all_results.append(result)
        finally:
            (
                self.hybrid_retriever.dense_weight,
                self.hybrid_retriever.sparse_weight,
                self.hybrid_retriever.bm25_weight,
            ) = old_weights

        return all_results

    def _score_evidence(self, query: str, result, policy: Dict) -> float:
        """轻量级证据重排：相关性 + section 命中 + 关键信号"""
        text = (result.text or "").lower()
        score = float(result.score)
        keywords = self._query_keywords(query)
        phrases = self._query_phrases(query)
        setup_like_query = bool(re.search(
            r"\b(set(?:\s|-)?up|config(?:uration)?|hyper-?parameter|parameter|training|implementation|"
            r"dataset\s+split|split|sample|sampling|augment(?:ation)?|epoch|batch\s+size|learning\s+rate|"
            r"optimizer|input\s+sparsity|label\s+determination|prun(?:e|ed|ing))\b",
            query.lower(),
        ))

        overlap = sum(1 for token in keywords if token in text)
        score += min(overlap, 4) * 0.08
        exact_phrase_hits = sum(1 for token in keywords if f" {token} " in f" {text} ")
        score += min(exact_phrase_hits, 3) * 0.05
        phrase_hits = sum(1 for phrase in phrases if phrase in text)
        score += min(phrase_hits, 3) * 0.12

        if self._section_matches(result.section_type, policy["preferred_sections"]):
            score += 0.22

        if setup_like_query and self._section_matches(
            result.section_type,
            ["experiment", "evaluation", "result", "dataset", "setup", "implementation", "training", "appendix"],
        ):
            score += 0.12

        query_type = policy["query_type"]
        query_lower = query.lower()
        policy_terms = {
            QueryType.EXPERIMENTAL.value: [
                "table", "optimizer", "learning rate", "batch size", "epoch", "configuration", "setting",
                "hyperparameter", "experiment", "training", "implementation",
            ],
            QueryType.METHOD.value: [
                "method", "approach", "architecture", "module", "framework", "algorithm", "pipeline",
            ],
            QueryType.REASONING.value: [
                "because", "motivation", "challenge", "limitation", "reason", "issue", "difficulty",
            ],
            QueryType.COMPARISON.value: [
                "baseline", "compared", "comparison", "outperform", "better", "worse", "versus",
            ],
            QueryType.FACTUAL.value: [
                "accuracy", "f1", "precision", "recall", "result", "score", "dataset", "table",
            ],
            QueryType.DEFINITION.value: [
                "defined as", "refers to", "means", "concept", "principle",
            ],
            QueryType.LIST.value: [
                "first", "second", "third", "steps", "components", "pipeline", "consists of",
            ],
            QueryType.SUMMARY.value: [
                "we propose", "in summary", "overall", "conclusion", "main contribution",
            ],
        }
        matched_terms = sum(1 for term in policy_terms.get(query_type, []) if term in text)
        score += min(matched_terms, 4) * 0.07

        if setup_like_query:
            setup_terms = [
                "setup", "configuration", "setting", "implementation", "training", "optimizer",
                "learning rate", "batch size", "epoch", "hyperparameter", "dataset split",
                "split", "augmentation", "sampling", "uniform distribution", "sparsity",
            ]
            setup_hits = sum(1 for term in setup_terms if term in text)
            score += min(setup_hits, 4) * 0.08

        if re.search(r"\btable\s*\d+\b", query_lower) and re.search(r"\btable\s*\d+\b", text):
            score += 0.25

        if query_type == QueryType.EXPERIMENTAL.value:
            if re.search(r"\b\d+(?:\.\d+)?%?\b", text):
                score += 0.12
            if any(token in text for token in ["dataset", "benchmark", "accuracy", "f1", "precision", "recall", "epoch", "batch", "learning rate"]):
                score += 0.10
        elif query_type == QueryType.METHOD.value:
            if any(token in text for token in ["method", "approach", "architecture", "module", "framework", "algorithm"]):
                score += 0.10
        elif query_type == QueryType.REASONING.value:
            if any(token in text for token in ["because", "motivation", "challenge", "limitation", "reason"]):
                score += 0.10
        elif query_type == QueryType.COMPARISON.value:
            if any(token in text for token in ["compared", "baseline", "outperform", "better", "worse", "versus"]):
                score += 0.12
        elif query_type == QueryType.DEFINITION.value:
            if any(token in text for token in ["defined as", "refers to", "means", "concept"]):
                score += 0.10

        if len(text) < 120:
            score -= 0.05

        citation_count = len(re.findall(r"\[[0-9,\-\s]+\]|\([A-Z][A-Za-z\-]+(?:\s+et al\.)?,?\s*\d{4}\)", result.text or ""))
        if citation_count >= 6:
            score -= 0.25

        # 参考文献强惩罚（即使绕过硬过滤也要压低）
        if (result.section_type or "").lower() in self._NOISE_SECTIONS:
            score -= 1.0

        if re.search(r"\b(arxiv|proceedings|journal|conference|vol\.|pp\.|doi)\b", text) and overlap <= 1:
            score -= 0.30

        if sum(1 for ch in text[:400] if ch in "=\\{}[]_^|") > 18 and overlap <= 1:
            score -= 0.18

        if query_type in {QueryType.FACTUAL.value, QueryType.DEFINITION.value, QueryType.EXPERIMENTAL.value} and len(text) > 1600:
            score -= 0.08

        return score

    @staticmethod
    def _text_signature(text: str, max_len: int = 220) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        return normalized[:max_len]

    def _should_expand_context(self, query: str, policy: Dict, result) -> bool:
        query_lower = query.lower()
        if policy["query_type"] in {QueryType.FACTUAL.value, QueryType.DEFINITION.value}:
            return False
        if re.search(r"\b(table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration|specific\s+conditions?)\b", query_lower):
            return False
        if len(result.text or "") < 140:
            return False
        # 直接使用已经 rescore 过的 result.score，避免双重计分
        if result.score < 0.15:
            return False
        return True

    _NOISE_SECTIONS = {"reference", "references", "bibliography", "acknowledgment", "acknowledgement", "acknowledgments", "acknowledgements"}

    def retrieve(self, query: str, top_k_retrieve: int = 50, top_k_rerank: int = 10,
                 policy: Optional[Dict] = None) -> list:
        """检索 + 路由召回 + 重排序 + 证据重排 + 局部扩展"""
        policy = policy or self._get_query_policy(query)
        results = self._retrieve_candidates(query, top_k_retrieve, policy)

        # ===== 硬过滤参考文献/致谢 chunks（占总索引 15%，是噪声主源）=====
        results = [r for r in results if (r.section_type or "").lower() not in self._NOISE_SECTIONS]

        if self.reranker and results:
            try:
                results = self.reranker.rerank(query, results, top_k=top_k_rerank)
            except Exception as e:
                print(f"    Rerank 失败, 用原始排序: {e}")
                results = results[:top_k_rerank]
        else:
            results = results[:top_k_rerank]

        rescored_results = []
        seen_signatures = set()
        # 归一化 reranker/RRF 分数到 [0, 1]，防止不同量纲的加分失效
        if results:
            raw_scores = [float(r.score) for r in results]
            min_s, max_s = min(raw_scores), max(raw_scores)
            score_range = max_s - min_s if max_s > min_s else 1.0
            for result in results:
                result.score = (float(result.score) - min_s) / score_range
        for result in results:
            result.score = self._score_evidence(query, result, policy)
            signature = self._text_signature(result.text)
            if signature and signature in seen_signatures:
                continue
            if signature:
                seen_signatures.add(signature)
            rescored_results.append(result)
        results = sorted(rescored_results, key=lambda item: item.score, reverse=True)

        # 证据筛选：最大化多样性，强制覆盖不同 section
        if self.use_llm:
            results = self._select_evidence(results, max_evidence=policy.get("max_evidence", 6), policy=policy)

        # 上下文窗口扩展：仅对 top anchor 做局部扩展
        if self.context_expander:
            anchor_limit = 2 if policy["query_type"] in {QueryType.METHOD.value, QueryType.REASONING.value, QueryType.SUMMARY.value} else 1
            anchor_results = [r for r in results if self._should_expand_context(query, policy, r)][:anchor_limit]
            if anchor_results:
                expanded = self.context_expander.expand_results(anchor_results, top_k=len(anchor_results))
                expanded_ids = set((r.paper_id, r.chunk_id) for r in expanded)
                remaining = [r for r in results if (r.paper_id, r.chunk_id) not in expanded_ids]
                results = expanded + remaining

        return results

    def _select_evidence(self, results: list, max_evidence: int = 10, policy: Optional[Dict] = None) -> list:
        """最大化多样性的证据选择：强制覆盖不同 section"""
        if len(results) <= max_evidence:
            return results

        policy = policy or {}
        query_type = policy.get("query_type", "")
        per_section_limit = 2 if query_type in {
            QueryType.FACTUAL.value,
            QueryType.DEFINITION.value,
            QueryType.COMPARISON.value,
        } else 3
        per_paper_limit = 3 if query_type in {
            QueryType.FACTUAL.value,
            QueryType.DEFINITION.value,
        } else 4

        selected = []
        seen_sections = {}  # (paper_id, section_type) -> count
        seen_papers = {}
        seen_signatures = set()

        # 第一轮：按重排后的结果优先选择，同时限制重复 section / paper
        for r in results:
            signature = self._text_signature(r.text)
            if signature and signature in seen_signatures:
                continue
            key = (r.paper_id, r.section_type)
            if seen_sections.get(key, 0) >= per_section_limit:
                continue
            if seen_papers.get(r.paper_id, 0) >= per_paper_limit:
                continue
            selected.append(r)
            seen_sections[key] = seen_sections.get(key, 0) + 1
            seen_papers[r.paper_id] = seen_papers.get(r.paper_id, 0) + 1
            if signature:
                seen_signatures.add(signature)
            if len(selected) >= max_evidence:
                return selected

        # 第二轮：还没满，放宽 section 限制但继续去重
        for r in results:
            if r in selected:
                continue
            signature = self._text_signature(r.text)
            if signature and signature in seen_signatures:
                continue
            key = (r.paper_id, r.section_type)
            if seen_sections.get(key, 0) < max(per_section_limit, 2):
                selected.append(r)
                seen_sections[key] = seen_sections.get(key, 0) + 1
                seen_papers[r.paper_id] = seen_papers.get(r.paper_id, 0) + 1
                if signature:
                    seen_signatures.add(signature)
                if len(selected) >= max_evidence:
                    return selected

        return selected

    def _call_llm(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """统一封装 DeepSeek 调用（带重试）"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    stream=False,
                    timeout=180,
                    max_tokens=2048,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"    LLM 调用失败 (attempt {attempt+1}/{max_retries}): {e}, {wait}s 后重试...")
                    import time as _t; _t.sleep(wait)
                else:
                    raise

    def _extract_evidence_local(self, query: str, contexts: List[RetrievalContext]) -> List[Dict]:
        """本地规则抽取关键证据句（不调 LLM，瞄间完成）"""
        keywords = self._query_keywords(query)
        q_lower = query.lower()
        evidence_items = []

        for idx, ctx in enumerate(contexts):
            text = ctx.text or ""
            # 按句号分句，同时支持分号分割的列表型句子
            sentences = re.split(r'(?<=[.!?;])\s+', text)
            # 合并过短的句子与前一句（避免断句导致信息丢失）
            merged = []
            for sent in sentences:
                if merged and len(merged[-1]) < 60:
                    merged[-1] = merged[-1] + " " + sent
                else:
                    merged.append(sent)
            scored_sents = []

            for sent in merged:
                if len(sent.strip()) < 15:
                    continue
                s_lower = sent.lower()
                score = 0.0
                # 关键词命中
                hits = sum(1 for kw in keywords if kw in s_lower)
                score += min(hits, 5) * 0.25
                # 精确短语匹配 (多词关键词)
                multi_word_kws = [kw for kw in keywords if ' ' in kw or len(kw) > 6]
                exact_hits = sum(1 for kw in multi_word_kws if kw in s_lower)
                score += exact_hits * 0.3
                # 含数字 / 百分比 / 表格引用
                if re.search(r'\b\d+(?:\.\d+)?%?\b', sent):
                    score += 0.15
                if re.search(r'\btable\s*\d+\b', s_lower):
                    score += 0.2
                # 含方法/数据集信号词
                if any(w in s_lower for w in ['method', 'approach', 'dataset', 'benchmark',
                                               'accuracy', 'f1', 'precision', 'recall',
                                               'propose', 'introduce', 'achieve', 'outperform',
                                               'experiment', 'result', 'performance', 'training',
                                               'evaluation', 'loss', 'objective', 'architecture']):
                    score += 0.15
                # 含定义/解释信号
                if any(w in s_lower for w in ['defined as', 'refers to', 'we define', 'is called',
                                               'formulated as', 'denoted as', 'represents']):
                    score += 0.25
                # 含因果/推理信号
                if any(w in s_lower for w in ['because', 'therefore', 'thus', 'hence',
                                               'due to', 'as a result', 'leads to', 'causes']):
                    score += 0.2
                # 句子足够长（更可能含完整信息）
                if len(sent) > 100:
                    score += 0.05
                if score > 0:
                    scored_sents.append((score, sent.strip()))

            # 每个 source 最多取 3 句（原来 2 句太少）
            scored_sents.sort(key=lambda x: x[0], reverse=True)
            for score, sent in scored_sents[:3]:
                evidence_items.append({
                    "source": idx + 1,
                    "quote": sent[:400],
                    "why": "keyword/signal match",
                })

        # 按分数全局排序，最多 8 条证据（原来 6 条太少）
        evidence_items_with_score = []
        for item in evidence_items:
            # 重新计算一下分数用于全局排序
            s_lower = item["quote"].lower()
            sc = sum(1 for kw in keywords if kw in s_lower) * 0.25
            evidence_items_with_score.append((sc, item))
        evidence_items_with_score.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in evidence_items_with_score[:8]]

    def generate_answer(self, query: str, results: list) -> str:
        """本地抽取证据 + 严格 prompt 生成答案（单次 API 调用）"""
        if not self.client:
            return "[LLM disabled]"

        prompt_template = PromptTemplate()
        contexts = [
            RetrievalContext(
                text=res.text,
                paper_id=res.paper_id,
                chunk_id=res.chunk_id,
                section_type=res.section_type,
                score=res.score,
                granularity=res.granularity,
            )
            for res in results
        ]

        try:
            # 本地规则抽取证据句（瞄间完成，不调 API）
            evidence_items = self._extract_evidence_local(query, contexts)

            if evidence_items:
                # 基于证据生成 grounded answer
                messages = prompt_template.build_grounded_answer_messages(
                    query, contexts, evidence_items
                )
            else:
                # fallback 到直接生成
                messages = prompt_template.build_qa_messages(query, contexts)

            return self._call_llm(messages)
        except Exception as e:
            try:
                messages = prompt_template.build_qa_messages(query, contexts)
                return self._call_llm(messages)
            except Exception as e2:
                return f"[LLM Error: {e2}]"

    def evaluate_single(self, item: dict, idx: int, total: int) -> dict:
        """评估单个问题"""
        question = item["question"]
        reference_answer = item["answer"]
        paper_url = item.get("url", "")
        label = item.get("label", "")
        difficulty = item.get("difficulty", {}).get("score", 0)
        target_paper_id = self.extract_paper_id(paper_url)
        query_policy = self._get_query_policy(question)

        # 1. 检索
        results = self.retrieve(question, policy=query_policy)

        # 2. 检索命中检查
        retrieval_info = check_retrieval_hit(results, target_paper_id)

        # MRR: reciprocal rank
        best_rank = retrieval_info.get("best_rank", -1)
        reciprocal_rank = 1.0 / best_rank if best_rank > 0 else 0.0

        # 3. 生成答案 & 答案质量指标 (仅 LLM 模式)
        if self.use_llm:
            generated_answer = self.generate_answer(question, results)
            rouge_l = compute_rouge_l(generated_answer, reference_answer)
            token_f1 = compute_token_f1(generated_answer, reference_answer)
        else:
            generated_answer = "[LLM disabled]"
            rouge_l = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            token_f1 = 0.0

        result = {
            "index": idx,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "target_paper_id": target_paper_id,
            "label": label,
            "difficulty": difficulty,
            "query_policy": query_policy,
            "retrieval": retrieval_info,
            "retrieval_texts": [r.text[:2000] for r in results],  # 保存更多检索文本供 LLM 评估
            "metrics": {
                "rouge_l_f1": rouge_l["f1"],
                "rouge_l_precision": rouge_l["precision"],
                "rouge_l_recall": rouge_l["recall"],
                "token_f1": token_f1,
                "reciprocal_rank": reciprocal_rank,
            },
        }

        # 简要输出
        hit_str = "HIT " if retrieval_info["hit"] else "MISS"
        rank_str = f"@{best_rank}" if best_rank > 0 else ""
        if self.use_llm:
            print(
                f"  [{idx + 1}/{total}] {hit_str}{rank_str} | "
                f"RR={reciprocal_rank:.3f} | ROUGE-L={rouge_l['f1']:.3f} | F1={token_f1:.3f} | "
                f"paper={target_paper_id} | label={label}"
            )
        else:
            print(
                f"  [{idx + 1}/{total}] {hit_str}{rank_str} | "
                f"RR={reciprocal_rank:.3f} | paper={target_paper_id} | label={label}"
            )

        return result

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_dir: str = "eval_results",
        max_questions: Optional[int] = None,
        resume_from: Optional[str] = None,
        save_every: int = 20,
    ) -> dict:
        """
        批量评估整个数据集 (支持断点续跑)

        Args:
            dataset_path: 数据集 JSON 路径
            output_dir: 结果输出目录
            max_questions: 最多评估几题 (None=全部)
            resume_from: 从之前的结果文件恢复
            save_every: 每 N 题保存一次中间结果
        """
        # 加载数据
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        # 自动检测嵌套格式 (evaluation_set) 并展平
        if dataset and isinstance(dataset[0], dict) and "qa_pairs" in dataset[0]:
            flat = []
            for paper in dataset:
                for qa in paper.get("qa_pairs", []):
                    flat.append(qa)
            print(f"检测到嵌套格式, 已展平: {len(dataset)} 篇论文 → {len(flat)} 个问题")
            dataset = flat

        if max_questions:
            dataset = dataset[:max_questions]

        total = len(dataset)
        dataset_name = Path(dataset_path).stem

        # 断点续跑
        all_results = []
        start_idx = 0
        if resume_from and os.path.exists(resume_from):
            with open(resume_from, "r", encoding="utf-8") as f:
                prev = json.load(f)
            all_results = prev.get("results", [])
            start_idx = len([r for r in all_results if "metrics" in r or "error" in r])
            print(f"从 {resume_from} 恢复, 跳过前 {start_idx} 题")

        # 准备输出文件
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"{dataset_name}_{timestamp}.json")

        print(f"\n{'=' * 60}")
        print(f"开始评估: {dataset_name}")
        print(f"总题数: {total} | 从第 {start_idx + 1} 题开始")
        print(f"结果文件: {result_file}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for idx in range(start_idx, total):
            item = dataset[idx]
            try:
                result = self.evaluate_single(item, idx, total)
                all_results.append(result)
            except Exception as e:
                print(f"  [{idx + 1}/{total}] ERROR: {e}")
                all_results.append({
                    "index": idx,
                    "question": item.get("question", ""),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

            # 增量保存
            if (idx + 1) % save_every == 0 or (idx + 1) == total:
                self._save_checkpoint(all_results, result_file, dataset_name, start_time)

            # 每 50 题打印汇总
            if (idx + 1) % 50 == 0:
                self._print_progress(all_results, idx + 1, total)

        elapsed = time.time() - start_time

        # 最终汇总
        completed = [r for r in all_results if "metrics" in r]
        summary = self._compute_summary(completed, elapsed, dataset_name, total)
        output = {"summary": summary, "results": all_results}

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self._print_summary(summary, result_file)
        return output

    def _save_checkpoint(self, results: list, filepath: str, dataset_name: str, start_time: float):
        """增量保存中间结果"""
        elapsed = time.time() - start_time
        completed = [r for r in results if "metrics" in r]
        summary = self._compute_summary(completed, elapsed, dataset_name, len(results))
        output = {"summary": summary, "results": results}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _print_progress(self, results: list, done: int, total: int):
        """定期打印进度"""
        completed = [r for r in results if "metrics" in r]
        errors = len(results) - len(completed)
        if completed:
            hit_rate = sum(1 for r in completed if r["retrieval"]["hit"]) / len(completed)
            mrr = sum(r["metrics"].get("reciprocal_rank", 0) for r in completed) / len(completed)
            if self.use_llm:
                avg_rouge = sum(r["metrics"]["rouge_l_f1"] for r in completed) / len(completed)
                avg_f1 = sum(r["metrics"]["token_f1"] for r in completed) / len(completed)
                print(
                    f"\n  === 进度 {done}/{total} | 成功={len(completed)} 出错={errors} | "
                    f"命中率={hit_rate:.1%} | MRR={mrr:.4f} | ROUGE-L={avg_rouge:.3f} | F1={avg_f1:.3f} ===\n"
                )
            else:
                print(
                    f"\n  === 进度 {done}/{total} | 成功={len(completed)} 出错={errors} | "
                    f"命中率={hit_rate:.1%} | MRR={mrr:.4f} ===\n"
                )

    def _compute_summary(self, results: list, elapsed: float, dataset_name: str, total_attempted: int) -> dict:
        """计算汇总统计"""
        n = len(results)
        if n == 0:
            return {
                "dataset": dataset_name,
                "total_questions": total_attempted,
                "completed": 0,
                "elapsed_seconds": round(elapsed, 1),
                "error": "No valid results",
            }

        hit_count = sum(1 for r in results if r["retrieval"]["hit"])
        rouge_scores = [r["metrics"]["rouge_l_f1"] for r in results]
        f1_scores = [r["metrics"]["token_f1"] for r in results]
        rr_scores = [r["metrics"].get("reciprocal_rank", 0) for r in results]

        # Recall@K
        recall_at = {}
        for k in [1, 3, 5, 10]:
            hits_at_k = sum(1 for r in results
                           if r["retrieval"]["hit"] and r["retrieval"].get("best_rank", 999) <= k)
            recall_at[f"recall@{k}"] = hits_at_k / n

        # Precision@1, F1@1 (仅保留 @1，高 K 下单相关文档场景无意义)
        hits_at_1 = sum(1 for r in results
                        if r["retrieval"]["hit"] and r["retrieval"].get("best_rank", 999) <= 1)
        prec1 = hits_at_1 / n
        rec1 = recall_at["recall@1"]
        f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) else 0.0

        # 按 label 分组
        label_stats = {}
        for r in results:
            label = r.get("label", "unknown")
            if label not in label_stats:
                label_stats[label] = {"count": 0, "hit": 0, "rr_sum": 0.0, "rouge_l_sum": 0.0, "f1_sum": 0.0}
            label_stats[label]["count"] += 1
            if r["retrieval"]["hit"]:
                label_stats[label]["hit"] += 1
            label_stats[label]["rr_sum"] += r["metrics"].get("reciprocal_rank", 0)
            label_stats[label]["rouge_l_sum"] += r["metrics"]["rouge_l_f1"]
            label_stats[label]["f1_sum"] += r["metrics"]["token_f1"]

        for label, stats in label_stats.items():
            c = stats["count"]
            stats["hit_rate"] = stats["hit"] / c if c else 0
            stats["mrr"] = stats["rr_sum"] / c if c else 0
            stats["avg_rouge_l"] = stats["rouge_l_sum"] / c if c else 0
            stats["avg_f1"] = stats["f1_sum"] / c if c else 0
            del stats["rouge_l_sum"]
            del stats["f1_sum"]
            del stats["rr_sum"]

        # 按难度分组
        difficulty_stats = {}
        for r in results:
            d = r.get("difficulty", 0)
            if d <= 0.5:
                level = "easy"
            elif d <= 0.7:
                level = "medium"
            else:
                level = "hard"
            if level not in difficulty_stats:
                difficulty_stats[level] = {"count": 0, "hit": 0, "rr_sum": 0.0, "rouge_l_sum": 0.0, "f1_sum": 0.0}
            difficulty_stats[level]["count"] += 1
            if r["retrieval"]["hit"]:
                difficulty_stats[level]["hit"] += 1
            difficulty_stats[level]["rr_sum"] += r["metrics"].get("reciprocal_rank", 0)
            difficulty_stats[level]["rouge_l_sum"] += r["metrics"]["rouge_l_f1"]
            difficulty_stats[level]["f1_sum"] += r["metrics"]["token_f1"]

        for level, stats in difficulty_stats.items():
            c = stats["count"]
            stats["hit_rate"] = stats["hit"] / c if c else 0
            stats["mrr"] = stats["rr_sum"] / c if c else 0
            stats["avg_rouge_l"] = stats["rouge_l_sum"] / c if c else 0
            stats["avg_f1"] = stats["f1_sum"] / c if c else 0
            del stats["rouge_l_sum"]
            del stats["f1_sum"]
            del stats["rr_sum"]

        return {
            "dataset": dataset_name,
            "total_questions": total_attempted,
            "completed": n,
            "errors": total_attempted - n,
            "elapsed_seconds": round(elapsed, 1),
            "avg_time_per_question": round(elapsed / n, 1) if n else 0,
            "retrieval": {
                "hit_rate": hit_count / n,
                "hit_count": hit_count,
                "miss_count": n - hit_count,
                "mrr": sum(rr_scores) / n,
                **recall_at,
                "precision@1": prec1,
                "f1@1": f1_1,
            },
            "answer_quality": {
                "avg_rouge_l_f1": sum(rouge_scores) / n,
                "avg_token_f1": sum(f1_scores) / n,
                "max_rouge_l_f1": max(rouge_scores),
                "min_rouge_l_f1": min(rouge_scores),
            },
            "by_label": label_stats,
            "by_difficulty": difficulty_stats,
        }

    def _print_summary(self, summary: dict, result_file: str):
        """打印总结"""
        print("\n" + "=" * 60)
        print("评估结果总结")
        print("=" * 60)

        print(f"\n数据集: {summary.get('dataset', 'N/A')}")
        print(f"完成: {summary.get('completed', 0)}/{summary.get('total_questions', 0)}")
        print(f"耗时: {summary.get('elapsed_seconds', 0)}s ({summary.get('avg_time_per_question', 0)}s/题)")

        if "retrieval" in summary:
            ret = summary["retrieval"]
            n = summary.get("completed", 1)
            print(f"\n检索指标:")
            print(f"  命中率 (Hit Rate): {ret['hit_rate']:.1%} ({ret['hit_count']}/{n})")
            print(f"  MRR:               {ret.get('mrr', 0):.4f}")
            for k in [1, 3, 5, 10]:
                r_key = f"recall@{k}"
                if r_key in ret:
                    print(f"  Recall@{k:<2d}:          {ret[r_key]:.1%}")
            if "precision@1" in ret:
                print(f"  Precision@1:        {ret['precision@1']:.4f}")
            if "f1@1" in ret:
                print(f"  F1@1:               {ret['f1@1']:.4f}")

        if "answer_quality" in summary and self.use_llm:
            ans = summary["answer_quality"]
            print(f"\n答案质量:")
            print(f"  ROUGE-L F1: {ans['avg_rouge_l_f1']:.4f} (min={ans['min_rouge_l_f1']:.4f}, max={ans['max_rouge_l_f1']:.4f})")
            print(f"  Token F1:   {ans['avg_token_f1']:.4f}")

        if "by_label" in summary:
            print(f"\n按题型分布:")
            for label, stats in sorted(summary["by_label"].items()):
                mrr_val = stats.get('mrr', 0)
                if self.use_llm:
                    print(f"  {label:25s} | n={stats['count']:3d} | hit={stats['hit_rate']:.0%} | MRR={mrr_val:.4f} | ROUGE-L={stats['avg_rouge_l']:.3f} | F1={stats['avg_f1']:.3f}")
                else:
                    print(f"  {label:25s} | n={stats['count']:3d} | hit={stats['hit_rate']:.0%} | MRR={mrr_val:.4f}")

        if "by_difficulty" in summary:
            print(f"\n按难度分布:")
            for level in ["easy", "medium", "hard"]:
                if level in summary["by_difficulty"]:
                    stats = summary["by_difficulty"][level]
                    mrr_val = stats.get('mrr', 0)
                    if self.use_llm:
                        print(f"  {level:10s} | n={stats['count']:3d} | hit={stats['hit_rate']:.0%} | MRR={mrr_val:.4f} | ROUGE-L={stats['avg_rouge_l']:.3f} | F1={stats['avg_f1']:.3f}")
                    else:
                        print(f"  {level:10s} | n={stats['count']:3d} | hit={stats['hit_rate']:.0%} | MRR={mrr_val:.4f}")

        print(f"\n完整结果已保存至: {result_file}")
        print("=" * 60)


# ============================================================
# 分层抽样
# ============================================================

def stratified_sample(dataset_path: str, n: int) -> str:
    """
    按 label × difficulty 分层抽样 n 题，保存为临时文件并返回路径。
    """
    import random
    random.seed(42)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 展平嵌套格式
    if dataset and isinstance(dataset[0], dict) and "qa_pairs" in dataset[0]:
        flat = []
        for paper in dataset:
            for qa in paper.get("qa_pairs", []):
                flat.append(qa)
        dataset = flat

    # 按 label × difficulty_level 分组
    groups = {}
    for item in dataset:
        label = item.get("label", "unknown")
        d = item.get("difficulty", {})
        score = d.get("score", 0) if isinstance(d, dict) else d
        if score <= 0.5:
            level = "easy"
        elif score <= 0.7:
            level = "medium"
        else:
            level = "hard"
        key = (label, level)
        groups.setdefault(key, []).append(item)

    # 均匀分配名额
    num_groups = len(groups)
    base_quota = n // num_groups if num_groups else 1
    remainder = n - base_quota * num_groups

    sampled = []
    sorted_keys = sorted(groups.keys())
    for i, key in enumerate(sorted_keys):
        items = groups[key]
        quota = base_quota + (1 if i < remainder else 0)
        quota = min(quota, len(items))
        sampled.extend(random.sample(items, quota))

    random.shuffle(sampled)

    # 保存到临时文件
    output_path = dataset_path.replace(".json", f"_sample{len(sampled)}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"[分层抽样] 从 {len(dataset)} 题中抽取 {len(sampled)} 题 ({num_groups} 组)")
    print(f"[分层抽样] 保存至: {output_path}")
    return output_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="批量评估 RAG Pipeline")
    parser.add_argument("--dataset", "-d", default="train_set_100papers.json",
                        help="评估数据集路径")
    parser.add_argument("--output", "-o", default="eval_results",
                        help="结果输出目录")
    parser.add_argument("--max", "-n", type=int, default=None,
                        help="最多评估几题 (默认全部)")
    parser.add_argument("--resume", "-r", default=None,
                        help="从之前的结果文件恢复")
    parser.add_argument("--no-reranker", action="store_true",
                        help="不使用 reranker")
    parser.add_argument("--no-llm", action="store_true",
                        help="不调用 LLM 生成答案 (只测检索, 轻量模式)")
    parser.add_argument("--no-sparse", action="store_true",
                        help="不使用 Sparse Retriever (更快, 二路检索)")
    parser.add_argument("--sample", type=int, default=None,
                        help="分层抽样 N 题 (按题型×难度均匀抽取)")

    args = parser.parse_args()

    # --no-llm 时自动启用轻量模式: 禁用 Sparse + ContextExpander + ROUGE/F1
    use_sparse = not args.no_sparse
    if args.no_llm and not args.no_sparse:
        print("[轻量模式] --no-llm 下已自动禁用 Sparse Retriever")
        use_sparse = False

    evaluator = RAGEvaluator(
        use_reranker=not args.no_reranker,
        use_llm=not args.no_llm,
        use_sparse=use_sparse,
    )

    # 分层抽样
    sample_path = args.dataset
    if args.sample:
        sample_path = stratified_sample(args.dataset, args.sample)

    evaluator.evaluate_dataset(
        dataset_path=sample_path,
        output_dir=args.output,
        max_questions=args.max,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
