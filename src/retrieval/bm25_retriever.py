"""
BM25 Sparse Retrieval 模块

功能：
- 基于 BM25 算法的稀疏检索
- 与 Dense Retriever 配合实现混合检索
- 对专有术语、模型名等精确匹配有优势
"""

import json
import re
import math
from typing import List, Dict, Optional
from collections import Counter
from .dense_retriever import RetrievalResult


class BM25Retriever:
    """
    BM25 稀疏检索器

    对学术论文中的专有术语（模型名、数据集名）等精确匹配场景有优势，
    可与 Dense Retriever 配合使用，弥补纯语义检索的短板。
    """

    # 内置英文停用词（避免 nltk 依赖）
    STOPWORDS = frozenset({
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
        'should', 'may', 'might', 'must', 'can', 'could', 'am',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'out', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if', 'while',
        'about', 'up', 'its', 'it', 'he', 'she', 'they', 'we', 'you',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'his', 'her', 'their', 'our', 'your', 'my', 'also', 'however',
    })

    def __init__(self, metadata_path: str, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            metadata_path: 元数据 JSON 文件路径（与 DenseRetriever 共用）
            k1: BM25 参数，控制词频饱和度
            b: BM25 参数，控制文档长度归一化
        """
        self.k1 = k1
        self.b = b

        print(f"加载 BM25 索引数据: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # 构建倒排索引
        self.doc_count = len(self.metadata)
        self.avg_doc_len = 0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # 每个 term 出现在多少个文档中
        self.inverted_index: Dict[str, List[tuple]] = {}  # term -> [(doc_idx, term_freq), ...]

        self._build_index()
        print(f"  BM25 索引构建完毕: {self.doc_count} 篇文档, {len(self.inverted_index)} 个唯一词")

    @staticmethod
    def _stem(word: str) -> str:
        """简单的英文词干提取（Porter-like 后缀剥离）"""
        if len(word) <= 3:
            return word
        # 常见后缀，按长度递减
        if word.endswith('ational'):
            return word[:-5] + 'e'
        if word.endswith('ization'):
            return word[:-5] + 'e'
        if word.endswith('iveness'):
            return word[:-4]
        if word.endswith('ement'):
            return word[:-4] if len(word) > 7 else word
        if word.endswith('tion') and len(word) > 5:
            return word[:-4] + 't' if not word.endswith('ation') else word[:-4] + 'ate'
        if word.endswith('ness') and len(word) > 5:
            return word[:-4]
        if word.endswith('ment') and len(word) > 6:
            return word[:-4]
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        if word.endswith('ing') and len(word) > 5:
            stem = word[:-3]
            if stem.endswith('at') or stem.endswith('bl') or stem.endswith('iz'):
                return stem + 'e'
            return stem
        if word.endswith('ed') and len(word) > 4:
            stem = word[:-2]
            if stem.endswith('at') or stem.endswith('bl') or stem.endswith('iz'):
                return stem + 'e'
            return stem
        if word.endswith('ly') and len(word) > 4:
            return word[:-2]
        if word.endswith('es') and len(word) > 3:
            if word.endswith('ses') or word.endswith('xes') or word.endswith('zes'):
                return word[:-2]
            return word[:-1]
        if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
            return word[:-1]
        return word

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """英文分词 + 停用词过滤 + 词干提取"""
        raw_tokens = re.findall(r'\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\b', text.lower())
        return [cls._stem(t) for t in raw_tokens if t not in cls.STOPWORDS and len(t) > 1]

    def _build_index(self):
        """构建 BM25 倒排索引"""
        total_len = 0

        for idx, meta in enumerate(self.metadata):
            text = meta.get('text', '')
            tokens = self.tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            term_counts = Counter(tokens)
            seen_terms = set()

            for term, freq in term_counts.items():
                # 更新倒排索引
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, freq))

                # 更新文档频率
                if term not in seen_terms:
                    self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                    seen_terms.add(term)

        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 1

    def _bm25_score(self, term: str, term_freq: int, doc_len: int) -> float:
        """计算单个 term 的 BM25 分数"""
        df = self.doc_freqs.get(term, 0)
        idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (term_freq * (self.k1 + 1)) / (
            term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
        )
        return idf * tf_norm

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        granularity_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回的结果数
            granularity_filter: 按粒度筛选

        Returns:
            List[RetrievalResult]: 检索结果
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # 计算每个文档的 BM25 分数
        doc_scores: Dict[int, float] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            for doc_idx, term_freq in self.inverted_index[term]:
                score = self._bm25_score(term, term_freq, self.doc_lengths[doc_idx])
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + score

        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs:
            meta = self.metadata[doc_idx]

            # 粒度筛选
            if granularity_filter and meta.get('granularity') != granularity_filter:
                continue

            result = RetrievalResult(
                chunk_id=meta.get('chunk_id', f'chunk_{doc_idx}'),
                text=meta.get('text', ''),
                paper_id=meta.get('paper_id', 'unknown'),
                granularity=meta.get('granularity', 'unknown'),
                section_type=meta.get('section_type', 'unknown'),
                score=score,
                rank=len(results) + 1
            )
            results.append(result)

            if len(results) >= top_k:
                break

        return results
