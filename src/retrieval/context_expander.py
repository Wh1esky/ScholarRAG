"""
Context Window Expansion (文档结构图上下文扩展)

功能：
- 检索到某个 chunk 后，沿文档结构自动扩展上下文
- 合并同一 paper + 同一 section 的相邻 chunk
- 为 LLM 提供更完整的段落级上下文

策略：
1. 对每个检索结果，找到同 paper_id + 同 section_type 的前后 N 个 chunk
2. 将相邻 chunk 文本拼接，替换原始单句文本
3. 去重：避免同一 chunk 被多次扩展
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ExpandedChunk:
    """扩展后的 chunk（可能包含多个原始 chunk 的文本）"""
    chunk_id: str          # 核心 chunk 的 ID
    text: str              # 扩展后的完整文本
    paper_id: str
    granularity: str
    section_type: str
    score: float
    rank: int
    context_chunk_ids: List[str]  # 包含的所有 chunk IDs
    window_size: int       # 实际扩展的窗口大小


class ContextExpander:
    """
    文档结构感知的上下文扩展器
    
    基于 metadata 中的 chunk 排列顺序,
    为检索到的 chunk 扩展同 paper + 同 section 的相邻上下文。
    """

    def __init__(self, metadata: List[Dict], window_size: int = 1):
        """
        Args:
            metadata: unified_metadata.json 加载后的列表
            window_size: 前后各扩展多少个 chunk (默认 1, 即前1后1)
        """
        self.window_size = window_size
        self.metadata = metadata
        
        # 构建索引: chunk_id -> global index
        self.chunk_id_to_idx: Dict[str, int] = {}
        # 构建索引: (paper_id, section_type) -> [global_idx, ...]  (同 section 内扩展)
        self.paper_section_chunks: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        
        for i, m in enumerate(metadata):
            cid = m['chunk_id']
            pid = m['paper_id']
            stype = m.get('section_type', 'unknown')
            self.chunk_id_to_idx[cid] = i
            self.paper_section_chunks[(pid, stype)].append(i)
        
        # 预排序每个 (paper, section) 的 chunk 列表
        for key in self.paper_section_chunks:
            self.paper_section_chunks[key].sort()
        
        print(f"ContextExpander 初始化: {len(metadata)} chunks, window_size={window_size}, 同section扩展")

    def _get_neighbor_indices(self, chunk_id: str) -> List[int]:
        """
        获取某个 chunk 在同 paper + 同 section 中的相邻 chunk 全局索引
        
        Returns:
            排序后的全局索引列表 (包含自身)
        """
        if chunk_id not in self.chunk_id_to_idx:
            return []
        
        global_idx = self.chunk_id_to_idx[chunk_id]
        meta = self.metadata[global_idx]
        pid = meta['paper_id']
        stype = meta.get('section_type', 'unknown')
        
        # 在同 paper + 同 section 的 chunk 列表中找位置
        section_indices = self.paper_section_chunks.get((pid, stype), [])
        if not section_indices:
            return [global_idx]
        
        # 二分查找当前 chunk 在 section 中的位置
        try:
            pos = section_indices.index(global_idx)
        except ValueError:
            return [global_idx]
        
        # 扩展窗口
        start = max(0, pos - self.window_size)
        end = min(len(section_indices), pos + self.window_size + 1)
        
        return section_indices[start:end]

    def expand_text(self, chunk_id: str) -> Tuple[str, List[str]]:
        """
        获取扩展后的文本
        
        Args:
            chunk_id: 原始 chunk ID
            
        Returns:
            (expanded_text, context_chunk_ids)
        """
        neighbor_indices = self._get_neighbor_indices(chunk_id)
        
        if not neighbor_indices:
            idx = self.chunk_id_to_idx.get(chunk_id)
            if idx is not None:
                return self.metadata[idx]['text'], [chunk_id]
            return "", [chunk_id]
        
        texts = []
        chunk_ids = []
        for idx in neighbor_indices:
            texts.append(self.metadata[idx]['text'])
            chunk_ids.append(self.metadata[idx]['chunk_id'])
        
        return " ".join(texts), chunk_ids

    def expand_results(self, results: List, top_k: Optional[int] = None) -> List:
        """
        对检索结果列表进行上下文扩展
        
        扩展策略:
        1. 每个 chunk 向前后各扩展 window_size 个同 section chunk
        2. 已被之前结果覆盖的 chunk 不重复扩展
        3. 保持原始排序和 score
        
        Args:
            results: RetrievalResult 列表
            top_k: 最多返回几个结果
            
        Returns:
            扩展后的 RetrievalResult 列表 (text 字段已替换为扩展文本)
        """
        from .dense_retriever import RetrievalResult
        
        if not results:
            return results
        
        seen_chunk_ids = set()  # 已覆盖的 chunk IDs, 避免重复
        expanded_results = []
        
        for r in results:
            # 跳过已被之前扩展覆盖的 chunk
            if r.chunk_id in seen_chunk_ids:
                continue
            
            expanded_text, context_ids = self.expand_text(r.chunk_id)
            
            # 记录已覆盖的 chunks
            seen_chunk_ids.update(context_ids)
            
            expanded_results.append(RetrievalResult(
                chunk_id=r.chunk_id,
                text=expanded_text,
                paper_id=r.paper_id,
                granularity=r.granularity,
                section_type=r.section_type,
                score=r.score,
                rank=len(expanded_results) + 1,
            ))
        
        if top_k:
            expanded_results = expanded_results[:top_k]
        
        return expanded_results
