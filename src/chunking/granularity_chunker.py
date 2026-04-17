"""
多粒度分块模块

功能：
1. 实现不同粒度的分块策略
2. 参考 MoG 论文的 half/single/double/quad/oct 分块思想
3. 支持 sentence, paragraph, section, document 四种粒度

MoG 论文原始分块策略：
- half:    原始 chunk 切成 2 份
- single:  原始 chunk（基准）
- double:  2 个 chunk 合并
- quad:    4 个 chunk 合并
- oct:     8 个 chunk 合并

我们简化后的策略：
- SENTENCE: ~100 tokens（最小粒度）
- PARAGRAPH: ~300-500 tokens（基准）
- SECTION: ~1000-2000 tokens
- DOCUMENT: ~4000+ tokens（整个文档或摘要）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum
import re


class ChunkGranularity(Enum):
    """分块粒度级别"""
    SENTENCE = "sentence"      # 句子级 (~100 tokens)
    PARAGRAPH = "paragraph"    # 段落级 (~300-500 tokens) - 基准
    SECTION = "section"        # 章节级 (~1000-2000 tokens)
    DOCUMENT = "document"      # 文档级 (~4000+ tokens)


@dataclass
class ChunkConfig:
    """分块配置"""
    min_tokens: int = 50       # 最小 token 数
    max_tokens: int = 500      # 最大 token 数（PARAGRAPH 基准）
    overlap_tokens: int = 100   # 相邻 chunk 重叠 token 数（增大以减少信息丢失）
    granularity: ChunkGranularity = ChunkGranularity.PARAGRAPH


@dataclass
class Chunk:
    """分块对象"""
    id: str
    text: str
    granularity: ChunkGranularity
    section_type: str = "unknown"
    token_count: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self.estimate_tokens()
    
    def estimate_tokens(self) -> int:
        """估算 token 数（英文约 4 字符 ≈ 1 token）"""
        return len(self.text) // 4


class GranularityChunker:
    """
    多粒度分块器
    
    根据配置的粒度级别进行分块
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.granularity = self.config.granularity
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子

        使用改进的启发式方法，正确处理学术论文中的常见缩写
        """
        # 先保护常见缩写，避免误切
        protected = text
        abbreviations = [
            'e.g.', 'i.e.', 'et al.', 'Fig.', 'fig.', 'Eq.', 'eq.',
            'Dr.', 'Mr.', 'Mrs.', 'vs.', 'etc.', 'approx.', 'Ref.',
            'ref.', 'Sec.', 'sec.', 'Vol.', 'vol.', 'No.', 'no.',
            'resp.', 'w.r.t.', 'cf.', 'Tab.', 'tab.', 'Def.', 'def.',
            'Thm.', 'Prop.', 'Cor.', 'Lem.', 'Alg.',
        ]
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)

        # 保护数字中的小数点 (e.g. 3.14, 0.001)
        protected = re.sub(r'(\d)\.(\d)', r'\1__DOT__\2', protected)

        # 句子分割：在 . ! ? 后跟空格+大写字母 或 换行 处切分
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z"\(])'
        parts = re.split(sentence_endings, protected)

        sentences = []
        for part in parts:
            # 还原缩写和小数点
            for placeholder, abbr in placeholders.items():
                part = part.replace(placeholder, abbr)
            part = part.replace('__DOT__', '.')
            part = part.strip()
            if len(part) > 10:
                sentences.append(part)

        return sentences
    
    def _get_overlap_text(self, text: str) -> str:
        """从文本尾部提取 overlap 部分（按句子边界截断，避免在单词中间断开）"""
        overlap_chars = self.config.overlap_tokens * 4  # 估算字符数
        if len(text) <= overlap_chars:
            return text
        # 取尾部 overlap_chars 字符
        tail = text[-overlap_chars:]
        # 尝试在句子边界或空格处截断，避免半截单词
        # 找第一个句号/问号/感叹号后的空格
        m = re.search(r'[.!?]\s+', tail)
        if m and m.start() < len(tail) // 2:
            # 从句子边界开始
            return tail[m.end():]
        # 否则找第一个空格
        sp = tail.find(' ')
        if sp > 0 and sp < len(tail) // 3:
            return tail[sp + 1:]
        return tail

    def split_into_sentence_chunks(self, text: str, chunk_id: str) -> List[Chunk]:
        """
        将文本分割成句子级 chunks

        保持句子完整性，同时合并过短的句子，相邻 chunk 之间有 overlap
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_text = ""
        overlap_prefix = ""
        idx = 0
        
        for sentence in sentences:
            if not current_text:
                current_text = overlap_prefix + sentence if overlap_prefix else sentence
            elif len(current_text) + len(sentence) < self.config.max_tokens * 4:
                current_text += " " + sentence
            else:
                # 保存当前 chunk
                if len(current_text) > self.config.min_tokens * 4:
                    chunks.append(Chunk(
                        id=f"{chunk_id}_{idx}",
                        text=current_text,
                        granularity=ChunkGranularity.SENTENCE,
                        token_count=len(current_text) // 4
                    ))
                    idx += 1
                    overlap_prefix = self._get_overlap_text(current_text) + " "
                current_text = overlap_prefix + sentence
        
        # 保存最后一个 chunk
        if len(current_text) > self.config.min_tokens * 4:
            chunks.append(Chunk(
                id=f"{chunk_id}_{idx}",
                text=current_text,
                granularity=ChunkGranularity.SENTENCE,
                token_count=len(current_text) // 4
            ))
        
        return chunks
    
    def split_into_paragraph_chunks(self, text: str, chunk_id: str) -> List[Chunk]:
        """
        将文本分割成段落级 chunks

        按段落分割，同时合并过短的段落，相邻 chunk 之间有 overlap
        """
        # 按换行符分割段落
        paragraphs = text.split('\n')
        chunks = []
        current_text = ""
        overlap_prefix = ""
        idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 列表项也作为有效内容保留（学术论文中的枚举型实验结果很重要）
            if para.startswith('-') or para.startswith('•') or para.startswith('*'):
                # 归入当前段落而非丢弃
                if not current_text:
                    current_text = overlap_prefix + para if overlap_prefix else para
                elif len(current_text) + len(para) < self.config.max_tokens * 4:
                    current_text += "\n" + para
                else:
                    if len(current_text) > self.config.min_tokens * 4:
                        chunks.append(Chunk(
                            id=f"{chunk_id}_{idx}",
                            text=current_text,
                            granularity=ChunkGranularity.PARAGRAPH,
                            token_count=len(current_text) // 4
                        ))
                        idx += 1
                        overlap_prefix = self._get_overlap_text(current_text) + " "
                    current_text = overlap_prefix + para if overlap_prefix else para
                continue
            
            if not current_text:
                current_text = overlap_prefix + para if overlap_prefix else para
            elif len(current_text) + len(para) < self.config.max_tokens * 4:
                current_text += " " + para
            else:
                # 保存当前 chunk
                if len(current_text) > self.config.min_tokens * 4:
                    chunks.append(Chunk(
                        id=f"{chunk_id}_{idx}",
                        text=current_text,
                        granularity=ChunkGranularity.PARAGRAPH,
                        token_count=len(current_text) // 4
                    ))
                    idx += 1
                    overlap_prefix = self._get_overlap_text(current_text) + " "
                current_text = overlap_prefix + para
        
        # 保存最后一个 chunk
        if len(current_text) > self.config.min_tokens * 4:
            chunks.append(Chunk(
                id=f"{chunk_id}_{idx}",
                text=current_text,
                granularity=ChunkGranularity.PARAGRAPH,
                token_count=len(current_text) // 4
            ))
        
        return chunks
    
    def merge_into_section_chunks(self, chunks: List[Chunk], section_id: str) -> List[Chunk]:
        """
        将多个 chunks 合并成章节级 chunk（带 overlap）
        
        Args:
            chunks: 较小的 chunks（如 paragraph 级）
            section_id: 章节 ID
            
        Returns:
            List[Chunk]: 合并后的 chunks
        """
        if not chunks:
            return []
        
        merged = []
        current_text = ""
        overlap_prefix = ""
        idx = 0
        
        for chunk in chunks:
            if not current_text:
                current_text = overlap_prefix + chunk.text if overlap_prefix else chunk.text
            elif len(current_text) + len(chunk.text) < self.config.max_tokens * 4:
                current_text += " " + chunk.text
            else:
                merged.append(Chunk(
                    id=f"{section_id}_{idx}",
                    text=current_text,
                    granularity=ChunkGranularity.SECTION,
                    section_type=chunks[0].section_type,
                    token_count=len(current_text) // 4
                ))
                idx += 1
                overlap_prefix = self._get_overlap_text(current_text) + " "
                current_text = overlap_prefix + chunk.text
        
        # 保存最后一个
        if current_text:
            merged.append(Chunk(
                id=f"{section_id}_{idx}",
                text=current_text,
                granularity=ChunkGranularity.SECTION,
                section_type=chunks[0].section_type if chunks else "unknown",
                token_count=len(current_text) // 4
            ))
        
        return merged
    
    def create_document_chunk(self, chunks: List[Chunk], doc_id: str, 
                               abstract: Optional[str] = None) -> List[Chunk]:
        """
        创建文档级 chunk
        
        通常用于摘要或整个文档的简短表示
        """
        result = []
        
        # 如果有摘要，创建一个 document 级 chunk
        if abstract:
            result.append(Chunk(
                id=f"{doc_id}_abstract",
                text=abstract,
                granularity=ChunkGranularity.DOCUMENT,
                section_type="abstract",
                token_count=len(abstract) // 4,
                metadata={"is_abstract": True}
            ))
        
        # 如果有足够的 chunks，也创建一个包含所有内容的 document chunk
        if len(chunks) > 0:
            all_text = " ".join(c.text for c in chunks)
            if len(all_text) > 1000:  # 只对较长的文档创建
                result.append(Chunk(
                    id=f"{doc_id}_full",
                    text=all_text[:50000],  # 限制长度
                    granularity=ChunkGranularity.DOCUMENT,
                    section_type="full_document",
                    token_count=len(all_text) // 4,
                    metadata={"is_full_document": True}
                ))
        
        return result
    
    def chunk_text(self, text: str, chunk_id: str, 
                   granularity: Optional[ChunkGranularity] = None) -> List[Chunk]:
        """
        根据粒度分块
        
        Args:
            text: 待分块文本
            chunk_id: chunk ID 前缀
            granularity: 分块粒度（默认使用配置的粒度）
            
        Returns:
            List[Chunk]: 分块结果
        """
        gran = granularity or self.granularity
        
        if gran == ChunkGranularity.SENTENCE:
            return self.split_into_sentence_chunks(text, chunk_id)
        elif gran == ChunkGranularity.PARAGRAPH:
            return self.split_into_paragraph_chunks(text, chunk_id)
        elif gran == ChunkGranularity.SECTION:
            # Section 需要先做 paragraph 分块，再合并
            para_chunks = self.split_into_paragraph_chunks(text, chunk_id)
            return self.merge_into_section_chunks(para_chunks, chunk_id)
        elif gran == ChunkGranularity.DOCUMENT:
            # Document 直接返回
            return [Chunk(
                id=f"{chunk_id}_doc",
                text=text,
                granularity=ChunkGranularity.DOCUMENT,
                token_count=len(text) // 4
            )]
        
        return []
    
    def chunk_document(self, chunks: List[Dict], doc_id: str,
                      granularity: Optional[ChunkGranularity] = None) -> Dict[str, List[Chunk]]:
        """
        对整个文档进行分块
        
        Args:
            chunks: 统一格式的 chunks 列表
            doc_id: 文档 ID
            granularity: 分块粒度
            
        Returns:
            Dict[str, List[Chunk]]: 按粒度分组的 chunks
        """
        gran = granularity or self.granularity
        
        # 按 section 分组
        section_chunks = {}
        for chunk in chunks:
            section = chunk.get('title', 'unknown')
            if section not in section_chunks:
                section_chunks[section] = []
            section_chunks[section].append(chunk)
        
        result = {}
        
        for section_name, section_items in section_chunks.items():
            # 合并同 section 的文本
            combined_text = " ".join(c.get('content', '') for c in section_items)
            
            if combined_text.strip():
                section_chunks_result = self.chunk_text(
                    combined_text,
                    f"{doc_id}_{section_name}",
                    granularity=gran
                )
                for c in section_chunks_result:
                    c.section_type = section_name
                result[section_name] = section_chunks_result
        
        return result


def main():
    """测试代码"""
    from src.chunking.unified_format import MinerUToUnifiedConverter
    
    # 测试分块
    test_text = """
    Deep learning has revolutionized many fields of artificial intelligence. 
    In recent years, transformer models have become the dominant architecture for NLP tasks.
    The attention mechanism allows models to focus on relevant parts of the input.
    BERT and GPT are examples of successful transformer-based models.
    """
    
    chunker = GranularityChunker()
    
    # 测试句子级分块
    print("=== Sentence Level ===")
    sentence_chunks = chunker.chunk_text(test_text, "test", ChunkGranularity.SENTENCE)
    for c in sentence_chunks:
        print(f"[{c.granularity.value}] {c.text[:60]}... ({c.token_count} tokens)")
    
    # 测试段落级分块
    print("\n=== Paragraph Level ===")
    para_chunks = chunker.chunk_text(test_text, "test", ChunkGranularity.PARAGRAPH)
    for c in para_chunks:
        print(f"[{c.granularity.value}] {c.text[:60]}... ({c.token_count} tokens)")


if __name__ == "__main__":
    main()
