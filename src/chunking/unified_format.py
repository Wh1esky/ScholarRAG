"""
统一数据格式转换模块

功能：
1. 将 MinerU 解析的 JSON 转换为统一格式
2. 统一格式参考 MoG 论文: id, title, content, contents
3. 处理文本、公式、表格、列表等不同元素类型

MinerU JSON 结构:
[
  [page_0_elements],
  [page_1_elements],
  ...
]

每个元素:
{
  "type": "title|paragraph|table|figure|list|...",
  "content": {...},
  "bbox": [x1, y1, x2, y2]
}
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum


class ElementType(Enum):
    """MinerU 元素类型"""
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    EQUATION_BLOCK = "equation_block"
    EQUATION_INLINE = "equation_inline"
    CITATION = "citation"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


@dataclass
class UnifiedChunk:
    """
    统一分块格式
    
    参考 MoG 论文格式:
    {
        "id": "paper_id|page_num|chunk_idx",
        "title": "章节标题或所属节",
        "content": "纯文本内容（不含标题）",
        "contents": "title + content 合并文本",
        "metadata": {...}
    }
    """
    id: str                    # 唯一标识: "2105.08233|0|1"
    title: str                 # 所属章节标题
    content: str               # 纯文本内容
    contents: str              # title + ". " + content
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __len__(self) -> int:
        """返回 content 的字符数"""
        return len(self.content)
    
    @property
    def token_count(self) -> int:
        """估算 token 数量（英文约 4 字符 ≈ 1 token）"""
        return len(self.content) // 4


@dataclass
class UnifiedDocument:
    """
    统一文档格式
    
    用于表示一篇完整的学术论文
    """
    paper_id: str                              # 论文 ID (如 arXiv ID)
    title: str                                 # 论文标题
    authors: Optional[str] = None              # 作者列表
    abstract: Optional[str] = None              # 摘要
    chunks: List[UnifiedChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata
        }
    
    def to_jsonl(self) -> str:
        """转换为 JSONL 格式（每行一个 chunk）"""
        lines = []
        for chunk in self.chunks:
            lines.append(json.dumps(chunk.to_dict(), ensure_ascii=False))
        return "\n".join(lines)


class MinerUToUnifiedConverter:
    """
    MinerU JSON 到统一格式的转换器
    
    使用方法:
        converter = MinerUToUnifiedConverter()
        document = converter.convert_file("path/to/file_content_list_v2.json")
        print(document.title)
        print(f"Chunks: {len(document.chunks)}")
    """
    
    # 章节标题关键词（用于识别论文结构）
    SECTION_KEYWORDS = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "1 introduction", "i. introduction"],
        "related_work": ["related work", "related works", "background", "literature review"],
        "method": ["method", "methodology", "method section", "approach", "model", "framework", "algorithm"],
        "experiment": ["experiment", "experiments", "evaluation", "results", "experimental results"],
        "conclusion": ["conclusion", "conclusions", "summary", "future work", "discussion"],
        "reference": ["reference", "references", "bibliography"],
        "appendix": ["appendix", "supplementary", "supplemental material"],
    }
    
    # 论文标题关键词（通常在文档开头）
    TITLE_PATTERNS = [
        r"^[A-Z].*[a-z].*[a-z]$",  # 普通标题
        r"^\d{4}\.\d{4,}$",  # arXiv ID 格式
    ]
    
    def __init__(self):
        self.current_section = "unknown"
        self.current_title = ""
        self.chunk_counter = 0
        
    def extract_text_from_element(self, element: Dict[str, Any]) -> str:
        """
        从 MinerU 元素中提取纯文本内容
        
        处理类型:
        - text: 直接提取
        - equation_interline/equation_inline: 提取 LaTeX 公式文本
        - table: 提取表格 HTML 内容或 caption
        - algorithm: 提取算法内容
        - figure/image: 用 [图片] 标记
        """
        element_type = element.get("type", "unknown")
        
        if element_type == "text":
            return element.get("content", "")
        
        content = element.get("content", {})
        
        if element_type in ("paragraph", "title"):
            # 段落或标题 - 需要递归提取
            return self._extract_text_from_compound(content)
        
        elif element_type == "list":
            # 列表 - 提取每个列表项
            items = content.get("list_items", [])
            text_parts = []
            for item in items:
                item_content = item.get("item_content", [])
                item_text = self._extract_text_list(item_content)
                if item_text:
                    text_parts.append(f"- {item_text}")
            return "\n".join(text_parts)
        
        elif element_type == "table":
            return self._extract_table_content(content)
        
        elif element_type in ("figure", "image"):
            return "[图片]"
        
        elif element_type in ("equation_inline", "equation_block", "equation_interline"):
            return self._extract_equation_content(content)
        
        elif element_type == "algorithm":
            return self._extract_algorithm_content(content)
        
        return ""
    
    def _extract_table_content(self, content: Dict) -> str:
        """从表格元素中提取内容（HTML → 纯文本 + caption）"""
        parts = []
        # 提取表格标题
        caption_items = content.get("table_caption", [])
        if caption_items:
            cap_text = self._extract_text_list(caption_items)
            if cap_text:
                parts.append(cap_text)
        # 提取 HTML 表格内容并转为纯文本
        html = content.get("html", "")
        if html:
            # 简单 HTML 表格 → 纯文本：去除标签，保留单元格内容
            import re as _re
            # 在 </td> 和 </tr> 处加分隔符
            text = _re.sub(r'</td>', ' | ', html)
            text = _re.sub(r'</tr>', '\n', text)
            text = _re.sub(r'<[^>]+>', '', text)  # 去除剩余标签
            text = text.strip()
            if text:
                parts.append(text)
        # 表格脚注
        footnote_items = content.get("table_footnote", [])
        if footnote_items:
            fn_text = self._extract_text_list(footnote_items)
            if fn_text:
                parts.append(fn_text)
        return "\n".join(parts) if parts else "[表格]"

    def _extract_equation_content(self, content) -> str:
        """从公式元素中提取 LaTeX 内容"""
        if isinstance(content, str):
            # equation_inline 直接包含 LaTeX 字符串
            return content
        if isinstance(content, dict):
            math_content = content.get("math_content", "")
            if math_content:
                return math_content
        return "[公式]"

    def _extract_algorithm_content(self, content: Dict) -> str:
        """从算法元素中提取内容"""
        parts = []
        # 算法标题
        caption = content.get("algorithm_caption", [])
        if caption:
            cap_text = self._extract_text_list(caption)
            if cap_text:
                parts.append(cap_text)
        # 算法内容
        algo_content = content.get("algorithm_content", [])
        if algo_content:
            body_text = self._extract_text_list(algo_content)
            if body_text:
                parts.append(body_text)
        return "\n".join(parts) if parts else "[算法]"

    def _extract_text_list(self, content_list: List[Dict]) -> str:
        """从内容列表中提取文本"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("content", ""))
                elif item.get("type") in ("equation_inline", "equation_block", "equation_interline"):
                    # 保留公式 LaTeX 内容而非占位符
                    eq_content = item.get("content", "")
                    if isinstance(eq_content, dict):
                        parts.append(eq_content.get("math_content", "[公式]"))
                    elif isinstance(eq_content, str):
                        parts.append(eq_content)
                    else:
                        parts.append("[公式]")
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    
    def _extract_text_from_compound(self, content: Dict) -> str:
        """从复合内容中提取文本"""
        # 处理 paragraph_content 或 title_content
        for key in ["paragraph_content", "title_content"]:
            if key in content:
                return self._extract_text_list(content[key])
        return ""
    
    def is_section_title(self, element: Dict[str, Any]) -> Tuple[bool, str]:
        """
        判断元素是否为章节标题
        
        返回:
            (is_title, section_type): 是否为标题及其类型
        """
        if element.get("type") != "title":
            return False, ""
        
        # 提取标题文本
        title_text = self.extract_text_from_element(element)
        title_lower = title_text.lower().strip()
        
        # 检查是否匹配章节关键词
        for section_type, keywords in self.SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return True, section_type
        
        return False, ""
    
    def extract_title(self, elements: List[Dict]) -> str:
        """
        提取论文标题
        
        通常是第一个 title 类型的元素，且不是 "Abstract" 等章节标题
        """
        for i, element in enumerate(elements[:10]):  # 只检查前10个元素
            if element.get("type") == "title":
                title_text = self.extract_text_from_element(element)
                title_lower = title_text.lower().strip()
                
                # 排除明显的章节标题
                skip_keywords = ["abstract", "introduction", "section", "chapter", "figure", "table"]
                if any(kw in title_lower for kw in skip_keywords):
                    continue
                
                # 排除太短的标题（可能是副标题）
                if len(title_text) < 10:
                    continue
                
                return title_text
        
        return "Unknown Title"
    
    def extract_abstract(self, elements: List[Dict]) -> Optional[str]:
        """提取摘要内容"""
        for element in elements:
            if element.get("type") == "title":
                title_text = self.extract_text_from_element(element)
                if "abstract" in title_text.lower():
                    # 找下一个 paragraph 元素作为摘要
                    idx = elements.index(element)
                    for next_elem in elements[idx+1:]:
                        if next_elem.get("type") == "paragraph":
                            return self.extract_text_from_element(next_elem)
        return None
    
    def extract_authors(self, elements: List[Dict]) -> Optional[str]:
        """提取作者信息（通常在标题后、摘要前的段落中）"""
        abstract_idx = None
        for i, element in enumerate(elements):
            if element.get("type") == "title":
                title_text = self.extract_text_from_element(element)
                if "abstract" in title_text.lower():
                    abstract_idx = i
                    break
        
        if abstract_idx is None:
            return None
        
        # 检查标题和摘要之间的段落
        for element in elements[:abstract_idx]:
            if element.get("type") == "paragraph":
                text = self.extract_text_from_element(element)
                # 作者信息通常包含多个名字和 Affiliation
                if "@" in text or re.search(r'[A-Z][a-z]+\s+[A-Z]', text):
                    return text
        
        return None
    
    def _generate_chunk_id(self, paper_id: str, page_idx: int) -> str:
        """生成 chunk ID"""
        chunk_id = f"{paper_id}|{page_idx}|{self.chunk_counter}"
        self.chunk_counter += 1
        return chunk_id
    
    def convert_page_elements(self, elements: List[Dict], paper_id: str, 
                             page_idx: int, prev_section: str = "unknown") -> Tuple[List[UnifiedChunk], str]:
        """
        转换一页的所有元素为 chunks
        
        返回:
            (chunks, last_section): 生成的 chunks 和最后的章节类型
        """
        chunks = []
        current_section = prev_section
        paragraph_buffer = []
        
        for element in elements:
            element_type = element.get("type", "")
            element_text = self.extract_text_from_element(element)
            
            # 跳过空文本
            if not element_text.strip():
                continue
            
            # 判断是否为章节标题
            is_title, section_type = self.is_section_title(element)
            
            if is_title:
                # 保存之前的段落缓冲
                if paragraph_buffer:
                    chunk_text = " ".join(paragraph_buffer)
                    chunk = UnifiedChunk(
                        id=self._generate_chunk_id(paper_id, page_idx),
                        title=current_section,
                        content=chunk_text,
                        contents=f"{current_section}. {chunk_text}",
                        metadata={"source": "paragraph_buffer", "page": page_idx}
                    )
                    chunks.append(chunk)
                    paragraph_buffer = []
                
                # 更新当前章节
                current_section = section_type
                self.current_title = element_text
                
            elif element_type == "paragraph":
                # 添加到段落缓冲
                paragraph_buffer.append(element_text)
                
            else:
                # 表格、图片、列表等 - 直接创建 chunk
                if paragraph_buffer:
                    chunk_text = " ".join(paragraph_buffer)
                    chunk = UnifiedChunk(
                        id=self._generate_chunk_id(paper_id, page_idx),
                        title=current_section,
                        content=chunk_text,
                        contents=f"{current_section}. {chunk_text}",
                        metadata={"source": "paragraph", "page": page_idx}
                    )
                    chunks.append(chunk)
                    paragraph_buffer = []
                
                # 非段落元素直接添加
                if element_text.strip():
                    chunk = UnifiedChunk(
                        id=self._generate_chunk_id(paper_id, page_idx),
                        title=current_section,
                        content=element_text,
                        contents=f"{current_section}. {element_text}",
                        metadata={
                            "source": element_type,
                            "page": page_idx,
                            "bbox": element.get("bbox", [])
                        }
                    )
                    chunks.append(chunk)
        
        # 处理最后的段落缓冲
        if paragraph_buffer:
            chunk_text = " ".join(paragraph_buffer)
            chunk = UnifiedChunk(
                id=self._generate_chunk_id(paper_id, page_idx),
                title=current_section,
                content=chunk_text,
                contents=f"{current_section}. {chunk_text}",
                metadata={"source": "paragraph_buffer", "page": page_idx}
            )
            chunks.append(chunk)
        
        return chunks, current_section
    
    def convert_file(self, file_path: str) -> UnifiedDocument:
        """
        转换单个 MinerU JSON 文件为统一格式
        
        Args:
            file_path: MinerU JSON 文件路径
            
        Returns:
            UnifiedDocument: 统一格式的文档
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            mineru_data = json.load(f)
        
        # 提取 paper_id（从文件名）
        paper_id = Path(file_path).stem.replace("_content_list_v2", "")
        
        # 重置计数器
        self.chunk_counter = 0
        
        # 提取文档级信息
        # MinerU 格式: [[page_0_elements], [page_1_elements], ...]
        all_elements = []
        for page_elements in mineru_data:
            if isinstance(page_elements, list):
                all_elements.extend(page_elements)
        
        title = self.extract_title(all_elements)
        abstract = self.extract_abstract(all_elements)
        authors = self.extract_authors(all_elements)
        
        # 转换每个页面
        all_chunks = []
        prev_section = "unknown"
        
        for page_idx, page_elements in enumerate(mineru_data):
            if isinstance(page_elements, list):
                chunks, prev_section = self.convert_page_elements(
                    page_elements, paper_id, page_idx, prev_section
                )
                all_chunks.extend(chunks)
        
        return UnifiedDocument(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            chunks=all_chunks,
            metadata={
                "source_file": file_path,
                "total_pages": len(mineru_data),
                "total_chunks": len(all_chunks)
            }
        )
    
    def convert_directory(self, directory: str, output_dir: Optional[str] = None,
                          save_json: bool = True) -> List[UnifiedDocument]:
        """
        批量转换目录中的所有 MinerU JSON 文件
        
        Args:
            directory: 包含 MinerU JSON 文件的目录
            output_dir: 输出目录（可选）
            save_json: 是否保存为 JSON 文件
            
        Returns:
            List[UnifiedDocument]: 所有转换后的文档列表
        """
        directory = Path(directory)
        json_files = list(directory.glob("*_content_list_v2.json"))
        
        documents = []
        for json_file in json_files:
            try:
                doc = self.convert_file(str(json_file))
                documents.append(doc)
                
                if save_json and output_dir:
                    output_path = Path(output_dir) / f"{doc.paper_id}_chunks.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
                        
            except Exception as e:
                print(f"Error converting {json_file}: {e}")
        
        return documents


def main():
    """测试代码"""
    import os
    
    # 测试单个文件
    test_file = "parsed_pdf/2105.08233_content_list_v2.json"
    
    if os.path.exists(test_file):
        converter = MinerUToUnifiedConverter()
        doc = converter.convert_file(test_file)
        
        print(f"Paper ID: {doc.paper_id}")
        print(f"Title: {doc.title}")
        print(f"Abstract: {doc.abstract[:100]}..." if doc.abstract else "No abstract")
        print(f"Total Chunks: {len(doc.chunks)}")
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(doc.chunks[:3]):
            print(f"  [{i+1}] {chunk.title}: {chunk.content[:80]}...")
    
    # 测试批量转换
    if os.path.exists("parsed_pdf"):
        output_dir = "src/chunking/chunks"
        os.makedirs(output_dir, exist_ok=True)
        
        converter = MinerUToUnifiedConverter()
        documents = converter.convert_directory("parsed_pdf", output_dir, save_json=True)
        
        print(f"\nConverted {len(documents)} documents")
        for doc in documents[:3]:
            print(f"  {doc.paper_id}: {len(doc.chunks)} chunks")


if __name__ == "__main__":
    main()
