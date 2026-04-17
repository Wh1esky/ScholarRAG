"""
论文结构识别模块

功能：
1. 识别论文的各个部分（Abstract, Introduction, Method, etc.）
2. 根据结构选择合适的分块策略
3. 处理章节层级关系

参考 MoG 论文的分块思想，为后续自适应分块提供基础
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SectionType(Enum):
    """论文章节类型"""
    TITLE = "title"                    # 论文标题
    AUTHOR_INFO = "author_info"        # 作者信息
    ABSTRACT = "abstract"             # 摘要
    INTRODUCTION = "introduction"       # 引言
    RELATED_WORK = "related_work"      # 相关工作
    METHOD = "method"                  # 方法
    EXPERIMENT = "experiment"          # 实验
    CONCLUSION = "conclusion"          # 结论
    REFERENCE = "reference"            # 参考文献
    APPENDIX = "appendix"             # 附录
    UNKNOWN = "unknown"                # 未知


class SectionPriority(Enum):
    """章节优先级（影响分块粒度）"""
    HIGH_DETAIL = 1    # 需要细粒度分块
    MEDIUM = 2         # 中等粒度
    LOW_DETAIL = 3     # 可以粗粒度
    
    @property
    def granularity_hint(self) -> str:
        hints = {
            1: "sentence, paragraph",
            2: "paragraph, section",
            3: "section, document"
        }
        return hints[self.value]


@dataclass
class Section:
    """论文章节"""
    title: str
    section_type: SectionType
    start_page: int
    end_page: int
    content: str = ""
    
    @property
    def priority(self) -> SectionPriority:
        """根据章节类型返回建议的分块优先级"""
        high_detail = [SectionType.ABSTRACT, SectionType.METHOD, SectionType.EXPERIMENT]
        medium = [SectionType.INTRODUCTION, SectionType.RELATED_WORK]
        low_detail = [SectionType.CONCLUSION, SectionType.APPENDIX, SectionType.REFERENCE]
        
        if self.section_type in high_detail:
            return SectionPriority.HIGH_DETAIL
        elif self.section_type in medium:
            return SectionPriority.MEDIUM
        else:
            return SectionPriority.LOW_DETAIL


class StructureRecognizer:
    """
    论文结构识别器
    
    功能：
    1. 从解析的论文内容中识别章节结构
    2. 确定每个章节的类型
    3. 为自适应分块提供基础
    """
    
    # 章节标题模式
    SECTION_PATTERNS = {
        SectionType.ABSTRACT: [
            r"^abstract$",
            r"^1?\s*abstract\s*$",
        ],
        SectionType.INTRODUCTION: [
            r"^1\s*\.?\s*introduction$",
            r"^introduction$",
        ],
        SectionType.RELATED_WORK: [
            r"^related\s*(work|works)$",
            r"^background$",
            r"^literature\s*review$",
            r"^2\s*\.?\s*\w*(related|work|background)",
        ],
        SectionType.METHOD: [
            r"^method(s|ology)?$",
            r"^approach$",
            r"^framework$",
            r"^model$",
            r"^algorithm",
            r"^3\s*\.?\s*\w*(method|approach)",
            r"^proposed\s*method",
        ],
        SectionType.EXPERIMENT: [
            r"^experiment(s|al\s*results)?$",
            r"^evaluation$",
            r"^results$",
            r"^4\s*\.?\s*\w*(experiment|evaluation|results)",
            r"^experimental\s*setup",
        ],
        SectionType.CONCLUSION: [
            r"^conclusion(s)?$",
            r"^summary$",
            r"^future\s*work$",
            r"^discussion$",
            r"^5\s*\.?\s*\w*(conclusion|summary|discussion)",
        ],
        SectionType.REFERENCE: [
            r"^references?$",
            r"^bibliography$",
        ],
        SectionType.APPENDIX: [
            r"^appendix",
            r"^supplementary",
            r"^proofs?",
        ],
    }
    
    def __init__(self):
        self.sections: List[Section] = []
        self.current_section: Optional[Section] = None
    
    def recognize_section_type(self, title: str) -> SectionType:
        """
        根据标题识别章节类型
        
        Args:
            title: 章节标题
            
        Returns:
            SectionType: 章节类型
        """
        import re
        title_lower = title.lower().strip()
        
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    return section_type
        
        return SectionType.UNKNOWN
    
    def recognize_from_chunks(self, chunks: List) -> List[Section]:
        """
        从 chunks 列表中识别章节结构
        
        Args:
            chunks: 包含 title 属性的 chunk 列表
            
        Returns:
            List[Section]: 识别出的章节列表
        """
        sections = []
        current_section = None
        start_page = 0
        
        for i, chunk in enumerate(chunks):
            title = getattr(chunk, 'title', 'unknown')
            
            # 检查是否是新的章节
            section_type = self.recognize_section_type(title)
            
            if section_type != SectionType.UNKNOWN:
                # 保存之前的章节
                if current_section:
                    current_section.end_page = i
                    sections.append(current_section)
                
                # 开始新章节
                current_section = Section(
                    title=title,
                    section_type=section_type,
                    start_page=i,
                    end_page=i
                )
            elif current_section:
                current_section.end_page = i
        
        # 保存最后一个章节
        if current_section:
            current_section.end_page = len(chunks)
            sections.append(current_section)
        
        self.sections = sections
        return sections
    
    def get_section_info(self) -> Dict[str, any]:
        """
        获取章节结构信息
        
        Returns:
            Dict: 包含章节统计信息的字典
        """
        if not self.sections:
            return {"total_sections": 0, "sections": []}
        
        section_counts = {}
        for section in self.sections:
            section_name = section.section_type.value
            if section_name not in section_counts:
                section_counts[section_name] = 0
            section_counts[section_name] += 1
        
        return {
            "total_sections": len(self.sections),
            "section_types": section_counts,
            "sections": [
                {
                    "title": s.title,
                    "type": s.section_type.value,
                    "pages": f"{s.start_page}-{s.end_page}",
                    "priority": s.priority.name
                }
                for s in self.sections
            ]
        }


def main():
    """测试代码"""
    from src.chunking.unified_format import MinerUToUnifiedConverter
    
    converter = MinerUToUnifiedConverter()
    doc = converter.convert_file("parsed_pdf/2105.08233_content_list_v2.json")
    
    recognizer = StructureRecognizer()
    sections = recognizer.recognize_from_chunks(doc.chunks)
    
    print(f"识别到 {len(sections)} 个章节:")
    for section in sections:
        print(f"  [{section.section_type.value}] {section.title}: pages {section.start_page}-{section.end_page}")


if __name__ == "__main__":
    main()
