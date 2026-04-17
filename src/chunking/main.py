"""
ScholarRAG - ScholarRAG 主入口脚本

自适应分块策略实现 - 第二阶段
参考论文: Mix-of-Granularity (COLING 2025)

功能：
1. MinerU JSON → 统一格式转换
2. 论文结构识别
3. 多粒度分块（Sentence/Paragraph/Section）
4. 自适应路由（根据查询类型选择粒度）
5. 质量评估

使用方法:
    python -m src.chunking.main --help
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

from .unified_format import MinerUToUnifiedConverter, UnifiedDocument
from .structure_recognizer import StructureRecognizer, SectionType
from .granularity_chunker import GranularityChunker, ChunkGranularity, ChunkConfig
from .adaptive_router import AdaptiveRouter, QueryType
from .evaluator import ChunkEvaluator


class ScholarRAGChunker:
    """
    ScholarRAG 分块主类
    
    整合所有分块功能，提供统一的接口
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.converter = MinerUToUnifiedConverter()
        self.chunker = GranularityChunker(config)
        self.router = AdaptiveRouter()
        self.evaluator = ChunkEvaluator()
    
    def process_paper(self, json_path: str) -> Dict:
        """
        处理单篇论文
        
        Args:
            json_path: MinerU JSON 文件路径
            
        Returns:
            Dict: 处理结果
        """
        # 1. 格式转换
        doc = self.converter.convert_file(json_path)
        
        # 2. 结构识别
        recognizer = StructureRecognizer()
        sections = recognizer.recognize_from_chunks(doc.chunks)
        
        # 3. 多粒度分块
        granularity_results = {}
        for granularity in [ChunkGranularity.SENTENCE, ChunkGranularity.PARAGRAPH]:
            chunks = []
            for unified_chunk in doc.chunks:
                result = self.chunker.chunk_text(
                    unified_chunk.content,
                    unified_chunk.id,
                    granularity
                )
                chunks.extend([
                    {
                        "id": c.id,
                        "text": c.text,
                        "granularity": c.granularity.value,
                        "section": unified_chunk.title,
                        "tokens": c.token_count
                    }
                    for c in result
                ])
            granularity_results[granularity.value] = chunks
        
        return {
            "paper_id": doc.paper_id,
            "title": doc.title,
            "abstract": doc.abstract,
            "sections": recognizer.get_section_info(),
            "granularity_results": granularity_results
        }
    
    def query_granularity(self, query: str) -> tuple:
        """
        根据查询确定最优分块粒度
        
        Args:
            query: 用户查询
            
        Returns:
            Tuple[str, QueryType, float]: (粒度建议, 查询类型, 置信度)
        """
        classification = self.router.classify_query(query)
        return (
            classification.suggested_granularity,
            classification.query_type,
            classification.confidence
        )
    
    def recommend_chunks(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        根据查询推荐最相关的 chunks
        
        Args:
            query: 用户查询
            chunks: chunk 列表
            top_k: 返回的 chunk 数量
            
        Returns:
            List[Dict]: 推荐的 chunks
        """
        granularity, query_type, confidence = self.query_granularity(query)
        
        # 简单的关键词匹配（实际应该用 embedding 模型）
        query_keywords = set(query.lower().split())
        
        chunk_scores = []
        for chunk in chunks:
            content = chunk.get('text', '').lower()
            chunk_keywords = set(content.split())
            overlap = len(query_keywords & chunk_keywords)
            chunk_scores.append((chunk, overlap))
        
        # 排序并返回 top-k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in chunk_scores[:top_k]]


def main():
    parser = argparse.ArgumentParser(
        description="ScholarRAG - 自适应论文分块工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单篇论文
  python -m src.chunking.main --input parsed_pdf/2105.08233_content_list_v2.json

  # 批量处理
  python -m src.chunking.main --batch --input parsed_pdf

  # 查询粒度建议
  python -m src.chunking.main --query "What is the accuracy of BERT?"
        """
    )
    
    parser.add_argument("--input", "-i", help="输入文件或目录")
    parser.add_argument("--output", "-o", default="src/chunking/output", help="输出目录")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式")
    parser.add_argument("--query", "-q", help="查询文本（用于获取粒度建议）")
    parser.add_argument("--granularity", "-g", 
                       choices=["sentence", "paragraph", "section"],
                       default="paragraph",
                       help="分块粒度")
    
    args = parser.parse_args()
    
    # 直接在这里初始化大管家（不管什么模式都先把它叫醒）
    scholar = ScholarRAGChunker()
    
    # 查询模式
    if args.query:
        granularity, qtype, confidence = scholar.query_granularity(args.query)
        print(f"\n查询: {args.query}")
        print(f"查询类型: {qtype.value}")
        print(f"置信度: {confidence:.2f}")
        print(f"推荐粒度: {granularity}")
        return
    
    # 单文件模式
    if not args.batch:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"文件不存在: {input_path}")
            return
        
        print(f"处理文件: {input_path}")
        result = scholar.process_paper(str(input_path))
        
        print(f"\n{'='*50}")
        print(f"Paper ID: {result['paper_id']}")
        print(f"Title: {result['title']}")
        print(f"Abstract: {result['abstract'][:100] if result['abstract'] else 'N/A'}...")
        print(f"\nSections: {result['sections']['total_sections']}")
        print("\nGranularity Results:")
        for gran, chunks in result['granularity_results'].items():
            print(f"  {gran}: {len(chunks)} chunks")
        
        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{result['paper_id']}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_file}")
        
        # 查询示例
        print(f"\n{'='*50}")
        print("查询示例:")
        for q in ["What is the main contribution?",
                  "How does the method work?",
                  "Compare with previous approaches"]:
            granularity, qtype, _ = scholar.query_granularity(q)
            print(f"  '{q}' → {qtype.value} → {granularity}")
        
        return
    
    # 批量处理模式
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"目录不存在: {input_dir}")
        return
    
    json_files = list(input_dir.glob("*_content_list_v2.json"))
    print(f"找到 {len(json_files)} 个文件")
    
    results = []
    for json_file in json_files:
        result = scholar.process_paper(str(json_file))
        results.append(result)
        print(f"✓ {result['paper_id']}: {sum(len(c) for c in result['granularity_results'].values())} chunks")
    
    # 保存批量结果
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "batch_results.json"
    
    # 简化结果
    simplified = [
        {
            "paper_id": r["paper_id"],
            "title": r["title"],
            "chunk_counts": {k: len(v) for k, v in r["granularity_results"].items()}
        }
        for r in results
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"批量处理完成: {len(results)} 篇论文")
    print(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
