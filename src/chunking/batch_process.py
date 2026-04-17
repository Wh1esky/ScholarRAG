"""
批量处理脚本

功能：
1. 批量转换所有 MinerU JSON 文件
2. 生成不同粒度的分块
3. 保存处理结果

使用方法:
    python src/chunking/batch_process.py
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# 添加项目根目录到路径
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chunking.unified_format import MinerUToUnifiedConverter, UnifiedDocument
from src.chunking.structure_recognizer import StructureRecognizer
from src.chunking.granularity_chunker import GranularityChunker, ChunkGranularity, ChunkConfig
from src.chunking.evaluator import ChunkEvaluator


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, 
                 parsed_dir: str = "parsed_pdf",
                 output_dir: str = "src/chunking/output"):
        self.parsed_dir = Path(parsed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.converter = MinerUToUnifiedConverter()
        self.chunker = GranularityChunker()
        self.evaluator = ChunkEvaluator()
        
    def process_single(self, json_file: Path) -> Optional[Dict]:
        """处理单个文件"""
        try:
            # 转换格式
            doc = self.converter.convert_file(str(json_file))
            
            # 结构识别
            recognizer = StructureRecognizer()
            sections = recognizer.recognize_from_chunks(doc.chunks)
            
            # 不同粒度分块
            chunk_sets = {}
            for granularity in ChunkGranularity:
                if granularity == ChunkGranularity.DOCUMENT:
                    continue  # 跳过 document 级（特殊处理）
                
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
                            "section_type": unified_chunk.title,
                            "token_count": c.token_count
                        }
                        for c in result
                    ])
                chunk_sets[granularity.value] = chunks
            
            return {
                "paper_id": doc.paper_id,
                "title": doc.title,
                "abstract": doc.abstract,
                "total_chunks": len(doc.chunks),
                "sections": recognizer.get_section_info(),
                "chunk_sets": {k: len(v) for k, v in chunk_sets.items()},
                "quality_metrics": self.evaluator.compute_basic_metrics(
                    [{"content": c["text"], "title": c["section_type"]} for c in chunk_sets.get("paragraph", [])]
                ).__dict__
            }
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            return None
    
    def process_all(self) -> List[Dict]:
        """批量处理所有文件"""
        json_files = list(self.parsed_dir.glob("*_content_list_v2.json"))
        print(f"Found {len(json_files)} JSON files")
        
        results = []
        for json_file in tqdm(json_files, desc="Processing papers"):
            result = self.process_single(json_file)
            if result:
                results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], filename: str = "batch_processing_results.json"):
        """保存处理结果"""
        output_path = self.output_dir / filename
        
        # 简化 metrics（移除 dataclass）
        simplified_results = []
        for r in results:
            r_copy = r.copy()
            if "quality_metrics" in r_copy:
                del r_copy["quality_metrics"]
            simplified_results.append(r_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # 保存统计摘要
        self.save_summary(results)
    
    def save_summary(self, results: List[Dict]):
        """保存统计摘要"""
        summary = {
            "total_papers": len(results),
            "papers_with_abstract": sum(1 for r in results if r.get("abstract")),
            "total_chunks_distribution": {},
            "granularity_distribution": {},
        }
        
        # 统计分块分布
        granularity_counts = {"sentence": 0, "paragraph": 0, "section": 0}
        for r in results:
            for gran, count in r.get("chunk_sets", {}).items():
                if gran in granularity_counts:
                    granularity_counts[gran] += count
        
        summary["granularity_distribution"] = granularity_counts
        
        # 各粒度平均 chunk 数
        avg_chunks = {}
        for gran in granularity_counts:
            if granularity_counts[gran] > 0:
                avg_chunks[gran] = granularity_counts[gran] / len(results)
        
        summary["avg_chunks_per_paper"] = avg_chunks
        
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Summary saved to {summary_path}")
        
        # 打印摘要
        print("\n" + "=" * 50)
        print("Batch Processing Summary")
        print("=" * 50)
        print(f"Total Papers: {summary['total_papers']}")
        print(f"Papers with Abstract: {summary['papers_with_abstract']}")
        print(f"\nGranularity Distribution (total chunks):")
        for gran, count in granularity_counts.items():
            print(f"  {gran}: {count} chunks ({avg_chunks.get(gran, 0):.1f} avg per paper)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process parsed PDFs")
    parser.add_argument("--parsed-dir", default="parsed_pdf", help="Directory with parsed PDFs")
    parser.add_argument("--output-dir", default="src/chunking/output", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir
    )
    
    results = processor.process_all()
    
    if args.limit:
        results = results[:args.limit]
    
    processor.save_results(results)


if __name__ == "__main__":
    main()
