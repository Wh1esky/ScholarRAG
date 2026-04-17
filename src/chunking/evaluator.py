"""
分块质量评估模块

功能：
1. 评估分块策略的有效性
2. 计算覆盖率、边界质量等指标
3. 与评估集（QA 对）对齐

参考 MoG 论文的评估方法
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


@dataclass
class ChunkQualityMetrics:
    """分块质量指标"""
    total_chunks: int
    avg_chunk_size: float           # 平均 chunk 大小（tokens）
    chunk_size_std: float          # chunk 大小标准差
    min_chunk_size: int           # 最小 chunk 大小
    max_chunk_size: int           # 最大 chunk 大小
    
    # 覆盖率指标
    text_coverage: float           # 文本覆盖率（保留的文本比例）
    section_coverage: Dict[str, float]  # 各章节覆盖率
    
    # 边界质量指标
    boundary_quality: float        # 边界质量分数
    
    # 检索相关指标
    retrieval_precision: float = 0.0   # 检索精度
    retrieval_recall: float = 0.0      # 检索召回率


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    retrieved_chunks: List[str]
    relevant_chunks: List[str]
    precision: float
    recall: float
    ndcg: float


class ChunkEvaluator:
    """
    分块质量评估器
    
    功能：
    1. 计算基础分块统计
    2. 评估分块对检索任务的影响
    3. 与 QA 评估集对齐
    """
    
    def __init__(self):
        self.metrics_history: List[ChunkQualityMetrics] = []
        self.retrieval_results: List[RetrievalResult] = []
    
    def compute_basic_metrics(self, chunks: List[Dict]) -> ChunkQualityMetrics:
        """
        计算基础分块指标
        
        Args:
            chunks: 分块列表
            
        Returns:
            ChunkQualityMetrics: 质量指标
        """
        if not chunks:
            return ChunkQualityMetrics(
                total_chunks=0,
                avg_chunk_size=0,
                chunk_size_std=0,
                min_chunk_size=0,
                max_chunk_size=0,
                text_coverage=0,
                section_coverage={},
                boundary_quality=0
            )
        
        # 计算 chunk 大小统计
        sizes = [c.get('token_count', len(c.get('content', '')) // 4) for c in chunks]
        
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_size = variance ** 0.5
        
        # 计算文本覆盖率
        total_chars = sum(len(c.get('content', '')) for c in chunks)
        # 假设原文长度（粗略估计）
        original_estimate = total_chars  # 简化：假设没有丢失
        
        # 计算各章节覆盖率
        section_coverage = {}
        sections = set(c.get('title', 'unknown') for c in chunks)
        for section in sections:
            section_chunks = [c for c in chunks if c.get('title') == section]
            section_total = sum(len(c.get('content', '')) for c in section_chunks)
            section_coverage[section] = section_total / original_estimate if original_estimate > 0 else 0
        
        # 计算边界质量（基于 chunk 大小的均匀性）
        # 理想情况下，chunk 大小应该均匀分布
        ideal_size = avg_size
        boundary_score = 1.0 - (std_size / (ideal_size * 2)) if ideal_size > 0 else 0
        boundary_score = max(0, min(1, boundary_score))
        
        return ChunkQualityMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
            chunk_size_std=std_size,
            min_chunk_size=min(sizes) if sizes else 0,
            max_chunk_size=max(sizes) if sizes else 0,
            text_coverage=1.0,  # 简化
            section_coverage=section_coverage,
            boundary_quality=boundary_score
        )
    
    def load_evaluation_dataset(self, dataset_path: str) -> List[Dict]:
        """
        加载评估数据集
        
        Args:
            dataset_path: 评估数据集路径
            
        Returns:
            List[Dict]: QA 对列表
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def evaluate_retrieval(self, chunks: List[Dict], 
                          questions: List[str],
                          ground_truth: List[List[str]]) -> List[RetrievalResult]:
        """
        评估检索性能
        
        Args:
            chunks: 分块列表
            questions: 问题列表
            ground_truth: 每个问题的相关 chunk IDs
            
        Returns:
            List[RetrievalResult]: 检索结果
        """
        results = []
        
        # 简单的检索评估（基于关键词匹配）
        for question, relevant_ids in zip(questions, ground_truth):
            # 模拟检索结果（实际应该用 embedding 模型）
            question_keywords = set(question.lower().split())
            
            # 计算每个 chunk 的相关性分数
            chunk_scores = []
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                chunk_keywords = set(content.split())
                overlap = len(question_keywords & chunk_keywords)
                chunk_scores.append((chunk.get('id'), overlap))
            
            # 排序并取 top-k
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            retrieved_ids = [c[0] for c in chunk_scores[:10]]
            
            # 计算指标
            relevant_set = set(relevant_ids)
            retrieved_set = set(retrieved_ids)
            
            intersection = relevant_set & retrieved_set
            precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
            recall = len(intersection) / len(relevant_set) if relevant_set else 0
            
            # 简化的 NDCG
            ndcg = recall  # 简化版本
            
            result = RetrievalResult(
                query=question,
                retrieved_chunks=retrieved_ids,
                relevant_chunks=list(relevant_ids),
                precision=precision,
                recall=recall,
                ndcg=ndcg
            )
            results.append(result)
        
        self.retrieval_results.extend(results)
        return results
    
    def compare_granularities(self, 
                             chunk_sets: Dict[str, List[Dict]]) -> Dict[str, ChunkQualityMetrics]:
        """
        比较不同粒度分块的质量
        
        Args:
            chunk_sets: 不同粒度的分块字典，格式 {granularity_name: chunks}
            
        Returns:
            Dict[str, ChunkQualityMetrics]: 各粒度的质量指标
        """
        results = {}
        for name, chunks in chunk_sets.items():
            metrics = self.compute_basic_metrics(chunks)
            results[name] = metrics
            self.metrics_history.append(metrics)
        
        return results
    
    def generate_report(self, metrics: ChunkQualityMetrics) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 质量指标
            
        Returns:
            str: 格式化的报告文本
        """
        report = []
        report.append("=" * 50)
        report.append("Chunking Quality Evaluation Report")
        report.append("=" * 50)
        report.append("")
        report.append("Basic Statistics:")
        report.append(f"  Total Chunks: {metrics.total_chunks}")
        report.append(f"  Average Chunk Size: {metrics.avg_chunk_size:.1f} tokens")
        report.append(f"  Chunk Size Std: {metrics.chunk_size_std:.1f}")
        report.append(f"  Min/Max Size: {metrics.min_chunk_size}/{metrics.max_chunk_size} tokens")
        report.append("")
        report.append("Coverage:")
        report.append(f"  Text Coverage: {metrics.text_coverage:.2%}")
        report.append("")
        report.append("Quality:")
        report.append(f"  Boundary Quality: {metrics.boundary_quality:.2%}")
        report.append("")
        
        if metrics.section_coverage:
            report.append("Section Coverage:")
            for section, coverage in sorted(metrics.section_coverage.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  {section}: {coverage:.2%}")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str):
        """保存评估结果"""
        results = {
            "metrics_history": [
                {
                    "total_chunks": m.total_chunks,
                    "avg_chunk_size": m.avg_chunk_size,
                    "boundary_quality": m.boundary_quality,
                    "text_coverage": m.text_coverage
                }
                for m in self.metrics_history
            ],
            "retrieval_results": [
                {
                    "query": r.query,
                    "precision": r.precision,
                    "recall": r.recall,
                    "ndcg": r.ndcg
                }
                for r in self.retrieval_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    """测试代码"""
    from src.chunking.unified_format import MinerUToUnifiedConverter
    from src.chunking.granularity_chunker import GranularityChunker, ChunkGranularity
    
    # 加载文档
    converter = MinerUToUnifiedConverter()
    doc = converter.convert_file("evaluation_set_100papers.json")
    
    # 不同粒度分块
    chunker = GranularityChunker()
    
    chunk_sets = {}
    for granularity in [ChunkGranularity.SENTENCE, ChunkGranularity.PARAGRAPH]:
        chunks = []
        for unified_chunk in doc.chunks[:20]:  # 只取前20个做测试
            result = chunker.chunk_text(
                unified_chunk.content,
                unified_chunk.id,
                granularity
            )
            chunks.extend([
                {"id": c.id, "content": c.text, "title": unified_chunk.title}
                for c in result
            ])
        chunk_sets[granularity.value] = chunks
    
    # 评估
    evaluator = ChunkEvaluator()
    results = evaluator.compare_granularities(chunk_sets)
    
    for name, metrics in results.items():
        print(f"\n=== {name.upper()} ===")
        print(evaluator.generate_report(metrics))


if __name__ == "__main__":
    main()
