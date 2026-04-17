"""
自适应路由模块

功能：
1. 根据查询类型选择最优的分块粒度
2. 参考 MoG 论文的 Router 思想（简化版）
3. 实现基于规则的路由策略

MoG 论文原始 Router：
- 使用 MLP 网络根据 query 预测各粒度的权重
- 使用 soft labels 训练

我们简化版：
- 使用基于规则的分类器判断查询类型
- 根据查询类型映射到最优粒度
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"              # 事实性问题 (What is X?)
    METHOD = "method"               # 方法描述 (How does X work?)
    COMPARISON = "comparison"        # 对比问题 (Compare X and Y)
    SUMMARY = "summary"             # 概述问题 (Summarize X)
    EXPERIMENTAL = "experimental"    # 实验设置 (What datasets were used?)
    DEFINITION = "definition"       # 定义问题 (What is X?)
    REASONING = "reasoning"         # 推理问题 (Why does X happen?)
    LIST = "list"                   # 列表问题 (What are the steps/parts?)


@dataclass
class QueryClassification:
    """查询分类结果"""
    query_type: QueryType
    confidence: float               # 置信度 0-1
    keywords: List[str]             # 匹配的关键词
    suggested_granularity: str      # 建议的分块粒度
    

# 查询类型到粒度的映射（参考 MoG 思想）
QUERY_TO_GRANULARITY = {
    QueryType.FACTUAL: ["sentence", "paragraph"],           # 细粒度找具体事实
    QueryType.METHOD: ["paragraph", "section"],             # 中等粒度理解方法
    QueryType.COMPARISON: ["section"],                       # 粗粒度获取完整对比
    QueryType.SUMMARY: ["section", "document"],             # 粗粒度获取概述
    QueryType.EXPERIMENTAL: ["paragraph", "section"],       # 中等粒度看实验设置
    QueryType.DEFINITION: ["paragraph"],                    # 中等粒度看定义
    QueryType.REASONING: ["paragraph", "section"],         # 中等粒度理解推理
    QueryType.LIST: ["paragraph"],                          # 中等粒度找列表
}


class AdaptiveRouter:
    """
    自适应路由（简化版 MoG Router）
    
    根据查询类型选择最优的分块粒度
    
    使用方法:
        router = AdaptiveRouter()
        classification = router.classify_query("What is the accuracy of model X?")
        print(f"Query type: {classification.query_type}")
        print(f"Suggested granularity: {classification.suggested_granularity}")
    """
    
    # 查询类型识别模式
    QUERY_PATTERNS = {
        QueryType.FACTUAL: [
            (r"\b(what\s+is|what\s+are|what\s+was|what\s+were)\b", 0.8),
            (r"\b(how\s+many|how\s+much)\b", 0.7),
            (r"\b(which)\b.*\b(result|accuracy|score|value)\b", 0.6),
            (r"\b(number\s+of|value\s+of|score\s+of|percentage|accuracy|f1|precision|recall)\b", 0.65),
        ],
        QueryType.METHOD: [
            (r"\b(how\s+does|how\s+do|how\s+can|how\s+will)\b", 0.8),
            (r"\b(what\s+is\s+the\s+(mechanism|approach|method|framework))\b", 0.7),
            (r"\b(describe\s+the|explain\s+the|illustrate)\b", 0.6),
            (r"\b(architecture|module|pipeline|workflow|component|algorithm)\b", 0.55),
        ],
        QueryType.COMPARISON: [
            (r"\b(compare|comparison|contrast|difference|between)\b", 0.9),
            (r"\b(similarity|same|unlike)\b", 0.6),
            (r"\b(better|worse|superior|inferior)\b", 0.5),
            (r"\b(compared\s+with|compared\s+to|versus|vs\.?|relative\s+to|outperform|baseline)\b", 0.85),
        ],
        QueryType.SUMMARY: [
            (r"\b(summarize|summary|overview|brief|concisely)\b", 0.9),
            (r"\b(give\s+me\s+a|provide\s+a)\s+(summary|brief)\b", 0.8),
            (r"\b(main\s+(point|idea|contribution))\b", 0.7),
        ],
        QueryType.EXPERIMENTAL: [
            (r"\b(dataset|datasets|benchmark|evaluation|experiment)\b", 0.8),
            (r"\b(what\s+(dataset|benchmark|experiment))\b", 0.7),
            (r"\b(training|test|validation)\s+(set|data|performance)\b", 0.6),
            (r"\b(table\s*\d+|hyper-?parameter|optimizer|learning\s+rate|batch\s+size|epochs?|configuration|setting|conditions?)\b", 0.92),
            (r"\b(specific\s+(conditions?|configurations?|settings?)|implemented|implementation\s+details)\b", 0.9),
        ],
        QueryType.DEFINITION: [
            (r"\b(define|definition|meaning|refers?\s+to)\b", 0.9),
            (r"\b(what\s+is\s+(a|an|the)?\s*[\w-]+\??$)", 0.5),  # 简单 what is 结尾
            (r"\b(principle|basis|concept)\b", 0.5),
        ],
        QueryType.REASONING: [
            (r"\b(why|reason|cause|because|therefore|thus|hence)\b", 0.9),
            (r"\b(explain\s+why|reason\s+for)\b", 0.8),
            (r"\b(how\s+come)\b", 0.7),
            (r"\b(challenge|limitation|drawback|issue|problem|bottleneck|difficulty|fail|failure|scaling)\b", 0.92),
        ],
        QueryType.LIST: [
            (r"\b(list|enumerate|step|procedure|pipeline|components?)\b", 0.8),
            (r"\b(what\s+are\s+the\s+(steps?|components?|parts?))\b", 0.7),
            (r"\b(name\s+the|tell\s+me\s+the)\b", 0.6),
        ],
    }

    TYPE_PRIORITIES = {
        QueryType.EXPERIMENTAL: 8,
        QueryType.REASONING: 7,
        QueryType.COMPARISON: 6,
        QueryType.LIST: 5,
        QueryType.DEFINITION: 4,
        QueryType.METHOD: 3,
        QueryType.FACTUAL: 2,
        QueryType.SUMMARY: 1,
    }
    
    # 粒度建议映射
    GRANULARITY_SUGGESTIONS = {
        QueryType.FACTUAL: "sentence, paragraph",
        QueryType.METHOD: "paragraph, section",
        QueryType.COMPARISON: "section",
        QueryType.SUMMARY: "section, document",
        QueryType.EXPERIMENTAL: "paragraph, section",
        QueryType.DEFINITION: "paragraph",
        QueryType.REASONING: "paragraph, section",
        QueryType.LIST: "paragraph",
    }
    
    def __init__(self):
        self.classification_history: List[QueryClassification] = []
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        分类查询类型
        
        Args:
            query: 用户查询
            
        Returns:
            QueryClassification: 分类结果
        """
        query_lower = query.lower()
        
        best_type = QueryType.FACTUAL  # 默认类型
        best_score = 0.0
        matched_keywords = []
        type_scores = {query_type: 0.0 for query_type in QueryType}
        type_matches = {query_type: [] for query_type in QueryType}
        
        for query_type, patterns in self.QUERY_PATTERNS.items():
            for pattern, weight in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    type_scores[query_type] += weight
                    type_matches[query_type].append(match.group())

        for query_type, score in type_scores.items():
            if not score:
                continue
            score += min(len(type_matches[query_type]), 3) * 0.08
            if (
                score > best_score or
                (
                    score == best_score and
                    self.TYPE_PRIORITIES[query_type] > self.TYPE_PRIORITIES[best_type]
                )
            ):
                best_score = score
                best_type = query_type
                matched_keywords = type_matches[query_type]
        
        # 如果没有匹配任何模式，使用默认策略
        if best_score == 0:
            # 根据查询长度和结构猜测
            word_count = len(query.split())
            if word_count < 10:
                best_type = QueryType.FACTUAL
            elif word_count < 25:
                best_type = QueryType.METHOD
            else:
                best_type = QueryType.SUMMARY
        
        if re.search(r"\b(table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration|specific\s+conditions?)\b", query_lower):
            best_type = QueryType.EXPERIMENTAL
            best_score = max(best_score, 1.1)
            matched_keywords.extend(re.findall(r"table\s*\d+|optimizer|learning\s+rate|batch\s+size|epochs?|hyper-?parameter|configuration|specific\s+conditions?", query_lower))
        elif re.search(r"\b(limitations?|challenges?|drawbacks?|issues?|bottleneck|difficulty|scaling)\b", query_lower):
            best_type = QueryType.REASONING
            best_score = max(best_score, 1.0)
            matched_keywords.extend(re.findall(r"limitations?|challenges?|drawbacks?|issues?|bottleneck|difficulty|scaling", query_lower))
        elif re.search(r"\b(compared\s+to|compared\s+with|versus|vs\.?|difference\s+between|baseline|outperform)\b", query_lower):
            best_type = QueryType.COMPARISON
            best_score = max(best_score, 1.0)
            matched_keywords.extend(re.findall(r"compared\s+to|compared\s+with|versus|vs\.?|difference\s+between|baseline|outperform", query_lower))

        # 计算最终置信度
        confidence = min(0.45 + best_score * 0.35, 1.0) if best_score > 0 else 0.5
        matched_keywords = list(dict.fromkeys(matched_keywords))
        
        result = QueryClassification(
            query_type=best_type,
            confidence=confidence,
            keywords=matched_keywords,
            suggested_granularity=self.GRANULARITY_SUGGESTIONS[best_type]
        )
        
        self.classification_history.append(result)
        return result
    
    def get_granularity_for_query(self, query: str) -> Tuple[str, float]:
        """
        获取查询对应的最优粒度
        
        Args:
            query: 用户查询
            
        Returns:
            Tuple[str, float]: (粒度建议, 置信度)
        """
        classification = self.classify_query(query)
        return classification.suggested_granularity, classification.confidence
    
    def batch_classify(self, queries: List[str]) -> List[QueryClassification]:
        """
        批量分类查询
        
        Args:
            queries: 查询列表
            
        Returns:
            List[QueryClassification]: 分类结果列表
        """
        return [self.classify_query(q) for q in queries]
    
    def get_statistics(self) -> Dict:
        """获取分类统计信息"""
        if not self.classification_history:
            return {"total_queries": 0}
        
        type_counts = {}
        for c in self.classification_history:
            type_name = c.query_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_queries": len(self.classification_history),
            "type_distribution": type_counts,
            "avg_confidence": sum(c.confidence for c in self.classification_history) / len(self.classification_history)
        }


def main():
    """测试代码"""
    router = AdaptiveRouter()
    
    test_queries = [
        "What is the accuracy of BERT on SQuAD?",
        "How does the attention mechanism work in transformers?",
        "Compare GPT-3 and GPT-4 in terms of model size",
        "Summarize the main contributions of this paper",
        "What datasets were used for evaluation?",
        "What is differential privacy?",
        "Why does dropout help prevent overfitting?",
        "What are the steps to fine-tune a language model?",
    ]
    
    print("=== Query Classification Results ===\n")
    for query in test_queries:
        result = router.classify_query(query)
        print(f"Query: {query}")
        print(f"  Type: {result.query_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Granularity: {result.suggested_granularity}")
        print()
    
    # 统计信息
    stats = router.get_statistics()
    print(f"\n=== Statistics ===")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Type distribution: {stats['type_distribution']}")


if __name__ == "__main__":
    main()
