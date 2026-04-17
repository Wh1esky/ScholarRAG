"""
PromptTemplate - Prompt 模板管理

提供多种 Prompt 模板用于 RAG 系统:
- 基础问答模板
- 带引用的问答模板
- 摘要生成模板
- 多轮对话模板

使用方法:
    from src.rag.prompt_template import PromptTemplate

    template = PromptTemplate()

    # 构建问答 prompt
    prompt = template.build_qa_prompt(
        query="What is the main contribution?",
        contexts=["Context 1...", "Context 2..."]
    )

    # 带引用的 prompt
    prompt = template.build_citation_prompt(
        query="Explain the method",
        contexts=["Context..."],
        citation_format="numeric"
    )
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class CitationFormat(Enum):
    """引用格式枚举"""
    NUMERIC = "numeric"      # [1], [2], [3]
    PARENTHETICAL = "parenthetical"  # (Author, 2024)
    SUPERSCRIPT = "superscript"  # ¹, ², ³


@dataclass
class RetrievalContext:
    """检索上下文"""
    text: str
    paper_id: str = ""
    chunk_id: str = ""
    section_type: str = ""
    score: float = 0.0
    granularity: str = ""
    citation: Optional[str] = None

    def __post_init__(self):
        if self.citation is None:
            self.citation = f"[{self.paper_id}]"


class PromptTemplate:
    """
    Prompt 模板管理类

    提供预定义的 Prompt 模板和灵活的构建方法
    """

    # 系统提示词 - 设定 AI 助手的角色
    SYSTEM_PROMPT = """You are a precise research assistant. Answer questions strictly based on the provided context from academic papers.

Rules:
1. ONLY use information explicitly stated in the provided context. Do NOT add facts, numbers, method names, or experimental details from your own knowledge.
2. For every key claim in your answer, indicate which [Source N] supports it.
3. If the context does not contain enough information to fully answer the question, clearly state: "Based on the provided context, [partial answer]. The context does not provide information about [missing aspect]."
4. Do NOT guess or infer specific values (e.g., learning rates, batch sizes, model names) unless they appear in the context.
5. Keep your answer concise and directly focused on the question.
6. If multiple sources are relevant, synthesize them but always cite.
"""

    EVIDENCE_EXTRACTION_SYSTEM_PROMPT = """You are an evidence extraction assistant for academic QA.

Your job is to identify only the most useful evidence snippets from the provided sources for answering the question.

Rules:
1. Select at most 4 evidence items.
2. Each evidence item must point to one [Source N].
3. Prefer evidence that directly answers the question, contains concrete facts, numbers, settings, results, motivations, or method details.
4. Do not invent evidence. If the source does not support the answer, do not include it.
5. Output ONLY valid JSON.

JSON format:
{"evidence": [{"source": 1, "quote": "short quoted or paraphrased evidence", "why": "why this supports the answer"}], "coverage": "high|medium|low"}
"""

    GROUNDED_ANSWER_SYSTEM_PROMPT = """You are a precise research assistant.

Answer the question using the extracted evidence provided.

Rules:
1. Always attempt to answer the question using the available evidence, even if coverage is partial. Synthesize relevant information from multiple sources when possible.
2. If the evidence only supports part of the question, answer the supported part thoroughly and briefly note what aspects lack direct evidence.
3. Every key claim must cite [Source N].
4. Do not introduce any fact that is not supported by the extracted evidence.
5. When the question asks for configurations, conditions, comparisons, limitations, or reasons, prioritize the most relevant details from the evidence.
6. Prefer concise, factual answers. When evidence provides related context, use it to construct the best possible answer.
7. Only state that evidence is insufficient if NONE of the provided evidence is even remotely relevant to the question.
"""

    # 简短回答系统提示
    SYSTEM_PROMPT_CONCISE = """You are a concise research assistant. Provide brief, accurate answers based only on the given context. Prefer short sentences and direct answers."""

    # 详细解释系统提示
    SYSTEM_PROMPT_DETAILED = """You are a detailed research assistant. Provide thorough explanations with examples when possible. Use structured format with headings, bullet points, and citations."""

    # 基础问答模板
    QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

    # 带引用的问答模板
    CITATION_QA_TEMPLATE = """Based on the following research papers, please answer the question.

References:
{context}

Question: {question}

Instructions:
- Answer based on the references provided
- Cite sources using [paper_id] notation
- If information is not in the references, state that clearly

Answer:"""

    # 带行号引用的模板
    LINE_CITATION_TEMPLATE = """Based on the following excerpts from research papers, answer the question.

{excerpts}

Question: {question}

Please provide an answer with inline citations [Source X] indicating which excerpt supports each claim."""

    # 摘要生成模板
    SUMMARIZATION_TEMPLATE = """Please summarize the following text in 2-3 sentences:

{text}

Summary:"""

    # 多文档综合模板
    MULTI_DOC_TEMPLATE = """You are analyzing multiple research papers on the same topic.

Papers:
{papers}

Task: {task}

Provide a comprehensive analysis that:
1. Identifies common themes and findings
2. Highlights differences and contradictions
3. Synthesizes the collective knowledge

Analysis:"""

    # 评估答案质量模板
    EVALUATION_TEMPLATE = """You are evaluating the quality of an answer to a research question.

Original Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Evaluate the generated answer based on:
1. Factual correctness (compared to reference)
2. Completeness
3. Clarity
4. Relevance to the question

Provide a score from 1-10 and brief feedback."""

    # RAGAS 评估模板
    RAGAS_FAITHFULNESS_TEMPLATE = """Given a question and answer, create one or more statements from the answer:

Question: {question}
Answer: {answer}

Then from each statement, assess whether it is fully entailed by the context (i.e., the statement can be inferred directly from the context without any additional information).

Format:
statements: [list of statements]
entailment_scores: [list of 0/1 scores where 1 means fully entailed]
final_score: [average of entailment scores]

Faithfulness assessment:"""

    RAGAS_ANSWER_RELEVANCE_TEMPLATE = """Evaluate the relevance of the answer to the question.

Question: {question}
Answer: {answer}

Consider:
- Does the answer address the core question?
- Are there irrelevant parts in the answer?
- Rate from 1 (not relevant) to 10 (highly relevant)

Provide a brief justification and score."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        citation_format: CitationFormat = CitationFormat.NUMERIC
    ):
        """
        初始化 PromptTemplate

        Args:
            system_prompt: 自定义系统提示词
            citation_format: 引用格式
        """
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.citation_format = citation_format

    def format_context(
        self,
        contexts: List[RetrievalContext],
        include_metadata: bool = True
    ) -> str:
        """
        格式化检索上下文

        Args:
            contexts: 检索结果列表
            include_metadata: 是否包含元数据

        Returns:
            格式化的上下文字符串
        """
        if not contexts:
            return "No relevant context found."

        formatted = []
        for i, ctx in enumerate(contexts, 1):
            if include_metadata:
                meta = f"[Source {i}] "
                if ctx.paper_id:
                    meta += f"Paper: {ctx.paper_id} | "
                if ctx.section_type:
                    meta += f"Section: {ctx.section_type} | "
                if ctx.granularity:
                    meta += f"Granularity: {ctx.granularity}"
                formatted.append(f"{meta}\n{ctx.text}")
            else:
                formatted.append(f"[{i}] {ctx.text}")

        return "\n\n".join(formatted)

    def format_excerpts(
        self,
        contexts: List[RetrievalContext]
    ) -> str:
        """
        格式化摘录用于引用

        Args:
            contexts: 检索结果列表

        Returns:
            格式化的摘录字符串
        """
        if not contexts:
            return "No relevant excerpts available."

        excerpts = []
        for i, ctx in enumerate(contexts, 1):
            excerpt = f"--- Excerpt {i} ---\n"
            excerpt += f"Paper: {ctx.paper_id or 'Unknown'}"
            if ctx.section_type:
                excerpt += f" | Section: {ctx.section_type}"
            excerpt += f"\n{ctx.text}"
            excerpts.append(excerpt)

        return "\n\n".join(excerpts)

    def build_qa_prompt(
        self,
        query: str,
        contexts: List[RetrievalContext],
        include_system: bool = True
    ) -> str:
        """
        构建问答 Prompt

        Args:
            query: 问题
            contexts: 检索上下文
            include_system: 是否包含系统提示

        Returns:
            完整的 Prompt
        """
        context_str = self.format_context(contexts, include_metadata=True)

        parts = []
        if include_system:
            parts.append(self.system_prompt)

        parts.append(f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:")

        return "\n\n".join(parts)

    def build_qa_messages(
        self,
        query: str,
        contexts: List[RetrievalContext]
    ) -> List[Dict[str, str]]:
        """
        构建消息列表格式的 Prompt (用于 Chat API)

        Args:
            query: 问题
            contexts: 检索上下文

        Returns:
            消息列表
        """
        context_str = self.format_context(contexts, include_metadata=True)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]

    def build_evidence_extraction_messages(
        self,
        query: str,
        contexts: List[RetrievalContext]
    ) -> List[Dict[str, str]]:
        """构建证据抽取 Prompt"""
        context_str = self.format_context(contexts, include_metadata=True)
        return [
            {"role": "system", "content": self.EVIDENCE_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nSources:\n{context_str}",
            },
        ]

    @staticmethod
    def format_extracted_evidence(evidence_items: List[Dict[str, Any]]) -> str:
        """格式化抽取后的证据列表"""
        if not evidence_items:
            return "No supporting evidence extracted."

        lines = []
        for item in evidence_items:
            source = item.get("source", "?")
            quote = item.get("quote", "").strip()
            why = item.get("why", "").strip()
            lines.append(f"[Source {source}] Evidence: {quote}\nWhy relevant: {why}")
        return "\n\n".join(lines)

    def build_grounded_answer_messages(
        self,
        query: str,
        contexts: List[RetrievalContext],
        evidence_items: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """构建基于抽取证据的最终回答 Prompt"""
        evidence_str = self.format_extracted_evidence(evidence_items)
        source_str = self.format_context(contexts, include_metadata=True)
        return [
            {"role": "system", "content": self.GROUNDED_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Extracted Evidence:\n{evidence_str}\n\n"
                    f"Original Sources (for reference only):\n{source_str}\n\n"
                    "Use all relevant evidence to construct the most complete answer possible. "
                    "Cite each source used. If only partial evidence is available, answer what you can and briefly note any gaps."
                ),
            },
        ]

    def build_citation_prompt(
        self,
        query: str,
        contexts: List[RetrievalContext],
        citation_format: Optional[CitationFormat] = None
    ) -> str:
        """
        构建带引用的 Prompt

        Args:
            query: 问题
            contexts: 检索上下文
            citation_format: 引用格式

        Returns:
            带引用的 Prompt
        """
        cf = citation_format or self.citation_format
        context_str = self.format_context(contexts, include_metadata=True)

        citation_instruction = {
            CitationFormat.NUMERIC: "Use [1], [2], etc. to cite sources",
            CitationFormat.PARENTHETICAL: "Use (Paper ID, Year) format for citations",
            CitationFormat.SUPERSCRIPT: "Use ¹, ², ³ for inline citations"
        }

        parts = [
            self.system_prompt,
            f"Context:\n{context_str}",
            f"Question: {query}",
            f"Instructions:\n- {citation_instruction[cf]}\n- Cite which source supports each claim"
        ]

        return "\n\n".join(parts)

    def build_summarization_prompt(
        self,
        text: str,
        max_length: int = 3
    ) -> str:
        """
        构建摘要 Prompt

        Args:
            text: 要摘要的文本
            max_length: 最大句子数

        Returns:
            摘要 Prompt
        """
        return self.SUMMARIZATION_TEMPLATE.format(text=text)

    def build_multi_doc_prompt(
        self,
        papers: List[Dict[str, Any]],
        task: str
    ) -> str:
        """
        构建多文档分析 Prompt

        Args:
            papers: 文档列表，每项包含 'id' 和 'content'
            task: 分析任务描述

        Returns:
            多文档 Prompt
        """
        papers_str = []
        for i, paper in enumerate(papers, 1):
            papers_str.append(f"[Paper {i}] {paper.get('id', 'Unknown')}\n{paper.get('content', '')}")

        return self.MULTI_DOC_TEMPLATE.format(
            papers="\n\n".join(papers_str),
            task=task
        )

    def build_ragas_faithfulness_prompt(
        self,
        question: str,
        answer: str
    ) -> str:
        """
        构建 RAGAS Faithfulness 评估 Prompt

        Args:
            question: 问题
            answer: 生成的答案

        Returns:
            评估 Prompt
        """
        return self.RAGAS_FAITHFULNESS_TEMPLATE.format(
            question=question,
            answer=answer
        )

    def build_ragas_relevance_prompt(
        self,
        question: str,
        answer: str
    ) -> str:
        """
        构建 RAGAS Answer Relevance 评估 Prompt

        Args:
            question: 问题
            answer: 生成的答案

        Returns:
            评估 Prompt
        """
        return self.RAGAS_ANSWER_RELEVANCE_TEMPLATE.format(
            question=question,
            answer=answer
        )

    def build_evaluation_prompt(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str
    ) -> str:
        """
        构建答案评估 Prompt

        Args:
            question: 问题
            reference_answer: 参考答案
            generated_answer: 待评估答案

        Returns:
            评估 Prompt
        """
        return self.EVALUATION_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_citation_format(self, fmt: CitationFormat):
        """设置引用格式"""
        self.citation_format = fmt


def create_template(
    style: str = "default",
    citation_format: str = "numeric"
) -> PromptTemplate:
    """
    创建 PromptTemplate 的工厂函数

    Args:
        style: 模板风格 ('default', 'concise', 'detailed')
        citation_format: 引用格式 ('numeric', 'parenthetical', 'superscript')

    Returns:
        PromptTemplate 实例
    """
    style_map = {
        "default": PromptTemplate.SYSTEM_PROMPT,
        "concise": PromptTemplate.SYSTEM_PROMPT_CONCISE,
        "detailed": PromptTemplate.SYSTEM_PROMPT_DETAILED,
    }

    citation_map = {
        "numeric": CitationFormat.NUMERIC,
        "parenthetical": CitationFormat.PARENTHETICAL,
        "superscript": CitationFormat.SUPERSCRIPT,
    }

    return PromptTemplate(
        system_prompt=style_map.get(style, PromptTemplate.SYSTEM_PROMPT),
        citation_format=citation_map.get(citation_format, CitationFormat.NUMERIC)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PromptTemplate 测试")
    print("=" * 60)

    # 创建模板
    template = PromptTemplate()

    # 测试上下文
    contexts = [
        RetrievalContext(
            text="The proposed method achieves 95% accuracy on benchmark datasets.",
            paper_id="paper_001",
            section_type="experiment",
            granularity="paragraph"
        ),
        RetrievalContext(
            text="We use a transformer-based architecture with 12 layers.",
            paper_id="paper_002",
            section_type="method",
            granularity="paragraph"
        )
    ]

    # 测试问答 Prompt
    print("\n1. 基础问答 Prompt:")
    print("-" * 40)
    qa_prompt = template.build_qa_prompt(
        query="What accuracy does the method achieve?",
        contexts=contexts
    )
    print(qa_prompt[:500] + "...")

    # 测试消息格式
    print("\n2. 消息列表格式:")
    print("-" * 40)
    messages = template.build_qa_messages(
        query="How is the model architecture designed?",
        contexts=contexts
    )
    for msg in messages:
        print(f"[{msg['role']}]: {msg['content'][:200]}...")

    # 测试带引用的 Prompt
    print("\n3. 带引用的 Prompt:")
    print("-" * 40)
    citation_prompt = template.build_citation_prompt(
        query="What is the model architecture?",
        contexts=contexts,
        citation_format=CitationFormat.NUMERIC
    )
    print(citation_prompt[:400] + "...")

    # 测试工厂函数
    print("\n4. 创建不同风格的模板:")
    print("-" * 40)
    concise_template = create_template(style="concise")
    detailed_template = create_template(style="detailed")
    print(f"Concise template: {concise_template.system_prompt[:50]}...")
    print(f"Detailed template: {detailed_template.system_prompt[:50]}...")
