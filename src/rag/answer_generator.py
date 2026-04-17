"""
AnswerGenerator - 答案生成器

负责根据检索结果和 Query 生成最终答案

功能:
- 调用 LLM 生成答案
- 支持多种生成模式 (简洁/详细)
- 处理答案后处理 (格式化、清洗)
- 支持流式输出

使用方法:
    from src.rag import AnswerGenerator, LLMClient, PromptTemplate

    llm = LLMClient(model_name="gpt-4")
    template = PromptTemplate()
    generator = AnswerGenerator(llm, template)

    answer = generator.generate(
        query="What is RAG?",
        contexts=[RetrievalContext(...)],
        temperature=0.7
    )
"""

import re
import time
from typing import List, Optional, Dict, Any, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum

from .llm_client import LLMClient, LLMResponse
from .prompt_template import PromptTemplate, RetrievalContext


class GenerationMode(Enum):
    """生成模式"""
    CONCISE = "concise"
    DEFAULT = "default"
    DETAILED = "detailed"


@dataclass
class GenerationResult:
    """生成结果"""
    answer: str
    prompt: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None

    # 元信息
    contexts_used: int = 0
    mode: str = "default"

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.total_tokens = self.usage.get('total_tokens', 0)
            self.prompt_tokens = self.usage.get('prompt_tokens', 0)
            self.completion_tokens = self.usage.get('completion_tokens', 0)


class AnswerGenerator:
    """
    答案生成器

    将检索结果转换为自然语言答案
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: PromptTemplate,
        default_temperature: float = 0.7,
        default_max_tokens: int = 2048
    ):
        """
        初始化 AnswerGenerator

        Args:
            llm_client: LLM 客户端
            prompt_template: Prompt 模板
            default_temperature: 默认温度
            default_max_tokens: 默认最大 token 数
        """
        self.llm = llm_client
        self.template = prompt_template
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def generate(
        self,
        query: str,
        contexts: List[RetrievalContext],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        mode: GenerationMode = GenerationMode.DEFAULT,
        return_prompt: bool = False
    ) -> GenerationResult:
        """
        生成答案

        Args:
            query: 问题
            contexts: 检索上下文列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            mode: 生成模式
            return_prompt: 是否返回 prompt

        Returns:
            GenerationResult: 生成结果
        """
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # 设置生成模式
        if mode == GenerationMode.CONCISE:
            system_prompt = PromptTemplate.SYSTEM_PROMPT_CONCISE
        elif mode == GenerationMode.DETAILED:
            system_prompt = PromptTemplate.SYSTEM_PROMPT_DETAILED
        else:
            system_prompt = PromptTemplate.SYSTEM_PROMPT

        # 构建 Chat 消息列表（正确使用 system/user role）
        messages = self.template.build_qa_messages(
            query=query,
            contexts=contexts,
        )
        # 用当前模式的 system prompt 替换默认的
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt

        # 调用 LLM（优先使用 chat_generate 以正确传递 system role）
        start_time = time.time()
        response = self.llm.chat_generate(
            messages=messages,
            temperature=temp,
            max_tokens=tokens
        )
        latency_ms = (time.time() - start_time) * 1000

        # 处理响应
        if response.error:
            return GenerationResult(
                answer="",
                prompt=str(messages) if return_prompt else "",
                model=self.llm.model_name,
                error=response.error,
                contexts_used=len(contexts),
                mode=mode.value
            )

        # 后处理答案
        answer = self._post_process(response.content)

        result = GenerationResult(
            answer=answer,
            prompt=str(messages) if return_prompt else "",
            model=self.llm.model_name,
            usage=response.usage,
            latency_ms=latency_ms + response.latency_ms,
            contexts_used=len(contexts),
            mode=mode.value
        )

        return result

    def generate_with_citation(
        self,
        query: str,
        contexts: List[RetrievalContext],
        citation_format: str = "numeric",
        **kwargs
    ) -> GenerationResult:
        """
        生成带引用的答案

        Args:
            query: 问题
            contexts: 检索上下文列表
            citation_format: 引用格式
            **kwargs: 其他参数

        Returns:
            GenerationResult: 生成结果
        """
        from .prompt_template import CitationFormat

        format_map = {
            "numeric": CitationFormat.NUMERIC,
            "parenthetical": CitationFormat.PARENTHETICAL,
            "superscript": CitationFormat.SUPERSCRIPT,
        }

        cf = format_map.get(citation_format, CitationFormat.NUMERIC)
        prompt = self.template.build_citation_prompt(
            query=query,
            contexts=contexts,
            citation_format=cf
        )

        start_time = time.time()
        response = self.llm.generate(
            prompt=prompt,
            temperature=kwargs.get('temperature', self.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.default_max_tokens)
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.error:
            return GenerationResult(
                answer="",
                prompt=prompt,
                model=self.llm.model_name,
                error=response.error,
                contexts_used=len(contexts)
            )

        answer = self._post_process(response.content)

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            model=self.llm.model_name,
            usage=response.usage,
            latency_ms=latency_ms + response.latency_ms,
            contexts_used=len(contexts)
        )

    def generate_stream(
        self,
        query: str,
        contexts: List[RetrievalContext],
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成答案

        Args:
            query: 问题
            contexts: 检索上下文列表
            callback: 每个 token 的回调函数
            **kwargs: 其他参数

        Yields:
            str: 生成的文本片段
        """
        prompt = self.template.build_qa_prompt(
            query=query,
            contexts=contexts,
            include_system=True
        )

        # 注意: 需要 LLM 客户端支持流式输出
        # 这里提供一个通用的实现框架
        response = self.llm.generate(
            prompt=prompt,
            temperature=kwargs.get('temperature', self.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.default_max_tokens),
            stream=True if hasattr(self.llm, 'generate_stream') else False
        )

        # 如果客户端支持流式输出
        if hasattr(self.llm, 'generate_stream'):
            for chunk in self.llm.generate_stream(prompt=prompt, **kwargs):
                if callback:
                    callback(chunk)
                yield chunk
        else:
            # 回退到非流式
            result = self.generate(query=query, contexts=contexts, **kwargs)
            yield result.answer

    def batch_generate(
        self,
        queries: List[str],
        contexts_list: List[List[RetrievalContext]],
        **kwargs
    ) -> List[GenerationResult]:
        """
        批量生成答案

        Args:
            queries: 问题列表
            contexts_list: 每个问题的上下文列表
            **kwargs: 其他参数

        Returns:
            List[GenerationResult]: 生成结果列表
        """
        results = []

        for query, contexts in zip(queries, contexts_list):
            result = self.generate(query=query, contexts=contexts, **kwargs)
            results.append(result)

        return results

    def _post_process(self, text: str) -> str:
        """
        后处理生成的文本

        Args:
            text: 原始生成文本

        Returns:
            处理后的文本
        """
        if not text:
            return ""

        # 移除多余的空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # 移除可能的 prompt 泄露
        if "Question:" in text:
            parts = text.split("Question:")
            if len(parts) > 1:
                text = parts[0].strip()

        # 规范化引号
        text = text.replace("""", '"').replace(""", '"')
        text = text.replace("''", "'").replace("'", "'")

        return text.strip()

    def _validate_contexts(self, contexts: List[RetrievalContext]) -> bool:
        """验证上下文是否有效"""
        if not contexts:
            return False

        # 检查是否有有效文本
        valid_count = sum(1 for ctx in contexts if ctx.text and ctx.text.strip())
        return valid_count > 0


class CitationAnswerGenerator(AnswerGenerator):
    """
    带引用的答案生成器

    继承自 AnswerGenerator，专注于生成带精确引用的答案
    """

    def generate(
        self,
        query: str,
        contexts: List[RetrievalContext],
        **kwargs
    ) -> GenerationResult:
        """
        生成带引用的答案

        Args:
            query: 问题
            contexts: 检索上下文列表
            **kwargs: 其他参数

        Returns:
            GenerationResult: 生成结果
        """
        return self.generate_with_citation(
            query=query,
            contexts=contexts,
            citation_format=kwargs.get('citation_format', 'numeric'),
            temperature=kwargs.get('temperature', self.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.default_max_tokens)
        )

    def format_with_inline_citations(
        self,
        answer: str,
        contexts: List[RetrievalContext]
    ) -> str:
        """
        格式化答案中的内联引用

        Args:
            answer: 原始答案
            contexts: 上下文列表

        Returns:
            带内联引用的答案
        """
        # 创建引用映射
        citation_map = {}
        for i, ctx in enumerate(contexts, 1):
            if ctx.paper_id:
                citation_map[f"[{i}]"] = f"[{ctx.paper_id}]"
            else:
                citation_map[f"[{i}]"] = f"[Source {i}]"

        # 替换引用
        formatted = answer
        for old, new in citation_map.items():
            formatted = formatted.replace(old, new)

        return formatted


def create_generator(
    model_name: str = "gpt-4",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    style: str = "default",
    **kwargs
) -> AnswerGenerator:
    """
    创建 AnswerGenerator 的工厂函数

    Args:
        model_name: 模型名称
        provider: Provider
        api_key: API Key
        style: 生成风格 ('concise', 'default', 'detailed')
        **kwargs: 其他参数

    Returns:
        AnswerGenerator 实例
    """
    from .llm_client import create_llm_client

    llm = create_llm_client(
        model_name=model_name,
        provider=provider,
        api_key=api_key
    )

    template = PromptTemplate()
    if style == "concise":
        template.set_system_prompt(PromptTemplate.SYSTEM_PROMPT_CONCISE)
    elif style == "detailed":
        template.set_system_prompt(PromptTemplate.SYSTEM_PROMPT_DETAILED)

    return AnswerGenerator(
        llm_client=llm,
        prompt_template=template,
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 60)
    print("AnswerGenerator 测试")
    print("=" * 60)

    # 创建测试上下文
    from src.rag.prompt_template import RetrievalContext

    test_contexts = [
        RetrievalContext(
            text="The proposed method achieves 95% accuracy on the benchmark dataset, outperforming existing approaches by 5%.",
            paper_id="paper_001",
            section_type="experiment",
            granularity="paragraph",
            score=0.92
        ),
        RetrievalContext(
            text="We use a transformer-based architecture with 12 layers and 768 hidden dimensions. The model is trained for 100 epochs with batch size 32.",
            paper_id="paper_002",
            section_type="method",
            granularity="paragraph",
            score=0.88
        ),
        RetrievalContext(
            text="Experimental results show significant improvements on multiple datasets including ImageNet, COCO, and LVIS.",
            paper_id="paper_001",
            section_type="experiment",
            granularity="paragraph",
            score=0.85
        )
    ]

    print("\n测试上下文:")
    for i, ctx in enumerate(test_contexts):
        print(f"  [{i+1}] {ctx.paper_id} - {ctx.section_type}: {ctx.text[:60]}...")

    # 测试生成器 (需要 API Key)
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        print("\n注意: 未设置 OPENAI_API_KEY，跳过实际生成测试")
        print("可以手动测试:")
        print("""
from src.rag import AnswerGenerator, LLMClient, PromptTemplate
from src.rag.prompt_template import RetrievalContext

llm = LLMClient(model_name="gpt-4")
template = PromptTemplate()
generator = AnswerGenerator(llm, template)

result = generator.generate(
    query="What accuracy does the method achieve?",
    contexts=test_contexts
)
print(result.answer)
        """)
    else:
        from src.rag import LLMClient, PromptTemplate

        llm = LLMClient(model_name="gpt-4")
        template = PromptTemplate()
        generator = AnswerGenerator(llm, template)

        print("\n生成答案 (default mode):")
        print("-" * 40)
        result = generator.generate(
            query="What accuracy does the method achieve and what architecture is used?",
            contexts=test_contexts
        )
        print(result.answer)
        print(f"\n使用 Token: {result.total_tokens}")
        print(f"延迟: {result.latency_ms:.2f}ms")
