"""
RAG Module - 检索增强生成模块

核心组件:
- LLMClient: 大语言模型调用封装
- PromptTemplate: Prompt 模板管理
- AnswerGenerator: 答案生成器
- RAGPipeline: 核心 RAG 流程

使用方法:
    from src.rag import RAGPipeline, LLMClient, PromptTemplate, AnswerGenerator

    # 方式 1: 手动初始化
    retriever = DenseRetriever.from_output_dir("src/embedding/output")
    llm = LLMClient(model_name="gpt-4")
    template = PromptTemplate()
    generator = AnswerGenerator(llm, template)
    pipeline = RAGPipeline(retriever, generator)

    # 方式 2: 使用工厂函数
    pipeline = create_pipeline(embedding_output_dir="src/embedding/output")
"""

from .llm_client import LLMClient, LLMResponse, create_llm_client
from .prompt_template import PromptTemplate, RetrievalContext, CitationFormat, create_template
from .answer_generator import AnswerGenerator, GenerationResult, GenerationMode, create_generator
from .rag_pipeline import RAGPipeline, RAGAnswer, create_pipeline

__all__ = [
    # LLM Client
    'LLMClient',
    'LLMResponse',
    'create_llm_client',
    # Prompt Template
    'PromptTemplate',
    'RetrievalContext',
    'CitationFormat',
    'create_template',
    # Answer Generator
    'AnswerGenerator',
    'GenerationResult',
    'GenerationMode',
    'create_generator',
    # RAG Pipeline
    'RAGPipeline',
    'RAGAnswer',
    'create_pipeline',
]
