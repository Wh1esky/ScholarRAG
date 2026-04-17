"""
demo.py - ScholarRAG 交互式演示

提供交互式的 RAG 问答界面

使用方法:
    # 完整模式 (需要 API Key)
    python src/rag/demo.py

    # 快速测试模式 (仅检索，不生成)
    python src/rag/demo.py --retrieval-only

    # 指定模型
    python src/rag/demo.py --model gpt-4 --top-k 5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.rag_pipeline import RAGPipeline, create_pipeline
from src.rag.answer_generator import GenerationMode
from src.retrieval.dense_retriever import DenseRetriever


def print_header():
    """打印欢迎信息"""
    print("\n" + "=" * 70)
    print("  ScholarRAG - 学术论文 RAG 问答系统")
    print("  Scholar Paper RAG Question Answering System")
    print("=" * 70)
    print()


def print_help():
    """打印帮助信息"""
    print("\n可用命令:")
    print("  help, h          - 显示帮助信息")
    print("  query, q          - 输入问题")
    print("  top-k <n>         - 设置 Top-K (默认: 5)")
    print("  mode <m>          - 设置生成模式 (concise/default/detailed)")
    print("  stats             - 显示统计信息")
    print("  reset             - 重置统计")
    print("  sources <on/off>  - 开启/关闭来源显示")
    print("  quit, exit, q     - 退出")
    print()


def check_setup() -> dict:
    """检查环境配置"""
    status = {
        "index_exists": False,
        "api_key_exists": False,
        "llm_model": None
    }

    # 检查索引
    output_dir = project_root / "src" / "embedding" / "output"
    index_path = output_dir / "unified_index.faiss"
    status["index_exists"] = index_path.exists()

    # 检查 API Key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    status["api_key_exists"] = bool(api_key)

    # 获取模型
    status["llm_model"] = os.environ.get("LLM_MODEL", "gpt-4")

    return status


def initialize_pipeline(
    retrieval_only: bool = False,
    model_name: Optional[str] = None,
    top_k: int = 5
) -> Optional[RAGPipeline]:
    """
    初始化 RAG Pipeline

    Args:
        retrieval_only: 是否仅使用检索模式
        model_name: LLM 模型名称
        top_k: Top-K 数

    Returns:
        RAGPipeline 或 None
    """
    from src.rag import LLMClient, PromptTemplate, AnswerGenerator

    output_dir = str(project_root / "src" / "embedding" / "output")

    # 初始化检索器
    try:
        retriever = DenseRetriever.from_output_dir(output_dir)
        print(f"  检索器初始化成功 (索引大小: {retriever.index.ntotal})")
    except Exception as e:
        print(f"  错误: 无法初始化检索器: {e}")
        return None

    # 仅检索模式
    if retrieval_only:
        print("  模式: 仅检索 (不生成答案)")
        class SimpleGenerator:
            def generate(self, query, contexts, **kwargs):
                class MockResult:
                    answer = "[检索模式已开启，请设置 OPENAI_API_KEY 来启用答案生成]"
                    total_tokens = 0
                    error = None
                return MockResult()

        class SimplePipeline:
            def __init__(self, retriever, generator, default_top_k):
                self.retriever = retriever
                self.generator = generator
                self.default_top_k = default_top_k
                self.stats = {"total_queries": 0}

            def answer(self, query, top_k=None, verbose=False, **kwargs):
                k = top_k or self.default_top_k
                results = self.retriever.retrieve(query, top_k=k)
                print(f"\n检索到 {len(results)} 个相关片段:\n")

                for i, r in enumerate(results, 1):
                    print(f"[{i}] Score: {r.score:.4f}")
                    print(f"    Paper: {r.paper_id}")
                    print(f"    Section: {r.section_type}")
                    print(f"    Text: {r.text[:200]}...")
                    print()

                class Answer:
                    answer = f"[检索到 {len(results)} 个相关片段]"
                    query = query
                    sources = [{
                        "rank": r.rank,
                        "paper_id": r.paper_id,
                        "text": r.text,
                        "score": r.score
                    } for r in results]
                    retrieval_metrics = {
                        "num_results": len(results),
                        "avg_score": sum(r.score for r in results) / len(results) if results else 0
                    }
                return Answer()

            def get_stats(self):
                return self.stats

        generator = SimpleGenerator()
        return SimplePipeline(retriever, generator, top_k)

    # 完整模式
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("  警告: 未设置 API Key，仅支持检索模式")
        print("  设置环境变量: export OPENAI_API_KEY=your_key")
        return initialize_pipeline(retrieval_only=True, top_k=top_k)

    try:
        model = model_name or os.environ.get("LLM_MODEL", "gpt-4")
        llm = LLMClient(model_name=model)
        template = PromptTemplate()
        generator = AnswerGenerator(llm, template)

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            default_top_k=top_k
        )
        print(f"  LLM 模型: {model}")
        print("  Pipeline 初始化成功")
        return pipeline

    except Exception as e:
        print(f"  错误: 无法初始化 Pipeline: {e}")
        return None


def load_sample_queries() -> List[str]:
    """加载示例问题"""
    return [
        "What is the main contribution of this paper?",
        "How does the proposed method work?",
        "What experimental results are reported?",
        "What datasets were used for evaluation?",
        "What are the limitations of the approach?",
        "How is the model architecture designed?",
        "What is the training procedure?",
        "How does this method compare to previous work?",
        "What are the key hyperparameters?",
        "What future work is suggested?"
    ]


def interactive_mode(pipeline: RAGPipeline, top_k: int = 5):
    """交互式问答模式"""
    print("\n" + "-" * 70)
    print("进入交互式问答模式")
    print("输入 'help' 查看可用命令，输入 'quit' 退出")
    print("-" * 70)

    show_sources = True
    mode = GenerationMode.DEFAULT
    mode_name = "default"

    while True:
        try:
            user_input = input("\n问题 (或命令): ").strip()

            if not user_input:
                continue

            # 解析命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用 ScholarRAG！再见！")
                break

            if user_input.lower() in ['help', 'h']:
                print_help()
                continue

            if user_input.lower() == 'stats':
                stats = pipeline.get_stats()
                print("\n统计信息:")
                print(f"  总问题数: {stats['total_queries']}")
                if stats['total_queries'] > 0:
                    print(f"  平均检索时间: {stats.get('avg_retrieval_time_ms', 0):.2f}ms")
                    print(f"  平均生成时间: {stats.get('avg_generation_time_ms', 0):.2f}ms")
                continue

            if user_input.lower() == 'reset':
                pipeline.reset_stats()
                print("统计已重置")
                continue

            if user_input.lower().startswith('top-k '):
                try:
                    new_k = int(user_input.split()[1])
                    top_k = new_k
                    print(f"Top-K 设置为: {top_k}")
                except:
                    print("无效的 Top-K 值")
                continue

            if user_input.lower().startswith('mode '):
                mode_str = user_input.split()[1].lower()
                if mode_str == 'concise':
                    mode = GenerationMode.CONCISE
                    mode_name = "简洁"
                elif mode_str == 'detailed':
                    mode = GenerationMode.DETAILED
                    mode_name = "详细"
                else:
                    mode = GenerationMode.DEFAULT
                    mode_name = "默认"
                print(f"生成模式设置为: {mode_name}")
                continue

            if user_input.lower().startswith('sources '):
                val = user_input.split()[1].lower()
                show_sources = val == 'on'
                print(f"来源显示: {'开启' if show_sources else '关闭'}")
                continue

            if user_input.lower() in ['sample', 'samples']:
                print("\n示例问题:")
                for i, q in enumerate(load_sample_queries(), 1):
                    print(f"  {i}. {q}")
                continue

            # 执行问答
            print("\n" + "-" * 70)
            result = pipeline.answer(
                user_input,
                top_k=top_k,
                mode=mode,
                verbose=False
            )

            print(f"\n答案:\n{result.answer}")

            if show_sources and result.sources:
                print(f"\n来源 ({len(result.sources)} 个):")
                for i, src in enumerate(result.sources[:3], 1):
                    paper = src.get('paper_id', 'Unknown')
                    section = src.get('section_type', 'Unknown')
                    score = src.get('score', 0)
                    text = src.get('text', '')[:100]
                    print(f"  [{i}] {paper} | {section} | Score: {score:.4f}")
                    print(f"      {text}...")

            if result.generation_metrics:
                tokens = result.generation_metrics.get('total_tokens', 0)
                if tokens:
                    print(f"\n(使用 {tokens} tokens)")

        except KeyboardInterrupt:
            print("\n\n使用 'quit' 命令退出")
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="ScholarRAG 交互式演示")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="仅使用检索模式，不生成答案")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM 模型名称")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K 检索数")
    parser.add_argument("--query", type=str, default=None,
                        help="单次查询")

    args = parser.parse_args()

    print_header()

    # 检查环境
    print("检查环境配置...")
    status = check_setup()

    if not status["index_exists"]:
        print("  警告: FAISS 索引不存在")
        print("  请先运行: python src/embedding/run_pipeline.py")
        print()

    if status["api_key_exists"]:
        print(f"  API Key: 已设置")
        print(f"  模型: {status['llm_model']}")
    else:
        print("  API Key: 未设置 (将仅支持检索模式)")
        args.retrieval_only = True

    print()

    # 初始化 Pipeline
    print("初始化 Pipeline...")
    pipeline = initialize_pipeline(
        retrieval_only=args.retrieval_only,
        model_name=args.model,
        top_k=args.top_k
    )

    if pipeline is None:
        print("\nPipeline 初始化失败")
        return

    print()

    # 单次查询模式
    if args.query:
        print(f"查询: {args.query}")
        print("-" * 70)
        result = pipeline.answer(args.query, top_k=args.top_k)
        print(f"\n答案:\n{result.answer}")

        if result.sources:
            print(f"\n来源:")
            for i, src in enumerate(result.sources[:3], 1):
                print(f"  [{i}] {src.get('paper_id', 'Unknown')}: {src.get('text', '')[:100]}...")
        return

    # 交互式模式
    interactive_mode(pipeline, top_k=args.top_k)


if __name__ == "__main__":
    main()
