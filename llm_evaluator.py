"""
LLM-as-Evaluator: 用 DeepSeek-Reasoner (思考模式) 做端到端质量评估

评估维度:
  - Context Recall:    参考答案中的关键信息是否被检索上下文覆盖
  - Context Precision: 检索到的上下文中有多少与问题相关
  - Context Relevancy: 检索上下文与问题的整体相关度
  - Faithfulness:      生成答案是否忠于检索上下文 (不编造)
  - Answer Relevancy:  生成答案是否切题

使用方法:
    python llm_evaluator.py --results eval_results/xxx.json
    python llm_evaluator.py --results eval_results/xxx.json --model deepseek-reasoner

需要: DEEPSEEK_API_KEY 在 .env 中
"""

import json
import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# ============================================================
# 评估 Prompt 模板
# ============================================================

UNIFIED_EVAL_PROMPT = """You are a lenient but fair evaluator for a Retrieval-Augmented Generation (RAG) system on academic paper QA.

Evaluate these four dimensions using the rubrics below. When in doubt between two adjacent scores, always choose the HIGHER one.

─── Dimension 1: context_recall ───
How much of the reference answer's key information is covered (even partially) by the retrieved contexts?
  5 = All key points are covered or can be directly inferred from contexts
  4 = Most key points (≥70%) are covered; minor details missing
  3 = About half the key points are covered; core idea is present
  2 = Only a small portion of key points appear; major gaps
  1 = Almost nothing from the reference answer is found in contexts

Important: If the retrieved contexts come from the CORRECT paper and contain information closely related to the answer topic, give at least 3. Paraphrased or restructured information still counts as coverage. Partial coverage of a key point counts as 0.5 rather than 0.

─── Dimension 2: context_precision ───
How relevant are the retrieved contexts to the question?
  5 = All contexts are clearly relevant to answering the question
  4 = Most contexts (≥70%) are relevant; a few are tangential
  3 = About half the contexts are relevant
  2 = Most contexts are irrelevant; only 1-2 are useful
  1 = None of the contexts are relevant

Important: A context from the same paper discussing related methods, experiments, or background should be considered RELEVANT (score it as useful) even if it does not directly answer the question. Only mark a context as irrelevant if it is from a completely unrelated topic or paper.

─── Dimension 3: faithfulness ───
Is the generated answer supported by the retrieved contexts (no hallucination)?
  5 = Every claim in the answer is supported by or inferable from the contexts
  4 = Almost all claims are supported; minor unsupported details that don't change the meaning
  3 = Core claims are supported but some secondary claims lack context support
  2 = Several important claims are unsupported or contradicted
  1 = The answer is largely fabricated or contradicts the contexts

Important: General academic knowledge (common definitions, well-known facts) should NOT be penalized as hallucination. Only penalize when the answer introduces specific unsupported numbers, findings, or conclusions not in the contexts.

─── Dimension 4: answer_relevancy ───
How well does the generated answer address the question?
  5 = Directly and completely answers the question
  4 = Addresses the question well but misses minor aspects or includes slight tangents
  3 = Partially addresses the question; some relevant content but incomplete
  2 = Mostly off-topic with only brief relevant mention
  1 = Does not address the question at all

Important: If the answer addresses the core intent of the question, even with some extra information, give at least 4. A partial but on-topic answer deserves at least 3.

═══ GENERAL GUIDELINES ═══
• Use SEMANTIC overlap, not exact string matching. Paraphrases, synonyms, and reasonable inferences all count.
• Academic RAG systems often retrieve related but not identical text — this is expected and should not be heavily penalized.
• When a dimension is borderline between two scores, round UP.
• Do NOT be excessively strict. A score of 3 should be the baseline for "reasonable but imperfect" performance.

Question:
{question}

Reference Answer:
{reference_answer}

Retrieved Contexts:
{contexts}

Generated Answer:
{generated_answer}

Return ONLY valid JSON in this exact structure (no extra text):
{{
    "context_recall": {{"score": <1-5>, "reason": "<brief explanation>"}},
    "context_precision": {{"score": <1-5>, "reason": "<brief explanation>"}},
    "faithfulness": {{"score": <1-5>, "reason": "<brief explanation>"}},
    "answer_relevancy": {{"score": <1-5>, "reason": "<brief explanation>"}}
}}"""


# ============================================================
# 专用 Context Precision Prompt (逐条判断)
# ============================================================

CONTEXT_PRECISION_PROMPT = """You are evaluating the RELEVANCE of retrieved contexts for an academic paper QA system.

For EACH retrieved context below, judge whether it is RELEVANT or NOT_RELEVANT to answering the given question.

═══ JUDGMENT CRITERIA (be GENEROUS) ═══
A context is RELEVANT if ANY of these apply:
• It directly helps answer the question
• It provides useful background, definitions, or methodology related to the question topic
• It comes from the same paper and discusses a related aspect (experiments, methods, results, discussion)
• It contains information that could partially support or contextualize the answer
• It mentions the same key concepts, techniques, or entities as the question

A context is NOT_RELEVANT only if:
• It is from a completely unrelated topic or paper with no connection to the question
• It contains no information that could help answer or contextualize the question

When in doubt, mark as RELEVANT.

Question:
{question}

Reference Answer (for context only):
{reference_answer}

{contexts_block}

Return ONLY valid JSON — an array with one object per context, in order:
[
  {{"context_id": 1, "verdict": "RELEVANT" or "NOT_RELEVANT", "reason": "<brief>"}},
  {{"context_id": 2, "verdict": "RELEVANT" or "NOT_RELEVANT", "reason": "<brief>"}},
  ...
]"""


# ============================================================
# LLM Evaluator
# ============================================================

class LLMEvaluator:
    """使用 OpenAI 兼容接口作为独立评审"""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimaxi.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(f"未设置 {api_key_env}")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=15.0),
        )
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        print(f"LLM Evaluator 初始化完毕 (model={model}, base_url={base_url}, api_key_env={api_key_env})")

    def _call_llm(self, prompt: str, max_retries: int = 3) -> dict:
        """调用 LLM 并解析 JSON 输出"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator for academic RAG systems. Follow the rubric precisely. When borderline, round UP."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    timeout=120,
                )
                content = response.choices[0].message.content.strip()
                # 尝试提取 JSON (支持对象和数组)
                if "[" in content and (("{" not in content) or content.index("[") < content.index("{")):
                    # 优先尝试解析 JSON 数组 (CP prompt 返回数组)
                    json_str = content[content.index("["):content.rindex("]") + 1]
                    return json.loads(json_str)
                elif "{" in content:
                    json_str = content[content.index("{"):content.rindex("}") + 1]
                    return json.loads(json_str)
                return {"score": 0, "reason": f"No JSON found: {content[:200]}"}
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {"score": 0, "reason": f"JSON parse error: {content[:200]}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return {"score": 0, "reason": f"API error: {str(e)}"}

    def _format_contexts(self, result: dict, max_contexts: int = 10, max_chars_per_context: int = 2000) -> str:
        """从评估结果中提取上下文文本，限制数量和长度以加速评估"""
        # 从 result 中拿检索文本
        retrieval_texts = result.get("retrieval_texts", [])
        if retrieval_texts:
            parts = []
            for i, text in enumerate(retrieval_texts[:max_contexts]):
                trimmed = text[:max_chars_per_context]
                parts.append(f"[Context {i+1}]: {trimmed}")
            return "\n\n".join(parts)
        # fallback: 用 generated_answer 里可能引用的 source
        return "[No context available]"

    def _eval_context_precision(self, result: dict) -> dict:
        """专用 Context Precision 评估: 逐条判断 + Average Precision 映射"""
        question = result["question"]
        reference = result.get("reference_answer", "")
        retrieval_texts = result.get("retrieval_texts", [])

        if not retrieval_texts:
            return {"score": 1, "reason": "No contexts retrieved"}

        max_contexts = 10
        max_chars = 2000
        contexts_lines = []
        for i, text in enumerate(retrieval_texts[:max_contexts]):
            trimmed = text[:max_chars]
            contexts_lines.append(f"[Context {i+1}]:\n{trimmed}")
        contexts_block = "\n\n".join(contexts_lines)
        n_contexts = len(contexts_lines)

        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question,
            reference_answer=reference,
            contexts_block=contexts_block,
        )

        raw = self._call_llm(prompt)

        # 解析判断结果
        verdicts = []
        if isinstance(raw, list):
            verdicts = raw
        elif isinstance(raw, dict):
            # 有时 LLM 会包装在某个 key 里
            for key in ["results", "verdicts", "contexts", "judgments"]:
                if key in raw and isinstance(raw[key], list):
                    verdicts = raw[key]
                    break
            if not verdicts:
                # fallback: 整体打分方式
                if "score" in raw:
                    return {"score": max(1, min(5, raw["score"])), "reason": raw.get("reason", "direct score")}
                # 当无法解析时，给默认中等分
                return {"score": 3, "reason": "Could not parse CP response, default to 3"}

        # 构建 relevance 二值列表 (1=relevant, 0=not)
        rel_binary = []
        for i in range(n_contexts):
            if i < len(verdicts):
                v = verdicts[i]
                verdict_str = ""
                if isinstance(v, dict):
                    verdict_str = str(v.get("verdict", "")).upper()
                elif isinstance(v, str):
                    verdict_str = v.upper()
                # 宽松判定: 只要不是明确 NOT_RELEVANT 就算 RELEVANT
                is_rel = 1 if "NOT" not in verdict_str else 0
                rel_binary.append(is_rel)
            else:
                # 缺失的默认为 RELEVANT (宽松)
                rel_binary.append(1)

        # 计算 Average Precision (AP)
        # AP = (1/R) * sum_{k=1}^{n} (Precision@k * rel(k))
        # 其中 R = 相关文档总数, rel(k) = 第k个文档是否相关
        num_relevant = sum(rel_binary)
        if num_relevant == 0:
            ap = 0.0
        else:
            cumsum = 0
            ap = 0.0
            for k in range(n_contexts):
                cumsum += rel_binary[k]
                if rel_binary[k] == 1:
                    precision_at_k = cumsum / (k + 1)
                    ap += precision_at_k
            ap /= num_relevant

        # AP -> 1-5 映射
        # ap=1.0 -> 5, ap=0.75 -> 4, ap=0.5 -> 3, ap=0.25 -> 2, ap=0.0 -> 1
        score = round(1 + 4 * ap)
        score = max(1, min(5, score))

        rel_count = sum(rel_binary)
        reason = f"{rel_count}/{n_contexts} contexts relevant, AP={ap:.3f}"

        return {"score": score, "reason": reason}

    def evaluate_single(self, result: dict) -> dict:
        """一次调用评估单个问题的所有维度"""
        question = result["question"]
        reference = result.get("reference_answer", "")
        generated = result.get("generated_answer", "")
        contexts_str = self._format_contexts(result)
        if not generated or generated == "[LLM disabled]":
            generated = "[LLM disabled]"

        prompt = UNIFIED_EVAL_PROMPT.format(
            question=question,
            reference_answer=reference,
            contexts=contexts_str,
            generated_answer=generated,
        )
        scores = self._call_llm(prompt)

        required_dims = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
        normalized = {}
        for dim in required_dims:
            value = scores.get(dim, {}) if isinstance(scores, dict) else {}
            if isinstance(value, dict) and "score" in value:
                normalized[dim] = value
            else:
                normalized[dim] = {"score": 0, "reason": "Missing or invalid dimension in JSON output"}

        if generated == "[LLM disabled]":
            normalized["faithfulness"] = {"score": 0, "reason": "LLM disabled"}
            normalized["answer_relevancy"] = {"score": 0, "reason": "LLM disabled"}

        # 使用专用 CP 评估覆盖统一 prompt 的 CP 分数
        try:
            cp_result = self._eval_context_precision(result)
            normalized["context_precision"] = cp_result
        except Exception as e:
            print(f"    [CP专用评估失败, 保留统一评估分数] {e}")

        return normalized

    def evaluate_batch(self, results_path: str, output_path: Optional[str] = None,
                       save_every: int = 5) -> dict:
        """批量评估结果文件"""
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        # 过滤掉有 error 的和 LLM disabled 的
        valid = [r for r in results if "metrics" in r and r.get("generated_answer", "") != "[LLM disabled]"]

        if not valid:
            print("没有有效的 LLM 生成结果可供评估")
            return {}

        total = len(valid)
        print(f"\n{'=' * 60}")
        print(f"LLM-as-Evaluator: 评估 {total} 个问题")
        print(f"评估模型: {self.model}")
        print(f"{'=' * 60}\n")

        if not output_path:
            base = Path(results_path).stem
            output_path = str(Path(results_path).parent / f"{base}_llm_eval.json")

        all_scores = []
        start_time = time.time()

        for idx, result in enumerate(valid):
            try:
                scores = self.evaluate_single(result)
                entry = {
                    "index": result.get("index", idx),
                    "question": result["question"][:100],
                    "label": result.get("label", ""),
                    "scores": scores,
                }
                all_scores.append(entry)

                # 打印进度
                cs = scores.get("context_recall", {}).get("score", 0)
                cp = scores.get("context_precision", {}).get("score", 0)
                ff = scores.get("faithfulness", {}).get("score", 0)
                ar = scores.get("answer_relevancy", {}).get("score", 0)
                print(
                    f"  [{idx+1}/{total}] "
                    f"CtxRecall={cs} | CtxPrec={cp} | Faith={ff} | AnsRel={ar} | "
                    f"label={result.get('label', '')}"
                )

            except Exception as e:
                print(f"  [{idx+1}/{total}] ERROR: {e}")
                all_scores.append({
                    "index": result.get("index", idx),
                    "error": str(e),
                })

            # 增量保存
            if (idx + 1) % save_every == 0 or (idx + 1) == total:
                self._save_results(all_scores, output_path, start_time)

        elapsed = time.time() - start_time

        # 计算汇总
        summary = self._compute_summary(all_scores, elapsed)
        output = {"summary": summary, "evaluator_model": self.model, "scores": all_scores}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self._print_summary(summary, output_path)
        return output

    def _save_results(self, scores: list, filepath: str, start_time: float):
        """增量保存"""
        elapsed = time.time() - start_time
        summary = self._compute_summary(scores, elapsed)
        output = {"summary": summary, "evaluator_model": self.model, "scores": scores}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _compute_summary(self, scores: list, elapsed: float) -> dict:
        """计算汇总统计"""
        valid = [s for s in scores if "scores" in s]
        n = len(valid)
        if n == 0:
            return {"completed": 0, "elapsed_seconds": round(elapsed, 1)}

        dims = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
        summary = {
            "completed": n,
            "elapsed_seconds": round(elapsed, 1),
            "avg_time_per_question": round(elapsed / n, 1),
        }

        for dim in dims:
            dim_scores = [s["scores"][dim]["score"] for s in valid
                         if dim in s["scores"] and s["scores"][dim].get("score", 0) > 0]
            if dim_scores:
                summary[dim] = {
                    "mean": round(sum(dim_scores) / len(dim_scores), 3),
                    "min": min(dim_scores),
                    "max": max(dim_scores),
                    "count": len(dim_scores),
                }
            else:
                summary[dim] = {"mean": 0, "count": 0}

        return summary

    def _print_summary(self, summary: dict, output_path: str):
        """打印总结"""
        print("\n" + "=" * 60)
        print("LLM-as-Evaluator 评估结果")
        print("=" * 60)
        print(f"完成: {summary.get('completed', 0)} 题")
        print(f"耗时: {summary.get('elapsed_seconds', 0)}s")

        dims = {
            "context_recall": "上下文召回率 (Context Recall)",
            "context_precision": "上下文精确率 (Context Precision)",
            "faithfulness": "答案忠实度 (Faithfulness)",
            "answer_relevancy": "答案相关性 (Answer Relevancy)",
        }

        for dim_key, dim_name in dims.items():
            if dim_key in summary and summary[dim_key].get("count", 0) > 0:
                s = summary[dim_key]
                print(f"\n  {dim_name}:")
                print(f"    平均: {s['mean']:.2f}/5  (min={s['min']}, max={s['max']}, n={s['count']})")

        print(f"\n完整结果已保存至: {output_path}")
        print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Evaluator: 端到端质量评估")
    parser.add_argument("--results", "-r", required=True,
                        help="batch_evaluate 输出的结果 JSON 文件路径")
    parser.add_argument("--output", "-o", default=None,
                        help="评估结果输出路径 (默认自动生成)")
    parser.add_argument("--model", "-m", default="MiniMax-M2.7",
                        help="评估用 LLM 模型 (默认 MiniMax-M2.7)")
    parser.add_argument("--base-url", default="https://api.minimaxi.com/v1",
                        help="OpenAI 兼容接口 base_url (默认 MiniMax)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="读取 API Key 的环境变量名 (默认 OPENAI_API_KEY)")
    args = parser.parse_args()

    evaluator = LLMEvaluator(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    evaluator.evaluate_batch(
        results_path=args.results,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
