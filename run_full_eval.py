"""
完整评估自动化流水线
====================

4 步评估流程：
  Step 1: train_set_100papers.json        全量  → 检索指标（不含 LLM）
  Step 2: train_set_100papers_sample50.json 50条  → 生成指标（含 LLM）+ LLM 评分
  Step 3: evaluation_set_100papers.json    全量  → 检索指标（不含 LLM）
  Step 4: evaluation_set_100papers.json    抽50  → 生成指标（含 LLM）+ LLM 评分

用法:
    python run_full_eval.py              # 完整跑 4 步
    python run_full_eval.py --test       # 每步只跑 2 题，快速验证流程
    python run_full_eval.py --step 1     # 只跑第 1 步
    python run_full_eval.py --step 2 4   # 只跑第 2、4 步
    python run_full_eval.py --from-step 3  # 从第 3 步开始跑
"""

import subprocess
import sys
import os
import json
import glob
import time
import argparse
from datetime import datetime
from pathlib import Path

# ── 常量 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval_results"
PYTHON = sys.executable

# ── 步骤定义 ──────────────────────────────────────────
STEPS = [
    {
        "id": 1,
        "name": "train_set 全量 → 检索指标",
        "dataset": "train_set_100papers.json",
        "no_llm": True,
        "sample": None,
        "llm_eval": False,
    },
    {
        "id": 2,
        "name": "train_set_sample50 → 生成指标 + LLM 评分",
        "dataset": "train_set_100papers_sample50.json",
        "no_llm": False,
        "sample": None,
        "llm_eval": True,
    },
    {
        "id": 3,
        "name": "evaluation_set 全量 → 检索指标",
        "dataset": "evaluation_set_100papers.json",
        "no_llm": True,
        "sample": None,
        "llm_eval": False,
    },
    {
        "id": 4,
        "name": "evaluation_set 抽50 → 生成指标 + LLM 评分",
        "dataset": "evaluation_set_100papers.json",
        "no_llm": False,
        "sample": 50,
        "llm_eval": True,
    },
]


def find_latest_result(dataset_stem_prefix: str) -> str | None:
    """找到 eval_results/ 下某数据集最新的结果文件（排除 _llm_eval）
    
    使用前缀匹配，支持 sample 数量不确定的情况（如 sample44 vs sample50）
    """
    pattern = str(EVAL_DIR / f"{dataset_stem_prefix}*.json")
    candidates = [f for f in glob.glob(pattern) if "_llm_eval" not in f]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_step(step: dict, test_mode: bool = False) -> dict:
    """
    执行一个评估步骤，返回 {result_file, llm_eval_file, success, error}
    """
    dataset_path = str(BASE_DIR / step["dataset"])
    step_result = {"result_file": None, "llm_eval_file": None, "success": False, "error": None}

    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        step_result["error"] = f"数据集不存在: {dataset_path}"
        print(f"  [ERROR] {step_result['error']}")
        return step_result

    # ── 阶段 A: batch_evaluate ──
    cmd = [PYTHON, str(BASE_DIR / "batch_evaluate.py"),
           "--dataset", dataset_path,
           "--output", str(EVAL_DIR)]

    if step["no_llm"]:
        cmd.append("--no-llm")

    if step["sample"]:
        cmd.extend(["--sample", str(step["sample"])])

    if test_mode:
        cmd.extend(["--max", "2"])

    print(f"  [CMD] {' '.join(cmd)}")
    t0 = time.time()

    ret = subprocess.run(cmd, cwd=str(BASE_DIR))

    elapsed = time.time() - t0
    print(f"  [耗时] {int(elapsed // 60)}分{int(elapsed % 60)}秒")

    if ret.returncode != 0:
        step_result["error"] = f"batch_evaluate 退出码 {ret.returncode}"
        print(f"  [ERROR] {step_result['error']}")
        return step_result

    # 找结果文件
    # --sample 会改变数据集名（如 evaluation_set_100papers_sample44）
    # 注意：实际抽样数可能 < 请求数（分层抽样时组不够），所以用前缀匹配
    if step["sample"]:
        stem_prefix = Path(step["dataset"]).stem + "_sample"
    else:
        stem_prefix = Path(step["dataset"]).stem

    result_file = find_latest_result(stem_prefix)
    if not result_file:
        step_result["error"] = f"找不到结果文件 (prefix={stem_prefix})"
        print(f"  [ERROR] {step_result['error']}")
        return step_result

    step_result["result_file"] = result_file
    print(f"  [结果] {result_file}")

    # 打印检索指标摘要
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        ret_info = summary.get("retrieval", {})
        if ret_info:
            print(f"  [检索] 命中率={ret_info.get('hit_rate', 0):.1%}  "
                  f"MRR={ret_info.get('mrr', 0):.4f}  "
                  f"Recall@1={ret_info.get('recall@1', 0):.1%}  "
                  f"Recall@5={ret_info.get('recall@5', 0):.1%}")
        ans_info = summary.get("answer_quality", {})
        if ans_info and ans_info.get("avg_rouge_l_f1", 0) > 0:
            print(f"  [答案] ROUGE-L={ans_info.get('avg_rouge_l_f1', 0):.4f}  "
                  f"Token-F1={ans_info.get('avg_token_f1', 0):.4f}")
    except Exception:
        pass

    # ── 阶段 B: LLM 评分（仅含 LLM 的步骤才跑）──
    if step["llm_eval"]:
        print(f"\n  ── 开始 LLM-as-Judge 评分 ──")
        cmd2 = [PYTHON, str(BASE_DIR / "llm_evaluator.py"),
                "--results", result_file]

        print(f"  [CMD] {' '.join(cmd2)}")
        t1 = time.time()

        ret2 = subprocess.run(cmd2, cwd=str(BASE_DIR))

        elapsed2 = time.time() - t1
        print(f"  [耗时] {int(elapsed2 // 60)}分{int(elapsed2 % 60)}秒")

        if ret2.returncode != 0:
            print(f"  [WARNING] llm_evaluator 退出码 {ret2.returncode}，但 batch_evaluate 结果已保存")
        else:
            eval_file = result_file.replace(".json", "_llm_eval.json")
            if os.path.exists(eval_file):
                step_result["llm_eval_file"] = eval_file
                print(f"  [LLM评分] {eval_file}")

                # 打印 LLM 评分摘要
                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)
                    es = eval_data.get("summary", {})
                    dims = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
                    parts = []
                    for d in dims:
                        if d in es and isinstance(es[d], dict) and es[d].get("count", 0) > 0:
                            parts.append(f"{d}={es[d]['mean']:.2f}")
                    if parts:
                        print(f"  [LLM评分] {' | '.join(parts)}")
                except Exception:
                    pass

    step_result["success"] = True
    return step_result


def main():
    parser = argparse.ArgumentParser(description="完整评估自动化流水线")
    parser.add_argument("--test", action="store_true",
                        help="测试模式：每步只跑 2 题，快速验证流程")
    parser.add_argument("--step", type=int, nargs="+", default=None,
                        help="只跑指定步骤，如 --step 1 3")
    parser.add_argument("--from-step", type=int, default=None,
                        help="从第 N 步开始跑，如 --from-step 3")
    args = parser.parse_args()

    # 确定要执行的步骤
    if args.step:
        steps_to_run = [s for s in STEPS if s["id"] in args.step]
    elif args.from_step:
        steps_to_run = [s for s in STEPS if s["id"] >= args.from_step]
    else:
        steps_to_run = STEPS

    if not steps_to_run:
        print("没有要执行的步骤")
        return

    EVAL_DIR.mkdir(exist_ok=True)

    mode_str = "测试模式 (每步 2 题)" if args.test else "完整模式"
    step_ids = [s["id"] for s in steps_to_run]

    print("=" * 70)
    print(f"  完整评估流水线 | {mode_str}")
    print(f"  执行步骤: {step_ids}")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    for step in steps_to_run:
        print(f"\n{'=' * 70}")
        print(f"  Step {step['id']}: {step['name']}")
        print(f"  数据集: {step['dataset']}", end="")
        if step["sample"]:
            print(f" (抽样 {step['sample']} 条)", end="")
        if args.test:
            print(f" [测试: 仅 2 题]", end="")
        print()
        print(f"  模式: {'仅检索' if step['no_llm'] else '检索 + 生成'}"
              f"{'  + LLM评分' if step['llm_eval'] else ''}")
        print("=" * 70)

        result = run_step(step, test_mode=args.test)
        all_results[step["id"]] = result

        status = "✓ 成功" if result["success"] else f"✗ 失败: {result['error']}"
        print(f"\n  [{status}] Step {step['id']} 完成")

    # ── 总结 ──
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  全部完成！总耗时: {int(total_elapsed // 60)}分{int(total_elapsed % 60)}秒")
    print(f"{'=' * 70}")

    print(f"\n  输出文件汇总:")
    for sid, res in all_results.items():
        step_info = next(s for s in STEPS if s["id"] == sid)
        status = "✓" if res["success"] else "✗"
        print(f"\n  {status} Step {sid}: {step_info['name']}")
        if res["result_file"]:
            print(f"    结果: {os.path.basename(res['result_file'])}")
        if res["llm_eval_file"]:
            print(f"    LLM评分: {os.path.basename(res['llm_eval_file'])}")
        if res["error"]:
            print(f"    错误: {res['error']}")


if __name__ == "__main__":
    main()
