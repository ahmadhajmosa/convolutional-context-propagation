from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import dspy

from .io_utils import read_jsonl, write_jsonl
from .judge import LLMJudge
from .module import CPP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a zero-shot CCP model tailored for long-context QA."
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path("data/longbench_eval.jsonl"),
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--use-llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LLM-as-judge scoring against the reference answer.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional judge model (defaults to --model).",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for judge model.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument(
        "--ccp-signature",
        choices=("legacy", "adaptive"),
        default="adaptive",
        help="CCP signature family: legacy (memo-style) or adaptive (task-aware).",
    )
    parser.add_argument(
        "--cap-signature",
        choices=("legacy", "adaptive"),
        default=None,
        help="Backward-compatible alias for --ccp-signature.",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-chars", type=int, default=4000)
    parser.add_argument("--chunk-overlap", type=int, default=400)
    parser.add_argument(
        "--extract-num-threads",
        type=int,
        default=1,
        help="Parallel threads for chunk extraction inside each example.",
    )
    parser.add_argument(
        "--extract-max-errors",
        type=int,
        default=20,
        help="Maximum tolerated chunk-extraction errors before cancellation.",
    )
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--save-traces", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print periodic progress with processed/total and evaluated counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset_jsonl.exists():
        print(f"Dataset not found: {args.dataset_jsonl}", file=sys.stderr)
        return 2
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key env var: {args.api_key_env}", file=sys.stderr)
        return 2
    if args.extract_num_threads < 1:
        print("--extract-num-threads must be >= 1", file=sys.stderr)
        return 2
    if args.extract_max_errors < 1:
        print("--extract-max-errors must be >= 1", file=sys.stderr)
        return 2

    rows = read_jsonl(args.dataset_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("No rows to evaluate.", file=sys.stderr)
        return 2
    total_rows = len(rows)
    if args.show_progress:
        print(f"Loaded {total_rows} rows from {args.dataset_jsonl}", file=sys.stderr, flush=True)

    lm = dspy.LM(model=args.model, api_key=api_key, temperature=args.temperature)
    dspy.configure(lm=lm)
    judge = LLMJudge() if args.use_llm_judge else None
    judge_lm = None
    if args.use_llm_judge:
        judge_model = args.judge_model or args.model
        if judge_model == args.model and args.judge_temperature == args.temperature:
            judge_lm = lm
        else:
            judge_lm = dspy.LM(
                model=judge_model,
                api_key=api_key,
                temperature=args.judge_temperature,
            )

    signature_mode = args.ccp_signature
    if args.cap_signature is not None:
        signature_mode = args.cap_signature

    ccp = CPP(
        layers=args.layers,
        window=args.window,
        stride=args.stride,
        signature_mode=signature_mode,
        enable_trace=args.save_traces is not None,
        extract_num_threads=args.extract_num_threads,
        extract_max_errors=args.extract_max_errors,
    )

    if args.save_traces is not None:
        args.save_traces.mkdir(parents=True, exist_ok=True)

    pred_rows: list[dict[str, object]] = []
    judge_sum = 0.0
    judge_count = 0
    count = 0
    task_judge_sum: dict[str, float] = defaultdict(float)
    task_judge_count: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(rows):
        processed = idx + 1
        if args.show_progress:
            print(
                f"[{processed}/{total_rows}] starting sample (evaluated_so_far={count})",
                file=sys.stderr,
                flush=True,
            )
        context = str(row.get("context", "")).strip()
        query = str(row.get("query", "")).strip()
        gold = str(row.get("answer", "")).strip()
        task = str(row.get("task", "")).strip() or "unknown"
        if not context or not query or not gold:
            if args.show_progress and args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"[{processed}/{total_rows}] evaluated={count} (skipping invalid rows as needed)",
                    file=sys.stderr,
                    flush=True,
                )
            continue

        pred = ccp(
            context=context,
            query=query,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
        )
        answer = str(getattr(pred, "answer", "")).strip()
        judge_score = None
        judge_verdict = ""
        judge_rationale = ""
        if judge is not None and judge_lm is not None:
            try:
                with dspy.context(lm=judge_lm):
                    judge_pred = judge(
                        query=query,
                        reference_answer=gold,
                        candidate_answer=answer,
                    )
                judge_score = float(getattr(judge_pred, "score_0_to_10", 0.0))
                judge_verdict = str(getattr(judge_pred, "verdict", "")).strip()
                judge_rationale = str(getattr(judge_pred, "rationale", "")).strip()
                judge_sum += judge_score
                judge_count += 1
                task_judge_sum[task] += judge_score
                task_judge_count[task] += 1
            except Exception as exc:  # noqa: BLE001
                judge_rationale = f"judge_error: {exc}"
        count += 1

        item_id = str(row.get("id", idx))
        pred_rows.append(
            {
                "id": item_id,
                "task": task,
                "query": query,
                "gold_answer": gold,
                "pred_answer": answer,
                "llm_judge_score_0_to_10": judge_score,
                "llm_judge_verdict": judge_verdict,
                "llm_judge_rationale": judge_rationale,
                "ccp_signature": signature_mode,
                "verifier_notes": str(getattr(pred, "verifier_notes", "")),
            }
        )

        if args.save_traces is not None:
            trace_payload = str(getattr(pred, "ccp_trace_json", "")).strip()
            if not trace_payload:
                trace_payload = str(getattr(pred, "cap_trace_json", "")).strip()
            if trace_payload:
                trace_path = args.save_traces / f"{idx:04d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', item_id)}.json"
                trace_path.write_text(trace_payload, encoding="utf-8")

        if args.show_progress and args.progress_every > 0 and processed % args.progress_every == 0:
            msg = f"[{processed}/{total_rows}] evaluated={count}"
            if judge_count > 0:
                msg += f" llm_judge={judge_sum / judge_count:.3f}/10"
            print(msg, file=sys.stderr, flush=True)

    if count == 0:
        print("No valid examples after filtering.", file=sys.stderr)
        return 2

    print(f"Rows processed: {total_rows}")
    print(f"Examples evaluated: {count}")
    if judge_count > 0:
        print(f"Average llm_judge_score_0_to_10: {judge_sum / judge_count:.4f}")
        print("Per-task llm_judge_score_0_to_10:")
        for task_name in sorted(task_judge_count.keys()):
            t_count = task_judge_count[task_name]
            t_avg = task_judge_sum[task_name] / t_count
            print(f"- {task_name}: {t_avg:.4f} (n={t_count})")
    elif args.use_llm_judge:
        print("Average llm_judge_score_0_to_10: N/A (no successful judge calls)")

    if args.save_predictions is not None:
        write_jsonl(args.save_predictions, pred_rows)  # type: ignore[arg-type]
        print(f"Saved predictions: {args.save_predictions}")

    return 0
