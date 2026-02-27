#!/usr/bin/env python3
"""
Convert LongBench task JSONL files into a unified JSONL format for CCP runners.

Output record format:
  {
    "id": "...",
    "task": "...",
    "query": "...",
    "context": "...",
    "answer": "...",
    "answers": ["..."],
    "language": "...",
    "length": "...",
    "source_file": "..."
  }
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_tasks(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def flatten_answer_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        val = value.strip()
        return [val] if val else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_answer_values(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key in ("answer", "answers", "answer_text", "text", "name", "title", "aliases"):
            if key in value:
                out.extend(flatten_answer_values(value.get(key)))
        if out:
            return out
        for _, v in value.items():
            out.extend(flatten_answer_values(v))
        return out
    val = str(value).strip()
    return [val] if val else []


def normalize_row(row: dict[str, Any], task: str, source_file: str, row_idx: int, max_context_chars: int | None) -> dict[str, Any] | None:
    query = str(row.get("input", "")).strip()
    context = str(row.get("context", "")).strip()
    if max_context_chars is not None and max_context_chars > 0:
        context = context[:max_context_chars]
    answers = flatten_answer_values(row.get("answers"))
    answers = [a for a in answers if a]
    if not query or not context or not answers:
        return None
    primary_answer = answers[0]
    item_id = str(row.get("_id", f"{task}-{row_idx}")).strip() or f"{task}-{row_idx}"
    return {
        "id": item_id,
        "task": task,
        "query": query,
        "context": context,
        "answer": primary_answer,
        "answers": answers,
        "language": str(row.get("language", "")).strip(),
        "length": str(row.get("length", "")).strip(),
        "source_file": source_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--longbench-dir",
        type=Path,
        default=Path("data/benchmarks/longbench/data"),
        help="Directory containing LongBench task JSONL files.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task stems (e.g. qasper,hotpotqa). Empty means all.",
    )
    parser.add_argument(
        "--include-e",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include *_e.jsonl files (LongBench-E variants).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/longbench_eval.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.longbench_dir.exists():
        print(f"LongBench dir not found: {args.longbench_dir}", file=sys.stderr)
        return 2

    selected_tasks = parse_tasks(args.tasks)
    rng = random.Random(args.seed)

    files = sorted(args.longbench_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in: {args.longbench_dir}", file=sys.stderr)
        return 2

    rows_out: list[dict[str, Any]] = []
    per_task_counts: dict[str, int] = {}

    for path in files:
        stem = path.stem
        if stem.endswith("_e") and not args.include_e:
            continue
        task_base = stem[:-2] if stem.endswith("_e") else stem
        if selected_tasks and task_base not in selected_tasks and stem not in selected_tasks:
            continue

        raw_rows = read_jsonl(path)
        task_rows: list[dict[str, Any]] = []
        for i, row in enumerate(raw_rows):
            rec = normalize_row(
                row=row,
                task=stem,
                source_file=path.name,
                row_idx=i,
                max_context_chars=args.max_context_chars,
            )
            if rec is not None:
                task_rows.append(rec)

        if args.shuffle:
            rng.shuffle(task_rows)
        if args.max_per_task is not None:
            task_rows = task_rows[: args.max_per_task]

        per_task_counts[stem] = len(task_rows)
        rows_out.extend(task_rows)

    if args.shuffle:
        rng.shuffle(rows_out)
    if args.max_total is not None:
        rows_out = rows_out[: args.max_total]

    if not rows_out:
        print("No records selected after applying filters.", file=sys.stderr)
        return 2

    write_jsonl(args.output_jsonl, rows_out)
    print(f"Wrote {len(rows_out)} rows -> {args.output_jsonl}")
    print("Per-task counts:")
    for task_name in sorted(per_task_counts.keys()):
        print(f"- {task_name}: {per_task_counts[task_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
