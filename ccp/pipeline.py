#!/usr/bin/env python3
"""
Run a zero-shot CCP model tailored for long-context QA.

This script does not use DSPy optimizers. It relies on task-specific
CCP signatures and evaluates outputs with an optional LLM-as-judge pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import dspy


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
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = text.strip()
    if not text:
        return [""]
    chunks: list[str] = []
    start = 0
    step = max_chars - overlap
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks or [text]


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split(" ")


def _parse_score_0_to_10(text: str) -> float:
    raw = str(text).strip()
    if not raw:
        return 0.0
    slash10 = "/10" in raw.replace(" ", "")
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return 0.0
    value = float(m.group(0))
    if not slash10 and 0.0 <= value <= 1.0:
        value *= 10.0
    if value < 0.0:
        return 0.0
    if value > 10.0:
        return 10.0
    return value


def _parse_float_0_to_1(value: object, default: float = 0.0) -> float:
    raw = str(value).strip()
    if not raw:
        return default
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return default
    v = float(m.group(0))
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def _is_trueish(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1", "supported", "pass"}


def _normalize_for_match(text: str) -> str:
    lowered = text.lower().replace("*", "")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _extract_quoted_snippets(text: str) -> list[str]:
    snippets = re.findall(r'"([^"]{3,400})"', text)
    return [s.strip() for s in snippets if s.strip()]


def _count_supported_quotes(chunk: str, quoted_evidence: str) -> int:
    chunk_norm = _normalize_for_match(chunk)
    count = 0
    for snippet in _extract_quoted_snippets(quoted_evidence):
        snippet_norm = _normalize_for_match(snippet)
        if snippet_norm and snippet_norm in chunk_norm:
            count += 1
    return count


class CCPPipeline(dspy.Module):
    """
    CCP tuned for long-context QA:
      plan query -> extract local answers per chunk -> fuse over layers -> verify.
    """

    def __init__(
        self,
        layers: int,
        window: int,
        stride: int,
        signature_mode: str = "adaptive",
        enable_trace: bool = False,
        extract_num_threads: int = 1,
        extract_max_errors: int = 20,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if extract_num_threads < 1:
            raise ValueError("extract_num_threads must be >= 1")
        if extract_max_errors < 1:
            raise ValueError("extract_max_errors must be >= 1")
        if signature_mode not in {"legacy", "adaptive"}:
            raise ValueError("signature_mode must be one of: legacy, adaptive")
        self.layers = layers
        self.window = window
        self.stride = stride
        self.signature_mode = signature_mode
        self.enable_trace = enable_trace
        self.extract_num_threads = extract_num_threads
        self.extract_max_errors = extract_max_errors

        if self.signature_mode == "legacy":
            self.plan = dspy.Predict(
                "query -> answer_type, extraction_focus, abstention_rule"
            )
            self.extract = dspy.Predict(
                "chunk, query, answer_type, extraction_focus, abstention_rule -> "
                "evidence_found, quoted_evidence, atomic_facts, local_answer, confidence, missing_info"
            )
            self.fuse = dspy.Predict(
                "window_findings, query, answer_type, extraction_focus, abstention_rule -> "
                "fused_answer, fused_evidence, confidence, unresolved_points"
            )
            self.verify = dspy.Predict(
                "query, candidate_answer, fused_evidence, unresolved_points -> "
                "is_supported, corrected_answer, verifier_notes"
            )
        else:
            self.plan = dspy.Predict(
                "query -> task_type, expected_output_format, scoring_focus, abstention_policy"
            )
            self.extract = dspy.Predict(
                "chunk, query, task_type, expected_output_format, scoring_focus, abstention_policy -> "
                "relevance_0_to_1, key_points, candidate_fragment, evidence_spans, unresolved"
            )
            self.fuse = dspy.Predict(
                "window_findings, query, task_type, expected_output_format, scoring_focus, abstention_policy -> "
                "fused_points, draft_answer, confidence_0_to_1, unresolved"
            )
            self.verify = dspy.Predict(
                "query, task_type, expected_output_format, draft_answer, fused_points, unresolved -> "
                "is_valid, corrected_answer, verifier_notes"
            )

    def forward(
        self,
        context: str,
        query: str,
        chunk_chars: int = 4000,
        chunk_overlap: int = 400,
    ) -> dspy.Prediction:
        if self.signature_mode == "legacy":
            return self._forward_legacy(
                context=context,
                query=query,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
            )
        return self._forward_adaptive(
            context=context,
            query=query,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )

    def _run_extract(self, extract_inputs: list[dict[str, str]]) -> list[dspy.Prediction | None]:
        if self.extract_num_threads > 1 and len(extract_inputs) > 1:
            parallel = dspy.Parallel(
                num_threads=self.extract_num_threads,
                max_errors=self.extract_max_errors,
                disable_progress_bar=True,
            )
            exec_pairs = [(self.extract, ex) for ex in extract_inputs]
            return parallel(exec_pairs)
        return [self.extract(**ex) for ex in extract_inputs]

    def _forward_legacy(
        self,
        context: str,
        query: str,
        chunk_chars: int = 4000,
        chunk_overlap: int = 400,
    ) -> dspy.Prediction:
        trace: dict[str, Any] = {}
        if self.enable_trace:
            trace = {
                "query": query,
                "signature_mode": self.signature_mode,
                "chunks": [],
                "fuse_layers": [],
                "final": {},
            }

        plan = self.plan(
            query=(
                f"{query}\n\n"
                "Return only extraction instructions. Do not answer the query."
            )
        )
        answer_type = str(getattr(plan, "answer_type", "")).strip() or "short factual answer"
        extraction_focus = str(getattr(plan, "extraction_focus", "")).strip() or (
            "extract only directly supported facts relevant to the query"
        )
        abstention_rule = str(getattr(plan, "abstention_rule", "")).strip() or (
            "if chunk lacks evidence for the query, set evidence_found=false"
        )

        chunks = chunk_text(context, max_chars=chunk_chars, overlap=chunk_overlap)
        nodes: list[str] = []
        fallback_nodes: list[tuple[int, str]] = []

        extract_inputs = [
            {
                "chunk": chunk,
                "query": query,
                "answer_type": answer_type,
                "extraction_focus": extraction_focus,
                "abstention_rule": abstention_rule,
            }
            for chunk in chunks
        ]
        extract_preds = self._run_extract(extract_inputs)

        for chunk_index, (chunk, pred) in enumerate(zip(chunks, extract_preds)):
            if pred is None:
                # Parallel returns None for failed items; treat as abstention.
                pred = dspy.Prediction(
                    evidence_found="false",
                    quoted_evidence="",
                    atomic_facts="",
                    local_answer="",
                    confidence="",
                    missing_info="extract_call_failed",
                )
            quoted_evidence = str(getattr(pred, "quoted_evidence", "")).strip()
            local_answer = str(getattr(pred, "local_answer", "")).strip()
            atomic_facts = str(getattr(pred, "atomic_facts", "")).strip()
            missing_info = str(getattr(pred, "missing_info", "")).strip()
            confidence = str(getattr(pred, "confidence", "")).strip()
            model_evidence_found = _is_trueish(getattr(pred, "evidence_found", ""))
            supported_quote_count = _count_supported_quotes(chunk, quoted_evidence)

            evidence_found = model_evidence_found and supported_quote_count > 0
            if not model_evidence_found and supported_quote_count >= 2:
                evidence_found = True
            if not evidence_found:
                local_answer = ""
                atomic_facts = ""
                if not missing_info:
                    missing_info = "missing_query_evidence"

            payload = {
                "chunk_index": chunk_index,
                "evidence_found": evidence_found,
                "model_evidence_found": model_evidence_found,
                "supported_quote_count": supported_quote_count,
                "quoted_evidence": quoted_evidence,
                "local_answer": local_answer,
                "atomic_facts": atomic_facts,
                "confidence": confidence,
                "missing_info": missing_info,
            }
            payload_text = json.dumps(payload, ensure_ascii=False)
            fallback_nodes.append((supported_quote_count, payload_text))
            if evidence_found and local_answer:
                nodes.append(payload_text)

            if self.enable_trace:
                cast_chunks = trace["chunks"]
                if isinstance(cast_chunks, list):
                    cast_chunks.append(payload)

        if not nodes:
            fallback_nodes.sort(key=lambda x: x[0], reverse=True)
            positive = [node for score, node in fallback_nodes if score > 0]
            if positive:
                nodes = positive[: min(3, len(positive))]
            elif fallback_nodes:
                nodes = [fallback_nodes[0][1]]
            else:
                nodes = []

        for layer_idx in range(self.layers):
            if len(nodes) <= 1:
                break
            next_nodes: list[str] = []
            layer_trace: list[dict[str, Any]] = []
            i = 0
            while i < len(nodes):
                window_nodes = nodes[i:i + self.window]
                if not window_nodes:
                    break
                packed = (
                    "Window findings in JSON.\n"
                    "Prefer entries with evidence_found=true and higher supported_quote_count.\n\n"
                    + "\n\n".join(f"[{j}] {node}" for j, node in enumerate(window_nodes, start=i))
                )
                fused = self.fuse(
                    window_findings=packed,
                    query=query,
                    answer_type=answer_type,
                    extraction_focus=extraction_focus,
                    abstention_rule=abstention_rule,
                )
                fused_answer = str(getattr(fused, "fused_answer", "")).strip()
                fused_evidence = str(getattr(fused, "fused_evidence", "")).strip()
                unresolved_points = str(getattr(fused, "unresolved_points", "")).strip()
                confidence = str(getattr(fused, "confidence", "")).strip()
                fused_payload = {
                    "window_start_index": i,
                    "fused_answer": fused_answer,
                    "fused_evidence": fused_evidence,
                    "unresolved_points": unresolved_points,
                    "confidence": confidence,
                }
                next_nodes.append(json.dumps(fused_payload, ensure_ascii=False))
                if self.enable_trace:
                    layer_trace.append(
                        {
                            "window_start_index": i,
                            "window_size": len(window_nodes),
                            "fused_answer": fused_answer,
                            "fused_evidence": fused_evidence,
                            "unresolved_points": unresolved_points,
                            "confidence": confidence,
                        }
                    )
                i += self.stride
            nodes = next_nodes
            if self.enable_trace:
                cast_layers = trace["fuse_layers"]
                if isinstance(cast_layers, list):
                    cast_layers.append({"layer_index": layer_idx + 1, "windows": layer_trace})

        candidate_answer = ""
        fused_evidence = ""
        unresolved_points = ""
        if nodes:
            top = nodes[0]
            try:
                payload = json.loads(top)
                if isinstance(payload, dict):
                    candidate_answer = str(
                        payload.get("fused_answer") or payload.get("local_answer") or ""
                    ).strip()
                    fused_evidence = str(
                        payload.get("fused_evidence") or payload.get("quoted_evidence") or ""
                    ).strip()
                    unresolved_points = str(payload.get("unresolved_points") or "").strip()
            except json.JSONDecodeError:
                candidate_answer = top.strip()
        if not candidate_answer:
            candidate_answer = "I cannot find enough supported evidence in the provided context."

        verified = self.verify(
            query=query,
            candidate_answer=candidate_answer,
            fused_evidence=fused_evidence,
            unresolved_points=unresolved_points,
        )
        if _is_trueish(getattr(verified, "is_supported", "")):
            final_answer = candidate_answer
        else:
            final_answer = str(getattr(verified, "corrected_answer", "")).strip() or candidate_answer
        verifier_notes = str(getattr(verified, "verifier_notes", "")).strip()

        if self.enable_trace:
            trace["final"] = {
                "candidate_answer": candidate_answer,
                "final_answer": final_answer,
                "fused_evidence": fused_evidence,
                "unresolved_points": unresolved_points,
                "verifier_notes": verifier_notes,
            }
            trace_json = json.dumps(trace, ensure_ascii=False)
        else:
            trace_json = ""

        return dspy.Prediction(
            answer=final_answer,
            fused_evidence=fused_evidence,
            unresolved_points=unresolved_points,
            verifier_notes=verifier_notes,
            ccp_trace_json=trace_json,
            cap_trace_json=trace_json,
        )

    def _forward_adaptive(
        self,
        context: str,
        query: str,
        chunk_chars: int = 4000,
        chunk_overlap: int = 400,
    ) -> dspy.Prediction:
        trace: dict[str, Any] = {}
        if self.enable_trace:
            trace = {
                "query": query,
                "signature_mode": self.signature_mode,
                "chunks": [],
                "fuse_layers": [],
                "final": {},
            }

        plan = self.plan(
            query=(
                f"{query}\n\n"
                "Classify task and define extraction guidance. "
                "Do not answer the question."
            )
        )
        task_type = str(getattr(plan, "task_type", "")).strip() or "open_qa"
        expected_output_format = str(getattr(plan, "expected_output_format", "")).strip() or (
            "short direct answer; if multiple items are requested, return a deduplicated list"
        )
        scoring_focus = str(getattr(plan, "scoring_focus", "")).strip() or (
            "maximize faithfulness first, then completeness"
        )
        abstention_policy = str(getattr(plan, "abstention_policy", "")).strip() or (
            "if support is weak, return partial answer and explicitly note uncertainty"
        )

        chunks = chunk_text(context, max_chars=chunk_chars, overlap=chunk_overlap)
        candidate_nodes: list[tuple[float, str]] = []

        extract_inputs = [
            {
                "chunk": chunk,
                "query": query,
                "task_type": task_type,
                "expected_output_format": expected_output_format,
                "scoring_focus": scoring_focus,
                "abstention_policy": abstention_policy,
            }
            for chunk in chunks
        ]
        extract_preds = self._run_extract(extract_inputs)

        for chunk_index, (chunk, pred) in enumerate(zip(chunks, extract_preds)):
            if pred is None:
                pred = dspy.Prediction(
                    relevance_0_to_1="0.0",
                    key_points="",
                    candidate_fragment="",
                    evidence_spans="",
                    unresolved="extract_call_failed",
                )

            relevance = _parse_float_0_to_1(getattr(pred, "relevance_0_to_1", "0.0"))
            key_points = str(getattr(pred, "key_points", "")).strip()
            candidate_fragment = str(getattr(pred, "candidate_fragment", "")).strip()
            evidence_spans = str(getattr(pred, "evidence_spans", "")).strip()
            unresolved = str(getattr(pred, "unresolved", "")).strip()
            supported_quote_count = _count_supported_quotes(chunk, evidence_spans)
            has_content = bool(candidate_fragment or key_points)

            relevance_score = relevance
            if supported_quote_count > 0:
                relevance_score = min(1.0, max(relevance_score, 0.40 + 0.10 * min(supported_quote_count, 3)))
            if not has_content:
                relevance_score = 0.0

            payload = {
                "chunk_index": chunk_index,
                "relevance_score": relevance_score,
                "supported_quote_count": supported_quote_count,
                "candidate_fragment": candidate_fragment,
                "key_points": key_points,
                "evidence_spans": evidence_spans,
                "unresolved": unresolved,
            }
            payload_text = json.dumps(payload, ensure_ascii=False)
            candidate_nodes.append((relevance_score, payload_text))

            if self.enable_trace:
                cast_chunks = trace["chunks"]
                if isinstance(cast_chunks, list):
                    cast_chunks.append(payload)

        candidate_nodes.sort(key=lambda x: x[0], reverse=True)
        retained = [node for score, node in candidate_nodes if score > 0.0]
        if not retained and candidate_nodes:
            retained = [candidate_nodes[0][1]]

        top_k = max(self.window * 4, 8)
        nodes = retained[:top_k]

        for layer_idx in range(self.layers):
            if len(nodes) <= 1:
                break
            next_nodes: list[str] = []
            layer_trace: list[dict[str, Any]] = []
            i = 0
            while i < len(nodes):
                window_nodes = nodes[i:i + self.window]
                if not window_nodes:
                    break
                packed = (
                    "Window findings in JSON.\n"
                    "Prefer high relevance_score, preserve diversity, and keep only supported content.\n\n"
                    + "\n\n".join(f"[{j}] {node}" for j, node in enumerate(window_nodes, start=i))
                )
                fused = self.fuse(
                    window_findings=packed,
                    query=query,
                    task_type=task_type,
                    expected_output_format=expected_output_format,
                    scoring_focus=scoring_focus,
                    abstention_policy=abstention_policy,
                )
                fused_points = str(getattr(fused, "fused_points", "")).strip()
                draft_answer = str(getattr(fused, "draft_answer", "")).strip()
                confidence_0_to_1 = _parse_float_0_to_1(getattr(fused, "confidence_0_to_1", "0.0"))
                unresolved = str(getattr(fused, "unresolved", "")).strip()

                # If model omits confidence, derive a weak proxy from input node scores.
                if confidence_0_to_1 == 0.0:
                    input_scores: list[float] = []
                    for node in window_nodes:
                        try:
                            payload = json.loads(node)
                            if isinstance(payload, dict):
                                input_scores.append(_parse_float_0_to_1(payload.get("relevance_score", 0.0)))
                        except json.JSONDecodeError:
                            continue
                    if input_scores:
                        confidence_0_to_1 = max(input_scores)

                fused_payload = {
                    "window_start_index": i,
                    "relevance_score": confidence_0_to_1,
                    "draft_answer": draft_answer,
                    "fused_points": fused_points,
                    "unresolved": unresolved,
                }
                next_nodes.append(json.dumps(fused_payload, ensure_ascii=False))
                if self.enable_trace:
                    layer_trace.append(
                        {
                            "window_start_index": i,
                            "window_size": len(window_nodes),
                            "relevance_score": confidence_0_to_1,
                            "draft_answer": draft_answer,
                            "fused_points": fused_points,
                            "unresolved": unresolved,
                        }
                    )
                i += self.stride
            nodes = next_nodes
            if self.enable_trace:
                cast_layers = trace["fuse_layers"]
                if isinstance(cast_layers, list):
                    cast_layers.append({"layer_index": layer_idx + 1, "windows": layer_trace})

        best_payload: dict[str, Any] = {}
        best_score = -1.0
        for node in nodes:
            try:
                payload = json.loads(node)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            score = _parse_float_0_to_1(payload.get("relevance_score", 0.0))
            if score > best_score:
                best_score = score
                best_payload = payload

        draft_answer = str(best_payload.get("draft_answer") or best_payload.get("candidate_fragment") or "").strip()
        fused_points = str(best_payload.get("fused_points") or best_payload.get("key_points") or "").strip()
        unresolved = str(best_payload.get("unresolved") or "").strip()
        if not draft_answer:
            draft_answer = "I cannot find enough supported evidence in the provided context."

        verified = self.verify(
            query=query,
            task_type=task_type,
            expected_output_format=expected_output_format,
            draft_answer=draft_answer,
            fused_points=fused_points,
            unresolved=unresolved,
        )
        if _is_trueish(getattr(verified, "is_valid", "")):
            final_answer = draft_answer
        else:
            final_answer = str(getattr(verified, "corrected_answer", "")).strip() or draft_answer
        verifier_notes = str(getattr(verified, "verifier_notes", "")).strip()

        if self.enable_trace:
            trace["final"] = {
                "task_type": task_type,
                "expected_output_format": expected_output_format,
                "candidate_answer": draft_answer,
                "final_answer": final_answer,
                "fused_points": fused_points,
                "unresolved": unresolved,
                "selected_score": best_score,
                "verifier_notes": verifier_notes,
            }
            trace_json = json.dumps(trace, ensure_ascii=False)
        else:
            trace_json = ""

        return dspy.Prediction(
            answer=final_answer,
            fused_evidence=fused_points,
            unresolved_points=unresolved,
            verifier_notes=verifier_notes,
            ccp_trace_json=trace_json,
            cap_trace_json=trace_json,
        )


class LLMJudge(dspy.Module):
    """
    LLM-as-judge scorer against the reference answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.score = dspy.Predict(
            "query, reference_answer, candidate_answer, rubric -> score_0_to_10, verdict, rationale"
        )

    def forward(self, query: str, reference_answer: str, candidate_answer: str) -> dspy.Prediction:
        rubric = (
            "Score the candidate answer against the reference answer for the given query.\n"
            "Scoring guide:\n"
            "- 9-10: semantically equivalent and complete\n"
            "- 7-8: mostly correct with minor omissions\n"
            "- 4-6: partially correct, key details missing or imprecise\n"
            "- 1-3: mostly incorrect\n"
            "- 0: irrelevant or contradictory\n"
            "Return numeric score_0_to_10, verdict in {correct, partial, incorrect}, and short rationale."
        )
        pred = self.score(
            query=query,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            rubric=rubric,
        )
        score = _parse_score_0_to_10(str(getattr(pred, "score_0_to_10", "")))
        verdict = str(getattr(pred, "verdict", "")).strip()
        rationale = str(getattr(pred, "rationale", "")).strip()
        return dspy.Prediction(score_0_to_10=score, verdict=verdict, rationale=rationale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

    ccp = CCPPipeline(
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

    pred_rows: list[dict[str, Any]] = []
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
        write_jsonl(args.save_predictions, pred_rows)
        print(f"Saved predictions: {args.save_predictions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
