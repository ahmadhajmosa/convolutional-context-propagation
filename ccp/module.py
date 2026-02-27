from __future__ import annotations

import json
from typing import Any

import dspy

from .io_utils import chunk_text
from .scoring import count_supported_quotes, is_trueish, parse_float_0_to_1
from .signatures import get_signature_set


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

        sigs = get_signature_set(signature_mode)
        self.plan = dspy.Predict(sigs.plan)
        self.extract = dspy.Predict(sigs.extract)
        self.fuse = dspy.Predict(sigs.fuse)
        self.verify = dspy.Predict(sigs.verify)

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
            model_evidence_found = is_trueish(getattr(pred, "evidence_found", ""))
            supported_quote_count = count_supported_quotes(chunk, quoted_evidence)

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
        if is_trueish(getattr(verified, "is_supported", "")):
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

            relevance = parse_float_0_to_1(getattr(pred, "relevance_0_to_1", "0.0"))
            key_points = str(getattr(pred, "key_points", "")).strip()
            candidate_fragment = str(getattr(pred, "candidate_fragment", "")).strip()
            evidence_spans = str(getattr(pred, "evidence_spans", "")).strip()
            unresolved = str(getattr(pred, "unresolved", "")).strip()
            supported_quote_count = count_supported_quotes(chunk, evidence_spans)
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
                confidence_0_to_1 = parse_float_0_to_1(getattr(fused, "confidence_0_to_1", "0.0"))
                unresolved = str(getattr(fused, "unresolved", "")).strip()

                if confidence_0_to_1 == 0.0:
                    input_scores: list[float] = []
                    for node in window_nodes:
                        try:
                            payload = json.loads(node)
                            if isinstance(payload, dict):
                                input_scores.append(parse_float_0_to_1(payload.get("relevance_score", 0.0)))
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
            score = parse_float_0_to_1(payload.get("relevance_score", 0.0))
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
        if is_trueish(getattr(verified, "is_valid", "")):
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

