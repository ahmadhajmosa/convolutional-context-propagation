from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CCPSignatureSet:
    plan: str
    extract: str
    fuse: str
    verify: str


LEGACY_SIGNATURES = CCPSignatureSet(
    plan="query -> answer_type, extraction_focus, abstention_rule",
    extract=(
        "chunk, query, answer_type, extraction_focus, abstention_rule -> "
        "evidence_found, quoted_evidence, atomic_facts, local_answer, confidence, missing_info"
    ),
    fuse=(
        "window_findings, query, answer_type, extraction_focus, abstention_rule -> "
        "fused_answer, fused_evidence, confidence, unresolved_points"
    ),
    verify=(
        "query, candidate_answer, fused_evidence, unresolved_points -> "
        "is_supported, corrected_answer, verifier_notes"
    ),
)


ADAPTIVE_SIGNATURES = CCPSignatureSet(
    plan="query -> task_type, expected_output_format, scoring_focus, abstention_policy",
    extract=(
        "chunk, query, task_type, expected_output_format, scoring_focus, abstention_policy -> "
        "relevance_0_to_1, key_points, candidate_fragment, evidence_spans, unresolved"
    ),
    fuse=(
        "window_findings, query, task_type, expected_output_format, scoring_focus, abstention_policy -> "
        "fused_points, draft_answer, confidence_0_to_1, unresolved"
    ),
    verify=(
        "query, task_type, expected_output_format, draft_answer, fused_points, unresolved -> "
        "is_valid, corrected_answer, verifier_notes"
    ),
)


def get_signature_set(signature_mode: str) -> CCPSignatureSet:
    if signature_mode == "legacy":
        return LEGACY_SIGNATURES
    if signature_mode == "adaptive":
        return ADAPTIVE_SIGNATURES
    raise ValueError("signature_mode must be one of: legacy, adaptive")

