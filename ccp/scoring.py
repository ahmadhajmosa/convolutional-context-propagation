from __future__ import annotations

import re


def parse_score_0_to_10(text: str) -> float:
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


def parse_float_0_to_1(value: object, default: float = 0.0) -> float:
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


def is_trueish(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1", "supported", "pass"}


def normalize_for_match(text: str) -> str:
    lowered = text.lower().replace("*", "")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def extract_quoted_snippets(text: str) -> list[str]:
    snippets = re.findall(r'"([^"]{3,400})"', text)
    return [s.strip() for s in snippets if s.strip()]


def count_supported_quotes(chunk: str, quoted_evidence: str) -> int:
    chunk_norm = normalize_for_match(chunk)
    count = 0
    for snippet in extract_quoted_snippets(quoted_evidence):
        snippet_norm = normalize_for_match(snippet)
        if snippet_norm and snippet_norm in chunk_norm:
            count += 1
    return count

