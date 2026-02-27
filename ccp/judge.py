from __future__ import annotations

import dspy

from .scoring import parse_score_0_to_10


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
        score = parse_score_0_to_10(str(getattr(pred, "score_0_to_10", "")))
        verdict = str(getattr(pred, "verdict", "")).strip()
        rationale = str(getattr(pred, "rationale", "")).strip()
        return dspy.Prediction(score_0_to_10=score, verdict=verdict, rationale=rationale)

