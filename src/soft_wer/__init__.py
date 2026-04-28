"""Soft WER: word error rate that forgives semantically-equivalent mismatched spans."""

from soft_wer.core import (
    JudgeFn,
    JudgeJob,
    SpanDecision,
    UtteranceResult,
    compute_soft_wer,
    merge_alignment_into_spans,
    write_result,
)

__all__ = [
    "JudgeFn",
    "JudgeJob",
    "SpanDecision",
    "UtteranceResult",
    "compute_soft_wer",
    "merge_alignment_into_spans",
    "write_result",
]
