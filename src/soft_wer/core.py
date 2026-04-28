"""Core Soft WER metric: alignment, span merging, and aggregation.

This module is judge-agnostic. A `JudgeFn` is supplied by the caller and is
responsible for setting `equivalent`/`reasoning`/`error` on each `SpanDecision`.
For an in-process vLLM judge, see `soft_wer.vllm_judge`.
"""

import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jiwer


@dataclass
class SpanDecision:
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int
    ref_span: str
    hyp_span: str
    chunk_types: list[str]
    n_sub: int
    n_del: int
    n_ins: int
    equivalent: bool | None = None
    reasoning: str | None = None
    error: str | None = None


@dataclass
class UtteranceResult:
    id: str | None
    reference: str
    hypothesis: str
    n_ref_words: int
    n_sub: int
    n_del: int
    n_ins: int
    n_sub_forgiven: int
    n_del_forgiven: int
    n_ins_forgiven: int
    wer: float
    soft_wer: float
    spans: list[SpanDecision] = field(default_factory=list)


# A single judge work-item: (utt_idx, span_idx, span_to_mutate, reference_sentence).
JudgeJob = tuple[int, int, SpanDecision, str]
# A judge function takes all jobs at once and mutates each span's equivalent/reasoning/error.
JudgeFn = Callable[[list[JudgeJob]], None]


def merge_alignment_into_spans(alignment, ref_words, hyp_words) -> list[SpanDecision]:
    """Group adjacent non-`equal` AlignmentChunks into contiguous mismatch spans."""
    spans: list[SpanDecision] = []
    current: SpanDecision | None = None
    for ch in alignment:
        if ch.type == "equal":
            if current is not None:
                spans.append(current)
                current = None
            continue
        if current is None:
            current = SpanDecision(
                ref_start=ch.ref_start_idx,
                ref_end=ch.ref_end_idx,
                hyp_start=ch.hyp_start_idx,
                hyp_end=ch.hyp_end_idx,
                ref_span="",
                hyp_span="",
                chunk_types=[ch.type],
                n_sub=0,
                n_del=0,
                n_ins=0,
            )
        else:
            current.chunk_types.append(ch.type)
            current.ref_end = ch.ref_end_idx
            current.hyp_end = ch.hyp_end_idx
        if ch.type == "substitute":
            current.n_sub += ch.ref_end_idx - ch.ref_start_idx
        elif ch.type == "delete":
            current.n_del += ch.ref_end_idx - ch.ref_start_idx
        elif ch.type == "insert":
            current.n_ins += ch.hyp_end_idx - ch.hyp_start_idx
    if current is not None:
        spans.append(current)
    for s in spans:
        s.ref_span = " ".join(ref_words[s.ref_start:s.ref_end])
        s.hyp_span = " ".join(hyp_words[s.hyp_start:s.hyp_end])
    return spans


def compute_soft_wer(
    refs: list[str],
    hyps: list[str],
    judge_fn: JudgeFn,
    *,
    ids: list[str] | None = None,
) -> tuple[dict, list[UtteranceResult]]:
    """Compute standard WER, soft WER, and per-utterance span decisions.

    `judge_fn` is called once with all span jobs across all utterances; it is
    expected to mutate each SpanDecision in place (setting equivalent / reasoning
    / error). Returns (aggregate_stats, per_utterance_results)."""
    wo = jiwer.process_words(refs, hyps)
    ref_word_lists = wo.references
    hyp_word_lists = wo.hypotheses
    alignments = wo.alignments

    utt_results: list[UtteranceResult] = []
    jobs: list[JudgeJob] = []

    for utt_idx, (ref, hyp, ref_words, hyp_words, alignment) in enumerate(
        zip(refs, hyps, ref_word_lists, hyp_word_lists, alignments)
    ):
        spans = merge_alignment_into_spans(alignment, ref_words, hyp_words)
        n_sub = sum(s.n_sub for s in spans)
        n_del = sum(s.n_del for s in spans)
        n_ins = sum(s.n_ins for s in spans)
        n_ref = len(ref_words)

        utt_results.append(UtteranceResult(
            id=ids[utt_idx] if ids is not None else None,
            reference=ref,
            hypothesis=hyp,
            n_ref_words=n_ref,
            n_sub=n_sub, n_del=n_del, n_ins=n_ins,
            n_sub_forgiven=0, n_del_forgiven=0, n_ins_forgiven=0,
            wer=(n_sub + n_del + n_ins) / n_ref if n_ref > 0 else 0.0,
            soft_wer=0.0,
            spans=spans,
        ))

        for span_idx, span in enumerate(spans):
            jobs.append((utt_idx, span_idx, span, ref))

    judge_fn(jobs)

    total_edits = 0
    total_forgiven = 0
    total_ref_words = 0

    for u in utt_results:
        forgiven_s = sum(s.n_sub for s in u.spans if s.equivalent)
        forgiven_d = sum(s.n_del for s in u.spans if s.equivalent)
        forgiven_i = sum(s.n_ins for s in u.spans if s.equivalent)
        u.n_sub_forgiven = forgiven_s
        u.n_del_forgiven = forgiven_d
        u.n_ins_forgiven = forgiven_i
        soft_edits = (u.n_sub - forgiven_s) + (u.n_del - forgiven_d) + (u.n_ins - forgiven_i)
        u.soft_wer = soft_edits / u.n_ref_words if u.n_ref_words > 0 else 0.0

        total_edits += u.n_sub + u.n_del + u.n_ins
        total_forgiven += forgiven_s + forgiven_d + forgiven_i
        total_ref_words += u.n_ref_words

    aggregate = {
        "wer": wo.wer,
        "soft_wer": (total_edits - total_forgiven) / total_ref_words if total_ref_words > 0 else 0.0,
        "n_substitutions": wo.substitutions,
        "n_deletions": wo.deletions,
        "n_insertions": wo.insertions,
        "n_ref_words": total_ref_words,
        "n_spans": sum(len(u.spans) for u in utt_results),
        "n_spans_forgiven": sum(1 for u in utt_results for s in u.spans if s.equivalent),
        "n_spans_judge_error": sum(1 for u in utt_results for s in u.spans if s.error is not None),
    }
    return aggregate, utt_results


def write_result(output_path: str, aggregate: dict, utt_results: list[UtteranceResult], config: dict) -> None:
    out = {
        "aggregate": aggregate,
        "config": config,
        "utterances": [asdict(u) for u in utt_results],
    }
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"wrote {output_path}", file=sys.stderr)
