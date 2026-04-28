"""CLI: compute Soft WER over an input log using an in-process vLLM judge."""

import argparse
import json
import sys
from pathlib import Path

from soft_wer.core import compute_soft_wer, write_result
from soft_wer.vllm_judge import build_vllm_judge_fn


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="JSON file mapping id -> {pred, label, ...}")
    ap.add_argument("--output", required=True, help="Output JSON path for the full result")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id or local path for vllm.LLM")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1,
                    help="vLLM top-k; -1 disables (default).")
    ap.add_argument("--thinking", action="store_true", help="Enable thinking tokens in the chat template")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N utterances")
    ap.add_argument("--pred-key", default="pred")
    ap.add_argument("--ref-key", default="label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    # Input may contain aggregate scalars (e.g. "total_wer") at the top level;
    # keep only per-utterance dict entries.
    items = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    if args.limit is not None:
        items = items[:args.limit]

    ids = [k for k, _ in items]
    hyps = [v[args.pred_key] for _, v in items]
    refs = [v[args.ref_key] for _, v in items]

    judge_fn = build_vllm_judge_fn(
        args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.thinking,
        seed=args.seed,
    )
    aggregate, utt_results = compute_soft_wer(refs, hyps, judge_fn, ids=ids)

    print(json.dumps(aggregate, indent=2), file=sys.stderr)
    write_result(args.output, aggregate, utt_results, config={
        "backend": "vllm",
        "model": args.model,
        "input": args.input,
        "pred_key": args.pred_key,
        "ref_key": args.ref_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "thinking": args.thinking,
    })


if __name__ == "__main__":
    main()
