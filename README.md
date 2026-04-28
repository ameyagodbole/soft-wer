# soft-wer

Word error rate that forgives semantically-equivalent mismatched spans.

Standard WER counts every surface-level edit between a reference and a hypothesis transcript. `soft-wer` aligns reference and hypothesis with [`jiwer`](https://github.com/jitsi/jiwer), groups adjacent non-equal alignment chunks into contiguous mismatch *spans*, and asks an LLM judge whether each span is semantically equivalent in context. Edits inside forgiven spans are not counted toward soft WER.

Cases the judge typically forgives:

- Compound vs. space-split: `LIFEBLOOD` vs `LIFE BLOOD`
- Numerals vs. spelled-out numbers: `1996` vs `nineteen ninety six`
- Common abbreviations: `MISTER` vs `MR`, `DOCTOR` vs `DR`
- Hyphenation / punctuation differences

The judge runs in-process via [vLLM](https://github.com/vllm-project/vllm) — no external server. All spans across all utterances are batched into a single `llm.chat(...)` call.

## Install

```bash
uv sync
```

Requires a CUDA-capable GPU (vLLM dependency). Python ≥ 3.11.

## CLI

```bash
uv run soft-wer \
    --input sample_wer_log.json \
    --output results/sample.json \
    --model Qwen/Qwen3-4B \
    --max-tokens 4096 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 50
```

Input format: a JSON file mapping `id -> {pred, label, ...}`. Aggregate scalars (e.g. `total_wer`) at the top level are ignored. See `sample_wer_log.json` for an example.

Output: a JSON file with `aggregate` (overall WER + soft WER + edit counts), `config` (the run parameters), and `utterances` (per-utterance results including each span's `equivalent` decision and `reasoning`).

Useful flags:

- `--pred-key` / `--ref-key` — override the field names in the input (defaults: `pred` / `label`)
- `--limit N` — only process the first N utterances (handy for smoke tests)
- `--thinking` — enable Qwen-style thinking tokens; reasoning gets stripped from `<think>...</think>` before parsing

## Library

```python
from soft_wer import compute_soft_wer
from soft_wer.vllm_judge import build_vllm_judge_fn

judge_fn = build_vllm_judge_fn("Qwen/Qwen3-4B")
aggregate, utt_results = compute_soft_wer(refs, hyps, judge_fn, ids=ids)
```

`compute_soft_wer` is judge-agnostic. A `JudgeFn` is any callable that takes a list of `JudgeJob` tuples and mutates each `SpanDecision` in place (setting `equivalent`, `reasoning`, optionally `error`). Bring your own judge — vLLM is one option, not a requirement of the core metric.

## SLURM

`submit_soft_wer_vllm.sh` is a SLURM submit script that runs the CLI on `sample_wer_log.json`. Override the model / paths with env vars:

```bash
MODEL=Qwen/Qwen3-8B sbatch submit_soft_wer_vllm.sh
```

## Layout

```
src/soft_wer/
  core.py        # SpanDecision, UtteranceResult, compute_soft_wer, write_result
  vllm_judge.py  # build_vllm_judge_fn + prompt + output parsing
  cli.py         # `soft-wer` console script
```
