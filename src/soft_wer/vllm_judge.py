"""In-process vLLM judge backend for Soft WER.

Loads the judge model directly with `vllm.LLM(...)` and issues a single batched
`llm.chat(...)` call over all spans across all utterances. No external server.
"""

import re

from soft_wer.core import JudgeFn, JudgeJob, SpanDecision


FIELD_HEADER = re.compile(r"\[\[\s*##\s*(\w+)\s*##\s*\]\]")


JUDGE_SYSTEM_PROMPT = """You judge whether a reference span and a hypothesis span from an ASR transcription are semantically equivalent *in the context of the full reference sentence*.

Answer True when the spans convey the same meaning even if the surface form differs. Typical equivalent cases:
- Compound vs. space-split: "LIFEBLOOD" vs "LIFE BLOOD"
- Numerals vs. spelled-out numbers: "1996" vs "nineteen ninety six"
- Common abbreviations: "MISTER" vs "MR", "DOCTOR" vs "DR"
- Hyphenation / punctuation differences

Answer False when the meaning changes, words are dropped/added that alter content, or the substitution conveys a different fact (e.g., "HAS FLED" vs "IS FLOODED", "WAS" vs "IS" when tense matters, "AT EM" vs "ADAM").

The reference span may be empty (pure insertion in hypothesis) or the hypothesis span may be empty (pure deletion). Judge whether the added/dropped content is semantically vacuous (e.g., filler) or meaningful.

Your output must follow this exact structure:

[[ ## reasoning ## ]]
<one or two sentences of reasoning>

[[ ## equivalent ## ]]
<True or False>

[[ ## completed ## ]]
"""

JUDGE_USER_TEMPLATE = """[[ ## reference_sentence ## ]]
{reference_sentence}

[[ ## reference_span ## ]]
{reference_span}

[[ ## hypothesis_span ## ]]
{hypothesis_span}

Respond with the corresponding output fields, starting with `[[ ## reasoning ## ]]`, then `[[ ## equivalent ## ]]`, ending with `[[ ## completed ## ]]`."""


def build_judge_messages(span: SpanDecision, ref_sentence: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            reference_sentence=ref_sentence,
            reference_span=span.ref_span or "<EMPTY>",
            hypothesis_span=span.hyp_span or "<EMPTY>",
        )},
    ]


def parse_structured_output(text: str) -> dict[str, str]:
    """Parse `[[ ## field ## ]]`-delimited output into {field: value}."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        m = FIELD_HEADER.match(line.strip())
        if m:
            header = m.group(1)
            remainder = line[m.end():].strip()
            sections.append((header, [remainder] if remainder else []))
        else:
            sections[-1][1].append(line)
    out: dict[str, str] = {}
    for k, v in sections:
        if k is None or k in out:
            continue
        out[k] = "\n".join(v).strip()
    return out


def parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    s = raw.strip().lower().strip('.,"\'` ')
    if s in {"true", "yes", "equivalent", "1"}:
        return True
    if s in {"false", "no", "not equivalent", "0"}:
        return False
    # Fallback: look at the first token
    first = s.split()[0] if s else ""
    if first in {"true", "yes"}:
        return True
    if first in {"false", "no"}:
        return False
    return None


def build_vllm_judge_fn(
    model: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = -1,
    enable_thinking: bool = False,
    seed: int = 0,
    vllm_kwargs: dict | None = None,
) -> JudgeFn:
    """Load a vLLM model in-process and return a JudgeFn that batches all span jobs."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, seed=seed, **(vllm_kwargs or {}))
    sampling = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, n=1,
    )

    def _judge_fn(jobs: list[JudgeJob]) -> None:
        if not jobs:
            return
        messages = [build_judge_messages(span, ref_sentence) for (_, _, span, ref_sentence) in jobs]
        try:
            outputs = llm.chat(
                messages,
                sampling_params=sampling,
                use_tqdm=True,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
        except TypeError:
            # Older vllm versions don't accept chat_template_kwargs.
            outputs = llm.chat(messages, sampling_params=sampling, use_tqdm=True)

        for (_, _, span, _ref), out in zip(jobs, outputs):
            text = out.outputs[0].text.strip()
            # Strip any <think>...</think> block (present when thinking is enabled).
            m = re.match(r"<think>\n(.+?)</think>\s*", text, flags=re.DOTALL)
            thinking = None
            if m:
                thinking = m.group(1).strip()
                text = text[m.end():].strip()
            fields = parse_structured_output(text)
            eq = parse_bool(fields.get("equivalent"))
            reasoning = fields.get("reasoning")
            if thinking and not reasoning:
                reasoning = thinking
            if eq is None:
                span.error = f"could not parse equivalent from: {text[:200]!r}"
            span.equivalent = eq
            span.reasoning = reasoning

    return _judge_fn
