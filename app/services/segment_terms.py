"""
逐段搜索词提取。

segment-first 流水线最初直接把片段原文作为搜索词。实践发现 stock 视频
API 对长中文句子的召回质量差且不稳定（超时/SSL 失败率显著上升），因此
在分段之后、素材搜索之前，先用 LLM 把每个片段提炼成 1-3 个词的英文
搜索词——沿用 `generate_terms` 的提示词结构（模型对该格式已稳定），
但把 Context 换成单个片段并绑定主题词。

为控制成本，多个片段合并成尽量少的 LLM 调用（batch）；单条解析失败
时回退为"片段原文/主题词"，由素材层的既有 fallback 链兜底。
"""

import json
import re
from typing import List

from loguru import logger

from app.services import llm

# 一次 LLM 请求最多提炼的分段数。过大既降低逐段质量，也更容易触发
# 模型截断；6 与 generate_terms 的 amount=5~8 规模一致。
_DEFAULT_BATCH_SIZE = 6
# 每段提炼的候选词数量；第一个词为主搜索词，其余留作未来扩展。
TERMS_PER_SEGMENT = 1

_MAX_RETRIES = 2


def build_segment_terms_prompt(
    video_subject: str,
    segment_text: str,
    amount: int = TERMS_PER_SEGMENT,
) -> str:
    """
    构造与 `llm.generate_terms` 同构的单段搜索词提示词。

    保持相同的 Role/Constrains/Output Example 结构，让模型在已验证的
    格式上工作；Context 缩小为单段文本，并始终携带主题词。
    """
    example_terms = [f"black hole visual topic {i}" for i in range(1, max(amount, 1))]
    example_terms.append("final visual topic")
    output_example = json.dumps(example_terms[:amount], ensure_ascii=False)

    return f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the content of one narration segment.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to what the segment describes, not just the subject.
5. reply with english search terms only.

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Narration Segment
{segment_text}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()


def _build_batch_prompt(
    video_subject: str,
    numbered_segments: List[tuple[int, str]],
    per_segment: int = TERMS_PER_SEGMENT,
) -> str:
    """把多个分段合并进一次请求，返回 JSON 数组与分段一一对应。"""
    segment_block = "\n".join(
        f"{index}. {text}" for index, text in numbered_segments
    )
    total = per_segment * len(numbered_segments)
    example = json.dumps([f"term {i}" for i in range(1, total + 1)], ensure_ascii=False)

    return f"""
# Role: Video Search Terms Generator

## Goals:
For each numbered narration segment below, generate {per_segment} stock-video search terms describing its visual content.

## Constrains:
1. return ONLY a json-array with exactly {total} strings.
2. the array must be in the same order as the numbered segments: segment 1's {per_segment} term(s) first, then segment 2's, and so on.
3. each search term should consist of 1-3 words, always add the main subject of the video.
4. each search term must describe what that segment's narration says (its visual moment), not the whole video.
5. reply with english search terms only.

## Output Example:
{example}

## Context:
### Video Subject
{video_subject}

### Numbered Narration Segments
{segment_block}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()


def _parse_term_array(response: str, expected: int) -> List[str]:
    """从模型回复中解析 json 数组；数量不符视为失败。"""
    match = re.search(r"\[.*]", response or "", re.DOTALL)
    if not match:
        return []
    try:
        terms = json.loads(match.group())
    except Exception:
        return []
    if not isinstance(terms, list):
        return []
    cleaned = [str(term).strip() for term in terms if str(term).strip()]
    if len(cleaned) < expected:
        return []
    return cleaned[:expected]


def extract_terms_for_segments(
    segments: List[dict],
    video_subject: str,
    amount: int = TERMS_PER_SEGMENT,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[int, str]:
    """
    为每个分段提炼英文搜索词。

    Args:
        segments: 至少含 {"index", "text"} 的分段记录；空文本分段跳过。
        video_subject: 视频主题，附加进每个搜索词。
        amount: 每段生成的词数；取第一个作为搜索词。
        batch_size: 一次 LLM 请求覆盖的分段数。

    Returns:
        {segment_index: search_term}。某段持续失败时该段不出现在
        结果里，由素材搜索层的回退链兜底。
    """
    per_segment = max(1, int(amount))
    terms: dict[int, str] = {}
    pending = [
        (int(segment.get("index", position)), str(segment.get("text") or "").strip())
        for position, segment in enumerate(segments)
    ]
    pending = [(index, text) for index, text in pending if text]

    for offset in range(0, len(pending), max(1, int(batch_size))):
        batch = pending[offset : offset + max(1, int(batch_size))]
        # batch 内保持原始 segment index；编号只用于提示词里的顺序说明。
        numbered = [(position + 1, index, text) for position, (index, text) in enumerate(batch)]
        prompt = _build_batch_prompt(
            video_subject, [(n, t) for n, _, t in numbered], per_segment=per_segment
        )

        parsed: List[str] = []
        for attempt in range(1, _MAX_RETRIES + 1):
            response = llm._generate_response(prompt)
            if response.startswith("Error: "):
                logger.warning(
                    f"segment terms generation failed: attempt={attempt}, "
                    f"error={response.removeprefix('Error: ').strip()[:120]}"
                )
                continue
            expected = per_segment * len(batch)
            parsed = _parse_term_array(response, expected)
            if parsed:
                break
            logger.warning(
                f"segment terms response unusable: attempt={attempt}, "
                f"expected={expected} terms"
            )

        if not parsed:
            logger.warning(
                "segment terms extraction failed for batch; segments will "
                "fall back to raw text/subject search: "
                f"indices={[index for index, _ in batch]}"
            )
            continue

        for position, (_, index, _) in enumerate(numbered):
            term = parsed[position * per_segment]
            terms[index] = f"{term} {video_subject}".strip() if video_subject else term

    logger.info(
        f"extracted segment search terms: requested={len(pending)}, resolved={len(terms)}"
    )
    return terms
