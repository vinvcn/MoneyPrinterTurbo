"""
逐段 LLM 查询生成（视频素材匹配三段漏斗的第一级）。

segment-first 流水线的素材匹配正在重构为三级漏斗：粗排（embedding 召回
排序）→ 精排（VL 模型重排）→ VLM 走查。本模块在分段之后为每个片段生成
一份查询包：terms（英文搜索词，主词在前，沿用 segment_terms 的既有语义）、
coarse_query（宽场景描述，供粗排 embedding 检索）、fine_query（精确视觉
时刻描述，供精排重排比对）。

与逐段搜索词提炼（segment_terms.py）的成本控制方式一致：一次
llm.generate_response 调用同时产出三类查询；响应无法解析出 JSON 对象时
按 segment_terms 的重试策略重试，重试耗尽后 fail-open 返回空查询包
（terms=[], coarse_query=None, fine_query=None），降级由调用方处理。
"""

import json
import re
from dataclasses import dataclass

from loguru import logger

from app.services import llm

# 解析失败的最大尝试次数（与 segment_terms._MAX_RETRIES 同构，镜像其重试
# 语义：range(1, _MAX_RETRIES + 1) 即最多 2 次调用）。
_MAX_RETRIES = 2

# 每段最多提炼的搜索词数量；第一个词为主搜索词，其余为备用词（沿用
# segment_terms.TERMS_PER_SEGMENT 的语义）。
_MAX_TERMS = 3

# 字符级 CJK 判定：与 segment_terms/segment_material 保持同一正则——搜索
# API（Pexels/Pixabay/Coverr）仅接受英文查询，含中日韩字符的词召回极差。
_CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")


@dataclass
class SegmentQueries:
    """单个片段的素材匹配查询包（三段漏斗共用一份输入）。"""

    # 1..3 个英文搜索词，主词在前；全部词条被过滤时为空列表（降级由调用方处理）。
    terms: list[str]
    # 宽场景描述（英文一句话，粗排 embedding 检索）；LLM 失败/字段缺失时为 None。
    coarse_query: str | None
    # 精确视觉时刻描述（英文一句话，精排重排）；LLM 失败/字段缺失时为 None。
    fine_query: str | None


def _contains_cjk(text: str) -> bool:
    """判断文本是否包含中日韩字符（搜索 API 仅接受英文查询）。"""
    return bool(_CJK_PATTERN.search(text or ""))


def _build_prompt(subject: str, segment_text: str) -> str:
    """
    构造单段三查询提示词。

    保持与 generate_terms / segment_terms 相同的 Role/Constrains/Output
    Example 结构（模型对该格式已稳定），Context 为单段文本并绑定主题词。
    """
    output_example = json.dumps(
        {
            "terms": ["panda eating bamboo", "bamboo forest", "panda cub playing"],
            "coarse_query": (
                "A giant panda sits in a lush green bamboo forest "
                "munching on bamboo stalks."
            ),
            "fine_query": (
                "Close-up of a giant panda gripping a bamboo stalk with its "
                "paws and stripping the leaves with its teeth."
            ),
        },
        ensure_ascii=False,
    )

    return f"""
# Role: Video Material Query Generator

## Goals:
For one narration segment, generate stock-video search terms plus two English scene descriptions used to match material clips.

## Constrains:
1. return ONLY a json object with exactly three keys: "terms", "coarse_query", "fine_query". you must not return anything else. you must not return the script.
2. "terms" must be a json-array of 1-3 search terms: each term consists of 1-3 words, always translate the main subject of the video into English and append it, and the terms must describe DIFFERENT visual angles, scenes, or shot types of the segment's moment — never synonyms or minor rephrasings, because near-duplicate terms return the same stock-footage candidate pool.
3. "coarse_query" must be one English sentence broadly describing the segment's visual scene (environment, subjects, mood) for coarse embedding retrieval.
4. "fine_query" must be one English sentence precisely describing the segment's key visual moment (action, composition, details) for fine visual reranking.
5. every value must be pure English (A-Z letters, spaces and punctuation only). never include any Chinese or other non-English characters, even if the subject or the narration is written in Chinese.

## Output Example:
{output_example}

## Context:
### Video Subject
{subject}

### Narration Segment
{segment_text}

Please note that you must use English for generating search terms and queries; Chinese is not accepted. If the subject or a segment is written in Chinese, translate its meaning into English.
""".strip()


def _strip_code_fences(text: str) -> str:
    """剥掉 markdown 代码围栏（```json ... ```）；无围栏时原样返回。"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def _clean_query(value: object) -> str | None:
    """查询字段规整：非字符串或空白视为缺失，返回 None。"""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_terms(raw_terms: object, subject: str) -> list[str]:
    """
    词条规整，逐字沿用 segment_terms 的语义：截断、追加英文主题词、
    逐词 CJK 过滤；全部词条被过滤时返回空列表（降级由调用方处理）。
    """
    if not isinstance(raw_terms, list):
        return []
    cleaned = [str(term).strip() for term in raw_terms if str(term).strip()]
    # 超量词条截断；主词在前，顺序与模型输出一致。
    cleaned = cleaned[:_MAX_TERMS]

    # 主题词含 CJK 时直接放弃追加（没有可追加的英文主题），让词条保持
    # 纯英文，而不是发出必然低召回的混合查询。
    append_subject = bool(subject)
    if subject and _contains_cjk(subject):
        logger.warning(
            "video subject contains CJK, appending it would break "
            f"English-only search: subject={subject!r}"
        )
        append_subject = False

    terms: list[str] = []
    for term in cleaned:
        final_term = f"{term} {subject}".strip() if append_subject else term
        # 逐词过滤：单个词条含 CJK 只丢弃该词，分段靠剩余词条存活。
        if _contains_cjk(final_term):
            logger.warning(f"segment term contains CJK, discarding: term={final_term!r}")
            continue
        terms.append(final_term)
    return terms


def _parse_segment_queries(response: object, subject: str) -> SegmentQueries | None:
    """
    从模型回复解析查询包；无法得到 JSON 对象时返回 None 触发重试。

    防御性解析：容忍 markdown 围栏与 JSON 前后的说明文字；字段缺失或
    类型不符时逐字段降级（terms → 空列表，查询 → None），只有 JSON 本身
    不可解析（包括 llm 层的 "Error: " 失败串）才算解析失败。
    """
    text = str(response or "").strip()
    if not text or text.startswith("Error: "):
        return None

    candidate = _strip_code_fences(text)
    match = re.search(r"\{.*}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    return SegmentQueries(
        terms=_normalize_terms(data.get("terms"), subject),
        coarse_query=_clean_query(data.get("coarse_query")),
        fine_query=_clean_query(data.get("fine_query")),
    )


def generate_segment_queries(subject: str, segment_text: str) -> SegmentQueries:
    """
    为单个片段生成素材匹配查询包（terms + coarse_query + fine_query）。

    Args:
        subject: 视频主题，翻译成英文后附加进每个搜索词（含 CJK 时不追加）。
        segment_text: 该片段的旁白原文。

    Returns:
        SegmentQueries。解析失败按 segment_terms 的策略重试，重试耗尽后
        fail-open 返回空查询包（terms=[], coarse_query=None,
        fine_query=None），绝不抛异常；空词条等降级由调用方处理。
    """
    subject = str(subject or "").strip()
    segment_text = str(segment_text or "").strip()
    prompt = _build_prompt(subject, segment_text)

    for attempt in range(1, _MAX_RETRIES + 1):
        response = llm.generate_response(prompt)
        parsed = _parse_segment_queries(response, subject)
        if parsed is not None:
            return parsed
        logger.warning(
            f"segment queries response unusable: attempt={attempt}, "
            f"response={str(response)[:120]!r}"
        )

    logger.warning(
        "segment query generation failed after retries; returning empty "
        "queries and leaving degradation to the caller"
    )
    return SegmentQueries(terms=[], coarse_query=None, fine_query=None)
