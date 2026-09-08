"""
Segment-first material primitives shared by the live video-match pipeline.

素材匹配主流程已迁移到 app/services/video_match.py（三段漏斗：粗排 →
精排 → VLM 走查 → image-gen 回填）。本模块只保留跨模块共享的数据结构
与常量：SegmentMaterials 及其持久化/序列化辅助、每段素材名额下限、
搜索翻页上限与 CJK 搜索词过滤。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import re

from loguru import logger

from app.services import task_artifacts

# Number of clips to download per segment. The assembler cycles through them
# when a segment lasts longer than one clip, keeping visual variety without
# global shuffling.
CLIPS_PER_SEGMENT = 3

# VLM 过滤拒收当前页全部候选时的最大翻页数（issue #9 D6）。Pixabay/Pexels
# 支持 page 参数；Coverr 分页无文档确认，远端实现按单页处理。
MAX_SEARCH_PAGES = 2

# 字符级 CJK 判定：搜索 API（Pexels/Pixabay/Coverr）仅接受英文查询，含
# 中日韩字符的词召回极差。素材层是最后一道防线——即使上游 LLM 词条、
# 片段原文或主题词带 CJK，也一律不作为搜索词发送。
_CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")

# VLM 过滤审计记录条数上限。一个分段在最坏情况下可能产生大量判定记录，
# 截断到合理长度避免任务清单被单段撑爆。
_MAX_FILTER_RECORDS = 12


def _english_search_term(term: str) -> str:
    """
    返回可安全发给搜索 API 的英文搜索词；含 CJK 字符时返回空串。

    空串会让该搜索词被跳过（例如中文片段原文），落到下一个自有词条或
    英文主题词，而不是把必然低召回的混合查询发给供应商。
    """
    candidate = (term or "").strip()
    if not candidate or _CJK_PATTERN.search(candidate):
        return ""
    return candidate


def english_search_term(term: str) -> str:
    """
    `_english_search_term` 的公开包装。

    供任务编排层（task.py 的早期失败守卫与 image-gen 回填回调）复用
    同一条 CJK 过滤规则，避免跨模块引用私有实现或在两处重复正则逻辑。
    """
    return _english_search_term(term)


@dataclass
class SegmentMaterials:
    """Structured material result for one narration segment."""

    index: int
    search_term: str
    clips: List[str] = field(default_factory=list)
    # The term that actually produced the clips ("" when nothing was found).
    # With quota-fill, this is the FIRST contributing term; later terms may
    # have contributed the remainder — see search_attempts for the chain.
    resolved_term: str = ""
    # Level of the first contributing term: "self", "subject", or "" when
    # all levels failed.
    fallback_level: str = ""
    # Audit trail of every fallback attempt: {"level", "term", "found"} in
    # tried order, so the manifest shows what each search returned even when
    # the level ultimately failed.
    search_attempts: List[dict] = field(default_factory=list)
    # Per-clip provenance: {"url", "local_file"} for every downloaded clip,
    # in the same order as `clips`.
    clip_sources: List[dict] = field(default_factory=list)
    # VLM filter audit trail (issue #9): {"term", "asset_id", "verdict",
    # "reason", "image_source", "attempts", "page"} per judged candidate,
    # in judged order. Empty when the filter is disabled.
    vlm_filter: List[dict] = field(default_factory=list)
    # Subject-level image generation audit trail: {"model", "prompt",
    # "source", "image_size", "image", "clip", "attempts"} — exactly one
    # record when the segment fell through to the generated concept image,
    # empty otherwise.
    image_gen: List[dict] = field(default_factory=list)


def persist_segment_material_sources(
    task_id: str,
    materials: List[SegmentMaterials],
) -> None:
    """
    Append per-segment material provenance to the task manifest.

    Best-effort like `material._persist_material_sources`: the manifest is an
    auxiliary record and must never break video generation.
    """
    records: List[dict[str, Any]] = []
    for segment_materials in materials:
        records.append(
            {
                "index": segment_materials.index,
                "search_term": segment_materials.search_term,
                "resolved_term": segment_materials.resolved_term,
                "fallback_level": segment_materials.fallback_level,
                "clips": [Path(clip).name for clip in segment_materials.clips],
                "search_attempts": segment_materials.search_attempts,
                "clip_sources": segment_materials.clip_sources,
                "vlm_filter": segment_materials.vlm_filter,
                "image_gen": segment_materials.image_gen,
            }
        )
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            segment_materials=records,
        )
        if saved:
            logger.info(
                f"saved segment material records: task_id={task_id}, "
                f"segments={len(records)}"
            )
    except Exception as exc:
        logger.warning(
            "failed to persist segment material records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def segments_to_records(materials: List[SegmentMaterials]) -> List[dict[str, Any]]:
    """Convert SegmentMaterials into JSON-safe dicts for state persistence."""
    return [
        {
            "index": m.index,
            "search_term": m.search_term,
            "resolved_term": m.resolved_term,
            "fallback_level": m.fallback_level,
            "clips": list(m.clips),
            "search_attempts": [dict(a) for a in m.search_attempts],
            "clip_sources": [dict(s) for s in m.clip_sources],
            "vlm_filter": [dict(f) for f in m.vlm_filter],
            "image_gen": [dict(g) for g in m.image_gen],
        }
        for m in materials
    ]
