"""
Per-segment material search/download with a deterministic fallback chain.

A segment-first pipeline needs more than a list of anonymous video paths:
each narration segment owns its own search, its own downloaded clips, and its
own fallback story when providers come back empty. This module keeps the
existing `download_videos` behavior untouched and adds a structured
per-segment flow on top of the shared search/cache/download primitives.

Fallback chain per segment (in order):
1. the segment's own text as the search term;
2. the previous segment's search term (its visuals are already on screen and
   still narratively adjacent);
3. the next segment's search term;
4. the video subject as a last resort.

Every attempt is recorded on the segment record so the task manifest can show
which term actually produced the visuals.

Optional VLM pre-download filter (issue #9): when the [vlm] config section is
enabled, every candidate is visually judged (thumbnail or first frame) before
download; irrelevant candidates are skipped, and exhausted pages roll into the
same fallback chain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List
import re

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import task_artifacts
from app.utils import utils

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

# VLM 过滤审计记录条数上限。一个分段在 2 页 × 多候选的最坏情况下可能产生
# 大量判定记录，截断到合理长度避免任务清单被单段撑爆。
_MAX_FILTER_RECORDS = 24


def _english_search_term(term: str) -> str:
    """
    返回可安全发给搜索 API 的英文搜索词；含 CJK 字符时返回空串。

    空串会让该回退层级被跳过（例如中文片段原文的 self 层），落到下一级
    （英文 LLM 词条或英文主题词），而不是把必然低召回的混合查询发给
    供应商。
    """
    candidate = (term or "").strip()
    if not candidate or _CJK_PATTERN.search(candidate):
        return ""
    return candidate


@dataclass
class SegmentMaterials:
    """Structured material result for one narration segment."""

    index: int
    search_term: str
    clips: List[str] = field(default_factory=list)
    # The term that actually produced the clips ("" when nothing was found).
    resolved_term: str = ""
    # Which fallback level produced the result: "self", "previous",
    # "next", "subject", or "" when all levels failed.
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


def _download_clips_for_term(
    items: List[MaterialInfo],
    needed_count: int,
    save_video: Callable[..., str],
    save_dir: str,
    judge_candidate: Callable[..., dict] | None = None,
    filter_records: List[dict] | None = None,
    segment_text: str = "",
    term: str = "",
    seen_urls: set[str] | None = None,
) -> tuple[List[str], List[dict]]:
    """
    Download up to `needed_count` unique clips; return paths and URL provenance.

    When `judge_candidate` is provided (VLM filter enabled), each candidate is
    visually judged before download; irrelevant candidates are skipped in favor
    of the next one (issue #9 D1/D7). `seen_urls` can be shared across calls
    (per fallback level) so page-2 candidates already downloaded from page 1
    are not fetched or re-judged twice.
    """
    saved_paths: List[str] = []
    clip_sources: List[dict] = []
    if seen_urls is None:
        seen_urls = set()
    for item in items:
        if len(saved_paths) >= needed_count:
            break
        if not item.url or item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        if judge_candidate is not None:
            verdict = judge_candidate(
                item=item,
                segment_text=segment_text,
                search_term=term,
            )
            if filter_records is not None:
                filter_records.append(verdict)
            if verdict.get("verdict") == "irrelevant":
                logger.info(
                    "vlm filter rejected candidate: "
                    f"asset_id={verdict.get('asset_id')}, "
                    f"reason={verdict.get('reason')!r}, "
                    f"image_source={verdict.get('image_source')}"
                )
                continue
            logger.info(
                "vlm filter accepted candidate: "
                f"asset_id={verdict.get('asset_id')}, "
                f"verdict={verdict.get('verdict')}, "
                f"image_source={verdict.get('image_source')}"
            )
        logger.info(f"downloading segment clip: {item.url}")
        try:
            saved_video_path = save_video(video_url=item.url, save_dir=save_dir)
        except Exception as exc:
            logger.warning(
                "failed to download segment clip: "
                f"provider={item.provider}, error={type(exc).__name__}, "
                f"detail={exc}"
            )
            continue
        if saved_video_path and saved_video_path not in saved_paths:
            logger.info(f"segment clip saved: {saved_video_path}")
            saved_paths.append(saved_video_path)
            clip_sources.append(
                {
                    "url": item.url,
                    "local_file": Path(saved_video_path).name,
                }
            )
    return saved_paths, clip_sources


def prepare_segment_materials(
    segments: List[dict],
    video_subject: str,
    search_videos: Callable[..., List[MaterialInfo]],
    save_video: Callable[..., str],
    video_aspect: VideoAspect,
    clip_duration: int = 5,
    clips_per_segment: int = CLIPS_PER_SEGMENT,
    save_dir: str = "",
    judge_candidate: Callable[..., dict] | None = None,
) -> List[SegmentMaterials]:
    """
    Search and download clips for every segment using the fallback chain.

    Args:
        segments: segment dicts with at least {"index", "text"}.
        video_subject: last-resort search term shared by all segments.
        search_videos: search callable (term, minimum_duration, video_aspect)
            -> List[MaterialInfo]; must already include caching. May accept a
            `page` kwarg when the provider supports pagination.
        save_video: download callable (url, save_dir) -> path ("" on failure).
        video_aspect: target orientation for remote filtering.
        clip_duration: minimum duration requested from providers.
        clips_per_segment: how many distinct clips to gather per segment.
        save_dir: download directory (empty = provider default cache).
        judge_candidate: optional VLM filter callable (issue #9); receives
            (item=MaterialInfo, segment_text=str, search_term=str) and returns
            a verdict record dict. None = filter disabled.

    Returns:
        One SegmentMaterials per input segment, in the same order.
    """
    subject = _english_search_term(str(video_subject or ""))
    # Cache search results across segments so neighbor fallbacks do not hit
    # the provider API again for a term that was already searched. Page-aware:
    # page 1 must exist before page 2 is fetched (issue #9 D6).
    search_cache: dict[tuple[str, int], List[MaterialInfo]] = {}
    def search_page_cached(term: str, page: int) -> List[MaterialInfo]:
        normalized = (term or "").strip()
        if not normalized:
            return []
        cache_key = (normalized, page)
        if cache_key not in search_cache:
            try:
                # 自定义/旧签名搜索函数（测试替身、第三方扩展）可能不接受
                # page 参数：先按带页码调用，不支持时退回无页码调用，保持
                # 旧实现按第一页结果继续工作。
                try:
                    found = list(
                        search_videos(
                            search_term=normalized,
                            minimum_duration=clip_duration,
                            video_aspect=video_aspect,
                            page=page,
                        )
                    )
                except TypeError as exc:
                    if "page" not in str(exc):
                        raise
                    found = list(
                        search_videos(
                            search_term=normalized,
                            minimum_duration=clip_duration,
                            video_aspect=video_aspect,
                        )
                    )
                # 逐条打印搜索返回的候选，供运行审计核对"搜到了什么"。
                logger.info(
                    f"segment search returned {len(found)} candidates for "
                    f"term={normalized!r}, page={page}"
                )
                for item in found:
                    logger.info(
                        f"  candidate: provider={item.provider}, "
                        f"duration={item.duration}s, url={item.url}"
                    )
                search_cache[cache_key] = found
            except Exception as exc:
                logger.warning(
                    "segment material search failed: "
                    f"term={normalized!r}, page={page}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                search_cache[cache_key] = []
        return search_cache[cache_key]

    def search_cached(term: str) -> List[MaterialInfo]:
        """Backward-compatible single-page access (page 1 only)."""
        return search_page_cached(term, 1)

    material_directory = save_dir
    if not material_directory:
        configured = str(config.app.get("material_directory", "")).strip()
        if configured and configured != "task":
            material_directory = configured

    results: List[SegmentMaterials] = []
    for position, segment in enumerate(segments):
        own_term = _english_search_term(
            str(segment.get("search_term") or segment.get("text") or "")
        )
        next_term = ""
        if position + 1 < len(segments):
            next_term = _english_search_term(
                str(
                    segments[position + 1].get("search_term")
                    or segments[position + 1].get("text")
                    or ""
                )
            )
        candidates: List[tuple[str, str]] = [
            ("self", own_term),
            ("previous", results[position - 1].resolved_term if position else ""),
            ("next", next_term),
        ]
        if subject:
            candidates.append(("subject", subject))

        segment_text = str(segment.get("text") or "")
        saved_paths: List[str] = []
        clip_sources: List[dict] = []
        resolved_term = ""
        fallback_level = ""
        search_attempts: List[dict] = []
        vlm_filter_records: List[dict] = []
        for level, term in candidates:
            term = (term or "").strip()
            if not term:
                continue
            # VLM 过滤启用时逐页尝试：当前页候选全部被拒收后翻下一页，
            # 翻页用尽仍凑不齐才落入下一个 fallback 层（issue #9 D6）。
            level_clips: List[str] = []
            level_sources: List[dict] = []
            level_seen_urls: set[str] = set()
            for page in range(1, MAX_SEARCH_PAGES + 1):
                page_items = search_page_cached(term, page)
                if not page_items:
                    break
                # 只有当当前页一个可用片段都没凑出来时才翻页；部分满足
                # （例如第 1 页下载成功但不足 3 个）时也继续翻页补齐。
                page_clips, page_sources = _download_clips_for_term(
                    items=page_items,
                    needed_count=clips_per_segment,
                    save_video=save_video,
                    save_dir=material_directory,
                    judge_candidate=judge_candidate,
                    filter_records=vlm_filter_records,
                    segment_text=segment_text,
                    term=term,
                    seen_urls=level_seen_urls,
                )
                level_clips.extend(page_clips)
                level_sources.extend(page_sources)
                if len(level_clips) >= clips_per_segment:
                    break
                # 下一页只在本页候选全部被拒收时才有意义（issue #9 D6）：
                # 本页下载到了片段但数量不足，说明本页素材可用，剩余缺口
                # 由下一级 fallback 补齐，不靠翻页硬凑。无过滤时同样只取
                # 一页，保持旧行为。
                if page_clips or not judge_candidate:
                    break
            search_attempts.append(
                {
                    "level": level,
                    "term": term,
                    "found": bool(level_clips),
                }
            )
            if level_clips:
                saved_paths = level_clips
                clip_sources = level_sources
                resolved_term = term
                fallback_level = level
                break

        results.append(
            SegmentMaterials(
                index=int(segment.get("index", position)),
                search_term=own_term,
                clips=saved_paths,
                resolved_term=resolved_term,
                fallback_level=fallback_level,
                search_attempts=search_attempts,
                clip_sources=clip_sources,
                vlm_filter=vlm_filter_records[:_MAX_FILTER_RECORDS],
            )
        )
        if not saved_paths:
            logger.warning(
                f"no materials found for segment {segment.get('index', position)} "
                f"after fallback chain (subject={subject!r})"
            )

    return results


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
        }
        for m in materials
    ]
