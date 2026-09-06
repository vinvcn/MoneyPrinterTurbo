"""
Per-segment material search/download with a deterministic fallback chain.

A segment-first pipeline needs more than a list of anonymous video paths:
each narration segment owns its own search, its own downloaded clips, and its
own fallback story when providers come back empty. This module keeps the
existing `download_videos` behavior untouched and adds a structured
per-segment flow on top of the shared search/cache/download primitives.

Fallback chain per segment (in order):
1. each of the segment's own search terms in order (the `search_terms`
   list, or the single `search_term`/text fallback when no list is given);
2. the video subject as a shared last resort. Self levels are deduplicated
   against clips already consumed by earlier segments (issue #10 finding 1);
   the subject level may reuse them on purpose.

Levels accumulate toward the per-segment clip quota: a partially filled
level keeps its clips and later levels contribute the rest, instead of the
segment settling for a shortfall (fix 5, UAT task c0044257 finding).

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

    供任务编排层（task.py 的早期失败守卫）复用同一条 CJK 过滤规则，
    避免跨模块引用私有实现或在两处重复正则逻辑。
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
    used_urls: set[str] | None = None,
    accept_uncertain: bool = True,
    enforce_used_urls: bool = True,
) -> tuple[List[str], List[dict]]:
    """
    Download up to `needed_count` unique clips; return paths and URL provenance.

    When `judge_candidate` is provided (VLM filter enabled), each candidate is
    visually judged before download; irrelevant candidates are skipped in favor
    of the next one (issue #9 D1/D7). `seen_urls` can be shared across calls
    (per fallback level) so page-2 candidates already downloaded from page 1
    are not fetched or re-judged twice.

    `used_urls` carries every URL already accepted by an earlier segment in
    this task (issue #10 finding 1). Enforcement is gated by
    `enforce_used_urls`: self levels skip already-used candidates without
    re-judging, so one universally on-topic asset cannot occupy several
    segments' clip slots; the subject level passes False and may reuse an
    already-used asset as a shared last resort. Registration is
    unconditional — every successful download adds its URL to `used_urls`,
    including subject-level downloads, so later segments' self levels still
    exclude them.

    Uncertain verdicts are deferred behind relevant ones (issue #10 finding 2,
    adopting the Q2 tightening): a page is first consumed accepting only
    `relevant` candidates; if the level still comes up short afterwards and
    `accept_uncertain` is True, deferred `uncertain` candidates are accepted
    as a last resort before moving to the next page/fallback level. This keeps
    a vague "eye close-up" out of a black-hole segment while anything better
    is available, without starving the pipeline.
    """
    saved_paths: List[str] = []
    clip_sources: List[dict] = []
    if seen_urls is None:
        seen_urls = set()

    def _download_one(item: MaterialInfo) -> str:
        logger.info(f"downloading segment clip: {item.url}")
        try:
            saved_video_path = save_video(video_url=item.url, save_dir=save_dir)
        except Exception as exc:
            logger.warning(
                "failed to download segment clip: "
                f"provider={item.provider}, error={type(exc).__name__}, "
                f"detail={exc}"
            )
            return ""
        if saved_video_path and saved_video_path not in saved_paths:
            logger.info(f"segment clip saved: {saved_video_path}")
            saved_paths.append(saved_video_path)
            clip_sources.append(
                {
                    "url": item.url,
                    "local_file": Path(saved_video_path).name,
                }
            )
            if used_urls is not None:
                used_urls.add(item.url)
            return saved_video_path
        return ""

    def _judge(item: MaterialInfo) -> dict | None:
        if judge_candidate is None:
            return None
        verdict = judge_candidate(
            item=item,
            segment_text=segment_text,
            search_term=term,
        )
        if filter_records is not None:
            filter_records.append(verdict)
        v = verdict.get("verdict")
        if v == "irrelevant":
            logger.info(
                "vlm filter rejected candidate: "
                f"asset_id={verdict.get('asset_id')}, "
                f"reason={verdict.get('reason')!r}, "
                f"image_source={verdict.get('image_source')}"
            )
        else:
            logger.info(
                "vlm filter accepted candidate: "
                f"asset_id={verdict.get('asset_id')}, "
                f"verdict={v}, "
                f"image_source={verdict.get('image_source')}"
            )
        return verdict

    # Pass 1: accept only `relevant` candidates; park `uncertain` ones unless
    # `accept_uncertain` allows pass 2 to run on the same call (last page of
    # the last fallback level).
    deferred_uncertain: List[MaterialInfo] = []
    for item in items:
        if len(saved_paths) >= needed_count:
            break
        if not item.url or item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        if enforce_used_urls and used_urls and item.url in used_urls:
            logger.info(
                "skipping candidate already used by an earlier segment: "
                f"url={item.url}"
            )
            continue
        verdict = _judge(item)
        if verdict is None:
            if _download_one(item):
                continue
            # 下载失败继续看下一个候选（与旧行为一致）。
            continue
        if verdict.get("verdict") == "irrelevant":
            continue
        if verdict.get("verdict") == "uncertain":
            if accept_uncertain:
                deferred_uncertain.append(item)
            else:
                # 本调用不允许兜底时，把 URL 从 seen_urls 回滚，让最后一层
                # 的收尾调用能重新见到并延期该候选（seen_urls 的语义是
                # "已下载"，不是"已判定"——判定状态由 deferred 名单跟踪）。
                seen_urls.discard(item.url)
            continue
        if not _download_one(item):
            continue

    # Pass 2 (last resort): the level ran dry on relevant candidates; accept
    # deferred uncertain ones so the segment is not starved. The caller sets
    # `accept_uncertain` only on the last fallback level's closing call, so
    # uncertain never jumps ahead of a fresh relevant candidate from another
    # page or level. Note: a candidate deferred on an earlier page/level was
    # judged already but never downloaded; the closing call re-judges it via
    # the normal pass-1 loop (accept_uncertain=True keeps it in the deferred
    # list of THIS call) and pass 2 then downloads it.
    if len(saved_paths) < needed_count and accept_uncertain:
        for item in deferred_uncertain:
            if len(saved_paths) >= needed_count:
                break
            logger.info(
                "accepting deferred uncertain candidate as last resort: "
                f"url={item.url}"
            )
            _download_one(item)
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
    # Cache search results across segments and across a segment's own terms
    # so a repeated term (or the subject shared by all segments) does not hit
    # the provider API again. Page-aware: page 1 must exist before page 2 is
    # fetched (issue #9 D6).
    search_cache: dict[tuple[str, int], List[MaterialInfo]] = {}
    def search_page_cached(term: str, page: int) -> List[MaterialInfo]:
        normalized = (term or "").strip()
        if not normalized:
            return []
        cache_key = (normalized, page)
        if cache_key not in search_cache:
            # 旧签名搜索函数（测试替身、第三方扩展）可能不接受 page 参数，
            # 或内部对缺失键返回 None 使 list(None) 抛 TypeError。两次尝试
            # 分别独立捕获：先带页码调用，TypeError 时退回无页码调用；退回
            # 调用自身的 TypeError（None 不可迭代）按空结果处理。
            found: List[MaterialInfo]
            try:
                found = list(
                    search_videos(
                        search_term=normalized,
                        minimum_duration=clip_duration,
                        video_aspect=video_aspect,
                        page=page,
                    )
                )
            except TypeError:
                try:
                    found = list(
                        search_videos(
                            search_term=normalized,
                            minimum_duration=clip_duration,
                            video_aspect=video_aspect,
                        )
                    )
                except TypeError:
                    found = []
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
    # 任务级已用素材 URL（issue #10 finding 1）：前面的 segment 已经采纳的
    # 候选不再进入后续 segment 的 self 层判定/下载，避免同一条"万能"素材
    # 被多个 segment 各自判 relevant 后重复占用片段名额。跳过只在 self 层
    # 生效（enforce_used_urls），subject 层作为共享兜底允许复用；但注册
    # 无条件——subject 层下载的 URL 同样入册，后续 segment 的 self 层仍会
    # 排除它。
    used_urls_across_segments: set[str] = set()
    for position, segment in enumerate(segments):
        # 自有词条链：优先取上游给出的 search_terms 列表（与词条通道的冻结
        # 契约，可能缺失），否则退回单个 search_term / 片段原文。统一过
        # 英文过滤，丢弃空串并按序去重——每个词条都是一个独立的 self 层。
        raw_terms = segment.get("search_terms") or [
            segment.get("search_term") or segment.get("text")
        ]
        own_terms: List[str] = []
        for raw_term in raw_terms:
            english = _english_search_term(str(raw_term or ""))
            if english and english not in own_terms:
                own_terms.append(english)
        candidates: List[tuple[str, str]] = [("self", t) for t in own_terms]
        if subject:
            candidates.append(("subject", subject))
        # 空词条的层不会参与尝试（下方 continue），真实的"最后一层"必须
        # 按非空层计算，否则唯一可用的层也拿不到 uncertain 兜底资格。
        non_empty_levels = sum(1 for _, t in candidates if (t or "").strip())

        segment_text = str(segment.get("text") or "")
        saved_paths: List[str] = []
        clip_sources: List[dict] = []
        resolved_term = ""
        fallback_level = ""
        search_attempts: List[dict] = []
        vlm_filter_records: List[dict] = []
        # 每层尝试时暂存本层被延期的 uncertain 候选（issue #10 finding 2）：
        # relevant 优先；仅当整条链的最后一层、最后一页仍未凑齐时，
        # _download_clips_for_term 才在层内回收 uncertain 兜底。
        seen_non_empty_levels = 0
        for level, term in candidates:
            term = (term or "").strip()
            if not term:
                continue
            needed = clips_per_segment - len(saved_paths)
            if needed <= 0:
                break
            seen_non_empty_levels += 1
            is_last_level = seen_non_empty_levels == non_empty_levels
            # VLM 过滤启用时逐页尝试：当前页没有 relevant 片段才翻下一页，
            # 翻页用尽仍凑不齐才落入下一个 fallback 层（issue #9 D6）。
            level_clips: List[str] = []
            level_sources: List[dict] = []
            level_seen_urls: set[str] = set()
            # 收集阶段：逐页把候选聚到 level_page_items（共享 level_seen_urls
            # 去重）。probe 调用（accept_uncertain=False）只下载 relevant，
            # used_urls 的跳过判断也在 probe 中生效——已用素材不会进名额；
            # 本层目标名额是 needed（全链剩余缺口），不是每层都重下满额。
            # 翻页条件沿用 issue #9 D6：本页连一个 relevant 片段都没凑出
            # 才继续翻。judge 未启用时 probe 一次即 break，保持旧行为。
            level_page_items: List[MaterialInfo] = []
            for page in range(1, MAX_SEARCH_PAGES + 1):
                page_items = search_page_cached(term, page)
                if not page_items:
                    break
                level_page_items.extend(page_items)
                probe_clips, probe_sources = _download_clips_for_term(
                    items=page_items,
                    needed_count=needed,
                    save_video=save_video,
                    save_dir=material_directory,
                    judge_candidate=judge_candidate,
                    filter_records=vlm_filter_records,
                    segment_text=segment_text,
                    term=term,
                    seen_urls=level_seen_urls,
                    used_urls=used_urls_across_segments,
                    accept_uncertain=False,
                    enforce_used_urls=(level == "self"),
                )
                level_clips.extend(probe_clips)
                level_sources.extend(probe_sources)
                if len(level_clips) >= needed:
                    break
                if probe_clips or not judge_candidate:
                    break
            # 层尾收尾：probe 阶段（accept_uncertain=False）下载的 relevant
            # 片段已在 level_clips。若 judge 未启用则到此为止；启用且名额
            # 未满时，把本层收集到的候选再过一遍：seen_urls 会跳过已下载
            # 的 URL，relevant 缺口只允许在最后一层由 uncertain 兜底补齐
            # （issue #10 finding 2）。judge 未启用时同样收尾一次，保证
            # 无过滤行为与旧版一致（候选顺序下载直到名额满）。
            if len(level_clips) < needed:
                extra_clips, extra_sources = _download_clips_for_term(
                    items=level_page_items,
                    needed_count=needed - len(level_clips),
                    save_video=save_video,
                    save_dir=material_directory,
                    judge_candidate=judge_candidate,
                    filter_records=vlm_filter_records,
                    segment_text=segment_text,
                    term=term,
                    seen_urls=level_seen_urls,
                    used_urls=used_urls_across_segments,
                    accept_uncertain=is_last_level,
                    enforce_used_urls=(level == "self"),
                )
                level_clips.extend(extra_clips)
                level_sources.extend(extra_sources)
            search_attempts.append(
                {
                    "level": level,
                    "term": term,
                    "found": bool(level_clips),
                }
            )
            if level_clips:
                # 名额补足：部分命中的层保留已得片段，缺口由后续层继续填补。
                # resolved_term/fallback_level 记录首个贡献层，完整贡献链见
                # search_attempts。
                saved_paths.extend(level_clips)
                clip_sources.extend(level_sources)
                if not resolved_term:
                    resolved_term = term
                    fallback_level = level
                if len(saved_paths) >= clips_per_segment:
                    break

        results.append(
            SegmentMaterials(
                index=int(segment.get("index", position)),
                search_term=own_terms[0] if own_terms else "",
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
