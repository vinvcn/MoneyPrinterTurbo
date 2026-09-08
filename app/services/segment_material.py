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
2. when `generate_image` is provided and every own term came up empty, a
   generated concept image (LLM-refined prompt → Kolors, provider photo
   fallback) rendered as a single static clip covering the segment.

Levels accumulate toward the per-segment clip quota: a partially filled
level keeps its clips and later levels contribute the rest, instead of the
segment settling for a shortfall (fix 5, UAT task c0044257 finding). The
image fallback only fires on a fully empty segment (G1 decision), so
partially filled segments keep their video shortfall.

Every attempt is recorded on the segment record so the task manifest can show
which term actually produced the visuals.

Optional VLM pre-download filter (issue #9): when the [vlm] config section is
enabled, each search page is first reranked with the Qwen3-VL reranker
(plan rerank-top5-vlm) and only its top-N candidates are visually judged
(thumbnail or first frame) before download; irrelevant candidates are skipped,
and exhausted pages roll into the same fallback chain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List
import re

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import material_rerank, task_artifacts
from app.services.video import segment_window_plan
from app.utils import utils

# allow: SIZE_OK — 模块承载配额/回退链/过滤接线多个既定契约，拆分归 F4 评审
# （同 image_embedding.py 的 wave 例外惯例）。

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
    # Subject-level image generation audit trail: {"model", "prompt",
    # "source", "image_size", "image", "clip", "attempts"} — exactly one
    # record when the segment fell through to the generated concept image,
    # empty otherwise.
    image_gen: List[dict] = field(default_factory=list)


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
    on_clip_accepted: Callable[[str], None] | None = None,
) -> tuple[List[str], List[dict]]:
    """
    Download up to `needed_count` unique clips; return paths and URL provenance.

    When `judge_candidate` is provided (VLM filter enabled), each candidate is
    visually judged before download; irrelevant candidates are skipped in favor
    of the next one (issue #9 D1/D7). `seen_urls` can be shared across calls
    (per fallback level) so page-2 candidates already downloaded from page 1
    are not fetched or re-judged twice.

    `used_urls` carries every URL already accepted by an earlier segment in
    this task (issue #10 finding 1): such candidates are skipped without
    re-judging, so one universally on-topic asset cannot occupy several
    segments' clip slots. Registration is unconditional — every successful
    download adds its URL to the set. The former subject level no longer
    searches videos (it generates a concept image instead), so the
    enforcement applies uniformly at every level.

    Uncertain verdicts are deferred behind relevant ones (issue #10 finding 2,
    adopting the Q2 tightening): a page is first consumed accepting only
    `relevant` candidates; if the level still comes up short afterwards and
    `accept_uncertain` is True, deferred `uncertain` candidates are accepted
    as a last resort before moving to the next page/fallback level. This keeps
    a vague "eye close-up" out of a black-hole segment while anything better
    is available, without starving the pipeline.

    A `duplicate` verdict (embedding gate hit, plan T5) is rejected outright:
    the candidate is never downloaded or accepted and has no last-resort
    tier — a near-duplicate of already-accepted material must not re-enter
    the timeline even when the segment would otherwise run dry (the existing
    image-gen/empty-segment fallbacks cover starvation instead). The record
    still lands in `filter_records` so the audit chain shows the gate fired.

    Candidate selection happens upstream in `prepare_segment_materials`
    (plan rerank-top5-vlm): when the VLM filter is enabled, each search page
    is reranked with the Qwen3-VL reranker and only the top-N candidates
    (plus thumbnail-less stragglers) reach this function, already in ranked
    order. This function therefore consumes an already-ranked list and runs
    no rerank of its own — the verdict rules below simply operate on that
    order.
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
            if on_clip_accepted is not None:
                # 跨 worker 契约（duplicate gate 注册）：clip 一经采纳即回调。
                # 回调异常只降级为告警，绝不中断素材下载链路。
                try:
                    on_clip_accepted(item.url)
                except Exception as exc:
                    logger.warning(
                        "on_clip_accepted callback failed: "
                        f"url={item.url}, error={type(exc).__name__}, "
                        f"detail={exc}"
                    )
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
        if v in ("irrelevant", "duplicate"):
            detail = ""
            if v == "duplicate":
                # 嵌入门命中：审计链需要 duplicate_of 与 cos 才能对上记录。
                detail = (
                    f", duplicate_of={verdict.get('duplicate_of')}, "
                    f"cos={verdict.get('cos')}"
                )
            logger.info(
                "vlm filter rejected candidate: "
                f"asset_id={verdict.get('asset_id')}, "
                f"verdict={v}, "
                f"reason={verdict.get('reason')!r}, "
                f"image_source={verdict.get('image_source')}"
                f"{detail}"
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
    # the last fallback level). `duplicate` (embedding gate hit) is rejected
    # outright like `irrelevant` — plan T5: judged duplicates are never
    # downloaded or accepted, and get no last-resort tier (admitting them as
    # a fallback would put the very near-duplicate family the gate exists to
    # block back on the timeline).
    deferred_uncertain: List[MaterialInfo] = []
    for item in items:
        if len(saved_paths) >= needed_count:
            break
        if not item.url or item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        if used_urls and item.url in used_urls:
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
        if verdict.get("verdict") in ("irrelevant", "duplicate"):
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
    generate_image: Callable[..., tuple[str, dict]] | None = None,
    on_clip_accepted: Callable[[str], None] | None = None,
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
        clip_duration: minimum duration requested from providers; also the
            window width used to derive the per-segment quota (B3).
        clips_per_segment: diversity floor for distinct clips per segment.
        save_dir: download directory (empty = provider default cache).
        judge_candidate: optional VLM filter callable (issue #9); receives
            (item=MaterialInfo, segment_text=str, search_term=str) and returns
            a verdict record dict. None = filter disabled.
        generate_image: optional subject-fallback callable (UAT a043f7bb
            follow-up); receives (segment_text=str, subject_term=str) and
            returns (clip_path, audit_record). Called only when every own
            term produced zero clips; returns clip_path="" on failure.
            None = no image fallback (segment stays empty).
        on_clip_accepted: optional callback invoked with the source URL of
            every clip accepted into a segment's quota, immediately after it
            is registered as used. Callback failures are logged and swallowed
            (never break the pipeline).

    Returns:
        One SegmentMaterials per input segment, in the same order.
    """
    subject = _english_search_term(str(video_subject or ""))
    # Cache search results across segments and across a segment's own terms
    # so a repeated term does not hit the provider API again. Page-aware:
    # page 1 must exist before page 2 is fetched (issue #9 D6).
    search_cache: dict[tuple[str, int], List[MaterialInfo]] = {}
    # 重排 walk-limit 与 (词条, 页) 备忘（plan rerank-top5-vlm）：walk_limit
    # 每次运行只读一次；备忘与 search_cache 同为本次调用的局部状态，不会跨调用
    # 泄漏。仅 VLM 判定启用时使用。
    rerank_walk_limit = material_rerank._walk_limit()
    rerank_cache: dict[tuple[str, int], List[MaterialInfo]] = {}
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
    # 候选不再进入后续 segment 的判定/下载，避免同一条"万能"素材被多个
    # segment 各自判 relevant 后重复占用片段名额。所有层都是 self 层，
    # 拦截无条件生效；注册无条件——每个成功下载都入册。
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
        # 空词条的层不会参与尝试（下方 continue），真实的"最后一层"必须
        # 按非空层计算，否则唯一可用的层也拿不到 uncertain 兜底资格。
        non_empty_levels = sum(1 for _, t in candidates if (t or "").strip())

        segment_text = str(segment.get("text") or "")
        # B3（F-H2）配额对齐：每段下载配额 = max(多样性下限, 装配层 A3 窗口
        # 数)。长段按需多下，让装配层轮播复用从常态变成 provider 干涸兜底；
        # 两层共用 segment_window_plan，配额与时间线切分不会错位。
        segment_duration = float(segment.get("duration") or 0)
        needed_clips = max(
            clips_per_segment,
            len(segment_window_plan(segment_duration, clip_duration)),
        )
        saved_paths: List[str] = []
        clip_sources: List[dict] = []
        resolved_term = ""
        fallback_level = ""
        search_attempts: List[dict] = []
        vlm_filter_records: List[dict] = []
        image_gen_records: List[dict] = []
        # 每层尝试时暂存本层被延期的 uncertain 候选（issue #10 finding 2）：
        # relevant 优先；仅当整条链的最后一层、最后一页仍未凑齐时，
        # _download_clips_for_term 才在层内回收 uncertain 兜底。
        seen_non_empty_levels = 0
        for level, term in candidates:
            term = (term or "").strip()
            if not term:
                continue
            needed = needed_clips - len(saved_paths)
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
            # 去重）。judge 启用时收集的是重排 top-N 选择而非整页原始候选。
            # probe 调用（accept_uncertain=False）只下载 relevant，
            # used_urls 的跳过判断也在 probe 中生效——已用素材不会进名额；
            # 本层目标名额是 needed（全链剩余缺口），不是每层都重下满额。
            # 翻页条件沿用 issue #9 D6：本页连一个 relevant 片段都没凑出
            # 才继续翻。judge 未启用时 probe 一次即 break，保持旧行为。
            level_page_items: List[MaterialInfo] = []
            for page in range(1, MAX_SEARCH_PAGES + 1):
                page_items = search_page_cached(term, page)
                if not page_items:
                    break
                if judge_candidate is None:
                    # 未启用 VLM 判定：整页候选原样收集、原样送审，不重排、
                    # 不建备忘（旧行为逐字节等价）。
                    level_page_items.extend(page_items)
                    probe_items = page_items
                else:
                    # 重排备忘（plan rerank-top5-vlm）：每个 (词条, 页) 只在
                    # 首次遇到时计算一次 top-N 选择；后续命中直接复用，
                    # 既不重调重排，也不重打重排审计行（重排模块只在真正
                    # 重排时落日志）。
                    cache_key = (term, page)
                    if cache_key not in rerank_cache:
                        # 已用/已判定 URL 在重排前剔除：重排名额不浪费在
                        # 本层已判定（seen_urls）或跨段已采纳（used_urls）
                        # 的候选上。union 产生新集合，不改动两处原状态。
                        excluded = level_seen_urls | used_urls_across_segments
                        fresh = [
                            item
                            for item in page_items
                            if item.url and item.url not in excluded
                        ]
                        try:
                            page_selection = material_rerank.rerank_page(
                                term, fresh, rerank_walk_limit
                            )
                        except Exception as exc:
                            # belt-and-braces：重排模块内部已 fail-open，这里
                            # 只兜"重排调用本身被替换/抛异常"的情况——告警后
                            # 按 provider 原序放行 fresh，流水线绝不中断。
                            logger.warning(
                                "material rerank page selection failed, "
                                "falling back to provider order: "
                                f"term={term!r}, page={page}, "
                                f"error={type(exc).__name__}, detail={exc}"
                            )
                            page_selection = fresh
                        rerank_cache[cache_key] = page_selection
                    page_selection = rerank_cache[cache_key]
                    level_page_items.extend(page_selection)
                    probe_items = page_selection
                probe_clips, probe_sources = _download_clips_for_term(
                    items=probe_items,
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
                    on_clip_accepted=on_clip_accepted,
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
                    on_clip_accepted=on_clip_accepted,
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
                if len(saved_paths) >= needed_clips:
                    break
            else:
                # 审计缺口 G2：整层跑完仍 0 clip 时补一行标记，让"哪一层
                # 空手"在日志流可见（search_attempts 只进清单不进日志）。
                # 配额打满的 break 都在层体之前/上方 if 分支内，不会到这里。
                logger.info(
                    f"segment {segment.get('index', position)}: "
                    f"level produced no clips, level={level}, term={term!r}"
                )

        # subject 图片生成（G1 决策）：仅当自有词条产出 0 clip 时触发，
        # 生成单张覆盖整段时长的概念图替代视频；partial-fill 段保持视频
        # 短缺现状（装配器可处理）。失败/未配置回调时段空手（原有语义）。
        if not saved_paths and generate_image is not None:
            image_clip, image_record = generate_image(segment_text, subject)
            if image_clip:
                saved_paths.append(image_clip)
                clip_sources.append(
                    {"url": "", "local_file": Path(image_clip).name}
                )
                resolved_term = subject
                fallback_level = "subject"
                # 审计缺口 G4：图片兜底成功时补一行，标明该段画面来自生成
                # 概念图而非视频素材；只记文件名，不记 prompt/图像内容。
                logger.info(
                    f"segment {segment.get('index', position)}: "
                    f"image-gen fallback engaged: clip={Path(image_clip).name}"
                )
            if image_record:
                image_gen_records.append(image_record)

        # 审计缺口 G1：每段一行汇总（成功/部分/空手都触发），把配额、命中
        # 词条、回退层级、尝试层数、VLM 判定量与图片兜底一次性落进日志流，
        # 供运行审计对账。vlm_judged 取截断前的真实判定条数——截断只发生
        # 在下方 SegmentMaterials 构造（_MAX_FILTER_RECORDS）。
        logger.info(
            f"segment {segment.get('index', position)}: "
            f"material resolution summary: clips={len(saved_paths)}/{needed_clips}, "
            f"resolved_term={resolved_term!r}, fallback_level={fallback_level}, "
            f"levels_tried={len(search_attempts)}, "
            f"vlm_judged={len(vlm_filter_records)}, "
            f"image_gen={len(image_gen_records)}"
        )

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
                image_gen=image_gen_records,
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
