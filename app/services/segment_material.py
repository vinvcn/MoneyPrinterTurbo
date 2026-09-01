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
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import task_artifacts

# Number of clips to download per segment. The assembler cycles through them
# when a segment lasts longer than one clip, keeping visual variety without
# global shuffling.
CLIPS_PER_SEGMENT = 3


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


def _download_clips_for_term(
    items: List[MaterialInfo],
    needed_count: int,
    save_video: Callable[..., str],
    save_dir: str,
) -> List[str]:
    """Download up to `needed_count` unique clips from already-searched items."""
    saved_paths: List[str] = []
    seen_urls: set[str] = set()
    for item in items:
        if len(saved_paths) >= needed_count:
            break
        if not item.url or item.url in seen_urls:
            continue
        seen_urls.add(item.url)
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
            saved_paths.append(saved_video_path)
    return saved_paths


def prepare_segment_materials(
    segments: List[dict],
    video_subject: str,
    search_videos: Callable[..., List[MaterialInfo]],
    save_video: Callable[..., str],
    video_aspect: VideoAspect,
    clip_duration: int = 5,
    clips_per_segment: int = CLIPS_PER_SEGMENT,
    save_dir: str = "",
) -> List[SegmentMaterials]:
    """
    Search and download clips for every segment using the fallback chain.

    Args:
        segments: segment dicts with at least {"index", "text"}.
        video_subject: last-resort search term shared by all segments.
        search_videos: search callable (term, minimum_duration, video_aspect)
            -> List[MaterialInfo]; must already include caching.
        save_video: download callable (url, save_dir) -> path ("" on failure).
        video_aspect: target orientation for remote filtering.
        clip_duration: minimum duration requested from providers.
        clips_per_segment: how many distinct clips to gather per segment.
        save_dir: download directory (empty = provider default cache).

    Returns:
        One SegmentMaterials per input segment, in the same order.
    """
    subject = str(video_subject or "").strip()
    # Cache search results across segments so neighbor fallbacks do not hit
    # the provider API again for a term that was already searched.
    search_cache: dict[str, List[MaterialInfo]] = {}

    def search_cached(term: str) -> List[MaterialInfo]:
        normalized = (term or "").strip()
        if not normalized:
            return []
        if normalized not in search_cache:
            try:
                search_cache[normalized] = list(
                    search_videos(
                        search_term=normalized,
                        minimum_duration=clip_duration,
                        video_aspect=video_aspect,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "segment material search failed: "
                    f"term={normalized!r}, error={type(exc).__name__}, detail={exc}"
                )
                search_cache[normalized] = []
        return search_cache[normalized]

    material_directory = save_dir
    if not material_directory:
        configured = str(config.app.get("material_directory", "")).strip()
        if configured and configured != "task":
            material_directory = configured

    results: List[SegmentMaterials] = []
    for position, segment in enumerate(segments):
        own_term = str(segment.get("search_term") or segment.get("text") or "").strip()
        next_term = (
            str(
                segments[position + 1].get("search_term")
                or segments[position + 1].get("text")
                or ""
            ).strip()
            if position + 1 < len(segments)
            else ""
        )
        candidates: List[tuple[str, str]] = [
            ("self", own_term),
            ("previous", results[position - 1].resolved_term if position else ""),
            ("next", next_term),
        ]
        if subject:
            candidates.append(("subject", subject))

        saved_paths: List[str] = []
        resolved_term = ""
        fallback_level = ""
        for level, term in candidates:
            term = (term or "").strip()
            if not term:
                continue
            saved_paths = _download_clips_for_term(
                items=search_cached(term),
                needed_count=clips_per_segment,
                save_video=save_video,
                save_dir=material_directory,
            )
            if saved_paths:
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
        }
        for m in materials
    ]


def find_neighbor_fallback_term(
    index: int,
    segment_terms: dict[int, str],
) -> Optional[str]:
    """
    Return the nearest available neighbor term for a failed segment search.

    Preference: previous segment first (narratively closer), then next.
    Pure helper so orchestration code stays testable without provider stubs.
    """
    previous_term = segment_terms.get(index - 1, "")
    if previous_term:
        return previous_term
    return segment_terms.get(index + 1) or None


if __name__ == "__main__":
    prepare_segment_materials(
        segments=[{"index": 0, "text": "money"}],
        video_subject="money",
        search_videos=lambda **_: [],
        save_video=lambda **_: "",
        video_aspect=VideoAspect.portrait,
    )
