"""prepare_segment_materials 配额（B3/F-H2）、on_clip_accepted 契约与
duplicate 判定拒收契约单测。

配额规则：每段下载配额 = max(clips_per_segment, len(segment_window_plan(D, W)))，
与装配层共用同一窗口计划，长段按需多下、短段保住多样性下限。
嵌入门（plan T5）：verdict="duplicate" 的候选在素材层被拒收——不下载、不采纳、
无 last-resort 兜底，但判定记录保留在 vlm_filter 审计链中。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services import material
from app.services import segment_material as sm


def _video_item(url, term, provider="pexels"):
    return material.MaterialInfo(
        provider=provider,
        url=url,
        duration=10,
        source_info={"provider": provider, "search_term": term, "asset_id": url},
    )


def _run(segments, search_results, clip_duration=3, on_clip_accepted=None):
    searched_terms = []
    saved_urls = []

    def fake_search(search_term, minimum_duration, video_aspect):
        searched_terms.append(search_term)
        return list(search_results.get(search_term, []))

    def fake_save_video(video_url, save_dir=""):
        saved_urls.append(video_url)
        return f"/saved/{video_url.rsplit('/', 1)[-1]}"

    results = sm.prepare_segment_materials(
        segments=segments,
        video_subject="",
        search_videos=fake_search,
        save_video=fake_save_video,
        video_aspect=VideoAspect.portrait,
        clip_duration=clip_duration,
        save_dir="/materials",
        on_clip_accepted=on_clip_accepted,
    )
    return results, searched_terms, saved_urls


def test_long_segment_downloads_window_count_clips():
    # D=12.816/W=3 → 计划 5 窗 → 配额 max(3, 5) = 5 条。
    results, _, saved = _run(
        segments=[{"index": 0, "text": "city", "duration": 12.816}],
        search_results={
            "city": [_video_item(f"https://v.example/{i}.mp4", "city") for i in range(6)]
        },
    )
    assert len(results[0].clips) == 5
    assert len(saved) == 5


def test_short_segment_keeps_diversity_floor():
    # D=3.744/W=3 → 计划 1 窗 → 配额 max(3, 1) = 3 条。
    results, _, saved = _run(
        segments=[{"index": 0, "text": "city", "duration": 3.744}],
        search_results={
            "city": [_video_item(f"https://v.example/{i}.mp4", "city") for i in range(6)]
        },
    )
    assert len(results[0].clips) == 3
    assert len(saved) == 3


def test_medium_segment_quota_is_window_count():
    # D=9.48/W=3 → 计划 [3,3,3.48] 共 3 窗 → 配额 max(3, 3) = 3 条。
    results, _, saved = _run(
        segments=[{"index": 0, "text": "city", "duration": 9.48}],
        search_results={
            "city": [_video_item(f"https://v.example/{i}.mp4", "city") for i in range(6)]
        },
    )
    assert len(results[0].clips) == 3
    assert len(saved) == 3


def test_missing_duration_keeps_legacy_quota():
    # 没有 duration 字段的 segment 退回旧配额（3 条），行为兼容。
    results, _, saved = _run(
        segments=[{"index": 0, "text": "city"}],
        search_results={
            "city": [_video_item(f"https://v.example/{i}.mp4", "city") for i in range(6)]
        },
    )
    assert len(results[0].clips) == sm.CLIPS_PER_SEGMENT
    assert len(saved) == sm.CLIPS_PER_SEGMENT


def test_on_clip_accepted_called_for_each_accepted_url():
    accepted = []
    results, _, _ = _run(
        segments=[{"index": 0, "text": "city", "duration": 12.816}],
        search_results={
            "city": [_video_item(f"https://v.example/{i}.mp4", "city") for i in range(5)]
        },
        on_clip_accepted=accepted.append,
    )
    # 每条采纳的 URL 恰好回调一次，顺序与 clips 一致。
    assert len(accepted) == 5
    assert accepted == [f"https://v.example/{i}.mp4" for i in range(5)]
    assert [f"/saved/{i}.mp4" for i in range(5)] == results[0].clips


def test_on_clip_accepted_exception_never_breaks_pipeline():
    def boom(url):
        raise RuntimeError("registry exploded")

    results, _, saved = _run(
        segments=[{"index": 0, "text": "city", "duration": 12.816}],
        search_results={
            "city": [_video_item(f"https://v.example/{i}.mp4", "city") for i in range(5)]
        },
        on_clip_accepted=boom,
    )
    # 回调炸了也不能影响下载：配额仍然拿满。
    assert len(results[0].clips) == 5
    assert len(saved) == 5


def test_on_clip_accepted_not_called_on_failed_download():
    accepted = []

    def failing_save(video_url, save_dir=""):
        raise OSError("network down")

    results = sm.prepare_segment_materials(
        segments=[{"index": 0, "text": "city", "duration": 12.816}],
        video_subject="",
        search_videos=lambda search_term, minimum_duration, video_aspect: [
            _video_item("https://v.example/bad.mp4", "city"),
            _video_item("https://v.example/good.mp4", "city"),
        ],
        # 首个 URL 下载失败、后续成功：回调只发给真正采纳的候选。
        save_video=lambda video_url, save_dir="": (
            failing_save(video_url, save_dir)
            if "bad" in video_url
            else f"/saved/good.mp4"
        ),
        video_aspect=VideoAspect.portrait,
        clip_duration=3,
        save_dir="/materials",
        on_clip_accepted=accepted.append,
    )
    assert accepted == ["https://v.example/good.mp4"]
    assert results[0].clips == ["/saved/good.mp4"]


def _record(verdict, **extra):
    return {"verdict": verdict, "reason": f"{verdict} reason", "image_source": "poster", **extra}


def _dup_record(duplicate_of, cos=0.9):
    return {
        "verdict": "duplicate",
        "reason": f"cos={cos:.3f} >= threshold",
        "image_source": "embedding",
        "duplicate_of": duplicate_of,
        "cos": cos,
    }


def _run_with_judge(segments, search_results, judge, on_clip_accepted=None):
    saved_urls = []
    judged = []

    def fake_search(search_term, minimum_duration, video_aspect):
        return list(search_results.get(search_term, []))

    def fake_save_video(video_url, save_dir=""):
        saved_urls.append(video_url)
        return f"/saved/{video_url.rsplit('/', 1)[-1]}"

    def wrapped_judge(item, segment_text, search_term):
        record = judge(item, segment_text, search_term)
        judged.append((item.url, record.get("verdict")))
        return record

    results = sm.prepare_segment_materials(
        segments=segments,
        video_subject="",
        search_videos=fake_search,
        save_video=fake_save_video,
        video_aspect=VideoAspect.portrait,
        clip_duration=3,
        save_dir="/materials",
        judge_candidate=wrapped_judge,
        on_clip_accepted=on_clip_accepted,
    )
    return results, saved_urls, judged


def test_duplicate_verdict_never_downloads_or_accepts():
    accepted = []

    def judge(item, segment_text, search_term):
        if item.url.endswith("dup.mp4"):
            return _dup_record(duplicate_of="https://v.example/a.mp4", cos=0.904)
        return _record("relevant")

    results, saved, _ = _run_with_judge(
        segments=[{"index": 0, "text": "city", "duration": 3.744}],
        search_results={
            "city": [
                _video_item("https://v.example/dup.mp4", "city"),
                _video_item("https://v.example/a.mp4", "city"),
                _video_item("https://v.example/b.mp4", "city"),
                _video_item("https://v.example/c.mp4", "city"),
            ]
        },
        judge=judge,
        on_clip_accepted=accepted.append,
    )
    # duplicate 候选：不下载、不进 clip_sources、不触发采纳回调。
    assert "https://v.example/dup.mp4" not in saved
    assert all(c["url"] != "https://v.example/dup.mp4" for c in results[0].clip_sources)
    assert "https://v.example/dup.mp4" not in accepted
    # 同页后续 relevant 候选正常下载，配额拿满（max(3, 1) = 3）。
    assert results[0].clips == ["/saved/a.mp4", "/saved/b.mp4", "/saved/c.mp4"]
    assert sorted(saved) == [
        "https://v.example/a.mp4",
        "https://v.example/b.mp4",
        "https://v.example/c.mp4",
    ]
    # 审计链完整：duplicate 判定记录仍在 vlm_filter 中。
    dup_records = [r for r in results[0].vlm_filter if r.get("verdict") == "duplicate"]
    assert len(dup_records) == 1
    assert dup_records[0]["duplicate_of"] == "https://v.example/a.mp4"
    assert dup_records[0]["cos"] == 0.904


def test_duplicate_in_middle_does_not_break_candidate_loop():
    def judge(item, segment_text, search_term):
        if item.url.endswith("dup.mp4"):
            return _dup_record(duplicate_of="https://v.example/a.mp4")
        return _record("relevant")

    results, saved, _ = _run_with_judge(
        segments=[{"index": 0, "text": "city", "duration": 3.744}],
        search_results={
            "city": [
                _video_item("https://v.example/a.mp4", "city"),
                _video_item("https://v.example/dup.mp4", "city"),
                _video_item("https://v.example/b.mp4", "city"),
                _video_item("https://v.example/c.mp4", "city"),
            ]
        },
        judge=judge,
    )
    assert results[0].clips == ["/saved/a.mp4", "/saved/b.mp4", "/saved/c.mp4"]
    assert "https://v.example/dup.mp4" not in saved


def test_duplicate_never_registered_in_used_urls():
    # seg0 判 duplicate 的 URL 未被下载，也就不得注册进 used_urls——
    # seg1 再见到同一 URL 时必须重新判定并可正常下载。
    # （若 bug 回归：seg0 错误下载会把它注册进 used_urls，seg2 直接跳过不判定。）
    def judge(item, segment_text, search_term):
        if segment_text == "first":
            return _dup_record(duplicate_of="https://v.example/x.mp4")
        return _record("relevant")

    results, saved, judged = _run_with_judge(
        segments=[
            {"index": 0, "text": "first", "duration": 3.744},
            {"index": 1, "text": "second", "duration": 3.744},
        ],
        search_results={
            "first": [_video_item("https://v.example/same.mp4", "city")],
            "second": [_video_item("https://v.example/same.mp4", "city")],
        },
        judge=judge,
    )
    # 同一 URL 被两段各判定一次：seg0 duplicate、seg1 relevant。
    judged_same = [v for u, v in judged if u == "https://v.example/same.mp4"]
    assert judged_same == ["duplicate", "relevant"]
    # seg0 无收：不下载、不采纳；seg1 正常下载同一 URL。
    assert results[0].clips == []
    assert results[0].clip_sources == []
    assert results[1].clips == ["/saved/same.mp4"]
    assert saved == ["https://v.example/same.mp4"]
