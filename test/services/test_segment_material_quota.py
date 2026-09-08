"""prepare_segment_materials 配额（B3/F-H2）、on_clip_accepted 契约与
duplicate 判定拒收契约单测。

配额规则：每段下载配额 = max(clips_per_segment, len(segment_window_plan(D, W)))，
与装配层共用同一窗口计划，长段按需多下、短段保住多样性下限。
嵌入门（plan T5）：verdict="duplicate" 的候选在素材层被拒收——不下载、不采纳、
无 last-resort 兜底，但判定记录保留在 vlm_filter 审计链中。
重排 top-N（plan rerank-top5-vlm）：judge 启用时每个 (词条, 页) 在送 VLM 前
先经 material_rerank.rerank_page 重排，只把 top-N（含无缩略图候选）按重排
顺序交判定；已用/已判定 URL 在重排前剔除；重排结果按 (词条, 页) 备忘，
同一运行内只算一次；重排调用抛异常时按 provider 原序 fail-open 并告警。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger as loguru_logger

from app.models.schema import VideoAspect
from app.services import material
from app.services import segment_material as sm


class _LogSink:
    """loguru 不走 stdlib logging 树，caplog 看不到——挂临时 sink 收集
    原始消息文本。"""

    def __init__(self):
        self.messages = []
        self._handler_id = None

    def __enter__(self):
        self._handler_id = loguru_logger.add(
            lambda message: self.messages.append(message.record["message"]),
            level="INFO",
        )
        return self

    def __exit__(self, *exc_info):
        loguru_logger.remove(self._handler_id)
        return False


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


def _patch_rerank(monkeypatch, behavior):
    """material_rerank.rerank_page 打桩：记录每次 (term, [url...], top_n)
    调用，返回值由 behavior(term, items, top_n) 决定。"""
    calls = []

    def fake_rerank(term, items, top_n):
        calls.append((term, [i.url for i in items], top_n))
        return behavior(term, items, top_n)

    monkeypatch.setattr(sm.material_rerank, "rerank_page", fake_rerank)
    return calls


def test_judge_receives_reranked_top5_in_ranked_order(monkeypatch):
    """重排桩返回固定顺序：VLM 判定恰好按 top-5 重排顺序进行；名额打满后
    尾部候选不再判定（plan rerank-top5-vlm 核心诉求）。"""
    items = [_video_item(f"https://v.example/{c}.mp4", "city") for c in "abcdefg"]

    def fixed_order(term, page_items, top_n):
        # 重排桩：top-5 打乱到 [e, a, c, b, d]，尾部原序跟在后面。
        ranked = [
            page_items[4], page_items[0], page_items[2],
            page_items[1], page_items[3],
        ]
        return ranked + page_items[5:]

    rerank_calls = _patch_rerank(monkeypatch, fixed_order)

    def judge(item, segment_text, search_term):
        return _record("relevant")

    results, saved, judged = _run_with_judge(
        segments=[{"index": 0, "text": "city", "duration": 12.816}],
        search_results={"city": items},
        judge=judge,
    )
    # 判定序列 = 重排后的 top-5 顺序；f/g 从未进判定。
    assert [u for u, _ in judged] == [
        f"https://v.example/{c}.mp4" for c in "eacbd"
    ]
    # 下载顺序与采纳结果同序，配额 5 拿满（D=12.816/W=3 → 5 窗）。
    assert results[0].clips == [f"/saved/{c}.mp4" for c in "eacbd"]
    assert saved == [f"https://v.example/{c}.mp4" for c in "eacbd"]
    # 重排恰好一次，输入是整页新鲜候选（本例无排除项）。
    assert [(term, urls) for term, urls, _ in rerank_calls] == [
        ("city", [f"https://v.example/{c}.mp4" for c in "abcdefg"])
    ]


def test_judge_disabled_items_pass_through_unrated(monkeypatch):
    """judge 未启用：不调用重排，候选按 provider 原序直接下载（旧行为）。"""
    items = [_video_item(f"https://v.example/{c}.mp4", "city") for c in "abcd"]
    rerank_calls = _patch_rerank(
        monkeypatch, lambda term, page_items, top_n: list(page_items)
    )

    results, _, saved = _run(
        segments=[{"index": 0, "text": "city", "duration": 3.744}],
        search_results={"city": items},
    )
    assert rerank_calls == []
    assert results[0].clips == [f"/saved/{c}.mp4" for c in "abc"]
    assert saved == [f"https://v.example/{c}.mp4" for c in "abc"]


def test_used_urls_excluded_before_rerank_call(monkeypatch):
    """跨段已采纳 URL 在重排调用前剔除：seg1 的重排输入不含 seg0 已下载
    的候选（重排名额不浪费在已用素材上）。"""
    tb_items = [_video_item("https://v.example/u1.mp4", "tb")] + [
        _video_item(f"https://v.example/b{i}.mp4", "tb") for i in (1, 2, 3, 4)
    ]
    rerank_calls = _patch_rerank(
        monkeypatch, lambda term, page_items, top_n: list(page_items)
    )

    def judge(item, segment_text, search_term):
        return _record("relevant")

    results, saved, _ = _run_with_judge(
        segments=[
            {"index": 0, "text": "first", "search_term": "ta", "duration": 3.744},
            {"index": 1, "text": "second", "search_term": "tb", "duration": 3.744},
        ],
        search_results={
            "ta": [_video_item("https://v.example/u1.mp4", "ta")],
            "tb": tb_items,
        },
        judge=judge,
    )
    # seg0 重排输入 = 整页 [u1]；seg1 重排输入 = [b1..b4]（u1 已被 seg0
    # 采纳，进入 used_urls 后在重排前剔除）。
    assert [(term, urls) for term, urls, _ in rerank_calls] == [
        ("ta", ["https://v.example/u1.mp4"]),
        ("tb", [
            "https://v.example/b1.mp4",
            "https://v.example/b2.mp4",
            "https://v.example/b3.mp4",
            "https://v.example/b4.mp4",
        ]),
    ]
    # seg1 不再判定/下载 u1，从新鲜候选拿满名额。
    assert results[1].clips == [
        "/saved/b1.mp4", "/saved/b2.mp4", "/saved/b3.mp4",
    ]
    assert saved.count("https://v.example/u1.mp4") == 1


def test_judged_urls_excluded_before_rerank_on_page_two(monkeypatch):
    """本层已判定 URL 在第 2 页重排前剔除：page1 判过的 p1 即使再次出现
    在 page2 原始结果里也不进重排输入。"""
    p1 = _video_item("https://v.example/p1.mp4", "t")
    pages = {
        1: [p1],
        2: [p1]
        + [_video_item(f"https://v.example/f{i}.mp4", "t") for i in (1, 2, 3)],
    }
    rerank_calls = _patch_rerank(
        monkeypatch, lambda term, page_items, top_n: list(page_items)
    )

    def fake_search(search_term, minimum_duration, video_aspect, page=1):
        return list(pages.get(page, []))

    def judge(item, segment_text, search_term):
        return _record("irrelevant" if "p1" in item.url else "relevant")

    results = sm.prepare_segment_materials(
        segments=[{"index": 0, "text": "t"}],
        video_subject="",
        search_videos=fake_search,
        save_video=lambda video_url, save_dir="": (
            f"/saved/{video_url.rsplit('/', 1)[-1]}"
        ),
        video_aspect=VideoAspect.portrait,
        clip_duration=3,
        save_dir="/materials",
        judge_candidate=judge,
    )
    # 第 1 页重排输入含 p1；第 2 页重排输入只剩新鲜候选（p1 已判定）。
    assert [(term, urls) for term, urls, _ in rerank_calls] == [
        ("t", ["https://v.example/p1.mp4"]),
        ("t", [
            "https://v.example/f1.mp4",
            "https://v.example/f2.mp4",
            "https://v.example/f3.mp4",
        ]),
    ]
    assert results[0].clips == [
        "/saved/f1.mp4", "/saved/f2.mp4", "/saved/f3.mp4",
    ]


def test_rerank_memoized_once_per_term_page(monkeypatch):
    """同一 (词条, 页) 跨段复用时重排只算一次（备忘命中不重调、不重记）。"""
    items = [_video_item(f"https://v.example/{c}.mp4", "city") for c in "abcdefg"]

    def top3_scrambled(term, page_items, top_n):
        return [page_items[2], page_items[0], page_items[1]] + page_items[3:]

    rerank_calls = _patch_rerank(monkeypatch, top3_scrambled)

    def judge(item, segment_text, search_term):
        return _record("relevant")

    results, _, _ = _run_with_judge(
        segments=[
            {"index": 0, "text": "city", "duration": 3.744},
            {"index": 1, "text": "city", "duration": 3.744},
        ],
        search_results={"city": items},
        judge=judge,
    )
    # 两段共用 (city, 1)：重排只在 seg0 首次遇到时发生一次。
    assert len(rerank_calls) == 1
    # seg0 按重排顺序拿 c,a,b；seg1 复用同一选择，used 跳过后拿 d,e,f。
    assert results[0].clips == ["/saved/c.mp4", "/saved/a.mp4", "/saved/b.mp4"]
    assert results[1].clips == ["/saved/d.mp4", "/saved/e.mp4", "/saved/f.mp4"]


def test_rerank_raise_falls_back_to_fresh_with_warning(monkeypatch):
    """重排调用抛异常：告警后按 provider 原序 fail-open，流水线继续；
    兜底结果进备忘，同 (词条, 页) 不再二次触发异常。"""

    def exploding_rerank(term, page_items, top_n):
        raise RuntimeError("reranker exploded")

    rerank_calls = _patch_rerank(monkeypatch, exploding_rerank)

    def judge(item, segment_text, search_term):
        return _record("relevant")

    items = [_video_item(f"https://v.example/{c}.mp4", "city") for c in "abcdef"]

    with _LogSink() as sink:
        results, saved, _ = _run_with_judge(
            segments=[
                {"index": 0, "text": "city", "duration": 3.744},
                {"index": 1, "text": "city", "duration": 3.744},
            ],
            search_results={"city": items},
            judge=judge,
        )
    # 抛异常的重排恰好被调用一次（seg0 首遇；seg1 复用兜底备忘）。
    assert len(rerank_calls) == 1
    # 两段都按 provider 原序拿满名额——选择退化为无重排，流水线不中断。
    assert results[0].clips == ["/saved/a.mp4", "/saved/b.mp4", "/saved/c.mp4"]
    assert results[1].clips == ["/saved/d.mp4", "/saved/e.mp4", "/saved/f.mp4"]
    assert saved == [f"https://v.example/{c}.mp4" for c in "abcdef"]
    # 告警行：term/page/error 字段齐全（loguru 需 sink 捕获，caplog 不可见）。
    assert any(
        "material rerank page selection failed" in m
        and "term='city'" in m
        and "page=1" in m
        and "error=RuntimeError" in m
        for m in sink.messages
    )


def test_resolution_summary_logged_for_success_and_empty():
    """审计缺口 G1/G2：每段一行汇总（成功与空手都触发；vlm_judged 取
    截断前真实判定数），整层空手时补一行空层标记——配额提前打满的段
    不产生空层行。"""

    def judge(item, segment_text, search_term):
        return _record("relevant")

    with _LogSink() as sink:
        results, _, _ = _run_with_judge(
            segments=[
                {"index": 0, "text": "city", "duration": 3.744},
                {"index": 1, "text": "void", "duration": 3.744},
            ],
            search_results={
                "city": [
                    _video_item("https://v.example/a.mp4", "city"),
                    _video_item("https://v.example/b.mp4", "city"),
                    _video_item("https://v.example/c.mp4", "city"),
                ]
            },
            judge=judge,
        )
    assert results[0].clips == ["/saved/a.mp4", "/saved/b.mp4", "/saved/c.mp4"]
    assert results[1].clips == []
    # 成功段汇总：配额、命中词条、回退层级、尝试层数、判定量齐全。
    assert (
        "segment 0: material resolution summary: clips=3/3, "
        "resolved_term='city', fallback_level=self, levels_tried=1, "
        "vlm_judged=3, image_gen=0"
    ) in sink.messages
    # 空手段同一格式：resolved_term/fallback_level 为空（!r 渲染 ''）。
    assert (
        "segment 1: material resolution summary: clips=0/3, "
        "resolved_term='', fallback_level=, levels_tried=1, "
        "vlm_judged=0, image_gen=0"
    ) in sink.messages
    # G2：空手段的 self 层补一行空层标记；成功段没有。
    assert (
        "segment 1: level produced no clips, level=self, term='void'"
    ) in sink.messages
    assert not any(
        "level produced no clips" in m and m.startswith("segment 0:")
        for m in sink.messages
    )


def test_imagegen_fallback_success_logged():
    """审计缺口 G4：自有词条全空手且图片兜底成功时补一行，标明画面来自
    生成概念图（只记文件名）；汇总行同步反映 fallback_level=subject。"""
    with _LogSink() as sink:
        results = sm.prepare_segment_materials(
            segments=[{"index": 2, "text": "void", "duration": 3.744}],
            video_subject="generic money",
            search_videos=lambda search_term, minimum_duration, video_aspect: [],
            save_video=lambda video_url, save_dir="": (
                f"/saved/{video_url.rsplit('/', 1)[-1]}"
            ),
            video_aspect=VideoAspect.portrait,
            clip_duration=3,
            save_dir="/materials",
            generate_image=lambda segment_text, subject_term: (
                "/saved/gen1.mp4",
                {"model": "Kwai-Kolors/Kolors", "source": "kolors"},
            ),
        )
    assert results[0].clips == ["/saved/gen1.mp4"]
    assert results[0].fallback_level == "subject"
    assert (
        "segment 2: image-gen fallback engaged: clip=gen1.mp4"
    ) in sink.messages
    # 汇总行与图片兜底状态一致：clips=1/3、image_gen=1。
    assert (
        "segment 2: material resolution summary: clips=1/3, "
        "resolved_term='generic money', fallback_level=subject, "
        "levels_tried=1, vlm_judged=0, image_gen=1"
    ) in sink.messages
