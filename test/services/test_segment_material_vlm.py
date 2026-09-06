"""
VLM 过滤接线测试（issue #9）：segment_material 候选循环 + 分页行为。

全部 mock，不发真实网络请求。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services import segment_material as sm
from app.services import material


def _video_item(url, term, provider="pexels", thumbnail_url=""):
    return material.MaterialInfo(
        provider=provider,
        url=url,
        duration=10,
        source_info={
            "provider": provider,
            "search_term": term,
            "asset_id": url,
            "thumbnail_url": thumbnail_url,
        },
    )


def _accept_all(item, segment_text="", search_term=""):
    return {
        "term": search_term,
        "asset_id": item.source_info.get("asset_id"),
        "verdict": "relevant",
        "reason": "ok",
        "image_source": "thumbnail",
        "attempts": 1,
        "page": 1,
    }


class TestPaginationAwareSearch(unittest.TestCase):
    def test_search_callable_receives_page_kwarg(self):
        """搜索回调必须收到 page 参数（默认第 1 页）。"""
        seen_pages = []

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            seen_pages.append(page)
            return [_video_item(f"https://v.example/{page}.mp4", search_term)]

        results = sm.prepare_segment_materials(
            segments=[{"index": 0, "text": "city skyline"}],
            video_subject="",
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=_accept_all,
        )
        self.assertEqual(seen_pages, [1])
        # fake 每页只返回 1 个候选：第 1 页下载到 1 个片段后即停止翻页
        # （剩余缺口由 fallback 层补齐，不靠翻页硬凑）。
        self.assertEqual(results[0].clips, ["/saved/1.mp4"])

    def test_filter_enabled_fetches_page_two_when_page_one_rejected(self):
        """第一页全部被拒收后必须翻到第二页（issue #9 D6）。"""
        searched = []

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            searched.append(page)
            return [_video_item(f"https://v.example/p{page}.mp4", search_term)]

        def reject_page_one(item, segment_text="", search_term=""):
            # asset_id 即 URL（p1/p2）；提取结尾的页码数字做判定。
            asset = str(item.source_info.get("asset_id") or "")
            page = int(asset.split("/p")[-1].split(".")[0])
            return {
                "verdict": "irrelevant" if page == 1 else "relevant",
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": page,
                "term": search_term,
            }

        results = sm.prepare_segment_materials(
            segments=[{"index": 0, "text": "city"}],
            video_subject="",
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=reject_page_one,
        )
        self.assertEqual(searched, [1, 2])
        self.assertEqual(results[0].clips, ["/saved/p2.mp4"])
        self.assertEqual(results[0].fallback_level, "self")

    def test_no_filter_stops_after_single_page(self):
        """未启用过滤时保持旧行为：只用第一页，不发起翻页请求。"""
        searched = []

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            searched.append(page)
            return [_video_item("https://v.example/a.mp4", search_term)]

        results = sm.prepare_segment_materials(
            segments=[{"index": 0, "text": "city"}],
            video_subject="",
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": "/saved/a.mp4",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
        )
        self.assertEqual(searched, [1])
        self.assertEqual(results[0].clips, ["/saved/a.mp4"])

    def test_pages_exhausted_returns_empty_without_subject(self):
        """两页候选全部被拒收且 subject 为空 ⇒ 无兜底层，空手而归。"""
        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            return [_video_item(f"https://v.example/{search_term}-{page}.mp4", search_term)]

        def reject_everything(item, segment_text="", search_term=""):
            return {
                "verdict": "irrelevant",
                "reason": "no",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results = sm.prepare_segment_materials(
            segments=[
                {"index": 0, "text": "good term"},
                {"index": 1, "text": "doomed term"},
            ],
            video_subject="",
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=reject_everything,
        )
        self.assertEqual(results[1].fallback_level, "")
        self.assertEqual(results[1].clips, [])


class TestFilterWiring(unittest.TestCase):
    def _run(self, segments, search_results, verdict_fn, subject="money"):
        searched_terms = []

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            searched_terms.append(search_term)
            return list(search_results.get(search_term, []))

        def fake_save_video(video_url, save_dir=""):
            return f"/saved/{video_url.rsplit('/', 1)[-1]}"

        results = sm.prepare_segment_materials(
            segments=segments,
            video_subject=subject,
            search_videos=fake_search,
            save_video=fake_save_video,
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=verdict_fn,
        )
        return results, searched_terms

    def test_irrelevant_candidate_skipped_next_used(self):
        """不相关候选被跳过，下一个候选顶上（issue #9 核心诉求）。"""
        items = [
            _video_item("https://v.example/bad.mp4", "t"),
            _video_item("https://v.example/good.mp4", "t"),
        ]

        def verdict_fn(item, segment_text="", search_term=""):
            return {
                "verdict": "irrelevant" if "bad" in item.url else "relevant",
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results, _ = self._run(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            search_results={"t": items},
            verdict_fn=verdict_fn,
            subject="",
        )
        self.assertEqual(results[0].clips, ["/saved/good.mp4"])

    def test_uncertain_verdict_accepted_as_last_resort(self):
        """uncertain 被延期：relevant 不足时才作为兜底接受（issue #10 finding 2）。"""

        def verdict_fn(item, segment_text="", search_term=""):
            return {
                "verdict": "uncertain",
                "reason": "blurry",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "first_frame",
                "attempts": 2,
                "page": 1,
                "term": search_term,
            }

        results, _ = self._run(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            search_results={"t": [_video_item("https://v.example/u.mp4", "t")]},
            verdict_fn=verdict_fn,
            subject="",
        )
        # 只有 uncertain 时仍然兜底接受，避免饿死。
        self.assertEqual(results[0].clips, ["/saved/u.mp4"])

    def test_uncertain_deferred_behind_relevant(self):
        """同页里 uncertain 排在 relevant 之后，relevant 够用时不收 uncertain。"""
        items = [
            _video_item("https://v.example/eye.mp4", "t"),      # uncertain
            _video_item("https://v.example/bh1.mp4", "t"),      # relevant
            _video_item("https://v.example/bh2.mp4", "t"),      # relevant
            _video_item("https://v.example/bh3.mp4", "t"),      # relevant
        ]

        def verdict_fn(item, segment_text="", search_term=""):
            v = "uncertain" if "eye" in item.url else "relevant"
            return {
                "verdict": v,
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results, _ = self._run(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            search_results={"t": items},
            verdict_fn=verdict_fn,
            subject="",
        )
        self.assertEqual(
            results[0].clips,
            ["/saved/bh1.mp4", "/saved/bh2.mp4", "/saved/bh3.mp4"],
        )

    def test_uncertain_deferred_skipped_when_page_has_enough_relevant(self):
        """relevant 补齐名额后，延期的 uncertain 不再回头补。"""
        items = [
            _video_item("https://v.example/eye.mp4", "t"),
            _video_item("https://v.example/bh1.mp4", "t"),
            _video_item("https://v.example/bh2.mp4", "t"),
        ]

        def verdict_fn(item, segment_text="", search_term=""):
            v = "uncertain" if "eye" in item.url else "relevant"
            return {
                "verdict": v,
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results, _ = self._run(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            search_results={"t": items},
            verdict_fn=verdict_fn,
            subject="",
        )
        # CLIPS_PER_SEGMENT=3，但只有 2 个 relevant + 1 个 uncertain：
        # subject 为空 ⇒ 本层即最后一层，relevant 缺口在收尾调用里由
        # uncertain 兜底补满（不会留空名额）。
        self.assertEqual(
            results[0].clips,
            ["/saved/bh1.mp4", "/saved/bh2.mp4", "/saved/eye.mp4"],
        )

    def test_uncertain_deferred_not_carried_across_pages_or_levels(self):
        """延期名单只在本页/本层有效，不跨页、不跨 fallback 层。"""
        # 页 1: 1 relevant + 1 uncertain；本层是最后一层 → 收尾调用兜底
        # 接受 uncertain。relevant 优先顺序不变（bh1 先下载）。
        items_p1 = [
            _video_item("https://v.example/eye.mp4", "t"),
            _video_item("https://v.example/bh1.mp4", "t"),
        ]
        searched_pages = []

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            searched_pages.append(page)
            return list(items_p1) if page == 1 else []

        def verdict_fn(item, segment_text="", search_term=""):
            v = "uncertain" if "eye" in item.url else "relevant"
            return {
                "verdict": v,
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results = sm.prepare_segment_materials(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            video_subject="",
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=verdict_fn,
        )
        # 最后一层兜底：relevant 1 个 + uncertain 兜底 1 个。
        self.assertEqual(results[0].clips, ["/saved/bh1.mp4", "/saved/eye.mp4"])

    def test_uncertain_last_resort_fills_after_relevant_exhausted(self):
        """relevant 不足且没有其它来源时（最后一层），延期 uncertain 兜底补齐名额。"""
        items = [
            _video_item("https://v.example/eye.mp4", "t"),
            _video_item("https://v.example/bh1.mp4", "t"),
            _video_item("https://v.example/bh2.mp4", "t"),
        ]

        def verdict_fn(item, segment_text="", search_term=""):
            v = "uncertain" if "eye" in item.url else "relevant"
            return {
                "verdict": v,
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results, _ = self._run(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            search_results={"t": items},
            verdict_fn=verdict_fn,
            subject="",
            # 只有本层可用（subject 为空）= 最后一层，兜底生效。
        )
        self.assertEqual(len(results[0].clips), 3)

    def test_uncertain_last_resort_fills_when_no_other_level(self):
        """本层 relevant=2、uncertain=1 且无下一层时兜底凑满 3。"""
        items = [
            _video_item("https://v.example/eye.mp4", "t"),
            _video_item("https://v.example/bh1.mp4", "t"),
            _video_item("https://v.example/bh2.mp4", "t"),
        ]

        def verdict_fn(item, segment_text="", search_term=""):
            v = "uncertain" if "eye" in item.url else "relevant"
            return {
                "verdict": v,
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            return list(items) if page == 1 else []

        results = sm.prepare_segment_materials(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            video_subject="",  # 无 subject 层可兜底
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=verdict_fn,
        )
        # relevant 2 个 + uncertain 兜底 1 个 = 满 3
        self.assertEqual(
            sorted(results[0].clips),
            ["/saved/bh1.mp4", "/saved/bh2.mp4", "/saved/eye.mp4"],
        )

    def test_used_asset_from_earlier_segment_skipped_later(self):
        """前面的 segment 已采纳的资产，后面的 segment 不再判定/下载。"""

        def verdict_fn(item, segment_text="", search_term=""):
            return {
                "verdict": "relevant",
                "reason": "ok",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        # 两个 segment 的搜索都返回同一批候选；第一个用 3 个后，
        # 第二个应跳过这 3 个（不给它第二次机会）。
        shared = [_video_item(f"https://v.example/shared{i}.mp4", "t") for i in range(3)]
        exclusive = [_video_item(f"https://v.example/ex{i}.mp4", "u") for i in range(4)]

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            if search_term == "ta":
                return list(shared)
            return list(exclusive)

        def fake_save(video_url, save_dir=""):
            return f"/saved/{video_url.rsplit('/', 1)[-1]}"

        results = sm.prepare_segment_materials(
            segments=[
                {"index": 0, "text": "ta", "search_term": "ta"},
                {"index": 1, "text": "tb", "search_term": "tb"},
            ],
            video_subject="",
            search_videos=fake_search,
            save_video=fake_save,
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=verdict_fn,
        )
        # seg0 收下 shared 0-2；seg1 不能再用它们，从自己的候选里拿 ex0-2。
        self.assertEqual(
            results[0].clips,
            ["/saved/shared0.mp4", "/saved/shared1.mp4", "/saved/shared2.mp4"],
        )
        self.assertEqual(
            results[1].clips,
            ["/saved/ex0.mp4", "/saved/ex1.mp4", "/saved/ex2.mp4"],
        )

    def test_used_asset_skipped_even_when_verdict_would_accept(self):
        """跨段跳过发生在判定之前：已用资产不会被再次判定。"""
        judged = []

        def verdict_fn(item, segment_text="", search_term=""):
            judged.append(item.url)
            return {
                "verdict": "relevant",
                "reason": "ok",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        shared = [_video_item("https://v.example/dup.mp4", "t")]
        second = [_video_item("https://v.example/other.mp4", "u")]

        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            return list(shared) if search_term == "ta" else list(second)

        def fake_save(video_url, save_dir=""):
            return f"/saved/{video_url.rsplit('/', 1)[-1]}"

        results = sm.prepare_segment_materials(
            segments=[
                {"index": 0, "text": "ta", "search_term": "ta"},
                {"index": 1, "text": "tb", "search_term": "tb"},
            ],
            video_subject="",
            search_videos=fake_search,
            save_video=fake_save,
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=verdict_fn,
        )
        # dup.mp4 只在 seg0 被判定过一次；seg1 跳过它不再判定。
        self.assertEqual(judged.count("https://v.example/dup.mp4"), 1)
        self.assertEqual(results[1].clips, ["/saved/other.mp4"])

    def test_filter_records_persisted_on_segment(self):
        """判定审计记录随分段落盘（verdict/reason/image_source/attempts）。"""
        captured = []

        def verdict_fn(item, segment_text="", search_term=""):
            record = {
                "verdict": "relevant",
                "reason": "ok",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }
            captured.append(record)
            return record

        results, _ = self._run(
            segments=[{"index": 0, "text": "t", "search_term": "t"}],
            search_results={"t": [_video_item("https://v.example/a.mp4", "t")]},
            verdict_fn=verdict_fn,
            subject="",
        )
        self.assertEqual(len(results[0].vlm_filter), 1)
        record = results[0].vlm_filter[0]
        self.assertEqual(record["verdict"], "relevant")
        self.assertEqual(record["image_source"], "thumbnail")
        self.assertEqual(record["attempts"], 1)
        records = sm.segments_to_records(results)
        self.assertIn("vlm_filter", records[0])

    def test_second_own_term_used_when_first_terms_candidates_rejected(self):
        """自有词条链 + VLM：term1 候选全判 irrelevant，term2 候选 relevant 顶上。"""
        def fake_search(search_term, minimum_duration, video_aspect, page=1):
            if page != 1:
                return []
            if search_term == "term-one":
                return [_video_item("https://v.example/bad1.mp4", "term-one")]
            if search_term == "term-two":
                return [_video_item("https://v.example/good2.mp4", "term-two")]
            return []

        def verdict_fn(item, segment_text="", search_term=""):
            return {
                "verdict": "irrelevant" if "bad" in item.url else "relevant",
                "reason": "x",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        results = sm.prepare_segment_materials(
            segments=[
                {
                    "index": 0,
                    "text": "narration",
                    "search_terms": ["term-one", "term-two"],
                }
            ],
            video_subject="",
            search_videos=fake_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
            judge_candidate=verdict_fn,
        )
        self.assertEqual(results[0].clips, ["/saved/good2.mp4"])
        self.assertEqual(results[0].resolved_term, "term-two")
        self.assertEqual(results[0].fallback_level, "self")
        # 审计记录按判定顺序先 term1 的拒收、后 term2 的接受，term 字段可区分。
        records = results[0].vlm_filter
        self.assertEqual([r["term"] for r in records], ["term-one", "term-two"])
        self.assertEqual([r["verdict"] for r in records], ["irrelevant", "relevant"])

    def test_segment_text_passed_to_judge(self):
        """旁白文本必须传入判定回调（issue #9 D5）。"""
        seen_texts = []

        def verdict_fn(item, segment_text="", search_term=""):
            seen_texts.append(segment_text)
            return {
                "verdict": "relevant",
                "reason": "ok",
                "asset_id": item.source_info.get("asset_id"),
                "image_source": "thumbnail",
                "attempts": 1,
                "page": 1,
                "term": search_term,
            }

        self._run(
            segments=[{"index": 0, "text": "narration about black holes", "search_term": "t"}],
            search_results={"t": [_video_item("https://v.example/a.mp4", "t")]},
            verdict_fn=verdict_fn,
            subject="",
        )
        self.assertEqual(seen_texts, ["narration about black holes"])


if __name__ == "__main__":
    unittest.main()
