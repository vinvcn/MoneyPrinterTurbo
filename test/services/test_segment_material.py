import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services import segment_material as sm
from app.services import material


def _video_item(url, term, provider="pexels"):
    item = material.MaterialInfo(
        provider=provider,
        url=url,
        duration=10,
        source_info={"provider": provider, "search_term": term, "asset_id": url},
    )
    return item


class TestPrepareSegmentMaterials(unittest.TestCase):
    def _run(self, segments, search_results, subject="money", subject_results=None):
        """Run prepare_segment_materials with deterministic stubs."""
        searched_terms = []
        saved_urls = []

        def fake_search(search_term, minimum_duration, video_aspect):
            searched_terms.append(search_term)
            return list(search_results.get(search_term, subject_results if search_term == subject else []))

        def fake_save_video(video_url, save_dir=""):
            saved_urls.append(video_url)
            return f"/saved/{video_url.rsplit('/', 1)[-1]}"

        results = sm.prepare_segment_materials(
            segments=segments,
            video_subject=subject,
            search_videos=fake_search,
            save_video=fake_save_video,
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
        )
        return results, searched_terms, saved_urls

    def test_self_term_downloads_clips_per_segment(self):
        results, searched, _ = self._run(
            segments=[{"index": 0, "text": "city skyline"}],
            search_results={
                "city skyline": [_video_item(f"https://v.example/{i}.mp4", "city skyline") for i in range(4)]
            },
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolved_term, "city skyline")
        self.assertEqual(results[0].fallback_level, "self")
        self.assertEqual(len(results[0].clips), sm.CLIPS_PER_SEGMENT)

    def test_second_own_term_used_when_primary_fails(self):
        """自有词条链：首选词条零结果时落到第二个自有词条（仍是 self 层）。"""
        results, searched, _ = self._run(
            segments=[
                {"index": 0, "text": "ignored", "search_terms": ["term-one", "term-two"]}
            ],
            search_results={
                "term-two": [_video_item("https://v.example/t2.mp4", "term-two")],
            },
        )
        # 名额补足（修 5）：t2 只贡献 1/3 名额，缺口会继续尝试 subject 层
        # （此处 subject 池为空，最终 1/3）；t1 仍被搜索且零贡献。
        self.assertEqual(searched, ["term-one", "term-two", "money"])
        self.assertEqual(results[0].fallback_level, "self")
        self.assertEqual(results[0].resolved_term, "term-two")
        self.assertEqual(results[0].clips, ["/saved/t2.mp4"])
        attempts = results[0].search_attempts
        self.assertEqual([a["level"] for a in attempts], ["self", "self", "subject"])
        self.assertEqual(
            [a["term"] for a in attempts], ["term-one", "term-two", "money"]
        )
        self.assertEqual([a["found"] for a in attempts], [False, True, False])

    def test_used_url_skipped_across_own_terms(self):
        """已下载 URL 在同任务后续 self 层被跳过：跨自有词条不重复下载。"""
        searched = []
        saved_urls = []

        def fake_search(search_term, minimum_duration, video_aspect):
            searched.append(search_term)
            if search_term == "term-one":
                return [_video_item("https://v.example/dup.mp4", "term-one")]
            if search_term == "term-two":
                return [
                    _video_item("https://v.example/dup.mp4", "term-two"),
                    _video_item("https://v.example/fresh.mp4", "term-two"),
                ]
            return []

        def fake_save_video(video_url, save_dir=""):
            saved_urls.append(video_url)
            return f"/saved/{video_url.rsplit('/', 1)[-1]}"

        results = sm.prepare_segment_materials(
            segments=[
                {"index": 0, "text": "term-one"},
                {
                    "index": 1,
                    "text": "narration",
                    "search_terms": ["term-one", "term-two"],
                },
            ],
            video_subject="",
            search_videos=fake_search,
            save_video=fake_save_video,
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
        )
        # seg0 经 term-one 下载 dup.mp4 并注册；seg1 的 term-one 层（缓存
        # 命中）与 term-two 层都再次遇到 dup.mp4，均被 self 层去重跳过，
        # 只有 fresh.mp4 进入 seg1。dup.mp4 全程只下载一次。
        self.assertEqual(saved_urls.count("https://v.example/dup.mp4"), 1)
        self.assertEqual(results[1].fallback_level, "self")
        self.assertEqual(results[1].resolved_term, "term-two")
        self.assertEqual(results[1].clips, ["/saved/fresh.mp4"])

    def test_own_terms_exhausted_falls_to_subject(self):
        """自有词条全部失败后才落 subject 层（层级顺序：self×N → subject）。"""
        results, searched, _ = self._run(
            segments=[
                {"index": 0, "text": "ignored", "search_terms": ["term-one", "term-two"]}
            ],
            search_results={},
            subject="generic money",
            subject_results=[_video_item("https://v.example/s.mp4", "generic money")],
        )
        self.assertEqual(searched, ["term-one", "term-two", "generic money"])
        self.assertEqual(results[0].fallback_level, "subject")
        self.assertEqual(results[0].resolved_term, "generic money")
        attempts = results[0].search_attempts
        self.assertEqual(
            [(a["level"], a["found"]) for a in attempts],
            [("self", False), ("self", False), ("subject", True)],
        )

    def test_last_resort_uses_subject(self):
        results, _, _ = self._run(
            segments=[{"index": 0, "text": "nothing matches"}],
            search_results={},
            subject="generic money",
            subject_results=[_video_item("https://v.example/s.mp4", "generic money")],
        )
        self.assertEqual(results[0].fallback_level, "subject")
        self.assertEqual(results[0].resolved_term, "generic money")

    def test_all_levels_fail_returns_empty_clips(self):
        results, _, _ = self._run(
            segments=[{"index": 0, "text": "nothing"}],
            search_results={},
            subject="also nothing",
        )
        self.assertEqual(results[0].clips, [])
        self.assertEqual(results[0].resolved_term, "")
        self.assertEqual(results[0].fallback_level, "")

    def test_cjk_terms_are_never_sent_to_search_api(self):
        """搜索 API 仅接受英文：含 CJK 的词条/原文/主题词一律不发。"""
        results, searched, _ = self._run(
            segments=[
                {
                    "index": 0,
                    "text": "这是中文原文",
                    "search_term": "city skyline 城市",
                }
            ],
            search_results={"city skyline": [_video_item("https://v.example/c.mp4", "city skyline")]},
            subject="人工智能",
        )
        # self 层含 CJK 被跳过，subject 层含 CJK 也被跳过——没有任何
        # 带 CJK 的查询到达搜索 API。
        for term in searched:
            self.assertRegex(term, r"^[A-Za-z0-9 ]+$", f"CJK leaked: {term!r}")
        self.assertEqual(results[0].fallback_level, "")
        self.assertEqual(results[0].clips, [])

    def test_english_self_term_still_used_when_subject_is_cjk(self):
        """纯英文词条不受主题词 CJK 影响，self 层正常工作。"""
        results, searched, _ = self._run(
            segments=[{"index": 0, "text": "city skyline"}],
            search_results={"city skyline": [_video_item("https://v.example/c.mp4", "city skyline")]},
            subject="人工智能",
        )
        self.assertEqual(results[0].fallback_level, "self")
        self.assertEqual(results[0].resolved_term, "city skyline")
        self.assertEqual(searched, ["city skyline"])

    def test_subject_search_cached_across_segments(self):
        """subject 词全任务共享：跨 segment 只搜一次，且 subject 层允许复用已用素材。"""
        results, searched, _ = self._run(
            segments=[
                {"index": 0, "text": "nothing here"},
                {"index": 1, "text": "also nothing"},
            ],
            search_results={},
            subject="generic money",
            subject_results=[_video_item("https://v.example/s.mp4", "generic money")],
        )
        # 两个 segment 的 self 层均无结果，先后落入 subject 层：subject 词
        # 只真正搜索一次（缓存跨 segment 共享）；已用素材去重不在 subject
        # 层生效，因此两个 segment 都拿到同一条兜底片段。
        self.assertEqual(searched.count("generic money"), 1)
        for result in results:
            self.assertEqual(result.fallback_level, "subject")
            self.assertEqual(result.resolved_term, "generic money")
            self.assertEqual(result.clips, ["/saved/s.mp4"])

    def test_download_failure_skips_item_and_continues(self):
        """A failing download should not abort the whole segment."""

        def fake_search(search_term, minimum_duration, video_aspect):
            return [_video_item("https://v.example/bad.mp4", "t")]

        calls = []

        def flaky_save(video_url, save_dir=""):
            calls.append(video_url)
            return ""  # save_video returns "" on invalid download

        results = sm.prepare_segment_materials(
            segments=[{"index": 0, "text": "t"}],
            video_subject="",
            search_videos=fake_search,
            save_video=flaky_save,
            video_aspect=VideoAspect.portrait,
            save_dir="/materials",
        )
        self.assertEqual(results[0].clips, [])


class TestRecordsConversion(unittest.TestCase):
    def test_segments_to_records_is_json_safe(self):
        materials = [
            sm.SegmentMaterials(
                index=0,
                search_term="a",
                clips=["/x/a1.mp4"],
                resolved_term="a",
                fallback_level="self",
            )
        ]
        records = sm.segments_to_records(materials)
        self.assertEqual(records[0]["clips"], ["/x/a1.mp4"])
        self.assertEqual(records[0]["index"], 0)


if __name__ == "__main__":
    unittest.main()
