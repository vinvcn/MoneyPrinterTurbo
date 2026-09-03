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

    def test_pages_exhausted_falls_to_next_level(self):
        """两页候选全部被拒收后落入下一个 fallback 层。"""
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

    def test_uncertain_verdict_accepts_candidate(self):
        """uncertain 判定放行候选（issue #9 D7）。"""

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
        self.assertEqual(results[0].clips, ["/saved/u.mp4"])

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
