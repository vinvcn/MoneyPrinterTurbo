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

    def test_fallback_to_neighbor_previous_term(self):
        """A segment whose own search fails uses the previous segment's resolved term."""
        results, _, _ = self._run(
            segments=[
                {"index": 0, "text": "city skyline"},
                {"index": 1, "text": "unsearchable gibberish xyz"},
            ],
            search_results={
                "city skyline": [_video_item("https://v.example/a.mp4", "city skyline")],
            },
        )
        self.assertEqual(results[1].resolved_term, "city skyline")
        self.assertEqual(results[1].fallback_level, "previous")
        self.assertEqual(results[1].clips, ["/saved/a.mp4"])

    def test_fallback_to_next_term_when_previous_also_failed(self):
        results, _, _ = self._run(
            segments=[
                {"index": 0, "text": "first fails too"},
                {"index": 1, "text": "second fails"},
                {"index": 2, "text": "ocean waves"},
            ],
            search_results={
                "ocean waves": [_video_item("https://v.example/o.mp4", "ocean waves")],
            },
        )
        self.assertEqual(results[1].fallback_level, "next")
        self.assertEqual(results[1].resolved_term, "ocean waves")

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

    def test_search_is_cached_across_fallback_levels(self):
        """The same term should only hit the search callable once."""
        results, searched, _ = self._run(
            segments=[
                {"index": 0, "text": "term-a"},
                {"index": 1, "text": "term-b-missing"},
            ],
            search_results={"term-a": [_video_item("https://v.example/1.mp4", "term-a")]},
        )
        # term-a searched once for segment 0, segment 1's previous-fallback reuses cache.
        self.assertEqual(searched.count("term-a"), 1)
        self.assertEqual(results[1].fallback_level, "previous")

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
