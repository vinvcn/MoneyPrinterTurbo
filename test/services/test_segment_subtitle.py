import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import segment_subtitle


class TestBuildSegmentSubtitles(unittest.TestCase):
    def test_empty_segments_produce_no_file(self):
        self.assertEqual(segment_subtitle.build_segment_subtitles([], "/tmp/x.srt"), "")

    def test_single_segment_gets_full_window(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "subtitle.srt")
            result = segment_subtitle.build_segment_subtitles(
                segments=[{"index": 0, "text": "Hello world.", "start_ms": 0, "duration_ms": 1500}],
                subtitle_file=out,
            )
            self.assertEqual(result, out)
            content = Path(out).read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,500", content)
            self.assertIn("Hello world.", content)

    def test_consecutive_segments_have_continuous_offsets(self):
        import tempfile

        segments = [
            {"index": 0, "text": "First.", "start_ms": 0, "duration_ms": 2000},
            {"index": 1, "text": "Second.", "start_ms": 2000, "duration_ms": 1000},
            {"index": 2, "text": "Third.", "start_ms": 3000, "duration_ms": 3500},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "subtitle.srt")
            segment_subtitle.build_segment_subtitles(segments, out)
            content = Path(out).read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:02,000", content)
            self.assertIn("00:00:02,000 --> 00:00:03,000", content)
            self.assertIn("00:00:03,000 --> 00:00:06,500", content)

    def test_empty_text_segments_are_skipped(self):
        segments = [
            {"index": 0, "text": "", "start_ms": 0, "duration_ms": 500},
            {"index": 1, "text": "Real line.", "start_ms": 500, "duration_ms": 1000},
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "subtitle.srt")
            segment_subtitle.build_segment_subtitles(segments, out)
            content = Path(out).read_text(encoding="utf-8")
            self.assertNotIn("00:00:00,000 --> 00:00:00,500", content)
            self.assertIn("Real line.", content)


if __name__ == "__main__":
    unittest.main()
