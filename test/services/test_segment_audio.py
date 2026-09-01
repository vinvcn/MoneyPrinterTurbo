import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import segment_audio


class TestPrepareSegmentAudio(unittest.TestCase):
    def _write_real_tone(self, path, duration=1.0, freq="440"):
        """Generate a real decodable mp3 via ffmpeg so pydub can read it."""
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"sine=frequency={freq}:sample_rate=24000:duration={duration}",
                "-codec:a", "libmp3lame", "-q:a", "4", str(path),
            ],
            capture_output=True,
            check=True,
        )

    def test_empty_segments_return_empty_result(self):
        result = segment_audio.prepare_segment_audio(
            segments=[],
            task_id="t",
            tts=lambda **_: None,
        )
        self.assertEqual(result.audio_file, "")
        self.assertEqual(result.total_duration_ms, 0)
        self.assertEqual(result.segments, [])

    def test_tts_failure_fails_prepare(self):
        segments = [{"index": 0, "text": "hello"}]

        def failing_tts(**_):
            return None

        result = segment_audio.prepare_segment_audio(
            segments=segments,
            task_id="tts-fail",
            tts=failing_tts,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_index, 0)

    def test_real_tone_segments_merge_with_accurate_offsets(self):
        """Two 1s tones must produce offsets of 0ms and ~1000ms (±40ms)."""
        task_id = "seg-audio-real"
        task_dir = Path(segment_audio.utils.task_dir(task_id))
        try:
            segments = [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}]

            def fake_tts(text, voice_file, **_):
                freq = "440" if text == "a" else "880"
                self._write_real_tone(voice_file, duration=1.0, freq=freq)
                return object()  # sub_maker placeholder

            result = segment_audio.prepare_segment_audio(
                segments=segments,
                task_id=task_id,
                tts=fake_tts,
            )

            self.assertTrue(result.ok)
            self.assertEqual(len(result.segments), 2)
            self.assertEqual(result.segments[0]["start_ms"], 0)
            # First segment is exactly 1000ms of frames.
            self.assertAlmostEqual(
                result.segments[1]["start_ms"], 1000, delta=40
            )
            self.assertAlmostEqual(result.total_duration_ms, 2000, delta=60)
            self.assertTrue(os.path.isfile(result.audio_file))
            self.assertGreater(os.path.getsize(result.audio_file), 0)
        finally:
            import shutil

            shutil.rmtree(task_dir, ignore_errors=True)

    def test_okay_result_includes_per_segment_durations(self):
        task_id = "seg-audio-durations"
        task_dir = Path(segment_audio.utils.task_dir(task_id))
        try:
            segments = [{"index": 0, "text": "only"}]

            def fake_tts(text, voice_file, **_):
                self._write_real_tone(voice_file, duration=1.0, freq="660")
                return object()

            result = segment_audio.prepare_segment_audio(
                segments=segments,
                task_id=task_id,
                tts=fake_tts,
            )
            self.assertTrue(result.ok)
            self.assertGreater(result.segments[0]["duration_ms"], 900)
            self.assertLess(result.segments[0]["duration_ms"], 1100)
        finally:
            import shutil

            shutil.rmtree(task_dir, ignore_errors=True)


class TestHmsConversion(unittest.TestCase):
    def test_offsets_convert_to_srt_timestamps(self):
        self.assertEqual(segment_audio.ms_to_srt_timestamp(0), "00:00:00,000")
        self.assertEqual(segment_audio.ms_to_srt_timestamp(3661500), "01:01:01,500")
        self.assertEqual(segment_audio.ms_to_srt_timestamp(999), "00:00:00,999")


if __name__ == "__main__":
    unittest.main()
