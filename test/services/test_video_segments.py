import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import video as vd


class _FakeAudioClip:
    def __init__(self, duration):
        self.duration = duration

    def close(self):
        pass


class _FakeVideoClip:
    def __init__(self, duration):
        self.duration = duration
        self.size = (1080, 1920)
        self.w = 1080
        self.h = 1920

    def subclipped(self, start_time, end_time):
        return _FakeVideoClip(end_time - start_time)

    def with_speed_scaled(self, factor):
        return _FakeVideoClip(self.duration / factor)

    def close(self):
        pass


class TestCombineSegmentVideos(unittest.TestCase):
    def _run_combine_segments(self, segments, video_clips, audio_duration=20.0):
        """Run combine_videos with timed segments; capture concat order."""
        concat_calls = []

        def fake_concat(clip_files, output_file, threads, output_dir, max_duration=None):
            concat_calls.append(list(clip_files))

        written_clips = []

        def fake_open(video_path, audio=False):
            return _FakeVideoClip(video_clips[video_path])

        def fake_write(clip, output_file, **kwargs):
            written_clips.append(output_file)
            Path(output_file).write_bytes(b"clip")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"fake")
            combined_path = str(Path(temp_dir) / "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip(audio_duration)),
                patch.object(vd, "_open_video_clip_quietly", side_effect=fake_open),
                patch.object(vd, "_write_videofile_with_codec_fallback", side_effect=fake_write),
                patch.object(vd, "concat_video_clips_with_ffmpeg", side_effect=fake_concat),
            ):
                vd.combine_videos(
                    combined_video_path=combined_path,
                    video_paths=[],
                    audio_file=str(audio_path),
                    segments=segments,
                )
            return concat_calls

    def test_segments_drive_concat_order(self):
        """Clips must be concatenated in segment order — no shuffle."""
        segments = [
            {"index": 0, "clips": ["/c/a.mp4", "/c/b.mp4"], "duration": 10.0},
            {"index": 1, "clips": ["/c/c.mp4"], "duration": 10.0},
        ]
        video_clips = {"/c/a.mp4": 10.0, "/c/b.mp4": 10.0, "/c/c.mp4": 10.0}
        concat_calls = self._run_combine_segments(segments, video_clips)

        self.assertEqual(len(concat_calls), 1)
        written = [Path(p).name for p in concat_calls[0]]
        # Segment 0 contributes 2 clips (each up to 5s covering 10s), then segment 1.
        self.assertTrue(all(os.path.isabs(p) or p for p in written))
        # Temp file order encodes processing order; segment 1's clip must
        # come after segment 0's.
        self.assertLess(written.index("temp-clip-1.mp4"), written.index("temp-clip-3.mp4"))

    def test_empty_clips_do_not_crash(self):
        segments = [{"index": 0, "clips": [], "duration": 10.0}]
        concat_calls = self._run_combine_segments(segments, {})
        self.assertEqual(concat_calls, [])


if __name__ == "__main__":
    unittest.main()
