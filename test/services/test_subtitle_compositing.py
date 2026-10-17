import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
)

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import video as vd
from app.utils import utils


FONT_PATH = os.path.join(utils.font_dir(), "STHeitiMedium.ttc")


def _make_overlay_rgba(height=6, width=8, seed=7):
    """Build a deterministic RGBA overlay with varied (semi-transparent) alpha."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return np.dstack([rgb, alpha])


class TestSubtitleOverlayParity(unittest.TestCase):
    """#311: the bbox blit is an optimisation, not a visual change."""

    def test_overlay_clip_matches_moviepy_composite_pixel_exact(self):
        rgba = _make_overlay_rgba()
        base = ColorClip((40, 30), color=(12, 34, 56)).with_duration(0.5)
        overlay = (
            ImageClip(rgba, transparent=True)
            .with_duration(0.5)
            .with_position((9, 11))
            .with_start(0.1)
            .with_end(0.4)
        )
        moviepy_composite = CompositeVideoClip([base, overlay])
        ours = vd._SubtitleOverlayClip(
            base, [vd._build_subtitle_overlay(overlay, base.size)]
        )
        try:
            for t in (0.0, 0.05, 0.1, 0.25, 0.39, 0.4, 0.45):
                np.testing.assert_array_equal(
                    ours.get_frame(t),
                    moviepy_composite.get_frame(t),
                    err_msg=f"frame mismatch at t={t}",
                )
        finally:
            moviepy_composite.close()
            ours.close()
            base.close()
            overlay.close()

    def test_two_overlapping_overlays_match_moviepy_order(self):
        base = ColorClip((24, 24), color=(5, 5, 5)).with_duration(0.3)
        first = (
            ImageClip(_make_overlay_rgba(6, 6, seed=1), transparent=True)
            .with_duration(0.3)
            .with_position((2, 2))
            .with_start(0.0)
            .with_end(0.3)
        )
        second = (
            ImageClip(_make_overlay_rgba(6, 6, seed=2), transparent=True)
            .with_duration(0.3)
            .with_position((4, 4))
            .with_start(0.0)
            .with_end(0.3)
        )
        moviepy_composite = CompositeVideoClip([base, first, second])
        ours = vd._SubtitleOverlayClip(
            base,
            [
                vd._build_subtitle_overlay(first, base.size),
                vd._build_subtitle_overlay(second, base.size),
            ],
        )
        try:
            np.testing.assert_array_equal(
                ours.get_frame(0.1), moviepy_composite.get_frame(0.1)
            )
        finally:
            moviepy_composite.close()
            ours.close()
            base.close()
            first.close()
            second.close()

    def test_realistic_rounded_subtitle_matches_moviepy_pixel_exact(self):
        """Inner CompositeVideoClip (rounded board + text) must stay identical."""
        base = ColorClip((320, 480), color=(30, 60, 90)).with_duration(0.4)
        box_w, clip_h = 220, 70
        text_clip = TextClip(
            font=FONT_PATH,
            text="字幕渲染对齐",
            font_size=28,
            color="white",
            bg_color=None,
            stroke_color="black",
            stroke_width=1,
            interline=7,
            size=(box_w, None),
            text_align="center",
            margin=(0, 8),
        )
        bg_clip = vd._rounded_subtitle_background_clip(
            box_w, clip_h, "#000000", alpha=140, radius=12
        )
        text_x, text_y = vd._get_visible_center_position(text_clip, box_w, clip_h)
        inner = (
            CompositeVideoClip(
                [bg_clip, text_clip.with_position((text_x, text_y))],
                size=(box_w, clip_h),
            )
            .with_start(0.05)
            .with_end(0.35)
            .with_duration(0.3)
            .with_position(("center", 480 * 0.95 - clip_h))
        )

        moviepy_composite = CompositeVideoClip([base, inner])
        ours = vd._SubtitleOverlayClip(
            base, [vd._build_subtitle_overlay(inner, base.size)]
        )
        try:
            for t in (0.0, 0.05, 0.2, 0.34, 0.35):
                np.testing.assert_array_equal(
                    ours.get_frame(t),
                    moviepy_composite.get_frame(t),
                    err_msg=f"frame mismatch at t={t}",
                )
        finally:
            moviepy_composite.close()
            ours.close()
            base.close()
            inner.close()
            text_clip.close()

    def test_duration_covers_overlay_that_outlives_base(self):
        base = ColorClip((16, 16), color=(0, 0, 0)).with_duration(0.2)
        overlay = (
            ImageClip(_make_overlay_rgba(4, 4), transparent=True)
            .with_duration(0.5)
            .with_position((0, 0))
            .with_start(0.0)
            .with_end(0.5)
        )
        moviepy_composite = CompositeVideoClip([base, overlay])
        ours = vd._SubtitleOverlayClip(
            base, [vd._build_subtitle_overlay(overlay, base.size)]
        )
        try:
            self.assertEqual(ours.duration, moviepy_composite.duration)
            self.assertEqual(ours.duration, 0.5)
        finally:
            moviepy_composite.close()
            ours.close()
            base.close()
            overlay.close()

    def test_build_overlay_resolves_center_position(self):
        overlay = (
            ImageClip(_make_overlay_rgba(4, 4), transparent=True)
            .with_duration(1)
            .with_position(("center", "center"))
        )
        try:
            built = vd._build_subtitle_overlay(overlay, (40, 30))
        finally:
            overlay.close()
        self.assertEqual(built.x, (40 - 4) // 2)
        self.assertEqual(built.y, (30 - 4) // 2)


    def test_build_overlay_is_deterministic_for_identical_input(self):
        """Same (text, style) must yield the same cached overlay."""
        rgba = _make_overlay_rgba()
        first = (
            ImageClip(rgba, transparent=True)
            .with_duration(1)
            .with_position((3, 4))
        )
        second = (
            ImageClip(rgba.copy(), transparent=True)
            .with_duration(1)
            .with_position((3, 4))
        )
        try:
            built_first = vd._build_subtitle_overlay(first, (40, 30))
            built_second = vd._build_subtitle_overlay(second, (40, 30))
        finally:
            first.close()
            second.close()
        np.testing.assert_array_equal(built_first.rgba, built_second.rgba)
        self.assertEqual(
            (built_first.x, built_first.y, built_first.start, built_first.end),
            (built_second.x, built_second.y, built_second.start, built_second.end),
        )

    def test_static_overlay_frames_are_identical_over_time(self):
        """A cached overlay is static, so its frames do not change with t."""
        base = ColorClip((16, 16), color=(9, 9, 9)).with_duration(1)
        overlay = (
            ImageClip(_make_overlay_rgba(5, 5), transparent=True)
            .with_duration(1)
            .with_position((2, 2))
            .with_start(0.0)
            .with_end(1.0)
        )
        clip = vd._SubtitleOverlayClip(
            base, [vd._build_subtitle_overlay(overlay, base.size)]
        )
        try:
            np.testing.assert_array_equal(clip.get_frame(0.0), clip.get_frame(0.7))
        finally:
            clip.close()
            base.close()
            overlay.close()

    def test_rendered_output_matches_moviepy_composite(self):
        """Re-encoded output of the new path equals the old path frame-for-frame."""
        base = ColorClip((64, 64), color=(20, 40, 60)).with_duration(0.4)
        overlay = (
            ImageClip(_make_overlay_rgba(10, 12), transparent=True)
            .with_duration(0.4)
            .with_position((8, 7))
            .with_start(0.0)
            .with_end(0.4)
        )
        old = CompositeVideoClip([base, overlay])
        new = vd._SubtitleOverlayClip(
            base, [vd._build_subtitle_overlay(overlay, base.size)]
        )
        with tempfile.TemporaryDirectory() as tmp:
            old_path = os.path.join(tmp, "old.mp4")
            new_path = os.path.join(tmp, "new.mp4")
            old.write_videofile(
                old_path, codec="libx264", fps=10, audio=False, logger=None
            )
            new.write_videofile(
                new_path, codec="libx264", fps=10, audio=False, logger=None
            )
            old_clip = VideoFileClip(old_path)
            new_clip = VideoFileClip(new_path)
            try:
                for t in (0.0, 0.15, 0.3):
                    np.testing.assert_array_equal(
                        old_clip.get_frame(t),
                        new_clip.get_frame(t),
                        err_msg=f"rendered frame mismatch at t={t}",
                    )
            finally:
                old_clip.close()
                new_clip.close()
        old.close()
        new.close()
        base.close()
        overlay.close()

class TestOverlayBlitBounds(unittest.TestCase):
    def test_blit_only_touches_overlay_bounding_box(self):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        overlay = vd._SubtitleOverlay(
            start=0.0,
            end=1.0,
            x=4,
            y=3,
            rgba=np.dstack(
                [np.full((5, 6, 3), 255, np.uint8), np.full((5, 6), 255, np.uint8)]
            ),
        )
        before = frame.copy()
        vd._blit_rgba_overlay(frame, overlay)

        np.testing.assert_array_equal(frame[3:8, 4:10], 255)
        untouched = np.ones(frame.shape[:2], dtype=bool)
        untouched[3:8, 4:10] = False
        np.testing.assert_array_equal(frame[untouched], before[untouched])

    def test_blit_clips_overlay_at_frame_edges(self):
        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        overlay = vd._SubtitleOverlay(
            start=0.0,
            end=1.0,
            x=-2,
            y=-2,
            rgba=np.dstack(
                [np.full((4, 4, 3), 200, np.uint8), np.full((4, 4), 255, np.uint8)]
            ),
        )
        vd._blit_rgba_overlay(frame, overlay)

        np.testing.assert_array_equal(frame[0:2, 0:2], 200)
        np.testing.assert_array_equal(frame[2:, :], 0)

    def test_blit_ignores_overlay_entirely_off_frame(self):
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        overlay = vd._SubtitleOverlay(
            start=0.0, end=1.0, x=-10, y=-10, rgba=np.full((3, 3, 4), 255, np.uint8)
        )
        before = frame.copy()
        vd._blit_rgba_overlay(frame, overlay)
        np.testing.assert_array_equal(frame, before)

    def test_blit_applies_pillow_alpha_blending(self):
        frame = np.full((4, 4, 3), 255, np.uint8)
        overlay = vd._SubtitleOverlay(
            start=0.0,
            end=1.0,
            x=0,
            y=0,
            rgba=np.dstack(
                [np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 128, np.uint8)]
            ),
        )
        expected = np.asarray(
            Image.alpha_composite(
                Image.new("RGBA", (4, 4), (255, 255, 255, 255)),
                Image.fromarray(overlay.rgba),
            )
        )[:, :, :3]
        vd._blit_rgba_overlay(frame, overlay)
        np.testing.assert_array_equal(frame, expected)

    def test_compose_frame_does_not_mutate_base_frames(self):
        base = ColorClip((8, 8), color=(1, 2, 3)).with_duration(0.2)
        reference = base.get_frame(0).copy()
        overlay = vd._SubtitleOverlay(
            start=0.0, end=1.0, x=0, y=0, rgba=np.full((8, 8, 4), 255, np.uint8)
        )
        clip = vd._SubtitleOverlayClip(base, [overlay])
        try:
            frame = clip.get_frame(0)
            np.testing.assert_array_equal(base.get_frame(0), reference)
            self.assertTrue((frame == 255).all())
        finally:
            clip.close()
            base.close()


if __name__ == "__main__":
    unittest.main()
