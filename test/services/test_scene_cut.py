"""scene_cut 契约测试（BYOM footage KB，ADR-0023 决策 3）。

用 ffmpeg lavfi 拼一段 6s、2/4s 处各一次硬切的视频当 fixture（红->绿->蓝
纯色段），断言边界落在硬切点附近；缺失路径 404、非视频 4xx。
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.asgi import get_application  # noqa: E402

SEGMENT_SECONDS = 2.0
SEGMENTS = 3
TOTAL_SECONDS = SEGMENT_SECONDS * SEGMENTS
# fixture 帧率低（8fps）+ ContentDetector 在色彩骤变处偶有半秒内漂移。
BOUNDARY_TOLERANCE_S = 0.75


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_hard_cut_video(path: Path) -> None:
    """生成 SEGMENTS 段纯色、段间硬切的短视频（总时长 TOTAL_SECONDS）。"""
    inputs: list[str] = []
    for color in ("red", "green", "blue"):
        inputs.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x64:d={SEGMENT_SECONDS}:r=8",
            ]
        )
    filter_complex = (
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]"
    )
    cmd = (
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "8",
            str(path),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True)


@unittest.skipUnless(_has_ffmpeg(), "ffmpeg not available on PATH")
class SceneCutContractTest(unittest.TestCase):
    """POST /api/v1/scene_cut 的 HTTP 契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mpt_scene_cut_")
        cls.root = Path(cls._tmp.name)
        cls.video_path = cls.root / "hard_cuts.mp4"
        _make_hard_cut_video(cls.video_path)
        cls.client = TestClient(get_application())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _post(self, input_path: str):
        return self.client.post("/api/v1/scene_cut", json={"input_path": input_path})

    def test_finds_hard_cuts_at_segment_boundaries(self) -> None:
        response = self._post(str(self.video_path))
        self.assertEqual(response.status_code, 200, response.text)

        body = response.json()
        self.assertIn("scenes", body)
        scenes = body["scenes"]
        self.assertGreaterEqual(len(scenes), SEGMENTS, scenes)

        # 场景按时间排序、区间为正，且总量覆盖整片。
        starts = [scene["t_start_s"] for scene in scenes]
        self.assertEqual(starts, sorted(starts))
        for scene in scenes:
            self.assertGreater(scene["t_end_s"], scene["t_start_s"])
        self.assertAlmostEqual(scenes[0]["t_start_s"], 0.0, delta=BOUNDARY_TOLERANCE_S)
        self.assertAlmostEqual(
            scenes[-1]["t_end_s"], TOTAL_SECONDS, delta=BOUNDARY_TOLERANCE_S
        )

        # 两个硬切点（2s / 4s）附近必须有边界。
        for expected in (SEGMENT_SECONDS, 2 * SEGMENT_SECONDS):
            self.assertTrue(
                any(
                    abs(scene["t_start_s"] - expected) <= BOUNDARY_TOLERANCE_S
                    for scene in scenes
                ),
                f"no scene boundary near {expected}s in {scenes}",
            )

    def test_response_shape_only_seconds(self) -> None:
        response = self._post(str(self.video_path))
        self.assertEqual(response.status_code, 200, response.text)
        for scene in response.json()["scenes"]:
            self.assertEqual(
                sorted(scene.keys()),
                ["t_end_s", "t_start_s"],
                "scene payload must carry seconds only — never bytes",
            )

    def test_missing_input_path_returns_404(self) -> None:
        missing = self.root / "does-not-exist.mp4"
        response = self._post(str(missing))
        self.assertEqual(response.status_code, 404, response.text)

    def test_non_video_file_returns_4xx(self) -> None:
        text_file = self.root / "not-a-video.txt"
        text_file.write_text("this is not a video", encoding="utf-8")
        response = self._post(str(text_file))
        self.assertIn(response.status_code, (400, 422), response.text)

    def test_service_layer_returns_whole_video_scene_when_no_cuts(self) -> None:
        """一次硬切都没有时退化为整片一个场景（上游至少拿到一个片段）。"""
        from app.services import scene_cut as scene_cut_service

        flat_path = self.root / "flat.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=2:r=8",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "8",
                str(flat_path),
            ],
            check=True,
            capture_output=True,
        )
        boundaries = scene_cut_service.detect_scene_cuts(str(flat_path))
        self.assertEqual(len(boundaries), 1, boundaries)
        self.assertEqual(boundaries[0].t_start_s, 0.0)
        self.assertAlmostEqual(boundaries[0].t_end_s, 2.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
