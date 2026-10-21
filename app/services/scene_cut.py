"""scene_cut：pyscenedetect 场景切分（BYOM footage KB，ADR-0023 决策 3）。

契约（与 VBL worker 索引管线对接）：
    POST /api/v1/scene_cut {"input_path": "<视频绝对路径>"}
    -> {"scenes": [{"t_start_s": 0.0, "t_end_s": 2.0}, ...]}

边界（ADR-0023 决策 3）：调用方（VBL worker）把源视频放到共享交换卷
`/shared` 上再传路径进来；本服务只读不写——不持久化视频、不派生缩略图、
不登记任何用户/会话/计费概念，出去的只有浮点秒边界，字节永远不进 HTTP。

切分策略：ContentDetector（帧内容差异，默认阈值 27.0）检硬切；一次都没
检出时退化为"整片一个场景"，让上游至少拿到一个可索引片段。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ContentDetector 默认阈值（帧间内容差异分数，pyscenedetect 规范值）。
DEFAULT_CONTENT_THRESHOLD = 27.0


class SceneCutError(Exception):
    """输入不可切分（打不开 / 不是视频 / 文件损坏）——控制器映射 4xx。"""


@dataclass(frozen=True)
class SceneBoundary:
    """一个场景的时间区间（秒，闭开区间 [t_start_s, t_end_s)）。"""

    t_start_s: float
    t_end_s: float

    def as_dict(self) -> dict[str, float]:
        return {"t_start_s": self.t_start_s, "t_end_s": self.t_end_s}


def _round_seconds(value: float) -> float:
    """秒值保留毫秒精度，避免 JSON 里出现浮点长尾。"""
    return round(float(value), 3)


def detect_scene_cuts(
    input_path: str, threshold: float = DEFAULT_CONTENT_THRESHOLD
) -> list[SceneBoundary]:
    """检测 ``input_path`` 的场景边界，返回按时间排序的区间列表。

    Args:
        input_path: 视频文件绝对路径（调用方置于 `/shared`，ADR-0023 决策 3）。
        threshold: ContentDetector 的内容差异阈值。

    Returns:
        至少一个 ``SceneBoundary``；无硬切时退化为整片一个场景。

    Raises:
        SceneCutError: 视频打不开 / 时长为零 / 内容不可解码。
    """
    # cv2 家族依赖较重且只在触达本服务时才需要——函数内延迟导入，
    # 让无关模块的测试环境不必背着 opencv 启动。
    from scenedetect import ContentDetector, SceneManager, open_video

    try:
        video = open_video(input_path)
    except Exception as exc:  # scenedetect 打不开即输入不合法（含非视频文件）
        raise SceneCutError(f"cannot open video: {input_path}") from exc

    try:
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        # 检测过程是 CPU 密集的同步调用；控制器经 anyio 线程池调进来。
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        duration_s = (
            video.duration.get_seconds() if video.duration is not None else 0.0
        )
    except Exception as exc:
        raise SceneCutError(f"scene detection failed: {input_path}") from exc
    finally:
        # VideoStreamCv2 不是上下文管理器；显式释放底层 cv2 捕获句柄。
        release = getattr(video, "release", None)
        if callable(release):
            release()

    if duration_s <= 0:
        raise SceneCutError(f"video has no duration: {input_path}")

    boundaries: list[SceneBoundary] = []
    for start, end in scene_list:
        t_start_s = _round_seconds(start.get_seconds())
        t_end_s = _round_seconds(end.get_seconds())
        if t_end_s > t_start_s:
            boundaries.append(SceneBoundary(t_start_s=t_start_s, t_end_s=t_end_s))

    if not boundaries:
        # 一次硬切都没检出：整片作为一个场景，上游仍可索引（ADR-0023 决策 3）。
        logger.info("no hard cuts detected, whole video as one scene: %s", input_path)
        boundaries = [SceneBoundary(t_start_s=0.0, t_end_s=_round_seconds(duration_s))]

    return boundaries
