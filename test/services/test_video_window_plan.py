"""segment_window_plan 单测（A3 截尾+下限合并，F-H1 的共享纯函数）。

期望值来自计划 .omo/plans/finding-g-fh1-fh2.md T1 的逐例标注：
- D=12.816/W=3 → [3,3,3,3,0.816]（末窗 0.816 保留，不合并）
- D=3.744/W=3 → [3.744]（末窗 0.744 < 阈值 → 合并为单窗）
两例把有效合并阈值约束在 (0.744, 0.816]，实现取 0.75。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.video import segment_window_plan


def _assert_plan(actual, expected):
    assert len(actual) == len(expected)
    assert actual == pytest.approx(expected, abs=1e-9)
    # 计划窗口之和必须精确等于 segment_duration（A3 的核心不变量）。
    assert sum(actual) == pytest.approx(sum(expected), abs=1e-9)


def test_long_segment_tail_kept_as_short_window():
    # 12.816/3 → ceil=5，末窗 0.816 ≥ 0.75 → 不合并，维持 5 窗（配额=5）。
    _assert_plan(segment_window_plan(12.816, 3), [3, 3, 3, 3, 0.816])


def test_tail_below_threshold_merges_into_single_window():
    # n=2、末窗 0.744 < 0.75 → 合并进前窗，n 降为 1，整段单窗。
    _assert_plan(segment_window_plan(3.744, 3), [3.744])


def test_exact_multiple_no_merge():
    _assert_plan(segment_window_plan(15.0, 3), [3, 3, 3, 3, 3])


def test_tiny_tail_merged():
    # n=5、末窗 0.1 → 合并：n=4，末窗 = 12.1 - 9 = 3.1。
    _assert_plan(segment_window_plan(12.1, 3), [3, 3, 3, 3.1])


def test_shorter_than_window_single_window():
    _assert_plan(segment_window_plan(2.0, 3), [2.0])


def test_zero_duration_returns_empty():
    assert segment_window_plan(0, 3) == []


def test_negative_duration_returns_empty():
    assert segment_window_plan(-5, 3) == []


def test_tail_merged_three_windows():
    # n=4、末窗 0.48 → 合并：n=3，末窗 = 9.48 - 6 = 3.48。
    _assert_plan(segment_window_plan(9.480, 3), [3, 3, 3.48])


def test_tail_above_threshold_no_merge():
    # 末窗 2.472 ≥ 1.0 → 不合并，n=4。
    _assert_plan(segment_window_plan(11.472, 3), [3, 3, 3, 2.472])


def test_zero_window_returns_empty():
    assert segment_window_plan(5.0, 0) == []


def test_floor_tightens_merge_threshold():
    # floor 收紧到 0.45：末窗 0.48 不再合并；末窗 0.744 也不合并。
    _assert_plan(segment_window_plan(9.480, 3, floor=0.45), [3, 3, 3, 0.48])
    _assert_plan(segment_window_plan(3.744, 3, floor=0.45), [3, 0.744])


def test_floor_looser_than_default_does_not_expand_merge():
    # floor 只能收紧（取 min）：默认行为不受放宽的 floor 影响。
    _assert_plan(segment_window_plan(12.816, 3, floor=5.0), [3, 3, 3, 3, 0.816])


@pytest.mark.parametrize(
    "duration,width",
    [(12.816, 3), (9.480, 3), (3.744, 3), (11.472, 3), (9.936, 3), (3.936, 3), (8.232, 3)],
)
def test_real_uat_durations_cover_duration_exactly(duration, width):
    plan = segment_window_plan(duration, width)
    assert plan
    assert sum(plan) == pytest.approx(duration, abs=1e-9)
    # 除（可能的）末窗外，每窗都是满窗。
    for window in plan[:-1]:
        assert window == pytest.approx(width, abs=1e-9)
    assert 0 < plan[-1] <= width + 0.75 + 1e-9
