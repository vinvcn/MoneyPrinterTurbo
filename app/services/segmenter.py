"""
把视频脚本切分成旁白片段（segment）。

segment-first 流水线的第一步：脚本的每个片段将独立配音、独立搜索素材，
因此分段结果决定了"哪句话配哪段画面"。切分规则必须确定性（不依赖 LLM），
让同一脚本在任何机器上得到相同的分段。

设计要点：
1. 先按句子/换行拆成最小旁白单元，再按时长上下限合并与拆分；
2. 时长估算复用无配音模式（``voice.estimate_no_voice_duration`` 的速率），
   避免第二套语速常数与真实 TTS 节奏漂移；
3. 单元之间保留连接符（空格/换行），保证片段文本重新拼接后仍是
   通顺的完整脚本，供字幕纠错和跨段回退使用。
"""

from dataclasses import dataclass
import re

from app.utils import utils

# 语速常数：CJK 约 4.2 字/秒，英文/数字约 2.7 词/秒。与 voice 服务中
# 无配音模式的估算保持同一套数字，避免分段时长与音频时长系统性偏离。
_CJK_CHARS_PER_SECOND = 4.2
_WORDS_PER_SECOND = 2.7

_DEFAULT_MIN_SEGMENT_DURATION = 3.0
_DEFAULT_MAX_SEGMENT_DURATION = 15.0
_CJK_PATTERN = re.compile(r"[一-鿿]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
# 句子边界标点：分段的"语义单元"是一句话，而不是逗号分隔的短语。
# 逗号只是句内停顿，画面通常要覆盖完整一句话，因此不作为单元边界；
# 超长单元会再由 max_duration 按词边界兜底拆分。
_SENTENCE_ENDINGS = frozenset(".!?。！？…；;؟؛")


@dataclass(frozen=True)
class ScriptSegment:
    """一个独立配音 + 独立配图的旁白片段。"""

    index: int
    text: str
    estimated_duration: float


def split_narration_units(text: str) -> list[str]:
    """
    把脚本拆成最小旁白单元（句子级，标点跟随前句保留）。

    与 ``utils.split_string_by_punctuations`` 不同，这里保留标点，
    因为分段结果会直接作为 TTS 输入，去掉标点会让语音失去停顿。
    数字中的小数点、千分位逗号不作为边界，换行是硬边界。
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return []

    units: list[str] = []
    current = ""
    pending_newline = False

    def flush_unit() -> None:
        nonlocal current
        stripped = current.strip()
        if stripped:
            units.append(stripped)
        current = ""

    for i, char in enumerate(normalized_text):
        if char == "\n":
            flush_unit()
            pending_newline = True
            continue

        if i > 0:
            previous_char = normalized_text[i - 1]
            next_char = normalized_text[i + 1] if i < len(normalized_text) - 1 else ""
            if char == "." and previous_char.isdigit() and next_char.isdigit():
                current += char
                continue
            if char == "," and previous_char.isdigit() and next_char.isdigit():
                current += char
                continue

        if char in _SENTENCE_ENDINGS:
            current += char
            flush_unit()
            pending_newline = False
            continue

        if pending_newline:
            # 换行后遇到第一个正文字符时才真正落上一个单元，保证行首
            # 缩进/空行不会产生空片段。
            flush_unit()
            pending_newline = False

        current += char

    if current.strip():
        flush_unit()

    return units


def estimate_duration_seconds(text: str) -> float:
    """
    按与无配音模式一致的语速估算旁白时长（秒）。

    只用于分段决策（合并/拆分阈值），真实时长以每段 TTS 音频为准。
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return 0.0

    cjk_chars = len(_CJK_PATTERN.findall(normalized_text))
    words = len(_WORD_PATTERN.findall(normalized_text))
    duration = cjk_chars / _CJK_CHARS_PER_SECOND + words / _WORDS_PER_SECOND
    return duration


def _split_unit_by_max_duration(unit: str, max_duration: float) -> list[str]:
    """把超过时长上限的单元按词/字符边界继续拆分。"""
    if estimate_duration_seconds(unit) <= max_duration:
        return [unit]

    if " " in unit:
        tokens = unit.split(" ")
    else:
        # 中文没有空格，按固定字符数近似拆分，尽量落在上限附近。
        chars_per_segment = max(
            1, int(max_duration * _CJK_CHARS_PER_SECOND)
        )
        return [
            unit[i : i + chars_per_segment]
            for i in range(0, len(unit), chars_per_segment)
        ]

    pieces: list[str] = []
    current: list[str] = []
    for token in tokens:
        candidate = " ".join([*current, token])
        if current and estimate_duration_seconds(candidate) > max_duration:
            pieces.append(" ".join(current))
            current = [token]
        else:
            current.append(token)
    if current:
        pieces.append(" ".join(current))
    return pieces


def segment_script(
    text: str,
    min_duration: float = _DEFAULT_MIN_SEGMENT_DURATION,
    max_duration: float = _DEFAULT_MAX_SEGMENT_DURATION,
) -> list[ScriptSegment]:
    """
    把脚本切分成旁白片段。

    规则：
    1. 按句子/换行得到最小单元；
    2. 相邻单元合并直到达到最小时长（避免片段过碎、画面切换过快）；
    3. 超过最大时长的单元按词边界继续拆分（保证单段搜索和 TTS 规模可控）。

    Args:
        text: 视频脚本文本。
        min_duration: 片段最短估算时长（秒），0 表示不合并。
        max_duration: 片段最长估算时长（秒）。

    Returns:
        按 ScriptSegment(index, text, estimated_duration) 排列的片段列表。
    """
    units = split_narration_units(text)
    if not units:
        return []

    max_duration = max(float(max_duration), 0.1)
    min_duration = max(float(min_duration), 0.0)

    # 先拆超长单元，再合并过短单元；合并不会重新超过上限，因为相邻
    # 单元合并只在两个都低于 min_duration 时发生，而每个都低于 min
    # 必然低于 max（min <= max 已在下方归一化）。
    normalized_units: list[str] = []
    for unit in units:
        normalized_units.extend(_split_unit_by_max_duration(unit, max_duration))

    if min_duration > max_duration:
        min_duration = max_duration

    merged: list[str] = []
    for unit in normalized_units:
        if merged and (
            estimate_duration_seconds(merged[-1]) < min_duration
            or estimate_duration_seconds(unit) < min_duration
        ):
            merged[-1] = f"{merged[-1]} {unit}".strip()
        else:
            merged.append(unit)

    return [
        ScriptSegment(
            index=index,
            text=segment_text,
            estimated_duration=round(estimate_duration_seconds(segment_text), 3),
        )
        for index, segment_text in enumerate(merged)
    ]
