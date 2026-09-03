"""
VLM 下载前相关性过滤（segment-first 流水线专用）。

背景：素材搜索按关键词返回候选，但搜索命中 ≠ 视觉相关。UAT 审计
（任务 17181fe7，issue #8）确认 vid-914ad5d9 这类"画面好看但主题不对"
的候选会被直接下载并进入成片，拉低质量。本模块在下载完整 mp4 之前，
用一张缩略图（或 mp4 首帧）让 VLM 判断候选是否与搜索词和旁白相关。

设计决策（issue #9，grilling session 2026-09-03）：
- 三值判定 relevant / irrelevant / uncertain；uncertain 放行，
  过滤器是质量增强，不能让流水线饿死。
- VLM 调用失败重试 3 次后 fail-open（放行），绝不阻塞任务。
- 图像来源优先缩略图（分辨率 ≥ 阈值才可用），否则 ffmpeg 提取候选
  mp4 首帧；同样重试 3 次后放行。
- 每次判定都写 INFO 日志（含判定、理由、图像来源），不静默。
- 端点/模型/密钥可配置（config.toml [vlm] 段），默认指向
  SiliconFlow + Qwen/Qwen3.5-4B，可切自建 vLLM 而不改代码。
"""

import base64
import json
import os
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.utils import utils

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
# 判定重试次数（含首次调用）。用尽后 fail-open 放行候选。
JUDGE_MAX_RETRIES = 3
# 缩略图下载超时（连接,读取）。缩略图远小于 mp4，超时从严。
THUMBNAIL_TIMEOUT = (10, 20)

VERDICT_RELEVANT = "relevant"
VERDICT_IRRELEVANT = "irrelevant"
VERDICT_UNCERTAIN = "uncertain"
_VALID_VERDICTS = {VERDICT_RELEVANT, VERDICT_IRRELEVANT, VERDICT_UNCERTAIN}

JUDGE_PROMPT_TEMPLATE = """
# Role: Stock Video Relevance Judge

You are given a single preview frame of a stock video candidate, the search
term it was found with, and the narration segment it would illustrate.

Decide whether this candidate is visually relevant to the narration segment:
- "relevant": the visuals clearly match what the narration describes.
- "irrelevant": the visuals clearly do NOT match (different subject, empty
  scene, watermark/text-only frame, unrelated imagery).
- "uncertain": you cannot confidently tell from a single frame.

Return ONLY a JSON object: {"verdict": "relevant|irrelevant|uncertain", "reason": "<short English reason>"}
""".strip()


def is_enabled() -> bool:
    """[vlm] enabled=false（或缺失段）时整体跳过过滤，行为与旧版一致。"""
    vlm_config = getattr(config, "vlm", None) or {}
    return bool(vlm_config.get("enabled", False))


def _vlm_setting(key: str, default: str) -> str:
    vlm_config = getattr(config, "vlm", None) or {}
    value = str(vlm_config.get(key, "") or "").strip()
    return value or default


def _min_thumbnail_pixels() -> tuple[int, int]:
    """缩略图最小分辨率阈值，默认 640x320（issue #9 D1）。"""
    vlm_config = getattr(config, "vlm", None) or {}
    try:
        width = int(vlm_config.get("min_thumbnail_width", 640))
        height = int(vlm_config.get("min_thumbnail_height", 320))
    except (TypeError, ValueError):
        return 640, 320
    return max(1, width), max(1, height)


def load_judge_config() -> dict[str, Any]:
    """
    汇总当前 [vlm] 配置，供调用方注入或审计。

    api_key 不进入返回值：调用方只负责把客户端函数注入 segment_material，
    密钥在本模块内部使用，避免随任务记录落盘。
    """
    return {
        "enabled": is_enabled(),
        "base_url": _vlm_setting("base_url", DEFAULT_BASE_URL),
        "model": _vlm_setting("model", DEFAULT_MODEL),
        "timeout": _judge_timeout(),
    }


def _judge_timeout() -> float:
    vlm_config = getattr(config, "vlm", None) or {}
    try:
        timeout = float(vlm_config.get("timeout", 30))
    except (TypeError, ValueError):
        return 30.0
    return timeout if timeout > 0 else 30.0


def build_judge_messages(
    image_data_uri: str,
    search_term: str,
    segment_text: str,
) -> list[dict[str, Any]]:
    """构造 VLM 判定消息：图像 + 搜索词 + 旁白文本。"""
    prompt = (
        f"{JUDGE_PROMPT_TEMPLATE}\n\n"
        f"## Search Term\n{search_term}\n\n"
        f"## Narration Segment\n{segment_text}\n"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _parse_verdict(response_text: str) -> tuple[str, str] | None:
    """从模型回复提取 (verdict, reason)；结构不合法返回 None。"""
    try:
        parsed = json.loads(response_text or "")
    except (json.JSONDecodeError, TypeError):
        # 宽松兜底：部分兼容端点在 json_object 模式外仍会夹带说明文字。
        match = None
        for candidate in (response_text or "").splitlines():
            stripped = candidate.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                match = stripped
                break
        if not match:
            return None
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        return None
    reason = str(parsed.get("reason", "")).strip()[:200]
    return verdict, reason


def judge_image(
    image_data_uri: str,
    search_term: str,
    segment_text: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
) -> tuple[str, str, int]:
    """
    对单张预览图做相关性判定。

    返回 (verdict, reason, attempts)。VLM 调用失败或回复不可解析时重试，
    重试用尽返回 (uncertain, "vlm unavailable after retries", attempts) ——
    即 fail-open 语义（issue #9 D2/D7）。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = build_judge_messages(image_data_uri, search_term, segment_text)
    last_error = ""
    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"},
                    "enable_thinking": False,
                },
                timeout=timeout,
            )
            if response.status_code >= 400:
                # 响应正文可能包含端点回显的鉴权上下文，只记状态码不记正文。
                last_error = f"status={response.status_code}"
                logger.warning(
                    "vlm judge http error: "
                    f"attempt={attempt}, model={model}, {last_error}"
                )
                continue
            content = response.json()["choices"][0]["message"]["content"] or ""
            parsed = _parse_verdict(content)
            if parsed is None:
                last_error = "unparseable verdict"
                logger.warning(
                    f"vlm judge returned unusable response: attempt={attempt}, "
                    f"model={model}, error={last_error}"
                )
                continue
            verdict, reason = parsed
            return verdict, reason, attempt
        except Exception as exc:
            last_error = type(exc).__name__
            logger.warning(
                f"vlm judge request failed: attempt={attempt}, model={model}, "
                f"error={last_error}"
            )
    return VERDICT_UNCERTAIN, "vlm unavailable after retries", JUDGE_MAX_RETRIES


def download_thumbnail_bytes(thumbnail_url: str) -> tuple[bytes, tuple[int, int]]:
    """
    下载缩略图并返回 (字节, 实际分辨率)。

    分辨率解析失败时返回 (0, 0)，由调用方按"不达标"处理并回退首帧。
    """
    response = requests.get(
        thumbnail_url,
        timeout=THUMBNAIL_TIMEOUT,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        },
    )
    response.raise_for_status()
    payload = response.content
    if not payload:
        raise ValueError("empty thumbnail response")
    width, height = _probe_image_size(payload)
    return payload, (width, height)


def _probe_image_size(payload: bytes) -> tuple[int, int]:
    """从 PNG/JPEG/WebP 头部解析像素尺寸，避免引入完整图像库依赖。"""
    if payload[:8] == b"\x89PNG\r\n\x1a\n" and len(payload) >= 24:
        width = int.from_bytes(payload[16:20], "big")
        height = int.from_bytes(payload[20:24], "big")
        return width, height
    if payload[:2] == b"\xff\xd8":
        # JPEG：扫描 SOF0/SOF2 标记段读取尺寸。
        index = 2
        size = len(payload)
        while index + 9 < size:
            if payload[index] != 0xFF:
                index += 1
                continue
            marker = payload[index + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height = int.from_bytes(payload[index + 5:index + 7], "big")
                width = int.from_bytes(payload[index + 7:index + 9], "big")
                return width, height
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            segment_length = int.from_bytes(payload[index + 2:index + 4], "big")
            index += 2 + segment_length
        return 0, 0
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        # 简化处理：VP8X/VP8/VP8L 均按"无法确认"处理，回退首帧。
        return 0, 0
    return 0, 0


def to_data_uri(payload: bytes) -> str:
    """把图像字节编码为 data URI（SiliconFlow/VLLM 实测支持）。"""
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_first_frame_jpeg(video_path: str) -> bytes:
    """
    用 ffmpeg 提取 mp4 首帧的 JPEG 字节（缩略图缺失时的判定输入）。

    输出写临时文件而不是 stdout 管道，规避 Windows/Docker 挂载下
    `-f image2pipe` 的偶发截断问题；提取失败抛出异常由调用方重试。
    """
    import subprocess
    import tempfile

    ffmpeg_binary = utils.get_ffmpeg_binary()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as frame_file:
        frame_path = frame_file.name
    try:
        subprocess.run(
            [
                ffmpeg_binary,
                "-y",
                "-v",
                "error",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                frame_path,
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )
        with open(frame_path, "rb") as fp:
            payload = fp.read()
        if not payload:
            raise ValueError("empty first frame")
        return payload
    finally:
        try:
            os.unlink(frame_path)
        except OSError:
            pass


def _candidate_local_path(item: Any) -> str:
    """从 MaterialInfo.source_info 里找已下载的本地文件（无则返回空串）。"""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    local_file = str(source.get("local_file") or "").strip()
    return local_file


def make_default_judge():
    """
    构造注入 segment_material 的默认判定回调（issue #9 D1/D2/D3/D5）。

    每个候选的判定顺序：
    1. 缩略图（存在且实际分辨率 ≥ [vlm] 阈值）；
    2. mp4 首帧（需要候选已下载——过滤发生在下载前，因此缩略图缺失时
       先下载到临时位置提取首帧，拒收后立即删除）；
    3. 两条路径都失败 → 重试用尽 → uncertain 放行（fail-open）。

    每次判定都写 INFO 日志；审计记录不含图像字节，只含来源与判定结果。
    """

    def judge_candidate(
        item: Any,
        segment_text: str = "",
        search_term: str = "",
    ) -> dict[str, Any]:
        source = item.source_info if isinstance(item.source_info, dict) else {}
        asset_id = str(source.get("asset_id") or "")
        thumbnail_url = str(source.get("thumbnail_url") or "").strip()
        min_width, min_height = _min_thumbnail_pixels()
        base_url = _vlm_setting("base_url", DEFAULT_BASE_URL)
        model = _vlm_setting("model", DEFAULT_MODEL)
        api_key = str((getattr(config, "vlm", None) or {}).get("api_key", "") or "")
        timeout = _judge_timeout()

        image_source = ""
        image_data_uri = ""

        # 路径 1：缩略图。
        if thumbnail_url:
            try:
                payload, (width, height) = download_thumbnail_bytes(thumbnail_url)
                if width >= min_width and height >= min_height:
                    image_source = "thumbnail"
                    image_data_uri = to_data_uri(payload)
                else:
                    logger.info(
                        "vlm filter thumbnail below resolution threshold: "
                        f"asset_id={asset_id}, size={width}x{height}, "
                        f"min={min_width}x{min_height}, fallback to first frame"
                    )
            except Exception as exc:
                logger.warning(
                    "vlm filter thumbnail download failed: "
                    f"asset_id={asset_id}, error={type(exc).__name__}"
                )

        # 路径 2：mp4 首帧。过滤发生在下载前，这里只对缩略图缺失/不达标的
        # 候选临时下载，拒收后立即删除临时文件。
        temp_path = ""
        if not image_data_uri:
            try:
                temp_path = _download_candidate_to_temp(item)
            except Exception as exc:
                logger.warning(
                    "vlm filter candidate predownload failed: "
                    f"asset_id={asset_id}, error={type(exc).__name__}"
                )
            if temp_path:
                try:
                    payload = extract_first_frame_jpeg(temp_path)
                    image_source = "first_frame"
                    image_data_uri = to_data_uri(payload)
                except Exception as exc:
                    logger.warning(
                        "vlm filter first frame extraction failed: "
                        f"asset_id={asset_id}, error={type(exc).__name__}"
                    )
                finally:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        # 路径 3：两种图像都不可得。判定必然失败 → 直接 uncertain 放行，
        # 不空耗 VLM 重试。
        if not image_data_uri:
            logger.warning(
                "vlm filter has no usable image input, fail-open: "
                f"asset_id={asset_id}, term={search_term!r}"
            )
            return {
                "term": search_term,
                "asset_id": asset_id,
                "verdict": VERDICT_UNCERTAIN,
                "reason": "no image input available",
                "image_source": "none",
                "attempts": 0,
                "page": _safe_page(source),
            }

        verdict, reason, attempts = judge_image(
            image_data_uri=image_data_uri,
            search_term=search_term,
            segment_text=segment_text,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        # 判定日志（issue #9 D3：不静默）。审计记录不含图像内容。
        logger.info(
            "vlm filter verdict: "
            f"asset_id={asset_id}, term={search_term!r}, verdict={verdict}, "
            f"image_source={image_source}, attempts={attempts}, "
            f"reason={reason!r}"
        )
        return {
            "term": search_term,
            "asset_id": asset_id,
            "verdict": verdict,
            "reason": reason,
            "image_source": image_source,
            "attempts": attempts,
            "page": _safe_page(source),
        }

    return judge_candidate


def _safe_page(source: dict) -> int:
    try:
        return max(1, int(source.get("page", 1)))
    except (TypeError, ValueError):
        return 1


def _download_candidate_to_temp(item: Any) -> str:
    """
    把候选 mp4 下载到临时文件供首帧提取（仅缩略图不可用时触发）。

    复用项目代理/TLS 配置；调用方负责在拒收后删除临时文件。
    """
    import tempfile

    response = requests.get(
        item.url,
        proxies=getattr(config, "proxy", {}) or {},
        verify=True,
        timeout=(30, 120),
        stream=True,
    )
    response.raise_for_status()
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        for chunk in response.iter_content(chunk_size=1024 * 512):
            if chunk:
                temp_file.write(chunk)
        temp_file.close()
        if os.path.getsize(temp_file.name) <= 0:
            raise ValueError("empty candidate download")
        return temp_file.name
    except Exception:
        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
        raise
