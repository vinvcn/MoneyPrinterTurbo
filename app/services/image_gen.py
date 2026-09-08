"""
subject 层图片生成（替代 subject 视频搜索，issue #10 follow-up）。

自有词条全线失败时，用 Kwai-Kolors/Kolors 按 LLM 精炼的画面描述生成
一张概念图并转成整段时长的静态 clip；Kolors 失败（重试 2 次）时退回
provider 图片搜索（Pexels/Pixabay photos API，同账号 key）。全部失败
返回空 clip，段空手由素材层记录 warning。

降级链：LLM 精炼失败 → 旁白原文作 prompt；生成图逐次独立，天然无跨段
复用问题（UAT a043f7bb 的 subject 复用问题在该路径上结构性消失）。

图片 URL 一小时过期（SiliconFlow 文档），因此生成后立即下载落盘。
"""

import os
import time
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.services import llm, material
from app.utils import utils

import re

_KOLORS_ENDPOINT = "https://api.siliconflow.cn/v1/images/generations"
_CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")
_PEXELS_PHOTO_ENDPOINT = "https://api.pexels.com/v1/search"
_PIXABAY_PHOTO_ENDPOINT = "https://pixabay.com/api/"
_CLIP_DURATION_SECONDS = 15.5  # segmenter 单段上限 15s + 装配安全余量
_KOLORS_ATTEMPTS = 3  # 初始 + 重试 2 次（G4 决策）
_REQUEST_TIMEOUT = (30, 90)


def _tls_verify() -> bool:
    # 与 material._get_tls_verify 同语义：默认校验，config.app.tls_verify=false
    # 仅供企业代理/自签证书场景临时关闭。
    value = config.app.get("tls_verify", True)
    if isinstance(value, str):
        value = value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _image_gen_api_key() -> str:
    # [image_gen] api_key 缺省回落 [vlm] api_key：同一 SiliconFlow 账号
    # 无需重复填写（G5 决策）。
    key = str(config.image_gen.get("api_key", "") or "").strip()
    return key or str(config.vlm.get("api_key", "") or "").strip()


def _aspect_orientation(video_aspect: Any) -> str:
    """返回 "portrait" / "landscape"；供尺寸键选择与图片搜索 orientation 复用。"""
    aspect = str(getattr(video_aspect, "value", video_aspect) or "")
    return "portrait" if aspect in ("9:16", "portrait") else "landscape"


def _image_size_for(video_aspect: Any) -> str:
    orientation = _aspect_orientation(video_aspect)
    key = f"image_size_{orientation}"
    # per-key 默认：缺失 landscape 键时不得静默回落竖屏尺寸。
    default = "720x1280" if orientation == "portrait" else "1280x720"
    return str(config.image_gen.get(key) or default)


def refine_scene_prompt(segment_text: str) -> str:
    """
    把旁白精炼成英文静态画面描述；LLM 失败或输出为空时降级旁白原文
    （Kolors 对中文 prompt 同样可用，降级不阻塞生成）。
    """
    narration = (segment_text or "").strip()
    if not narration:
        return ""
    prompt = (
        "# Role: Text-to-Image Scene Writer\n\n"
        "Condense the narration below into ONE short English visual scene "
        "description (max 40 words): concrete subject, setting, lighting, "
        "camera shot. Photographic style. Output ONLY the description "
        "itself — no quotes, no explanations, no text-in-image requests.\n\n"
        f"Narration: {narration}"
    )
    response = llm.generate_response(prompt)
    if response.startswith("Error: "):
        logger.warning(
            "image scene refinement failed, falling back to raw narration: "
            f"{response.removeprefix('Error: ').strip()[:120]}"
        )
        return narration
    refined = response.strip().strip('"').strip()
    return refined or narration


def generate_kolors_image(prompt: str, image_size: str, save_dir: str) -> str:
    """
    调 Kolors 生成一张图并立即下载落盘（响应 URL 一小时过期）。
    传输错误 / 429 / 5xx 重试共 3 次尝试（退避 1s、2s）；401/403 不重试。
    成功返回本地图片路径，全部失败返回 ""。
    """
    model = str(config.image_gen.get("model") or "Kwai-Kolors/Kolors")
    payload = {
        "model": model,
        "prompt": prompt,
        "image_size": image_size,
        "num_inference_steps": int(config.image_gen.get("num_inference_steps", 20)),
        "guidance_scale": float(config.image_gen.get("guidance_scale", 7.5)),
        "batch_size": 1,
    }
    headers = {
        "Authorization": f"Bearer {_image_gen_api_key()}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(1, _KOLORS_ATTEMPTS + 1):
        try:
            response = requests.post(
                _KOLORS_ENDPOINT,
                json=payload,
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=_REQUEST_TIMEOUT,
            )
            if response.status_code in (401, 403):
                logger.warning(
                    f"kolors auth rejected (no retry): status={response.status_code}"
                )
                return ""
            response.raise_for_status()
            images = (response.json() or {}).get("images") or []
            image_url = (images[0] or {}).get("url", "") if images else ""
            if not image_url:
                raise ValueError("empty image url in response")
            return _download_image(image_url, save_dir, prefix="kolors")
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # 仅 429/5xx/传输错误可重试；其余 4xx（400/404/422 等）重试无益。
            retryable = status is None or status == 429 or status >= 500
            logger.warning(
                f"kolors generation failed: attempt={attempt}/{_KOLORS_ATTEMPTS}, "
                f"retryable={retryable}, error={last_error[:140]}"
            )
            if not retryable or attempt >= _KOLORS_ATTEMPTS:
                return ""
            time.sleep(attempt)
    logger.warning(f"kolors generation exhausted retries: {last_error[:140]}")
    return ""


def _download_image(image_url: str, save_dir: str, prefix: str) -> str:
    response = requests.get(
        image_url,
        proxies=config.proxy,
        verify=_tls_verify(),
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    os.makedirs(save_dir, exist_ok=True)
    extension = ".png" if "png" in str(response.headers.get("Content-Type", "")) else ".jpg"
    path = os.path.join(save_dir, f"{prefix}-{utils.md5(image_url)}{extension}")
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def search_provider_photo(query: str, video_aspect: Any, save_dir: str) -> str:
    """
    provider 图片搜索兜底：Pexels photos API 优先，Pixabay image API 次之
    （Coverr 无图片 API，不参与）。返回本地图片路径，全部失败返回 ""。
    """
    query = (query or "").strip()
    if not query or _CJK_PATTERN.search(query):
        # 图片搜索 API 仅接受英文：含 CJK 的查询必然低召回，直接放弃回退。
        return ""
    orientation = _aspect_orientation(video_aspect)
    pixabay_orientation = "vertical" if orientation == "portrait" else "horizontal"

    pexels_key = material.get_api_key("pexels_api_keys")
    if pexels_key:
        try:
            response = requests.get(
                _PEXELS_PHOTO_ENDPOINT,
                params={
                    "query": query,
                    "per_page": 3,
                    "orientation": orientation,
                },
                headers={"Authorization": pexels_key},
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            photos = (response.json() or {}).get("photos") or []
            if photos:
                image_url = ((photos[0] or {}).get("src") or {}).get("original", "")
                if image_url:
                    return _download_image(image_url, save_dir, prefix="pexels-photo")
        except Exception as exc:
            logger.warning(f"pexels photo search failed: {type(exc).__name__}: {exc}")

    pixabay_key = material.get_api_key("pixabay_api_keys")
    if pixabay_key:
        try:
            response = requests.get(
                _PIXABAY_PHOTO_ENDPOINT,
                params={
                    "key": pixabay_key,
                    "q": query,
                    "image_type": "photo",
                    "per_page": 3,
                    "orientation": pixabay_orientation,
                },
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            hits = (response.json() or {}).get("hits") or []
            if hits:
                image_url = (hits[0] or {}).get("largeImageURL", "")
                if image_url:
                    return _download_image(image_url, save_dir, prefix="pixabay-photo")
        except Exception as exc:
            logger.warning(f"pixabay photo search failed: {type(exc).__name__}: {exc}")

    return ""


def still_to_clip(image_path: str, save_dir: str, duration: float = _CLIP_DURATION_SECONDS) -> str:
    """
    静态图 → 整段时长 mp4（ffmpeg loop）。装配器按段窗口裁切，因此固定
    时长即可，无需向素材层传递段时长。转换失败返回 ""。
    """
    if not image_path or not os.path.exists(image_path):
        return ""
    os.makedirs(save_dir, exist_ok=True)
    clip_path = os.path.join(
        save_dir,
        f"imgclip-{utils.md5(image_path)}.mp4",
    )
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-t",
        f"{float(duration):.3f}",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        clip_path,
    ]
    try:
        import subprocess

        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        if result.returncode != 0 or not os.path.exists(clip_path):
            logger.warning(
                f"still-to-clip ffmpeg failed: rc={result.returncode}, "
                f"stderr={result.stderr[-160:]}"
            )
            return ""
        return clip_path
    except Exception as exc:
        logger.warning(f"still-to-clip failed: {type(exc).__name__}: {exc}")
        return ""


def make_subject_clip(
    segment_text: str,
    subject_term: str,
    video_aspect: Any,
    save_dir: str,
    duration: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    subject 层图片 clip 获取的编排入口（由 task.py 注入为
    video_match.match_segments 的 generate_image 回调）。

    返回 (clip_path, audit_record)；clip_path 为 "" 时段空手。audit_record
    随 manifest 的 image_gen 字段落盘（prompt/model/source/路径）。
    duration 为目标 clip 时长（秒）；None 保持既有默认时长，video-match
    的名额回填按剩余窗口时长和覆盖此值。
    """
    image_size = _image_size_for(video_aspect)
    model = str(config.image_gen.get("model") or "Kwai-Kolors/Kolors")
    record: dict[str, Any] = {
        "model": model,
        "image_size": image_size,
        "source": "failed",
        "prompt": "",
        "image": "",
        "clip": "",
        "attempts": 0,
        "error": "",
    }

    prompt = refine_scene_prompt(segment_text)
    record["prompt"] = prompt
    if not prompt:
        record["error"] = "empty_prompt"
        return "", record

    image_path = ""
    attempts = 0
    for source in ("kolors", "provider_photo"):
        if source == "kolors":
            attempts += 1
            record["attempts"] = attempts
            image_path = generate_kolors_image(prompt, image_size, save_dir)
        else:
            if not subject_term:
                record["error"] = "no_image_source"
                break
            image_path = search_provider_photo(subject_term, video_aspect, save_dir)
        if image_path:
            record["source"] = source
            break

    if not image_path:
        record["error"] = record.get("error") or "all_sources_failed"
        logger.warning(
            "subject image clip failed for all sources: "
            f"prompt={prompt[:60]!r}"
        )
        return "", record

    record["image"] = os.path.basename(image_path)
    clip_path = still_to_clip(
        image_path,
        save_dir,
        duration=_CLIP_DURATION_SECONDS if duration is None else float(duration),
    )
    record["clip"] = os.path.basename(clip_path) if clip_path else ""
    if not clip_path:
        record["error"] = "still_to_clip_failed"
        return "", record
    return clip_path, record
