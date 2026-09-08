"""
图像向量嵌入 + 任务级查重门（segment-first 流水线专用，plan Finding G）。

背景：同上传者的系列素材画面高度相似，跨段重复下载后会以"换段不换画"
的形式进入成片。本模块在 VLM 相关性判定之前，先对候选预览图做多模态
嵌入，与本任务已收下素材的嵌入逐对计算余弦相似度，命中重复直接拒收，
省一次 VLM 调用与整段重复画面。DashScope 嵌入只用于近重复检测
（duplicate-only）：页面级相关性排序由 [material_rerank] 承担，本门
不再做文本-图像粗筛。

设计决策（.omo/FINDINGS-SUMMARY.md，finding G）：
- 模型 tongyi-embedding-vision-flash，DashScope 形状 A 请求体
  （{"model", "input": {"contents": [{"image": data_uri}]}}），经验审计
  确认返回 768 维、逐位确定、约 243ms/次。
- fail-open 是硬契约：仅 429 限流退避重试 3 次（1.5/3/6s），其余任何
  HTTP 错误、超时、连接错误或响应解析失败一律返回 None 放行候选，
  查重门绝不阻塞流水线。
- accepted-only 注册表：只与"本任务已收下"的素材比对；候选向量在判定
  时缓存，素材被采纳时由 register_accepted 无 API 调用挪入注册表。
- gate 对象按任务创建（task.py 接线注入），注册表与缓存不跨任务泄漏。
"""

import math
import time
from typing import Any

import requests
from loguru import logger

from app.config import config

DEFAULT_EMBEDDING_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
DEFAULT_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"
# 429 限流退避序列（秒）。共发起 4 次请求（首调 + 3 次重试），间隔逐次翻倍；
# 仍被限流则 fail-open 放行，不做无限退避。
RATE_LIMIT_BACKOFF_SECONDS = (1.5, 3.0, 6.0)
DEFAULT_DUPLICATE_THRESHOLD = 0.68


def embed_image(
    data_uri: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> list[float] | None:
    """
    对单张 data URI 图像做多模态嵌入，失败返回 None（fail-open）。

    端点默认 DashScope 多模态嵌入服务，可用 base_url 指向兼容网关。
    429 按退避序列重试；其余非 2xx、超时、连接错误、响应缺字段或向量
    为空一律返回 None，由调用方放行候选。
    """
    endpoint = (base_url or "").strip() or DEFAULT_EMBEDDING_BASE_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": {"contents": [{"image": data_uri}]},
    }
    for attempt in range(1, len(RATE_LIMIT_BACKOFF_SECONDS) + 2):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=body,
                timeout=timeout,
                proxies=getattr(config, "proxy", {}) or {},
            )
        except Exception as exc:
            logger.warning(
                "image embedding request failed: "
                f"attempt={attempt}, model={model}, error={type(exc).__name__}"
            )
            return None
        if (
            response.status_code == 429
            and attempt <= len(RATE_LIMIT_BACKOFF_SECONDS)
        ):
            delay = RATE_LIMIT_BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "image embedding rate limited: "
                f"attempt={attempt}, model={model}, retry in {delay}s"
            )
            time.sleep(delay)
            continue
        if response.status_code >= 400:
            # 正文可能回显鉴权上下文，只记状态码不记正文。
            logger.warning(
                "image embedding http error: "
                f"attempt={attempt}, model={model}, status={response.status_code}"
            )
            return None
        try:
            embedding = response.json()["output"]["embeddings"][0]["embedding"]
        except Exception as exc:
            logger.warning(
                "image embedding unusable response: "
                f"attempt={attempt}, model={model}, error={type(exc).__name__}"
            )
            return None
        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(isinstance(v, (int, float)) for v in embedding)
        ):
            logger.warning(
                "image embedding vector malformed: "
                f"attempt={attempt}, model={model}"
            )
            return None
        return [float(v) for v in embedding]
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；维度不一致或零向量时返回 0（视为最不相似）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class EmbeddingGate:
    """
    任务级图像查重门（accepted-only 注册表，duplicate-only）。

    生命周期与单个视频任务一致：task.py 每次生成为其新建实例，注册表与
    候选缓存随对象回收，不跨任务泄漏。候选向量在判定时写入缓存（判定
    语义为"先嵌入再比对"），素材真正被采纳时 register_accepted 把缓存
    向量挪入注册表，不再发起任何 API 调用。本门只做近重复检测：
    DashScope 嵌入仅用于与已采纳素材的余弦比对，不做任何文本-图像
    相关性预筛。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        threshold: float,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.threshold = float(threshold)
        self.base_url = (base_url or "").strip() or None
        # 已采纳素材：url -> 向量。只有这里的向量参与查重比对。
        self._accepted: dict[str, list[float]] = {}
        # 判定过但尚未采纳的候选：url -> 向量。拒收的候选残留在此无害
        # （同 URL 再次出现时免一次重复嵌入），随任务结束一并回收。
        self._candidates: dict[str, list[float]] = {}

    def judge_candidate_embedding(
        self,
        url: str,
        data_uri: str,
        term: str = "",
    ) -> dict[str, Any] | None:
        """
        嵌入候选并与已采纳注册表查重。

        (1) 候选向量经 _candidates 缓存，每个 URL 只嵌入一次；(2) 与
        注册表逐对算余弦，命中重复返回审计记录（verdict="duplicate"），
        重复是终审拒绝。其余情况返回 None 放行；register_accepted 只对
        真正被采纳的 URL 生效。term 保留在调用契约上（审计记录携带搜索
        词），门内不再使用。
        """
        vec = self._candidates.get(url)
        if vec is None:
            vec = embed_image(
                data_uri=data_uri,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            if vec is None:
                return None
            self._candidates[url] = vec
        duplicate_of, cos = self._closest_accepted(vec)
        if duplicate_of is not None:
            return {
                "verdict": "duplicate",
                "reason": f"cos={cos:.3f} >= threshold",
                "image_source": "embedding",
                "duplicate_of": duplicate_of,
                "cos": cos,
            }
        return None

    def register_accepted(self, url: str) -> None:
        """把候选缓存向量挪入已采纳注册表（无 API 调用）。"""
        vec = self._candidates.pop(url, None)
        if vec is not None:
            self._accepted[url] = vec

    def _closest_accepted(self, vec: list[float]) -> tuple[str | None, float]:
        best_url = None
        best_cos = -1.0
        for accepted_url, accepted_vec in self._accepted.items():
            cos = _cosine_similarity(vec, accepted_vec)
            if cos > best_cos:
                best_url, best_cos = accepted_url, cos
        if best_url is not None and best_cos >= self.threshold:
            return best_url, best_cos
        return None, -1.0


def is_duplicate_gate_enabled() -> bool:
    """[image_embedding] duplicate_gate=false（或缺失段）时整体关闭查重门。"""
    section = getattr(config, "image_embedding", None) or {}
    return bool(section.get("duplicate_gate", False))


def _gate_setting(key: str, default: str) -> str:
    section = getattr(config, "image_embedding", None) or {}
    value = str(section.get(key, "") or "").strip()
    return value or default


def _gate_threshold() -> float:
    section = getattr(config, "image_embedding", None) or {}
    try:
        threshold = float(section.get("duplicate_threshold", DEFAULT_DUPLICATE_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_DUPLICATE_THRESHOLD
    if not 0 < threshold <= 1:
        return DEFAULT_DUPLICATE_THRESHOLD
    return threshold


def make_default_gate() -> EmbeddingGate:
    """按 [image_embedding] 配置构造查重门（task.py 每任务调用一次）。"""
    return EmbeddingGate(
        model=_gate_setting("model", DEFAULT_EMBEDDING_MODEL),
        api_key=str((getattr(config, "image_embedding", None) or {}).get("api_key", "") or ""),
        threshold=_gate_threshold(),
        base_url=_gate_setting("base_url", "") or None,
    )
