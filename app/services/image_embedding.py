"""
图像向量嵌入 + 任务级查重门与文本-图像粗筛（segment-first 流水线专用，
plan Finding G；粗筛为 embed-prefilter T2）。

背景：同上传者的系列素材画面高度相似，跨段重复下载后会以"换段不换画"
的形式进入成片。本模块在 VLM 相关性判定之前，先对候选预览图做多模态
嵌入，与本任务已收下素材的嵌入逐对计算余弦相似度，命中重复直接拒收，
省一次 VLM 调用与整段重复画面。

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
# 粗筛阈值默认值：T5 用 UAT 标注数据（102 条 VLM 记录 → 75 条 eligible，
# 43 个 asset）做零误拒校准——取 relevant 余弦最小值 0.109445 再留 0.02
# 余量。证据：.scratch/uat-storage/coarse-calibration-20260907/
# （summary.txt / candidates.tsv / f1_curve.tsv）。官方无固定阈值；该值只
# 保证"相关素材零误拒"（irrelevant 仍有约 35/45 放行），粗筛只是 VLM 之前
# 的廉价预过滤，VLM 仍是终审。
DEFAULT_COARSE_THRESHOLD = 0.089445
# allow: SIZE_OK — 任务契约把 embed_text（T1）与 EmbeddingGate 粗筛（T2）钉在
# 本模块，纯 LOC 超 250 为已记录例外；拆分（如 embedding_clients.py）留待 F4 评审。


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


def embed_text(
    text: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> list[float] | None:
    """
    对一段文本做多模态嵌入，失败返回 None（fail-open）。

    与 embed_image 使用同一端点（DashScope 多模态嵌入，官方文档
    help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference）。
    官方文档明确：所有模态的向量位于同一语义空间，文本向量可与图像向量
    直接做余弦跨模态比较；官方未提供固定相似度阈值，须用自有数据校准。
    端点可用 base_url 指向兼容网关。429 按退避序列重试；其余非 2xx、
    超时、连接错误、响应缺字段或向量为空一律返回 None，由调用方放行。
    """
    if not text.strip():
        return None
    endpoint = (base_url or "").strip() or DEFAULT_EMBEDDING_BASE_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": {"contents": [{"text": text}]},
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
                "text embedding request failed: "
                f"attempt={attempt}, model={model}, error={type(exc).__name__}"
            )
            return None
        if (
            response.status_code == 429
            and attempt <= len(RATE_LIMIT_BACKOFF_SECONDS)
        ):
            delay = RATE_LIMIT_BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "text embedding rate limited: "
                f"attempt={attempt}, model={model}, retry in {delay}s"
            )
            time.sleep(delay)
            continue
        if response.status_code >= 400:
            # 正文可能回显鉴权上下文，只记状态码不记正文。
            logger.warning(
                "text embedding http error: "
                f"attempt={attempt}, model={model}, status={response.status_code}"
            )
            return None
        try:
            embedding = response.json()["output"]["embeddings"][0]["embedding"]
        except Exception as exc:
            logger.warning(
                "text embedding unusable response: "
                f"attempt={attempt}, model={model}, error={type(exc).__name__}"
            )
            return None
        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(isinstance(v, (int, float)) for v in embedding)
        ):
            logger.warning(
                "text embedding vector malformed: "
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
    任务级图像查重门（accepted-only 注册表）。

    生命周期与单个视频任务一致：task.py 每次生成为其新建实例，注册表与
    候选缓存随对象回收，不跨任务泄漏。候选向量在判定时写入缓存（判定
    语义为"先嵌入再比对"），素材真正被采纳时 register_accepted 把缓存
    向量挪入注册表，不再发起任何 API 调用。开启粗筛（coarse_enabled）
    时，未被查重拒绝的候选再过一道文本-图像余弦预筛，term 的查询向量
    同样只在进程内缓存。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        threshold: float,
        base_url: str | None = None,
        coarse_enabled: bool = False,
        coarse_threshold: float = 0.0,
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
        # 粗筛开关与阈值：cos < coarse_threshold 的候选提前拒绝，省一次
        # VLM 调用。默认关闭，行为与引入粗筛之前完全一致。
        self.coarse_enabled = bool(coarse_enabled)
        self.coarse_threshold = float(coarse_threshold)
        # 粗筛查询向量缓存：term -> 向量。同一段落的多个候选共用同一
        # term，只嵌入一次；随任务结束一并回收，不持久化。
        self._query_vecs: dict[str, list[float]] = {}

    def judge_candidate_embedding(
        self,
        url: str,
        data_uri: str,
        term: str = "",
        skip_coarse: bool = False,
    ) -> dict[str, Any] | None:
        """
        嵌入候选并依次过查重门与粗筛。

        (1) 候选向量经 _candidates 缓存，每个 URL 只嵌入一次；(2) 先查重：
        命中重复返回审计记录（verdict="duplicate"），重复是终审拒绝，不再
        进入粗筛；(3) 未被拒绝且粗筛生效时（coarse_enabled、term 非空白且
        未 skip_coarse），查询向量经 _query_vecs 缓存或 embed_text 现算
        （失败 fail-open，跳过粗筛），cos < coarse_threshold 返回审计记录
        （verdict="prefiltered"），此时不清除候选缓存。其余情况返回 None
        放行；register_accepted 只对真正被采纳的 URL 生效。
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
        if self.coarse_enabled and term.strip() and not skip_coarse:
            query_vec = self._query_vecs.get(term)
            if query_vec is None:
                query_vec = embed_text(
                    text=term,
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                # 失败的查询向量不写缓存：下次同 term 仍会重试嵌入。
                if query_vec is not None:
                    self._query_vecs[term] = query_vec
            if query_vec is not None:
                coarse_cos = _cosine_similarity(query_vec, vec)
                if coarse_cos < self.coarse_threshold:
                    return {
                        "verdict": "prefiltered",
                        "reason": (
                            f"coarse cos={coarse_cos:.3f} < threshold "
                            f"{self.coarse_threshold:.3f}"
                        ),
                        "image_source": "embedding",
                        "cos": coarse_cos,
                    }
                # 审计缺口 G3：粗筛实际运行且放行时补一行通过记录，与
                # prefiltered 行成对，让"粗筛看过并放行"在日志流可见。
                # 查重-only（粗筛关闭）、空白 term、skip_coarse 复判或查询
                # 向量嵌入失败时不打——放行候选由下游 vlm filter verdict
                # 行覆盖，门内保持静默。
                logger.info(
                    "embedding gate passed candidate: "
                    f"term={term!r}, cos={coarse_cos}"
                )
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


def is_coarse_filter_enabled() -> bool:
    """[image_embedding] coarse_filter=false（或缺失段）时关闭粗筛预过滤。"""
    section = getattr(config, "image_embedding", None) or {}
    return bool(section.get("coarse_filter", False))


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


def _coarse_threshold() -> float:
    section = getattr(config, "image_embedding", None) or {}
    try:
        threshold = float(section.get("coarse_threshold", DEFAULT_COARSE_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_COARSE_THRESHOLD
    if not 0 < threshold <= 1:
        return DEFAULT_COARSE_THRESHOLD
    return threshold


def make_default_gate() -> EmbeddingGate:
    """按 [image_embedding] 配置构造查重门（task.py 每任务调用一次）。"""
    return EmbeddingGate(
        model=_gate_setting("model", DEFAULT_EMBEDDING_MODEL),
        api_key=str((getattr(config, "image_embedding", None) or {}).get("api_key", "") or ""),
        threshold=_gate_threshold(),
        base_url=_gate_setting("base_url", "") or None,
        coarse_enabled=is_coarse_filter_enabled(),
        coarse_threshold=_coarse_threshold(),
    )
