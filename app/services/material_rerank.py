"""
素材搜索页重排客户端（Qwen/Qwen3-VL-Reranker-8B，SiliconFlow /v1/rerank）。

segment-first 流水线在把每个搜索页的候选送 VLM 判定之前，先用重排器按
"与搜索 term 的视觉相关性"对候选排序，只把 top_n 个（外加无缩略图而无法
重排的候选）交给 VLM，显著减少 VLM 调用次数（plan rerank-top5-vlm 实测
约省 70-75%）。

设计决策：
- fail-open 是硬契约：term 为空、候选为空、开关关闭、无凭据、网络错误、
  429 限流退避耗尽、400 兜底重试仍失败、响应体不可解析——任何失败一律
  原样返回传入的 items（回到"无重排"的 provider 顺序），绝不阻塞流水线；
  重排只影响送审顺序，VLM 仍是终审。
- 凭据继承 [vlm]（同一 SiliconFlow 账号无需重复填写），[material_rerank]
  的 api_key/base_url 非空时覆盖。
- SiliconFlow 的 400 "Field required" 常常并非请求体缺字段，而是它服务端
  拉取缩略图 URL 失败时的误导性报错（.scratch/rerank-eval/
  qwen3vl_rerank_eval.py 实测）；此时改发 base64 data URI 文档重试一次，
  得分一致。
- 网络错误先直连再走 config.proxy 各试一次（eval 脚本验证过的模式，
  国内端点直连为主）。

导入方向约束：本模块只允许依赖 app.config / app.models /
app.services.vlm_judge；禁止导入 app.services.segment_material（todo 3 将
由 segment_material 反向导入本模块，会成环）。
"""

import dataclasses
import time
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo
from app.services.vlm_judge import download_thumbnail_bytes, to_data_uri

DEFAULT_RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
DEFAULT_TOP_N = 5
DEFAULT_TIMEOUT_SECONDS = 120
# 连接超时固定 30s（与缩略图下载同级），读超时取 [material_rerank] timeout。
CONNECT_TIMEOUT_SECONDS = 30
# 429 限流退避序列（秒）。共发起 4 次请求（首调 + 3 次重试），间隔逐次翻倍；
# 仍被限流则 fail-open 放行，不做无限退避。
RATE_LIMIT_BACKOFF_SECONDS = (1.5, 3.0, 6.0)


class _RerankUnavailableError(Exception):
    """重排请求最终不可用；仅被 rerank_page 捕获打 fail-open 行，类型名进入日志 error= 字段。"""

    def __init__(self, status: int, detail: str = ""):
        self.status = status
        self.detail = detail
        super().__init__(f"status={status}" + (f", detail={detail}" if detail else ""))


@dataclasses.dataclass(frozen=True, slots=True)
class _RerankRequest:
    """一次重排请求的全部发送参数（凭据只在 headers，不落日志）。"""

    endpoint: str
    headers: dict[str, str]
    body: dict[str, Any]
    timeout: tuple[int, float]


def _section() -> dict[str, Any]:
    """[material_rerank] 段；缺失时返回空 dict，各读取点自行给默认值。"""
    return getattr(config, "material_rerank", None) or {}


def is_rerank_enabled() -> bool:
    """[material_rerank] enabled=false（或缺失段）时整体关闭重排。"""
    return bool(_section().get("enabled", True))


def _top_n() -> int:
    """[material_rerank] top_n；缺失、非法或小于 1 时回落 5。"""
    try:
        top_n = int(_section().get("top_n", DEFAULT_TOP_N))
    except (TypeError, ValueError):
        return DEFAULT_TOP_N
    if top_n < 1:
        return DEFAULT_TOP_N
    return top_n


def _rerank_timeout() -> float:
    """[material_rerank] timeout（读超时，秒）；缺失、非法或非正时回落 120。"""
    try:
        timeout = float(_section().get("timeout", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    if timeout <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return timeout


def _credentials() -> tuple[str, str]:
    """返回 (api_key, base_url)：继承 [vlm]，[material_rerank] 非空时覆盖。"""
    rerank_section = _section()
    vlm_section = getattr(config, "vlm", None) or {}
    api_key = str(rerank_section.get("api_key", "") or "") or str(
        vlm_section.get("api_key", "") or ""
    )
    base_url = str(rerank_section.get("base_url", "") or "") or str(
        vlm_section.get("base_url", "") or ""
    )
    return api_key.strip(), base_url.strip()


def _split_by_thumbnail(
    items: list[MaterialInfo],
) -> tuple[list[tuple[MaterialInfo, str]], list[MaterialInfo]]:
    """按缩略图有无拆分：可重排（缩略图非空）与其余（保持原相对顺序）。"""
    rankable: list[tuple[MaterialInfo, str]] = []
    unrankable: list[MaterialInfo] = []
    for item in items:
        source = item.source_info if isinstance(item.source_info, dict) else {}
        thumbnail = str(source.get("thumbnail_url") or "").strip()
        if thumbnail:
            rankable.append((item, thumbnail))
        else:
            unrankable.append(item)
    return rankable, unrankable


def _asset_id(item: MaterialInfo) -> str:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    return str(source.get("asset_id") or "")


def _configured_proxies() -> dict[str, str]:
    """config.proxy 中非空的代理配置；空 dict 等价直连。"""
    proxy_config = getattr(config, "proxy", None) or {}
    return {key: value for key, value in dict(proxy_config).items() if value}


def _post_once(request: _RerankRequest, attempt: int) -> requests.Response:
    """单次尝试：先直连，失败走 config.proxy 重试一次；两次都失败抛最后一次异常。"""
    proxies = _configured_proxies()
    last_error: requests.RequestException | None = None
    for index, proxies_try in enumerate((None, proxies)):
        try:
            return requests.post(
                request.endpoint,
                headers=request.headers,
                json=request.body,
                timeout=request.timeout,
                proxies=proxies_try,
            )
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "material rerank request failed: "
                f"attempt={attempt}, via={'direct' if index == 0 else 'proxy'}, "
                f"error={type(exc).__name__}"
            )
    assert last_error is not None
    raise last_error


def _post_with_rate_limit_retry(request: _RerankRequest) -> requests.Response:
    """发送请求：429 按退避序列重试（共 4 次尝试），耗尽抛专用异常。"""
    for attempt in range(1, len(RATE_LIMIT_BACKOFF_SECONDS) + 2):
        response = _post_once(request, attempt)
        if (
            response.status_code == 429
            and attempt <= len(RATE_LIMIT_BACKOFF_SECONDS)
        ):
            delay = RATE_LIMIT_BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "material rerank rate limited: "
                f"attempt={attempt}, retry in {delay}s"
            )
            time.sleep(delay)
            continue
        return response
    raise _RerankUnavailableError(status=429, detail="rate limited after backoff")


def _data_uri_documents(
    rankable: list[tuple[MaterialInfo, str]],
) -> list[dict[str, str]]:
    """把缩略图下载后编码为 base64 data URI 文档（400 兜底重试用）。"""
    documents: list[dict[str, str]] = []
    for _item, thumbnail in rankable:
        payload, _size = download_thumbnail_bytes(thumbnail)
        documents.append({"image": to_data_uri(payload)})
    return documents


def _parse_scores(
    response: requests.Response,
    rankable: list[tuple[MaterialInfo, str]],
) -> list[tuple[MaterialInfo, float]]:
    """解析 (候选, relevance_score)；index 按文档位置映射，形状异常直接抛出。

    200 但没有任何分数同样视为不可用（否则会静默丢弃全部候选）。
    """
    results = response.json()["results"]
    scored: list[tuple[MaterialInfo, float]] = []
    for result in results:
        item, _thumbnail = rankable[result["index"]]
        scored.append((item, float(result["relevance_score"])))
    if not scored:
        raise _RerankUnavailableError(status=response.status_code, detail="no results")
    return scored


def _request_scores(
    term: str,
    rankable: list[tuple[MaterialInfo, str]],
) -> tuple[list[tuple[MaterialInfo, float]], bool]:
    """发起重排请求并解析分数；失败抛异常由 rerank_page 兜底。

    首选 URL 文档；收到 400 时改发 base64 data URI 文档重试一次（缘由见
    模块 docstring）。返回 (scored 列表, 是否走了 base64 兜底)。
    """
    api_key, base_url = _credentials()
    body: dict[str, Any] = {
        "model": str(_section().get("model", "") or DEFAULT_RERANK_MODEL),
        "query": term,
        "documents": [{"image": thumbnail} for _item, thumbnail in rankable],
        "top_n": len(rankable),
        "return_documents": False,
    }
    request = _RerankRequest(
        endpoint=f"{base_url}/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        body=body,
        timeout=(CONNECT_TIMEOUT_SECONDS, _rerank_timeout()),
    )
    response = _post_with_rate_limit_retry(request)
    fallback_used = False
    if response.status_code == 400:
        # 重建请求体而不是原地改 documents：首次发送的请求体保持原样，
        # 便于审计与测试捕获每一次实际发出的内容。
        retry_body = {**body, "documents": _data_uri_documents(rankable)}
        fallback_used = True
        response = _post_with_rate_limit_retry(
            dataclasses.replace(request, body=retry_body)
        )
    if response.status_code >= 400:
        raise _RerankUnavailableError(status=response.status_code)
    return _parse_scores(response, rankable), fallback_used


def rerank_page(
    term: str,
    items: list[MaterialInfo],
    top_n: int,
) -> list[MaterialInfo]:
    """对一页候选做视觉重排，返回交给下游（VLM 判定）的列表。

    返回顺序：重排前 top_n 名 → 无缩略图候选（原顺序）→ 其余可重排候选
    （分数降序，同分保持原序）。任何失败一律原样返回 items——fail-open。
    """
    normalized = (term or "").strip()
    if not normalized or not items or not is_rerank_enabled():
        return items
    rankable, unrankable = _split_by_thumbnail(items)
    if not rankable:
        return items
    api_key, base_url = _credentials()
    if not api_key or not base_url:
        logger.info(f"material rerank skipped, no credentials: term={normalized!r}")
        return items
    try:
        scored, fallback_used = _request_scores(normalized, rankable)
    except Exception as exc:
        logger.error(
            "material rerank failed, fail-open: "
            f"term={normalized!r}, error={type(exc).__name__}"
        )
        return items
    # Python 的 sorted 稳定：reverse=True 时同分候选仍保持原相对顺序。
    ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
    for rank, (item, score) in enumerate(ordered, 1):
        logger.info(
            "material rerank score: "
            f"asset_id={_asset_id(item)}, score={score}, rank={rank}"
        )
    top_block = ordered[:top_n]
    logger.info(
        "material rerank selected: "
        f"term={normalized!r}, ranked={len(ordered)}, "
        f"top={len(top_block)}, fallback={fallback_used}"
    )
    return (
        [item for item, _score in top_block]
        + unrankable
        + [item for item, _score in ordered[top_n:]]
    )
