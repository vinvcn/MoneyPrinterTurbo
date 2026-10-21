"""BYOM 素材知识库（footage KB）的 premise 候选池：VPC 向量检索 + footage_map 快照。

BYOM 生成（video_source="premise"）不再由 MPT 自持素材登记册——宿主侧的
user_materials / user_material_vectors 三件套连同 HTTP 文件端点已按
ADR-0023 决策 1 删除，选材事实源归 VBL：dispatch 时业务层把选中素材集的
READY footage 快照成 TaskVideoRequest 的 footage_map，连同范围 kb_scope
（{owner_id, collection_ids}）一起送达（决策 9）。

每段候选池由此模块构建（决策 5，pool builder）：

1. 用该段的 coarse_query 调 VPC ``POST /index/embed`` 取查询向量；
2. ``POST /index/search`` 限定 namespace=kb_scope.owner_id、
   group_keys=footage_map 的 footage id（即所选素材集的快照——外人的
   footage 组绝不进入检索范围）；
3. 帧命中按 group_key 聚合，每个 footage 取其最高相似度（决策 4 的映射：
   namespace=owner、group_key=footage_id、vector_id=frame_idx）；
4. 映射成 MaterialInfo：url 即该 footage 在 /shared 交换卷上的 cache clip
   路径（决策 2 的路径寻址），时长与场景区间取自 footage_map 快照。

VPC 是纯索引服务（决策 4）：namespace/group_key/vector_id 对它都是不透明
字符串，本模块同样只做 id 映射，不解释、不派生任何用户/会话/计费/素材生命
周期概念。零命中、嵌入不可用、检索失败、快照缺失一律 fail-open 返回空池，
由调用方的 image-gen 回填接管，绝不回退 stock 检索。
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import requests
from loguru import logger

from app.config import config
from app.models.schema import FootageMapEntry, KBScope, MaterialInfo

# 候选 provider 标识（原 user_material）：日志与审计记录的来源标记。
FOOTAGE_PROVIDER = "footage"

# 单段检索的帧命中上限 k：VPC 按帧返回 top-k（同一 footage 多帧各占一格），
# 每个片段至多需要 max(CLIPS_PER_SEGMENT, 窗口数) 个候选，50 帧足以覆盖
# 正常密度的多帧 footage 且不超过 VPC 的 MAX_K=200。
DEFAULT_SEARCH_K = 50

_INDEX_TAG = "footage kb pool"


def _index_settings() -> tuple[str, float]:
    """（base_url, timeout）来自 [index] 配置；base_url 空 = 未部署索引服务。"""
    section = getattr(config, "index", None) or {}
    base_url = str(section.get("base_url", "") or "").strip().rstrip("/")
    try:
        timeout = float(section.get("timeout", 30) or 30)
    except (TypeError, ValueError):
        timeout = 30.0
    return base_url, timeout


def _search_k() -> int:
    """单段检索 k：读 [index] search_k，非法值回落默认并钳在 [1, VPC MAX_K]。"""
    section = getattr(config, "index", None) or {}
    try:
        value = int(section.get("search_k", DEFAULT_SEARCH_K))
    except (TypeError, ValueError):
        return DEFAULT_SEARCH_K
    return max(1, min(value, 200))


class FootageIndexClient(Protocol):
    """VPC 索引客户端协议：生产走 [index] 端点的 HTTP 实现，测试注入 fake。"""

    def embed_query(self, text: str) -> list[float] | None:
        """查询文本 -> 向量；失败返回 None（fail-open）。"""
        ...

    def search_footage(
        self,
        namespace: str,
        vector: Sequence[float],
        group_keys: Sequence[str],
        k: int,
    ) -> list[dict[str, Any]]:
        """namespace + group_keys 限定的原始帧命中；失败返回 []（fail-open）。"""
        ...


def embed_query(text: str) -> list[float] | None:
    """``POST /index/embed`` {text} -> 查询向量；任何失败返回 None（fail-open）。

    服务端缓存优先，命中不调嵌入供应商；向量维度以响应为准（2026-10-21 的
    模型切换后为多模态 TokenHub 端点，对本模块仍是“发文本、得向量”）。
    端点未配置（base_url 空）/ 网络失败 / 非 200 / 响应损坏一律 None，由
    调用方降级。
    """
    query = str(text or "").strip()
    if not query:
        return None
    base_url, timeout = _index_settings()
    if not base_url:
        logger.warning(
            f"{_INDEX_TAG}: [index] base_url is empty — premise search is off, "
            "segments degrade to image-gen backfill"
        )
        return None
    try:
        response = requests.post(
            f"{base_url}/index/embed",
            json={"text": query},
            timeout=(timeout, timeout),
            proxies={},
        )
    except Exception as exc:
        logger.warning(
            f"{_INDEX_TAG}: embed request failed: error={type(exc).__name__}"
        )
        return None
    if response.status_code != 200:
        logger.warning(
            f"{_INDEX_TAG}: embed returned non-200: status={response.status_code}"
        )
        return None
    try:
        vector = [float(v) for v in response.json().get("vector") or []]
    except Exception as exc:
        logger.warning(
            f"{_INDEX_TAG}: embed response malformed: error={type(exc).__name__}"
        )
        return None
    return vector or None


def search_footage(
    namespace: str,
    vector: Sequence[float],
    group_keys: Sequence[str],
    k: int,
) -> list[dict[str, Any]]:
    """``POST /index/search`` -> 原始帧命中 [{group_key, vector_id, score}]。

    namespace 与 group_keys 双双限定：namespace 是 owner 范围，group_keys
    是本次生成可用的 footage id（footage_map 快照）——外人的组即便猜中 id
    也不在检索范围里。失败（端点未配置 / 网络 / 非 200 / 维度不匹配的 400）
    一律返回 []，由调用方降级。
    """
    scope = str(namespace or "").strip()
    keys = [str(key) for key in group_keys if str(key)]
    values = [float(v) for v in vector]
    if not scope or not keys or not values:
        return []
    base_url, timeout = _index_settings()
    if not base_url:
        logger.warning(
            f"{_INDEX_TAG}: [index] base_url is empty — premise search is off, "
            "segments degrade to image-gen backfill"
        )
        return []
    try:
        response = requests.post(
            f"{base_url}/index/search",
            json={"namespace": scope, "vector": values, "k": int(k), "group_keys": keys},
            timeout=(timeout, timeout),
            proxies={},
        )
    except Exception as exc:
        logger.warning(
            f"{_INDEX_TAG}: search request failed: error={type(exc).__name__}"
        )
        return []
    if response.status_code != 200:
        logger.warning(
            f"{_INDEX_TAG}: search returned non-200: status={response.status_code}"
        )
        return []
    try:
        hits = response.json().get("hits") or []
        if not isinstance(hits, list):
            raise TypeError("hits is not a list")
        return [hit for hit in hits if isinstance(hit, dict)]
    except Exception as exc:
        logger.warning(
            f"{_INDEX_TAG}: search response malformed: error={type(exc).__name__}"
        )
        return []


class _HttpIndexClient:
    """[index] 端点的 HTTP 客户端：经模块级函数间接发请求。

    测试 patch 模块级 ``embed_query`` / ``search_footage`` 即生效（函数在
    调用时才经全局名解析），因此接线层测试无需构造 HTTP fake。
    """

    def embed_query(self, text: str) -> list[float] | None:
        return embed_query(text)

    def search_footage(
        self,
        namespace: str,
        vector: Sequence[float],
        group_keys: Sequence[str],
        k: int,
    ) -> list[dict[str, Any]]:
        return search_footage(namespace, vector, group_keys, k)


_DEFAULT_CLIENT = _HttpIndexClient()


def thumb_data_uri(thumb_path: str) -> str:
    """本地缩略图 -> data URI（查重门 / VLM 走查的判定输入）。

    footage_map 的 thumb_path 指向 /shared 交换卷上的 cache 文件，直读即可
    （决策 2：路径寻址，绝不为此走 HTTP）。缺失/读取失败返回空串，走调用方
    既有的空 URI 契约。
    """
    path = str(thumb_path or "").strip()
    if not path:
        return ""
    try:
        with open(path, "rb") as thumb_file:
            payload = thumb_file.read()
    except OSError:
        logger.warning(f"{_INDEX_TAG}: thumb read failed (fail-open): path={path}")
        return ""
    if not payload:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def _best_score_per_footage(hits: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """帧命中 -> footage id -> 该 footage 的最高相似度（同组多帧取 max）。"""
    best: dict[str, float] = {}
    for hit in hits:
        group_key = str(hit.get("group_key") or "")
        if not group_key:
            continue
        try:
            score = float(hit.get("score"))
        except (TypeError, ValueError):
            continue
        current = best.get(group_key)
        if current is None or score > current:
            best[group_key] = score
    return best


def _ranked_footage_ids(hits: Sequence[Mapping[str, Any]]) -> list[tuple[str, float]]:
    """按最高相似度降序的 (footage_id, score)；同分按 id 升序保证确定性。"""
    return sorted(
        _best_score_per_footage(hits).items(), key=lambda pair: (-pair[1], pair[0])
    )


def _footage_item(footage_id: str, entry: FootageMapEntry, score: float) -> MaterialInfo:
    """快照条目 + 检索相似度 -> 漏斗候选 MaterialInfo。

    url 是 /shared cache clip 路径（全链路唯一的候选身份：查重门键、跨段
    已用集合、clip_sources 溯源都认它）；thumbnail_url 留空——BYOM 候选的
    预览图是本地文件，缩略图绝不走 HTTP 供给方路径。
    """
    return MaterialInfo(
        provider=FOOTAGE_PROVIDER,
        url=entry.clip_path,
        duration=int(round(entry.duration_s)),
        source_info={
            "provider": FOOTAGE_PROVIDER,
            "asset_id": footage_id,
            "search_term": "",
            "thumbnail_url": "",
            "thumb_path": entry.thumb_path,
            "score": round(score, 6),
            "rendition": {
                "id": footage_id,
                "footage_id": footage_id,
                "t_start_ms": entry.t_start_ms,
                "t_end_ms": entry.t_end_ms,
                "clip_path": entry.clip_path,
                "thumb_path": entry.thumb_path,
            },
        },
    )


def build_pool(
    kb_scope: KBScope | None,
    footage_map: Mapping[str, FootageMapEntry] | None,
    query_text: str,
    client: FootageIndexClient | None = None,
) -> list[MaterialInfo]:
    """为一个片段构建 BYOM 候选池（ADR-0023 决策 5 的 pool builder）。

    Args:
        kb_scope: 请求的 premise 范围；owner_id 即检索 namespace。None 或
            owner_id 为空 = 请求不携带范围，直接空池。
        footage_map: dispatch 快照（footage_id -> 条目）；它同时是检索的
            group_keys 与候选的字节来源。空快照 = 无素材可用，空池。
        query_text: 该段的 coarse_query（粗排检索用的宽场景描述）。空查询
            不发起任何检索调用。
        client: 索引客户端；None 用 [index] 端点的 HTTP 实现（测试可注入
            fake 或 patch 模块级函数）。

    Returns:
        按“每个 footage 的最高相似度”降序的候选列表；零命中 / 嵌入不可用 /
        检索失败 / 快照缺失一律返回空列表（fail-open，绝不抛异常）。
    """
    query = str(query_text or "").strip()
    if not query:
        logger.warning(f"{_INDEX_TAG}: empty segment query, degrade to backfill")
        return []
    if kb_scope is None:
        logger.warning(f"{_INDEX_TAG}: request carries no kb_scope, degrade to backfill")
        return []
    namespace = str(kb_scope.owner_id or "").strip()
    if not namespace:
        logger.warning(
            f"{_INDEX_TAG}: kb_scope without owner_id, refuse to search any namespace"
        )
        return []
    entries = {str(key): value for key, value in dict(footage_map or {}).items()}
    if not entries:
        logger.warning(
            f"{_INDEX_TAG}: empty footage_map snapshot, degrade to backfill"
        )
        return []

    # group_keys 仅限本次快照的 footage id（即 kb_scope.collection_ids 选中
    # 素材集的 READY 集合，VBL 已在 dispatch 侧做归属过滤）：外人的 footage
    # 组连候选集都进不来。
    group_keys = sorted(entries)
    resolved = client or _DEFAULT_CLIENT

    vector = resolved.embed_query(query)
    if not vector:
        logger.warning(
            f"{_INDEX_TAG}: query embedding unavailable, degrade to backfill"
        )
        return []
    try:
        hits = resolved.search_footage(namespace, vector, group_keys, _search_k())
    except Exception as exc:
        logger.warning(
            f"{_INDEX_TAG}: search failed, degrade to backfill: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return []

    pool: list[MaterialInfo] = []
    for footage_id, score in _ranked_footage_ids(hits):
        entry = entries.get(footage_id)
        if entry is None:
            # 防御：快照里没有的 footage 绝不成为候选（namespace 之外的命中
            # 同理——VPC 已按 namespace 过滤，这里再兜一层）。
            continue
        pool.append(_footage_item(footage_id, entry, score))
    if not pool:
        logger.warning(
            f"{_INDEX_TAG}: zero hits for the segment "
            f"(groups={len(group_keys)}), degrade to backfill"
        )
        return []
    logger.info(
        f"{_INDEX_TAG}: assembled candidates={len(pool)} "
        f"groups={len(group_keys)}, top_score={pool[0].source_info['score']}"
    )
    return pool
