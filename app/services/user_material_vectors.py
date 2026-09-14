"""premise:// 向量的持久化缓存（plan todo 6）：url-keyed、懒读、批量写、fail-soft。

跨任务复用故事：粗排/查重门对共享缓存的普通 `cache[url] = vec` 写入，经本类
缓冲；`flush()`（match_segments 每段粗排后调用）只把 `premise://` 条目批量
upsert 进 todo-2 的 `user_material_vectors` 表。stock http url 仅存于本实例
内存——用户素材才允许落盘（Metis #7：only successful premise:// vectors
persist）。model/dim 与当前 [image_embedding] 配置失配时读 = miss（cosine 对
维度不一致静默 0.0 沉底，绝不能拿旧空间的向量冒充新空间的结果）。

为什么继承 dict 而非裸 MutableMapping：task.py 把同一实例注入
`make_default_gate(vector_cache=...)`，既有接线测试断言该参数 isinstance
dict，且 EmbeddingGate/video_match 的取回链都做过 isinstance(shared, dict)
判定——dict 子类同时满足两处契约，显式 vector_cache 参数（video_match）仍
保留以绕开 gate-off 私有取回路径。代价是 dict 的 C 实现绕过实例方法，
故 `get`/`__getitem__`/`__setitem__`/`__delitem__`/`clear` 全部显式覆盖；
`update`/`pop`/`setdefault` 等 C 路径不参与持久化 buffer（漏斗只用
get/set，已按 grep 核实）。

DB 访问全部经 user_materials 的单例连接 + RLock 纪律；evict_material 复用
todo-2 的转义 LIKE 前缀（身份里的 `_` 是通配符，必须转义），且只删向量。
"""

from __future__ import annotations

import logging
import struct
from typing import Any

from app.services import image_embedding, user_materials

logger = logging.getLogger(__name__)

_PREMISE_PREFIX = "premise://"


class PersistentVectorCache(dict[str, list[float]]):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self._dim = dim
        self._pending: dict[str, list[float]] = {}
        self._ready = False

    @property
    def dim(self) -> int | None:
        return self._dim

    @staticmethod
    def _current_model() -> str:
        return image_embedding._gate_setting("model", image_embedding.DEFAULT_EMBEDDING_MODEL)

    def _ensure(self) -> None:
        if not self._ready:
            user_materials.ensure_schema()
            self._ready = True

    @staticmethod
    def _is_premise(url: object) -> bool:
        return isinstance(url, str) and url.startswith(_PREMISE_PREFIX)

    # dict.get 的基类重载（default: 任意 T -> 返回 V|T）与懒加载语义在类型
    # 层不可兼容：本实现只会返回 V|None。Any 是唯一同时满足覆写检查与
    # 调用方（coarse_rank/gate 均以 `is None` 判 miss）的标注。
    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, url: str) -> list[float]:
        try:
            return super().__getitem__(url)
        except KeyError:
            pass
        if not self._is_premise(url):
            raise KeyError(url)
        row = self._read_row(url)
        if row is None:
            raise KeyError(url)
        model, dim, blob = row
        if model != self._current_model():
            raise KeyError(url)
        if self._dim is not None and dim != self._dim:
            raise KeyError(url)
        try:
            vec = [float(v) for v in struct.unpack(f"<{dim}f", blob)]
        except struct.error:
            logger.warning("user_materials vector row corrupt, treating as miss: url=%s", url)
            raise KeyError(url) from None
        super().__setitem__(url, vec)
        return vec

    @staticmethod
    def _read_row(url: str) -> tuple[str, int, bytes] | None:
        try:
            with user_materials._db_lock:
                conn = user_materials._get_conn()
                user_materials.ensure_schema()
                row = conn.execute(
                    "SELECT model, dim, embedding FROM user_material_vectors WHERE url=?",
                    (url,),
                ).fetchone()
        except Exception:
            logger.warning("user_materials vector read failed (fail-soft miss): url=%s", url, exc_info=True)
            return None
        if row is None:
            return None
        return str(row[0]), int(row[1]), bytes(row[2])

    def __setitem__(self, url: str, vec: list[float]) -> None:
        values = [float(v) for v in vec]
        super().__setitem__(url, values)
        self._dim = len(values)
        if self._is_premise(url):
            self._pending[url] = values

    def __delitem__(self, url: str) -> None:
        super().__delitem__(url)
        self._pending.pop(url, None)

    def clear(self) -> None:
        super().clear()
        self._pending.clear()

    def flush(self) -> None:
        """批量 upsert 缓冲的 premise 向量；任何 DB 异常只记 WARNING（fail-soft）。

        失败时保留 buffer：后续段/后续调用可重试；upsert 幂等（ON CONFLICT 覆写）。
        """
        if not self._pending:
            return
        try:
            self._ensure()
            model = self._current_model()
            now = user_materials._now_iso()
            rows = [
                (url, struct.pack(f"<{len(vec)}f", *vec), model, len(vec), now)
                for url, vec in self._pending.items()
            ]
            with user_materials._write_txn() as conn:
                conn.executemany(
                    "INSERT INTO user_material_vectors (url, embedding, model, dim, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(url) DO UPDATE SET embedding=excluded.embedding,"
                    " model=excluded.model, dim=excluded.dim, updated_at=excluded.updated_at",
                    rows,
                )
            self._pending.clear()
        except Exception:
            logger.warning(
                "user_materials vector cache flush failed (fail-soft, task continues): %d vectors kept in buffer",
                len(self._pending),
                exc_info=True,
            )

    def evict_material(self, owner: str, material_id: str) -> None:
        """删除该素材全部 premise 向量（转义前缀），只碰向量表。

        todo-2 的 complete()/delete() 已在各自事务里做同前缀 purge——本方法供
        仅向量需要失效（内容未变、embedding 模型变更等）的将来调用方使用。
        """
        prefix = user_materials._vector_prefix(owner, material_id)
        try:
            self._ensure()
            with user_materials._write_txn() as conn:
                conn.execute("DELETE FROM user_material_vectors WHERE url LIKE ? ESCAPE '\\'", (prefix,))
        except Exception:
            logger.warning("user_materials vector evict failed (fail-soft): %s", prefix, exc_info=True)
        raw_prefix = f"{_PREMISE_PREFIX}{owner}/{material_id}/"
        for url in [k for k in super().__iter__() if isinstance(k, str) and k.startswith(raw_prefix)]:
            del self[url]
