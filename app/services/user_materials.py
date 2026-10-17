"""用户素材（premise）本地登记册：stdlib sqlite3 台账 + 本地文件暂存 + 惰性 TTL 清理。

Go 侧（VBL）是素材事实源，本模块只是 dispatch 时的本地镜像缓存（plan 决策 7）：
派生 footage 推送到 `.partial/{rev}/`，经 `complete` 原子转正；`premise://` URL
只是逻辑身份，全部字节走本地路径，绝不通过 HTTP 自服务（Metis #10）。
线程安全：MPT 任务跑在线程池，模块级单例连接（`check_same_thread=False`）配一把
可重入锁守护所有 DB 读写（Metis #6，plan todo 2）。

allow: SIZE_OK — plan todo 2 契约钦点单一模块 `app/services/user_materials.py`
（todo 3/6/7 按此名导入），登记 + 路径 + GC + resolver 是同一台账的不可拆面。
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Mapping, Sequence

from app.config import config as app_config
from app.utils import file_security, utils

# 项目日志惯例是 loguru；本模块按验收要求保持仅 stdlib + app.* 导入，改用 stdlib
# logger。WARNING 经 root lastResort 走 stderr，清理/IO 失败不会静默。
logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
_KIND_EXT = {"clip": ".mp4", "thumb": ".jpg"}
_SWEEP_INTERVAL_SECONDS = 3600.0
_PARTIAL_MAX_AGE_SECONDS = 24 * 3600.0
_DEFAULT_MAX_AGE_DAYS = 15
_DEFAULT_MAX_CANDIDATES = 500

_db_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_last_sweep_monotonic: float | None = None

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS user_materials ("
    " owner TEXT NOT NULL, material_id TEXT NOT NULL, rev INTEGER NOT NULL,"
    " status TEXT NOT NULL CHECK (status IN ('partial','ready')),"
    " footage_count INTEGER NOT NULL DEFAULT 0, total_bytes INTEGER NOT NULL DEFAULT 0,"
    " pushed_at TEXT NOT NULL, PRIMARY KEY (owner, material_id));"
    "CREATE TABLE IF NOT EXISTS user_material_footages ("
    " owner TEXT NOT NULL, material_id TEXT NOT NULL, idx INTEGER NOT NULL,"
    " t_start REAL NOT NULL, t_end REAL NOT NULL, duration REAL NOT NULL,"
    " PRIMARY KEY (owner, material_id, idx));"
    "CREATE TABLE IF NOT EXISTS user_material_vectors ("
    " url TEXT PRIMARY KEY, embedding BLOB NOT NULL, model TEXT NOT NULL,"
    " dim INTEGER NOT NULL, updated_at TEXT NOT NULL);"
)


def _storage_root(create: bool = False) -> str:
    return utils.storage_dir("user_materials", create=create)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    # 统一“秒精度 + UTC”格式，pushed_at 按字符串比较即按时间比较（TTL 用）。
    return _utcnow().isoformat(timespec="seconds")


def _conf_int(key: str, default: int) -> int:
    section = getattr(app_config, "user_materials", None) or {}
    raw = section.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("user_materials.%s=%r invalid, fallback %s", key, raw, default)
        return default


def _validate_part(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _validate_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _material_dir(owner: str, material_id: str) -> str:
    return os.path.join(
        _storage_root(), _validate_part(owner, "owner"), _validate_part(material_id, "material_id")
    )


def _rev_dir(owner: str, material_id: str, rev: int) -> str:
    return os.path.join(_material_dir(owner, material_id), str(_validate_int(rev, "rev")))


def _partial_rev_dir(owner: str, material_id: str, rev: int) -> str:
    return os.path.join(_material_dir(owner, material_id), ".partial", str(_validate_int(rev, "rev")))


def _vector_prefix(owner: str, material_id: str) -> str:
    # 身份字符含 `_`，而 `_`/`%` 是 LIKE 通配符；不转义会跨同名前缀误删向量。
    head = f"premise://{_validate_part(owner, 'owner')}/{_validate_part(material_id, 'material_id')}/"
    return head.replace("%", r"\%").replace("_", r"\_") + "%"


def _db_file() -> str:
    return os.path.join(_storage_root(create=True), "user_materials.db")


def _get_conn() -> sqlite3.Connection:
    global _conn
    with _db_lock:
        if _conn is None:
            _conn = sqlite3.connect(_db_file(), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.commit()
    return _conn


@contextmanager
def _write_txn() -> Iterator[sqlite3.Connection]:
    conn = _get_conn()
    with _db_lock:
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def ensure_schema() -> None:
    conn = _get_conn()
    with _db_lock:
        conn.executescript(_SCHEMA)
    maybe_sweep()


def maybe_sweep() -> None:
    """惰性 TTL 清理：每个公开入口都调用；进程内距上次 >=1h 才真正执行（Metis #9）。"""
    global _last_sweep_monotonic
    with _db_lock:
        now = time.monotonic()
        if _last_sweep_monotonic is not None and now - _last_sweep_monotonic < _SWEEP_INTERVAL_SECONDS:
            return
        _last_sweep_monotonic = now  # 先落时间戳，防止嵌套调用递归重入。
    try:
        _sweep_expired_materials()
        _sweep_stale_partials()
    except Exception:  # 清理是尽力而为的后台动作，绝不拖垮业务调用。
        logger.warning("user_materials sweep failed", exc_info=True)


def _sweep_expired_materials() -> None:
    days = _conf_int("max_age_days", _DEFAULT_MAX_AGE_DAYS)
    cutoff = (_utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
    with _db_lock:
        expired = _get_conn().execute(
            "SELECT owner, material_id FROM user_materials WHERE status='ready' AND pushed_at < ?",
            (cutoff,),
        ).fetchall()
    for row in expired:
        _purge(row["owner"], row["material_id"])


def _scandir_dirs(path: str) -> list[os.DirEntry[str]]:
    try:
        return [e for e in os.scandir(path) if e.is_dir()]
    except OSError:
        return []


def _sweep_stale_partials() -> None:
    root = _storage_root()
    if not os.path.isdir(root):
        return
    deadline = time.time() - _PARTIAL_MAX_AGE_SECONDS
    for owner_dir in _scandir_dirs(root):
        for material_dir in _scandir_dirs(owner_dir.path):
            for rev_dir in _scandir_dirs(os.path.join(material_dir.path, ".partial")):
                try:
                    if rev_dir.stat().st_mtime < deadline:
                        shutil.rmtree(rev_dir.path)
                except OSError:
                    logger.warning("failed to purge stale partial dir %s", rev_dir.path, exc_info=True)


def _purge(owner: str, material_id: str) -> None:
    with _write_txn() as conn:
        conn.execute(
            "DELETE FROM user_material_footages WHERE owner=? AND material_id=?", (owner, material_id)
        )
        conn.execute(
            "DELETE FROM user_material_vectors WHERE url LIKE ? ESCAPE '\\'",
            (_vector_prefix(owner, material_id),),
        )
        conn.execute("DELETE FROM user_materials WHERE owner=? AND material_id=?", (owner, material_id))
    shutil.rmtree(_material_dir(owner, material_id), ignore_errors=True)


def _ready_material(owner: str, material_id: str) -> sqlite3.Row | None:
    with _db_lock:
        return _get_conn().execute(
            "SELECT * FROM user_materials WHERE owner=? AND material_id=? AND status='ready'",
            (_validate_part(owner, "owner"), _validate_part(material_id, "material_id")),
        ).fetchone()


def begin_push(owner: str, material_id: str, rev: int) -> str:
    """建 `.partial/{rev}/` 并把登记行占位为 status='partial'。返回暂存目录。"""
    ensure_schema()
    partial = _partial_rev_dir(owner, material_id, rev)
    os.makedirs(partial, exist_ok=True)
    with _write_txn() as conn:
        conn.execute(
            "INSERT INTO user_materials (owner, material_id, rev, status, footage_count, total_bytes, pushed_at)"
            " VALUES (?, ?, ?, 'partial', 0, 0, ?)"
            " ON CONFLICT(owner, material_id) DO UPDATE SET rev=excluded.rev,"
            " status='partial', pushed_at=excluded.pushed_at",
            (owner, material_id, _validate_int(rev, "rev"), _now_iso()),
        )
    return partial


def put_file(owner: str, material_id: str, rev: int, idx: int, kind: str) -> str:
    """返回单文件在 `.partial/{rev}/{idx}.mp4|jpg` 的落盘路径；调用方负责写字节。"""
    ensure_schema()
    if kind not in _KIND_EXT:
        raise ValueError(f"invalid kind: {kind!r}, expected one of {sorted(_KIND_EXT)}")
    target = os.path.join(_partial_rev_dir(owner, material_id, rev), f"{_validate_int(idx, 'idx')}{_KIND_EXT[kind]}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return file_security.resolve_path_within_directory(_storage_root(), target, require_file=False)


def complete(
    owner: str, material_id: str, rev: int, manifest: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """all-or-nothing 转正：逐 idx 验齐 clip+jpg → os.replace 发布 → 单事务写台账 + 清旧向量。

    任何文件缺失立即 ValueError，台账保持 'partial'；旧 rev 目录与旧向量随之清除。
    """
    ensure_schema()
    rev = _validate_int(rev, "rev")
    entries: list[tuple[int, float, float, float]] = []
    seen: set[int] = set()
    for item in manifest:
        try:
            idx = _validate_int(int(item["idx"]), "idx")
            timing = (float(item["t_start"]), float(item["t_end"]), float(item["duration"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid manifest entry: {item!r}") from exc
        if idx in seen:
            raise ValueError(f"duplicate idx {idx} in manifest")
        seen.add(idx)
        entries.append((idx, *timing))
    if not entries:
        raise ValueError("manifest must not be empty")

    partial = _partial_rev_dir(owner, material_id, rev)
    published = _rev_dir(owner, material_id, rev)
    stage = partial if os.path.isdir(partial) else published
    if not os.path.isdir(stage):
        raise ValueError(f"nothing staged under rev {rev}: {partial!r} missing")
    total_bytes = 0
    for idx, *_rest in entries:
        for kind, ext in _KIND_EXT.items():
            path = os.path.join(stage, f"{idx}{ext}")
            if not os.path.isfile(path):
                raise ValueError(f"missing {kind} for idx {idx}: {path!r}")
            total_bytes += os.path.getsize(path)
    if stage is partial:
        if os.path.isdir(published):
            shutil.rmtree(published)
        os.replace(partial, published)

    with _write_txn() as conn:
        old = conn.execute(
            "SELECT rev FROM user_materials WHERE owner=? AND material_id=?", (owner, material_id)
        ).fetchone()
        conn.execute(
            "INSERT INTO user_materials (owner, material_id, rev, status, footage_count, total_bytes, pushed_at)"
            " VALUES (?, ?, ?, 'ready', ?, ?, ?)"
            " ON CONFLICT(owner, material_id) DO UPDATE SET rev=excluded.rev, status='ready',"
            " footage_count=excluded.footage_count, total_bytes=excluded.total_bytes, pushed_at=excluded.pushed_at",
            (owner, material_id, rev, len(entries), total_bytes, _now_iso()),
        )
        conn.execute("DELETE FROM user_material_footages WHERE owner=? AND material_id=?", (owner, material_id))
        conn.executemany(
            "INSERT INTO user_material_footages (owner, material_id, idx, t_start, t_end, duration)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(owner, material_id, idx, t_start, t_end, duration) for idx, t_start, t_end, duration in entries],
        )
        conn.execute(
            "DELETE FROM user_material_vectors WHERE url LIKE ? ESCAPE '\\'", (_vector_prefix(owner, material_id),)
        )

    if old is not None and old["rev"] != rev:
        base = _material_dir(owner, material_id)
        shutil.rmtree(os.path.join(base, str(old["rev"])), ignore_errors=True)
        shutil.rmtree(os.path.join(base, ".partial", str(old["rev"])), ignore_errors=True)
    return {"material_id": material_id, "rev": rev, "footage_count": len(entries), "total_bytes": total_bytes}


def delete(owner: str, material_id: str) -> None:
    """purge 三表行（含 url LIKE premise://owner/material_id/% 的向量）+ 本地目录；幂等。"""
    ensure_schema()
    _purge(owner, material_id)


def list_ready(owner: str) -> list[dict[str, Any]]:
    ensure_schema()
    with _db_lock:
        rows = _get_conn().execute(
            "SELECT material_id, rev, footage_count, total_bytes FROM user_materials"
            " WHERE owner=? AND status='ready' ORDER BY material_id",
            (_validate_part(owner, "owner"),),
        ).fetchall()
    return [dict(row) for row in rows]


def pool(owner: str, cap: int | None = None) -> list[dict[str, Any]]:
    """owner 全部 ready footage（含绝对本地路径与 premise url），cap 限量（默认 [user_materials].max_candidates）。"""
    ensure_schema()
    owner = _validate_part(owner, "owner")
    limit = _validate_int(cap, "cap", minimum=1) if cap is not None else _conf_int("max_candidates", _DEFAULT_MAX_CANDIDATES)
    with _db_lock:
        rows = _get_conn().execute(
            "SELECT m.rev, f.material_id, f.idx, f.t_start, f.t_end, f.duration"
            " FROM user_material_footages f JOIN user_materials m"
            " ON m.owner=f.owner AND m.material_id=f.material_id"
            " WHERE f.owner=? AND m.status='ready' ORDER BY f.material_id, f.idx LIMIT ?",
            (owner, limit),
        ).fetchall()
    items = []
    for row in rows:
        rev_dir = _rev_dir(owner, row["material_id"], row["rev"])
        items.append(
            {
                "material_id": row["material_id"],
                "rev": row["rev"],
                "idx": row["idx"],
                "t_start": row["t_start"],
                "t_end": row["t_end"],
                "duration": row["duration"],
                "url": url(owner, row["material_id"], row["idx"]),
                "clip_path": os.path.join(rev_dir, f"{row['idx']}.mp4"),
                "thumb_path": os.path.join(rev_dir, f"{row['idx']}.jpg"),
            }
        )
    return items


def url(owner: str, material_id: str, idx: int) -> str:
    return f"premise://{_validate_part(owner, 'owner')}/{_validate_part(material_id, 'material_id')}/{_validate_int(idx, 'idx')}"


def resolve_premise_url(premise_url: str) -> tuple[str, str, dict[str, Any]] | None:
    """premise://url → (clip 本地路径, thumb 本地路径, 台账行)；未知/未 ready/文件缺失返回 None。"""
    ensure_schema()
    prefix = "premise://"
    if not isinstance(premise_url, str) or not premise_url.startswith(prefix):
        return None
    parts = premise_url[len(prefix) :].split("/")
    if len(parts) != 3:
        return None
    try:
        owner = _validate_part(parts[0], "owner")
        material_id = _validate_part(parts[1], "material_id")
        idx = _validate_int(int(parts[2]), "idx")
    except ValueError:
        return None
    with _db_lock:
        row = _get_conn().execute(
            "SELECT m.rev, f.t_start, f.t_end, f.duration FROM user_materials m"
            " JOIN user_material_footages f ON f.owner=m.owner AND f.material_id=m.material_id AND f.idx=?"
            " WHERE m.owner=? AND m.material_id=? AND m.status='ready'",
            (idx, owner, material_id),
        ).fetchone()
    if row is None:
        return None
    try:
        rev_dir = _rev_dir(owner, material_id, row["rev"])
        clip = file_security.resolve_path_within_directory(_storage_root(), os.path.join(rev_dir, f"{idx}.mp4"))
        thumb = file_security.resolve_path_within_directory(_storage_root(), os.path.join(rev_dir, f"{idx}.jpg"))
    except ValueError:
        return None
    return clip, thumb, {
        "owner": owner,
        "material_id": material_id,
        "idx": idx,
        "rev": row["rev"],
        "t_start": row["t_start"],
        "t_end": row["t_end"],
        "duration": row["duration"],
    }


def read_thumb_b64(owner: str, material_id: str, idx: int) -> str:
    """从本地读首帧 jpg，返回 data URI（候选缩略图直接内嵌，绝不走 HTTP）。"""
    ensure_schema()
    row = _ready_material(owner, material_id)
    if row is None:
        raise ValueError(f"material not ready: {owner}/{material_id}")
    thumb = file_security.resolve_path_within_directory(
        _storage_root(), os.path.join(_rev_dir(owner, material_id, row["rev"]), f"{_validate_int(idx, 'idx')}.jpg")
    )
    with open(thumb, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
