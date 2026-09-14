"""user_materials 登记册测试：plan todo 2 的三个 QA 场景 + id 校验、WAL、向量 blob 存储。"""

from __future__ import annotations

import base64
import os
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.services import user_materials as um

OWNER = "owner-1"
MID = "mat_2026-09-14-aaaa"
CLIP_BYTES = b"C" * 20
THUMB_BYTES = b"T" * 10


def _direct_conn(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(os.path.join(str(tmp_path), "user_materials.db"))
    conn.row_factory = sqlite3.Row
    return conn


def _insert_vector(tmp_path, url: str, vec: list[float], model: str = "tongyi-embedding-vision-flash") -> None:
    conn = _direct_conn(tmp_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO user_material_vectors (url, embedding, model, dim, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (url, struct.pack(f"<{len(vec)}f", *vec), model, len(vec), um._now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def _read_vectors(tmp_path) -> list[sqlite3.Row]:
    conn = _direct_conn(tmp_path)
    try:
        return conn.execute("SELECT url, embedding, model, dim FROM user_material_vectors ORDER BY url").fetchall()
    finally:
        conn.close()


def _stage(owner: str, material_id: str, rev: int, count: int = 2) -> None:
    um.begin_push(owner, material_id, rev)
    for idx in range(count):
        with open(um.put_file(owner, material_id, rev, idx, "clip"), "wb") as f:
            f.write(CLIP_BYTES)
        with open(um.put_file(owner, material_id, rev, idx, "thumb"), "wb") as f:
            f.write(THUMB_BYTES)


def _manifest(count: int = 2) -> list[dict[str, float]]:
    return [
        {"idx": idx, "t_start": idx * 5.0, "t_end": idx * 5.0 + 5.0, "duration": 5.0}
        for idx in range(count)
    ]


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "_storage_root", lambda create=False: str(tmp_path))
    monkeypatch.setattr(um, "_conn", None)
    monkeypatch.setattr(um, "_last_sweep_monotonic", 0.0)
    yield tmp_path
    if um._conn is not None:
        um._conn.close()


def test_happy_lifecycle_put_complete_list_pool_resolve_delete_purges_vectors(registry):
    """QA 场景 1（happy）: put -> complete -> list_ready -> pool -> resolve -> delete，向量随之清空。"""
    _stage(OWNER, MID, 3)
    _insert_vector(registry, um.url(OWNER, MID, 0), [0.5, -0.25, 1.0])  # complete 即 purge 本素材旧向量
    _insert_vector(registry, um.url("other", "mat-x", 0), [0.1, 0.2])

    result = um.complete(OWNER, MID, 3, _manifest())
    assert result == {
        "material_id": MID,
        "rev": 3,
        "footage_count": 2,
        "total_bytes": 2 * (len(CLIP_BYTES) + len(THUMB_BYTES)),
    }

    assert um.list_ready(OWNER) == [
        {"material_id": MID, "rev": 3, "footage_count": 2, "total_bytes": result["total_bytes"]}
    ]
    assert um.list_ready("other") == []

    entries = um.pool(OWNER, cap=10)
    assert [e["idx"] for e in entries] == [0, 1]
    assert entries[0]["url"] == f"premise://{OWNER}/{MID}/0"
    assert all(os.path.isfile(e["clip_path"]) and os.path.isfile(e["thumb_path"]) for e in entries)
    rev_dir = os.path.join(str(registry), OWNER, MID, "3")
    assert os.path.isdir(rev_dir) and not os.path.isdir(os.path.join(str(registry), OWNER, MID, ".partial", "3"))

    resolved = um.resolve_premise_url(um.url(OWNER, MID, 1))
    assert resolved is not None
    clip, thumb, row = resolved
    assert clip == os.path.realpath(os.path.join(rev_dir, "1.mp4"))
    assert thumb == os.path.realpath(os.path.join(rev_dir, "1.jpg"))
    assert row["t_start"] == 5.0 and row["rev"] == 3
    assert um.resolve_premise_url("premise://no/such/0") is None
    assert um.resolve_premise_url("https://example.com/x") is None
    assert um.resolve_premise_url("nonsense") is None

    data_uri = um.read_thumb_b64(OWNER, MID, 0)
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1]) == THUMB_BYTES

    um.delete(OWNER, MID)
    assert um.list_ready(OWNER) == []
    assert um.pool(OWNER, cap=10) == []
    assert um.resolve_premise_url(um.url(OWNER, MID, 0)) is None
    with pytest.raises(ValueError):
        um.read_thumb_b64(OWNER, MID, 0)
    assert not os.path.exists(os.path.join(str(registry), OWNER, MID))
    assert [r["url"] for r in _read_vectors(registry)] == [um.url("other", "mat-x", 0)]


def test_complete_missing_idx_keeps_partial_then_delete_clears(registry):
    """QA 场景 2（failure）: 缺 idx 文件 -> ValueError，行保持 'partial'，随后 delete 清空。"""
    um.begin_push(OWNER, MID, 1)
    with open(um.put_file(OWNER, MID, 1, 0, "clip"), "wb") as f:
        f.write(CLIP_BYTES)
    with open(um.put_file(OWNER, MID, 1, 0, "thumb"), "wb") as f:
        f.write(THUMB_BYTES)

    with pytest.raises(ValueError, match="idx 1"):
        um.complete(OWNER, MID, 1, _manifest(count=2))

    with _direct_conn(registry) as conn:
        status = conn.execute(
            "SELECT status FROM user_materials WHERE owner=? AND material_id=?", (OWNER, MID)
        ).fetchone()[0]
    assert status == "partial"
    assert um.list_ready(OWNER) == []
    assert os.path.isdir(os.path.join(str(registry), OWNER, MID, ".partial", "1"))
    assert not os.path.isdir(os.path.join(str(registry), OWNER, MID, "1"))

    um.delete(OWNER, MID)
    with _direct_conn(registry) as conn:
        assert conn.execute("SELECT COUNT(*) FROM user_materials").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM user_material_footages").fetchone()[0] == 0
    assert not os.path.exists(os.path.join(str(registry), OWNER, MID))


def test_sweep_purges_expired_ready_materials_and_stale_partials(registry, monkeypatch):
    """QA 场景 3（GC）: 早于 max_age_days(15) 的 ready 行 + >24h 的 .partial 目录被 maybe_sweep 清掉。"""
    _stage(OWNER, MID, 0)
    um.complete(OWNER, MID, 0, _manifest())
    _insert_vector(registry, um.url(OWNER, MID, 1), [0.5])

    fake_now = datetime.now(timezone.utc) + timedelta(days=16)
    monkeypatch.setattr(um, "_utcnow", lambda: fake_now)
    _stage("other", "mat-old", 7)
    stale_partial = os.path.join(str(registry), "other", "mat-old", ".partial", "7")
    old_ts = datetime.now(timezone.utc).timestamp() - 25 * 3600  # mtime 判定走真实墙钟
    os.utime(stale_partial, (old_ts, old_ts))
    fresh_partial = os.path.join(str(registry), "other", "mat-fresh", ".partial", "7")
    os.makedirs(fresh_partial)

    monkeypatch.setattr(um, "_last_sweep_monotonic", None)
    um.maybe_sweep()

    assert um.list_ready(OWNER) == []
    assert not os.path.exists(os.path.join(str(registry), OWNER, MID))
    assert _read_vectors(registry) == []
    assert not os.path.isdir(stale_partial)
    assert os.path.isdir(fresh_partial)
    with _direct_conn(registry) as conn:
        assert conn.execute("SELECT status FROM user_materials WHERE material_id='mat-old'").fetchone()[0] == "partial"


def test_sweep_memo_skips_repeat_within_one_hour(registry, monkeypatch):
    """1h 进程内 memo：连续多次公开调用只触发一次扫描。"""
    swept: list[int] = []
    monkeypatch.setattr(um, "_sweep_expired_materials", lambda: swept.append(1))
    monkeypatch.setattr(um, "_sweep_stale_partials", lambda: None)
    um.begin_push(OWNER, MID, 0)
    for _ in range(3):
        um.list_ready(OWNER)
    assert swept == [1]


def test_id_kind_and_manifest_validation(registry):
    bad_values: list[Any] = ["", "a b", "bad/../..", "x" * 65, "ow!ner", 123]
    for bad in bad_values:
        with pytest.raises(ValueError):
            um.url(bad, MID, 0)
        with pytest.raises(ValueError):
            um.url(OWNER, bad, 0)
    with pytest.raises(ValueError):
        um.put_file(OWNER, MID, 0, -1, "clip")
    with pytest.raises(ValueError):
        um.put_file(OWNER, MID, 0, 0, "exe")
    with pytest.raises(ValueError):
        um.begin_push(OWNER, MID, -5)
    with pytest.raises(ValueError):
        um.complete(OWNER, MID, 0, [])
    with pytest.raises(ValueError):
        um.complete(
            OWNER,
            MID,
            0,
            [
                {"idx": 0, "t_start": 0, "t_end": 5, "duration": 5},
                {"idx": 0, "t_start": 1, "t_end": 4, "duration": 3},
            ],
        )
    with pytest.raises(ValueError):
        um.pool(OWNER, cap=0)


def test_wal_pragma_and_vector_float32_blob_roundtrip(registry):
    """WAL 生效 + 向量以 little-endian float32 struct.pack 存储；不同 dim 行可共存（失配=miss 的存储前提）。"""
    um.ensure_schema()
    with _direct_conn(registry) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    vec3 = [0.25, -1.5, 3.0]
    vec4 = [1.0, 2.0, 4.0, 8.0]
    _insert_vector(registry, um.url(OWNER, MID, 0), vec3)
    _insert_vector(registry, um.url(OWNER, MID, 1), vec4, model="other-model")
    rows = {r["url"]: r for r in _read_vectors(registry)}
    r0 = rows[um.url(OWNER, MID, 0)]
    assert r0["dim"] == 3 and r0["model"] == "tongyi-embedding-vision-flash"
    assert struct.unpack(f"<{r0['dim']}f", r0["embedding"]) == pytest.approx(vec3)
    assert struct.pack("<f", 1.0) == b"\x00\x00\x80?"  # little-endian 编码确认
    r1 = rows[um.url(OWNER, MID, 1)]
    assert r1["dim"] == 4 and struct.unpack("<4f", r1["embedding"]) == pytest.approx(vec4)


def test_config_defaults_and_pool_cap(registry, monkeypatch):
    from app.config import config as config_module

    monkeypatch.setattr(config_module, "user_materials", {}, raising=False)
    assert um._conf_int("max_age_days", um._DEFAULT_MAX_AGE_DAYS) == 15
    assert um._conf_int("max_candidates", um._DEFAULT_MAX_CANDIDATES) == 500
    monkeypatch.setattr(config_module, "user_materials", {"max_candidates": "bad"}, raising=False)
    assert um._conf_int("max_candidates", 500) == 500

    _stage(OWNER, MID, 0)
    um.complete(OWNER, MID, 0, _manifest())
    monkeypatch.setattr(config_module, "user_materials", {"max_age_days": 15, "max_candidates": 1}, raising=False)
    assert len(um.pool(OWNER)) == 1
