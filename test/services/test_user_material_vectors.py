"""PersistentVectorCache 测试（plan todo 6）：跨任务 premise:// 向量复用。

TDD red-first：全部行为按 plan 契约断言——run1 计算 run2 复用、model/dim
失配 = miss + 覆写、stock url 永不落库、fail-soft flush、gate-off 漏斗仍持久化、
evict_material 只清向量且尊重 `_` LIKE 转义。嵌入用桩计数，DB 走 tmp_path。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
from typing import Any
from unittest.mock import patch

import pytest

from app.config import config
from app.models.schema import MaterialInfo
from app.services import image_embedding, user_materials as um, video_match
from app.services.user_material_vectors import PersistentVectorCache

PREMISE_URL = "premise://owner-1/mat-2026/0"
OTHER_URL = "premise://owner-1/mat-2026/1"
SIBLING_URL = "premise://fooxbar/mat-2026/0"
STOCK_URL = "https://stock.example/v/1.mp4"


class _StubEmbedImage:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = dict(vectors)
        self.calls: list[str] = []

    def __call__(self, data_uri, model, api_key, base_url=None, timeout=30.0):
        self.calls.append(data_uri)
        return self.vectors.get(data_uri)


def _db_rows(tmp_path) -> list[dict[str, Any]]:
    db = os.path.join(str(tmp_path), "user_materials.db")
    if not os.path.exists(db):
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT url, model, dim, embedding FROM user_material_vectors ORDER BY url")]
    finally:
        conn.close()


def _seed_row(tmp_path, url: str, vec: list[float], model: str) -> None:
    conn = sqlite3.connect(os.path.join(str(tmp_path), "user_materials.db"))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_material_vectors ("
            " url TEXT PRIMARY KEY, embedding BLOB NOT NULL, model TEXT NOT NULL,"
            " dim INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO user_material_vectors (url, embedding, model, dim, updated_at)"
            " VALUES (?, ?, ?, ?, '2026-09-14T00:00:00+00:00')",
            (url, struct.pack(f"<{len(vec)}f", *vec), model, len(vec)),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "_storage_root", lambda create=False: str(tmp_path))
    monkeypatch.setattr(um, "_conn", None)
    monkeypatch.setattr(um, "_last_sweep_monotonic", float("inf"))
    monkeypatch.setattr(config, "image_embedding", {"model": "M1", "api_key": "k"}, raising=False)
    yield tmp_path
    if um._conn is not None:
        um._conn.close()


def _rank(pool, cache, vectors):
    stub = _StubEmbedImage(vectors)
    with (
        patch.object(image_embedding, "embed_text", return_value=[1.0, 0.0]),
        patch.object(image_embedding, "embed_image", stub),
    ):
        video_match.coarse_rank(pool, "a scene", cache, None)
    return stub


def _pool():
    return [
        {"asset_id": "a0", "url": PREMISE_URL, "data_uri": "thumb-0"},
        {"asset_id": "a1", "url": STOCK_URL, "data_uri": "thumb-1"},
    ]


def test_run1_embeds_and_persists_run2_reuses(registry):
    """AC 核心：run1 计算 2 次嵌入并 flush 落库；run2 新实例同库 = 0 次嵌入。"""
    vectors = {"thumb-0": [0.9, 0.1], "thumb-1": [0.8, 0.2]}
    cache1 = PersistentVectorCache()
    stub1 = _rank(_pool(), cache1, vectors)
    assert stub1.calls == ["thumb-0", "thumb-1"]
    cache1.flush()
    rows = {r["url"]: r for r in _db_rows(registry)}
    assert list(rows) == [PREMISE_URL], "stock url 绝不允许进 DB"
    assert rows[PREMISE_URL]["model"] == "M1" and rows[PREMISE_URL]["dim"] == 2
    assert struct.unpack("<2f", bytes(rows[PREMISE_URL]["embedding"])) == pytest.approx((0.9, 0.1))

    stub2 = _rank(_pool(), PersistentVectorCache(), vectors)
    assert stub2.calls == ["thumb-1"], "premise 命中持久化向量（0 次重嵌），stock 仅内存外仍需计算"


def test_model_change_is_miss_and_row_replaced(registry, monkeypatch):
    cache = PersistentVectorCache()
    cache[PREMISE_URL] = [1.0, 2.0]
    cache.flush()
    assert _db_rows(registry)[0]["model"] == "M1"

    monkeypatch.setattr(config, "image_embedding", {"model": "M2", "api_key": "k"}, raising=False)
    fresh = PersistentVectorCache()
    assert fresh.get(PREMISE_URL) is None, "model 失配必须 miss"
    stub = _rank([{"asset_id": "a0", "url": PREMISE_URL, "data_uri": "thumb-0"}], fresh, {"thumb-0": [0.0, 1.0, 2.0]})
    assert stub.calls == ["thumb-0"]
    fresh.flush()
    row = _db_rows(registry)[0]
    assert row["model"] == "M2" and row["dim"] == 3
    assert struct.unpack("<3f", bytes(row["embedding"])) == pytest.approx((0.0, 1.0, 2.0))


def test_known_dim_mismatch_is_miss_then_replaced(registry):
    _seed_row(registry, PREMISE_URL, [1.0, 2.0], "M1")
    cache = PersistentVectorCache(dim=3)
    assert cache.get(PREMISE_URL) is None, "dim 失配 = miss（cosine 会静默 0.0 沉底，必须 miss）"
    cache[PREMISE_URL] = [9.0, 8.0, 7.0]
    cache.flush()
    row = _db_rows(registry)[0]
    assert row["dim"] == 3 and struct.unpack("<3f", bytes(row["embedding"])) == pytest.approx((9.0, 8.0, 7.0))


def test_dim_learned_on_first_write(registry):
    cache = PersistentVectorCache()
    cache[PREMISE_URL] = [1.0, 2.0, 3.0, 4.0]
    assert cache.dim == 4, "dim=None 实例首次写入后学习"
    assert cache.get(OTHER_URL) is None


def test_flush_empty_is_cheap_noop_and_touches_no_db(registry):
    PersistentVectorCache().flush()
    assert not os.path.exists(os.path.join(str(registry), "user_materials.db"))


def test_stock_url_only_in_memory(registry):
    cache = PersistentVectorCache()
    cache[STOCK_URL] = [5.0, 6.0]
    assert cache.get(STOCK_URL) == [5.0, 6.0]
    cache.flush()
    assert _db_rows(registry) == []
    assert PersistentVectorCache().get(STOCK_URL) is None, "stock 向量不跨实例"


def test_flush_db_failure_is_fail_soft(registry, caplog):
    """DB 写抛异常 -> flush 只记 WARNING 不上抛；什么都没持久化，run2 重算。"""
    with caplog.at_level(logging.WARNING, logger="app.services.user_material_vectors"):
        with patch.object(um, "_get_conn", side_effect=RuntimeError("injected db failure")):
            cache = PersistentVectorCache()
            cache[PREMISE_URL] = [1.0, 2.0]
            cache.flush()  # 不抛异常 = fail-soft 契约
    assert "flush" in caplog.text.lower()
    assert not os.path.exists(os.path.join(str(registry), "user_materials.db"))

    stub = _rank(
        [{"asset_id": "a0", "url": PREMISE_URL, "data_uri": "thumb-0"}],
        PersistentVectorCache(),
        {"thumb-0": [1.0, 0.0]},
    )
    assert stub.calls == ["thumb-0"], "上一轮失败没落库 -> 本轮照常计算"


def test_evict_material_purges_only_matching_vectors(registry):
    """evict_material 只删向量；`_` 转义防同名前缀误删；台账行与其它 owner 向量无损。"""
    cache = PersistentVectorCache()
    cache["premise://foo_bar/mat-2026/0"] = [1.0]
    cache[SIBLING_URL] = [2.0]
    cache[PREMISE_URL] = [3.0]
    cache.flush()
    um.begin_push("foo_bar", "mat-2026", 0)
    assert len(_db_rows(registry)) == 3

    cache.evict_material("foo_bar", "mat-2026")
    urls = {r["url"] for r in _db_rows(registry)}
    assert urls == {PREMISE_URL, SIBLING_URL}
    assert cache.get("premise://foo_bar/mat-2026/0") is None
    with um._db_lock:
        row = um._get_conn().execute(
            "SELECT status FROM user_materials WHERE owner=? AND material_id=?", ("foo_bar", "mat-2026")
        ).fetchone()
    assert row is not None, "evict_material 绝不允许动台账行（那是 delete 的职责）"


def test_gate_off_funnel_persists_through_explicit_param(registry):
    """duplicate_gate=false（无 EmbeddingGate）：match_segments 拿显式 vector_cache
    绕开 isinstance(dict) 私有取回链，coarse 段后 flush 真实落库。

    todo-7 起 premise:// 候选的 data_uri 来自本地 registry（read_thumb_b64），
    测试同步播种素材并断言任何 HTTP 缩略图路径都不参与（boom 桩）。
    """
    um.begin_push("owner-1", "mat-2026", 0)
    with open(um.put_file("owner-1", "mat-2026", 0, 0, "clip"), "wb") as f:
        f.write(b"X")
    with open(um.put_file("owner-1", "mat-2026", 0, 0, "thumb"), "wb") as f:
        f.write(b"TT")
    um.complete("owner-1", "mat-2026", 0, [{"idx": 0, "t_start": 0.0, "t_end": 5.0, "duration": 5.0}])
    thumb_uri = um.read_thumb_b64("owner-1", "mat-2026", 0)

    item = MaterialInfo(
        provider="user_material", url=PREMISE_URL, duration=10,
        source_info={"asset_id": "mat-2026/0", "thumbnail_url": ""},
    )

    def fake_llm(prompt):
        return json.dumps({"terms": ["terma"], "coarse_query": "A broad scene.", "fine_query": "fp"})

    def fake_search(search_term, page=1, **_legacy):
        return [item] if (search_term, page) == ("terma", 1) else []

    def fake_judge(item, segment_text="", search_term=""):
        return {"verdict": "relevant", "reason": "ok"}

    def fake_save(video_url, save_dir=""):
        return f"/saved/{video_url.rsplit('/', 1)[-1]}"

    def _no_http(*_a, **_k):
        raise AssertionError("premise bytes must never go over HTTP (Metis #10)")

    cache = PersistentVectorCache()
    with (
        patch.object(video_match.llm, "generate_response", side_effect=fake_llm),
        patch.object(video_match, "download_thumbnail_bytes", side_effect=_no_http),
        patch.object(video_match, "to_data_uri", side_effect=_no_http),
        patch.object(image_embedding, "embed_text", return_value=[1.0, 0.0]),
        patch.object(image_embedding, "embed_image", _StubEmbedImage({thumb_uri: [1.0, 0.0]})),
        patch.object(video_match.material_rerank, "is_rerank_enabled", return_value=False),
        patch.object(video_match.material_rerank, "_walk_limit", return_value=5),
    ):
        results = video_match.match_segments(
            segments=[{"index": 0, "text": "A panda eats.", "duration": 3.0}],
            video_subject="panda",
            search_videos=fake_search,
            save_video=fake_save,
            video_aspect="9:16",
            clip_duration=3,
            judge_candidate=fake_judge,
            embedding_gate=None,
            generate_image=None,
            vector_cache=cache,
        )
    assert results and results[0].clips, "relevant 候选应被采纳"
    rows = _db_rows(registry)
    assert [r["url"] for r in rows] == [PREMISE_URL], "gate-off 也必须经显式参数持久化 premise 向量"
    assert rows[0]["model"] == "M1"
