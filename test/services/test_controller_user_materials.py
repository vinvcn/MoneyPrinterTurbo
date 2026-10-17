"""user_materials v1 路由的 TestClient 测试（plan todo 3 QA 场景全集）。

全部外部 I/O 走 monkeypatch：storage root -> tmp_path，时钟/连接同 todo 2；
413 用把 MAX_PUSH_BYTES 缩小注入的方式验证，绝不真写 1 GiB。
"""

from __future__ import annotations

import os
import sqlite3
import struct

import pytest
from fastapi.testclient import TestClient

from app.controllers.v1 import user_materials as routes
from app.services import user_materials as um

OWNER = "owner-1"
MID = "mat_2026-09-14-aaaa"
BASE = "/api/v1/user_materials"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "_storage_root", lambda create=False: str(tmp_path))
    monkeypatch.setattr(um, "_conn", None)
    monkeypatch.setattr(um, "_last_sweep_monotonic", None)
    from app import asgi

    yield tmp_path, TestClient(asgi.app)
    if um._conn is not None:
        um._conn.close()


def _put_file(
    client, idx: object, kind: str | None, body: bytes = b"x", rev: int | str | None = 0,
    owner: str = OWNER, material: str = MID, **params: object,
) -> object:
    query = {"kind": kind, "rev": rev}
    query.update(params)
    query = {k: v for k, v in query.items() if v is not None}  # None = 刻意省略该 query 参数
    return client.put(f"{BASE}/{owner}/{material}/files/{idx}", params=query, content=body)


def _insert_vector(tmp_path, url: str) -> None:
    conn = sqlite3.connect(os.path.join(str(tmp_path), "user_materials.db"))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO user_material_vectors (url, embedding, model, dim, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (url, struct.pack("<2f", 0.5, -0.5), "tongyi-embedding-vision-flash", 2, "2026-09-14T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()


def _vector_count(tmp_path) -> int:
    conn = sqlite3.connect(os.path.join(str(tmp_path), "user_materials.db"))
    try:
        return conn.execute("SELECT COUNT(*) FROM user_material_vectors").fetchone()[0]
    finally:
        conn.close()


def test_happy_three_files_complete_list_delete_purges_vectors(env):
    """QA happy：3 个文件 PUT -> complete -> GET 显示 ready -> DELETE 204 且向量随之清空。"""
    _, client = env
    assert _put_file(client, 0, "clip", b"C" * 20).status_code == 200
    resp = _put_file(client, 0, "thumb", b"T" * 10)
    assert resp.status_code == 200
    assert resp.json() == {"idx": 0, "kind": "thumb", "rev": 0, "bytes_written": 10}
    assert _put_file(client, 1, "clip", b"C" * 20).status_code == 200  # 多余暂存文件，不入 manifest

    done = client.post(
        f"{BASE}/{OWNER}/{MID}/complete",
        json={"rev": 0, "footages": [{"idx": 0, "t_start": 0.0, "t_end": 5.0, "duration": 5.0}]},
    )
    assert done.status_code == 200
    assert done.json() == {"material_id": MID, "rev": 0, "footage_count": 1, "total_bytes": 30}

    listed = client.get(BASE, params={"owner_id": OWNER})
    assert listed.status_code == 200
    assert listed.json() == {"materials": [{"material_id": MID, "rev": 0, "footage_count": 1, "total_bytes": 30}]}
    assert client.get(BASE, params={"owner_id": "other"}).json() == {"materials": []}

    _insert_vector(env[0], um.url(OWNER, MID, 0))
    deleted = client.delete(f"{BASE}/{OWNER}/{MID}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(BASE, params={"owner_id": OWNER}).json() == {"materials": []}
    assert _vector_count(env[0]) == 0
    assert not os.path.exists(os.path.join(str(env[0]), OWNER, MID))


def test_invalid_kind_idx_rev_are_422(env):
    _, client = env
    assert _put_file(client, 0, "exe", b"x").status_code == 422  # 未知 kind
    assert _put_file(client, 0, None, b"x").status_code == 422  # 缺 kind
    assert _put_file(client, 0, "clip", b"x", rev=None).status_code == 422  # 缺 rev
    assert _put_file(client, "abc", "clip", b"x", rev=0).status_code == 422  # idx 非数字
    assert _put_file(client, -1, "clip", b"x", rev=0).status_code == 422  # 负 idx
    assert _put_file(client, 0, "clip", b"x", rev="-1").status_code == 422  # 负 rev
    assert _put_file(client, 0, "clip", b"x", owner="bad owner").status_code == 422  # 空格 id
    assert client.get(BASE, params={"owner_id": "bad/../.."}).status_code == 422


def test_over_cap_returns_413_and_removes_partial_file(env, monkeypatch):
    """413：缩小注入的字节上限（不写 1 GiB）；溢出后暂存文件必须被清掉。"""
    tmp_path, client = env
    monkeypatch.setattr(routes, "MAX_PUSH_BYTES", 8)
    resp = _put_file(client, 0, "clip", b"Z" * 16)
    assert resp.status_code == 413
    staged = os.path.join(str(tmp_path), OWNER, MID, ".partial", "0", "0.mp4")
    assert not os.path.exists(staged)
    assert um.list_ready(OWNER) == []


def test_complete_before_upload_is_409(env):
    _, client = env
    resp = client.post(
        f"{BASE}/{OWNER}/{MID}/complete",
        json={"rev": 0, "footages": [{"idx": 0, "t_start": 0.0, "t_end": 5.0, "duration": 5.0}]},
    )
    assert resp.status_code == 409
    assert resp.json()["status"] == 409
    assert client.get(BASE, params={"owner_id": OWNER}).json() == {"materials": []}


def test_manifest_missing_staged_thumb_is_409(env):
    _, client = env
    assert _put_file(client, 0, "clip", b"C" * 20).status_code == 200
    resp = client.post(
        f"{BASE}/{OWNER}/{MID}/complete",
        json={"rev": 0, "footages": [{"idx": 0, "t_start": 0.0, "t_end": 5.0, "duration": 5.0}]},
    )
    assert resp.status_code == 409
    assert um.list_ready(OWNER) == []


def test_path_traversal_percent2f_is_422(env):
    """`..%2Fx` 在路由前被解码成额外段 -> 兜底路由统一 422（绝不落到 404/目录逃逸）。"""
    _, client = env
    resp = client.put(f"{BASE}/{OWNER}/..%2F{MID}/files/0", params={"kind": "clip", "rev": 0}, content=b"x")
    assert resp.status_code == 422
    assert client.delete(f"{BASE}/{OWNER}/..%2Fx").status_code == 422
    assert client.post(
        f"{BASE}/{OWNER}/..%2Fx/complete",
        json={"rev": 0, "footages": [{"idx": 0, "t_start": 0.0, "t_end": 5.0, "duration": 5.0}]},
    ).status_code == 422
    assert client.get(BASE, params={"owner_id": f"{OWNER}/../../etc"}).status_code == 422


def test_delete_is_idempotent_204(env):
    _, client = env
    assert client.delete(f"{BASE}/{OWNER}/{MID}").status_code == 204
    assert client.delete(f"{BASE}/{OWNER}/{MID}").status_code == 204


def test_routes_registered_on_app(env):
    """注册链完整性：router.py 接入后 openapi 暴露 4 个 user_materials 路径。"""
    _, client = env
    paths = client.get("/openapi.json").json()["paths"]
    assert f"{BASE}/{{owner_id}}/{{material_id}}/files/{{idx}}" in paths
    assert f"{BASE}/{{owner_id}}/{{material_id}}/complete" in paths
    assert BASE in paths
    assert f"{BASE}/{{owner_id}}/{{material_id}}" in paths
