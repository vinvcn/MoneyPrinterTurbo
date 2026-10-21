"""BYOM premise 候选池测试（ADR-0023 决策 1/5/9）：VPC 检索 + footage_map 快照。

宿主侧 user_materials 三件套（登记册 / 向量缓存 / HTTP 文件端点）已按
ADR-0023 决策 1 删除，premise 候选池改为每段一次 VPC 检索：embed 查询词 ->
search(ns + group_keys) -> 按 group_key 取 max 相似度 -> url 即 /shared cache
clip 路径。

覆盖：pool builder 的聚合/排序/order（fake VPC 客户端注入）、owner scope
（namespace 与 group_keys 的隔离）、零命中与各级失败的 backfill 降级、HTTP
客户端的请求契约、task 接线（owner_id 守卫 / kb_scope 归属 / save_video 本地
拷贝）、match_segments premise 漏斗行为，以及 T6 线上形状的字段保真。
全部桩 + fake，无任何真实 VPC/HTTP 调用。
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import FootageMapEntry, KBScope, TaskVideoRequest
from app.services import kb_footage, task, video_match, vlm_judge

OWNER = "owner-1"
CLIP_BYTES = b"clip-bytes-0"
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg"
CLIP_PATH = "/shared/cache/m1/f0/clip.mp4"
THUMB_PATH = "/shared/cache/m1/f0/thumb.jpg"


def _params(source, owner_id=OWNER, **kwargs):
    return TaskVideoRequest(
        video_subject="s", video_source=source, owner_id=owner_id, video_clip_duration=3, **kwargs
    )


def _scope(owner_id: str = OWNER, collection_ids=("c1", "c2")) -> KBScope:
    return KBScope(owner_id=owner_id, collection_ids=list(collection_ids))


def _footage_map(tmp_path, ids=("f0", "f1", "f2")) -> dict[str, FootageMapEntry]:
    """把每个 footage 的 clip/thumb 真实落到 tmp_path，返回 footage_map 条目。"""
    entries: dict[str, FootageMapEntry] = {}
    for idx, footage_id in enumerate(ids):
        clip = tmp_path / footage_id / "clip.mp4"
        thumb = tmp_path / footage_id / "thumb.jpg"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(CLIP_BYTES)
        thumb.write_bytes(JPEG_BYTES if idx % 2 == 0 else b"\xff\xd8\xff\xe0other")
        entries[footage_id] = FootageMapEntry(
            duration_s=4.0 + idx,
            t_start_ms=idx * 1000,
            t_end_ms=idx * 1000 + 4000,
            clip_path=str(clip),
            thumb_path=str(thumb),
        )
    return entries


class _FakeVPC:
    """记录调用的 fake 索引客户端：构造时给定 hits / 向量，可强制失败。

    embed_result=None 表示嵌入失败（返回 None）；缺省返回构造向量。
    """

    _UNSET = object()

    def __init__(self, vector=(0.1, 0.2), hits=None, embed_result=_UNSET, search_error=None):
        self._vector = list(vector)
        self._hits = list(hits or [])
        self._embed_result = embed_result
        self._search_error = search_error
        self.embed_calls: list[str] = []
        self.search_calls: list[dict] = []

    def embed_query(self, text: str):
        self.embed_calls.append(text)
        if self._embed_result is not self._UNSET:
            return self._embed_result
        return list(self._vector)

    def search_footage(self, namespace, vector, group_keys, k):
        self.search_calls.append(
            {
                "namespace": namespace,
                "vector": list(vector),
                "group_keys": list(group_keys),
                "k": k,
            }
        )
        if self._search_error is not None:
            raise self._search_error
        return [dict(hit) for hit in self._hits]


# ---------------------------------------------------------------------------
# pool builder：聚合 / max / 排序
# ---------------------------------------------------------------------------


def test_pool_groups_frames_by_footage_and_takes_max(tmp_path):
    """同 footage 多帧命中取最高相似度；候选 url 即快照 clip 路径。"""
    footage_map = _footage_map(tmp_path)
    fake = _FakeVPC(
        hits=[
            {"group_key": "f0", "vector_id": "0", "score": 0.2},
            {"group_key": "f1", "vector_id": "3", "score": 0.5},
            {"group_key": "f0", "vector_id": "7", "score": 0.9},
            {"group_key": "f2", "vector_id": "1", "score": 0.4},
        ]
    )

    pool = kb_footage.build_pool(_scope(), footage_map, "rainy night street", fake)

    assert [item.url for item in pool] == [
        footage_map["f0"].clip_path,
        footage_map["f1"].clip_path,
        footage_map["f2"].clip_path,
    ], "按每 footage 的最高相似度降序：f0(0.9) > f1(0.5) > f2(0.4)"
    assert [item.source_info["score"] for item in pool] == [0.9, 0.5, 0.4]
    assert [item.source_info["asset_id"] for item in pool] == ["f0", "f1", "f2"]
    assert [item.duration for item in pool] == [4, 5, 6], "时长取自 footage_map 快照"
    first = pool[0].source_info["rendition"]
    assert first["footage_id"] == "f0"
    assert first["t_start_ms"] == 0 and first["t_end_ms"] == 4000
    assert first["clip_path"] == footage_map["f0"].clip_path
    assert first["thumb_path"] == footage_map["f0"].thumb_path
    assert all(item.provider == kb_footage.FOOTAGE_PROVIDER for item in pool)


def test_pool_ties_ordered_by_footage_id(tmp_path):
    """同分按 footage id 升序，保证池序确定（不打乱、可复现）。"""
    footage_map = _footage_map(tmp_path, ids=("fb", "fa", "fc"))
    fake = _FakeVPC(
        hits=[
            {"group_key": "fb", "vector_id": "0", "score": 0.7},
            {"group_key": "fc", "vector_id": "0", "score": 0.7},
            {"group_key": "fa", "vector_id": "0", "score": 0.7},
        ]
    )

    pool = kb_footage.build_pool(_scope(), footage_map, "a wide street", fake)

    assert [item.source_info["asset_id"] for item in pool] == ["fa", "fb", "fc"]


def test_pool_drops_footage_absent_from_snapshot(tmp_path):
    """快照里没有的 group（外人的 footage）绝不成为候选——第二道隔离闸。"""
    footage_map = _footage_map(tmp_path, ids=("f0", "f1"))
    fake = _FakeVPC(
        hits=[
            {"group_key": "f0", "vector_id": "0", "score": 0.8},
            {"group_key": "stranger-footage", "vector_id": "0", "score": 0.99},
        ]
    )

    pool = kb_footage.build_pool(_scope(), footage_map, "query", fake)

    assert [item.source_info["asset_id"] for item in pool] == ["f0"]


def test_pool_footage_without_hits_is_absent(tmp_path):
    """快照覆盖但检索零命中的 footage 不入池（无向量可召回）。"""
    footage_map = _footage_map(tmp_path, ids=("f0", "f1", "f2"))
    fake = _FakeVPC(hits=[{"group_key": "f0", "vector_id": "0", "score": 0.6}])

    pool = kb_footage.build_pool(_scope(), footage_map, "query", fake)

    assert [item.source_info["asset_id"] for item in pool] == ["f0"]


# ---------------------------------------------------------------------------
# owner scope：namespace 来自 kb_scope.owner_id，group_keys 来自快照
# ---------------------------------------------------------------------------


def test_pool_search_scope_comes_from_kb_scope_and_snapshot(tmp_path):
    """检索范围 = kb_scope.owner_id 作 namespace + footage_map 的 footage id
    作 group_keys（即所选素材集的快照）；外人的 namespace 无从谈起。"""
    footage_map = _footage_map(tmp_path, ids=("f0", "f1", "f2"))
    fake = _FakeVPC(hits=[{"group_key": "f1", "vector_id": "0", "score": 0.5}])

    kb_footage.build_pool(_scope(owner_id="u-42"), footage_map, "rainy street", fake)

    assert fake.embed_calls == ["rainy street"]
    assert len(fake.search_calls) == 1
    call = fake.search_calls[0]
    assert call["namespace"] == "u-42", "namespace 必须来自 kb_scope.owner_id"
    assert call["group_keys"] == ["f0", "f1", "f2"], "group_keys 即快照的 footage id"
    assert call["vector"] == [0.1, 0.2]
    assert call["k"] == kb_footage._search_k()


def test_pool_uses_configured_search_k(monkeypatch, tmp_path):
    """k 读 [index] search_k，配置变更立即生效。"""
    footage_map = _footage_map(tmp_path, ids=("f0",))
    fake = _FakeVPC(hits=[{"group_key": "f0", "vector_id": "0", "score": 0.5}])
    monkeypatch.setattr(config, "index", {"base_url": "", "timeout": 30, "search_k": 12})

    kb_footage.build_pool(_scope(), footage_map, "query", fake)

    assert fake.search_calls[0]["k"] == 12


def test_pool_refuses_scope_without_owner(tmp_path):
    """kb_scope 缺 owner_id：不发起任何检索调用，直接空池。"""
    footage_map = _footage_map(tmp_path)
    fake = _FakeVPC(hits=[{"group_key": "f0", "vector_id": "0", "score": 0.9}])

    pool = kb_footage.build_pool(KBScope(collection_ids=["c1"]), footage_map, "q", fake)

    assert pool == []
    assert fake.embed_calls == [] and fake.search_calls == [], "没有 owner 就没有 namespace"


# ---------------------------------------------------------------------------
# 零命中 / 各级失败：一律降级为空池（backfill 接管），绝不抛异常
# ---------------------------------------------------------------------------


def test_pool_zero_hits_degrades_to_empty(tmp_path):
    footage_map = _footage_map(tmp_path)
    fake = _FakeVPC(hits=[])

    assert kb_footage.build_pool(_scope(), footage_map, "query", fake) == []


def test_pool_embed_failure_degrades_without_search(tmp_path):
    footage_map = _footage_map(tmp_path)
    fake = _FakeVPC(embed_result=None)

    assert kb_footage.build_pool(_scope(), footage_map, "query", fake) == []
    assert fake.search_calls == [], "嵌入不可用就不该发起检索"


def test_pool_search_failure_degrades_to_empty(tmp_path):
    footage_map = _footage_map(tmp_path)
    fake = _FakeVPC(search_error=RuntimeError("index down"))

    assert kb_footage.build_pool(_scope(), footage_map, "query", fake) == []


@pytest.mark.parametrize(
    "kb_scope,footage_map,query",
    [
        (None, {"f0": FootageMapEntry(clip_path="/x.mp4")}, "query"),
        (_scope(), None, "query"),
        (_scope(), {}, "query"),
        (_scope(), {"f0": FootageMapEntry(clip_path="/x.mp4")}, ""),
        (_scope(), {"f0": FootageMapEntry(clip_path="/x.mp4")}, "   "),
    ],
)
def test_pool_missing_inputs_degrade_without_client_calls(kb_scope, footage_map, query):
    fake = _FakeVPC(hits=[{"group_key": "f0", "vector_id": "0", "score": 0.9}])

    assert kb_footage.build_pool(kb_scope, footage_map, query, fake) == []
    assert fake.embed_calls == [] and fake.search_calls == []


def test_pool_default_client_uses_module_http_functions(monkeypatch, tmp_path):
    """不注入 client 时走 [index] 端点的 HTTP 函数（patch 模块函数即生效）。"""
    footage_map = _footage_map(tmp_path, ids=("f0",))
    seen = {}

    def fake_embed(text):
        seen["embed"] = text
        return [0.5]

    def fake_search(namespace, vector, group_keys, k):
        seen["search"] = (namespace, list(group_keys), k)
        return [{"group_key": "f0", "vector_id": "0", "score": 0.75}]

    monkeypatch.setattr(kb_footage, "embed_query", fake_embed)
    monkeypatch.setattr(kb_footage, "search_footage", fake_search)

    pool = kb_footage.build_pool(_scope(), footage_map, "rainy street")

    assert seen["embed"] == "rainy street"
    assert seen["search"] == (OWNER, ["f0"], kb_footage._search_k())
    assert [item.source_info["asset_id"] for item in pool] == ["f0"]


# ---------------------------------------------------------------------------
# HTTP 客户端契约（requests.post 打桩，无真实调用）
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_embed_query_posts_text_and_returns_vector(monkeypatch):
    calls = {}

    def fake_post(url, json=None, timeout=None, proxies=None):
        calls.update({"url": url, "json": json, "proxies": proxies})
        return _Resp(payload={"model": "m", "dim": 2, "vector": [0.1, 0.2], "cached": False})

    monkeypatch.setattr(kb_footage.config, "index", {"base_url": "http://vpc:8787", "timeout": 5})
    monkeypatch.setattr(kb_footage.requests, "post", fake_post)

    assert kb_footage.embed_query("rainy street") == [0.1, 0.2]
    assert calls["url"] == "http://vpc:8787/index/embed"
    assert calls["json"] == {"text": "rainy street"}
    assert calls["proxies"] == {}, "容器网络名直连，不走宿主代理"


def test_embed_query_non_200_and_empty_base_url_return_none(monkeypatch):
    posted = []

    def fake_post(url, json=None, timeout=None, proxies=None):
        posted.append(url)
        return _Resp(status_code=502, payload={})

    monkeypatch.setattr(kb_footage.requests, "post", fake_post)
    monkeypatch.setattr(kb_footage.config, "index", {"base_url": "", "timeout": 30})

    assert kb_footage.embed_query("q") is None
    assert posted == [], "端点未配置不发起请求"

    monkeypatch.setattr(kb_footage.config, "index", {"base_url": "http://vpc:8787", "timeout": 30})
    assert kb_footage.embed_query("q") is None, "502 fail-open"


def test_search_footage_posts_namespace_and_group_keys(monkeypatch):
    calls = {}

    def fake_post(url, json=None, timeout=None, proxies=None):
        calls.update({"url": url, "json": json})
        return _Resp(
            payload={"namespace": "u", "hits": [{"group_key": "f0", "vector_id": "0", "score": 0.5}]}
        )

    monkeypatch.setattr(kb_footage.config, "index", {"base_url": "http://vpc:8787", "timeout": 30})
    monkeypatch.setattr(kb_footage.requests, "post", fake_post)

    hits = kb_footage.search_footage("u-42", [0.1, 0.2], ["f1", "f0"], 10)

    assert calls["url"] == "http://vpc:8787/index/search"
    assert calls["json"] == {
        "namespace": "u-42",
        "vector": [0.1, 0.2],
        "k": 10,
        "group_keys": ["f1", "f0"],
    }
    assert hits == [{"group_key": "f0", "vector_id": "0", "score": 0.5}]


def test_search_footage_dimension_mismatch_400_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(kb_footage.config, "index", {"base_url": "http://vpc:8787", "timeout": 30})
    monkeypatch.setattr(kb_footage.requests, "post", lambda *a, **k: _Resp(status_code=400))

    assert kb_footage.search_footage("u", [0.1], ["f0"], 10) == []


def test_thumb_data_uri_reads_local_file(tmp_path):
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(JPEG_BYTES)

    assert kb_footage.thumb_data_uri(str(thumb)) == (
        "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode()
    )
    assert kb_footage.thumb_data_uri("") == ""
    assert kb_footage.thumb_data_uri(str(tmp_path / "missing.jpg")) == ""


# ---------------------------------------------------------------------------
# owner_id 校验（premise 与 mixed；stock/legacy 不要求）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("video_source", ["premise", "mixed"])
def test_owner_id_required_raises(video_source):
    with pytest.raises(ValueError, match="owner_id_required"):
        task._require_owner_id(_params(video_source, None))
    with pytest.raises(ValueError, match="owner_id_required"):
        task._require_owner_id(_params(video_source, "   "))


def test_owner_id_not_required_for_stock_or_legacy():
    for source in ("stock", "pexels", "builtin", ""):
        task._require_owner_id(_params(source, None))


@pytest.mark.parametrize("video_source", ["premise", "mixed"])
def test_owner_id_required_marks_task_failed(video_source, monkeypatch):
    monkeypatch.setattr(task.sm.state, "get_task", lambda *a, **k: None)
    monkeypatch.setattr(task.sm.state, "update_task", lambda *a, **k: None)
    result = task._run_segment_first_pipeline(
        "t-uid", _params(video_source, None), "script text", stop_at="materials"
    )
    assert result["failed_stage"] == "materials"
    assert "owner_id_required" in str(result["error"])


# ---------------------------------------------------------------------------
# 搜索路由：premise 零 stock 调用；stock/legacy 原样
# ---------------------------------------------------------------------------


def test_premise_search_videos_returns_empty_and_never_touches_providers():
    called = {"assert": 0, "multi": 0}

    def fake_assert():
        called["assert"] += 1

    def fake_multi(**_kw):
        called["multi"] += 1
        return []

    with (
        patch.object(task.material, "assert_search_provider_available", fake_assert),
        patch.object(task.material, "search_videos_multi_provider", fake_multi),
    ):
        search = task._make_search_videos(_params("premise", OWNER))
        assert search("ignored keyword", page=1) == []
        assert search("anything", page=2) == []
        task._preflight_search_provider(_params("premise", OWNER))
    assert called == {"assert": 0, "multi": 0}, "premise 绝不触在线供应商路径"


def test_stock_and_legacy_keep_multi_provider_search():
    calls = []

    def fake_multi(**kw):
        calls.append(kw)
        return [task.MaterialInfo(provider="pexels", url="https://x/1.mp4", duration=10)]

    with patch.object(task.material, "search_videos_multi_provider", fake_multi):
        for source in ("stock", "pexels", ""):
            search = task._make_search_videos(_params(source, None))
            out = search("term", minimum_duration=4, video_aspect="9:16", page=2)
            assert [i.url for i in out] == ["https://x/1.mp4"]
    assert [c["page"] for c in calls] == [2, 2, 2]


def test_preflight_asserts_for_stock_when_vlm_enabled():
    with (
        patch.object(task.vlm_judge, "is_enabled", lambda: True),
        patch.object(
            task.material, "assert_search_provider_available", side_effect=AssertionError("no provider")
        ),
    ):
        with pytest.raises(AssertionError, match="no provider"):
            task._preflight_search_provider(_params("pexels", None))


def test_preflight_skipped_for_mixed_and_premise():
    with (
        patch.object(task.vlm_judge, "is_enabled", lambda: True),
        patch.object(
            task.material, "assert_search_provider_available", side_effect=AssertionError("no provider")
        ),
    ):
        task._preflight_search_provider(_params("premise", "u1"))
        task._preflight_search_provider(_params("mixed", "u1"))


# ---------------------------------------------------------------------------
# premise_pool 接线：scope 归属 / 快照 / fail-open
# ---------------------------------------------------------------------------


def test_make_premise_pool_only_for_premise_sources():
    assert task._make_premise_pool(_params("premise")) is not None
    assert task._make_premise_pool(_params("mixed")) is not None
    for source in ("stock", "pexels", "library", "local", ""):
        assert task._make_premise_pool(_params(source)) is None


def test_make_premise_pool_builds_vpc_pool_per_segment(monkeypatch, tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0", "f1"))
    seen = {}

    def fake_embed(text):
        seen["embed"] = text
        return [0.5]

    def fake_search(namespace, vector, group_keys, k):
        seen["search"] = (namespace, list(group_keys))
        return [
            {"group_key": "f1", "vector_id": "0", "score": 0.8},
            {"group_key": "f0", "vector_id": "1", "score": 0.6},
        ]

    monkeypatch.setattr(kb_footage, "embed_query", fake_embed)
    monkeypatch.setattr(kb_footage, "search_footage", fake_search)
    pool_cb = task._make_premise_pool(
        _params("premise", kb_scope=_scope(), footage_map=footage_map)
    )

    items = pool_cb("rainy night street")

    assert seen["embed"] == "rainy night street", "每段查询词经 VPC embed"
    assert seen["search"] == (OWNER, ["f0", "f1"])
    assert [i.url for i in items] == [
        footage_map["f1"].clip_path,
        footage_map["f0"].clip_path,
    ]
    assert [i.source_info["asset_id"] for i in items] == ["f1", "f0"]


def test_make_premise_pool_rejects_scope_of_another_owner(monkeypatch, tmp_path):
    """kb_scope.owner_id 与任务 owner 不一致：绝不检索别人的 namespace。"""
    footage_map = _footage_map(tmp_path, ids=("f0",))
    called = {"embed": 0, "search": 0}

    def counting_embed(text):
        called["embed"] += 1
        return [0.5]

    def counting_search(*a):
        called["search"] += 1
        return []

    monkeypatch.setattr(kb_footage, "embed_query", counting_embed)
    monkeypatch.setattr(kb_footage, "search_footage", counting_search)
    pool_cb = task._make_premise_pool(
        _params("premise", kb_scope=_scope(owner_id="someone-else"), footage_map=footage_map)
    )

    assert pool_cb("query") == []
    assert called == {"embed": 0, "search": 0}


def test_make_premise_pool_degrades_without_scope_or_snapshot():
    for params in (
        _params("premise"),
        _params("premise", kb_scope=_scope()),
        _params("premise", footage_map={}),
    ):
        pool_cb = task._make_premise_pool(params)
        assert pool_cb is not None
        assert pool_cb("rainy street") == []


def test_make_premise_pool_mixed_has_no_snapshot_in_v1():
    """mixed v1 不实现：VBL 不为 mixed 快照 footage_map，池恒空、只有 stock 兜底。"""
    pool_cb = task._make_premise_pool(_params("mixed", kb_scope=_scope()))

    assert pool_cb is not None
    assert pool_cb("rainy street") == []


# ---------------------------------------------------------------------------
# save_video：footage_map 的 /shared clip 本地拷贝，其余委派
# ---------------------------------------------------------------------------


def test_save_video_copies_shared_clip_for_footage(tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0",))
    save_dir = tmp_path / "task-materials"
    save_dir.mkdir()
    cb = task._make_save_video(
        _params("premise", footage_map=footage_map), str(save_dir)
    )

    out = cb(footage_map["f0"].clip_path)

    assert out == str(save_dir / "footage-f0.mp4")
    assert Path(out).read_bytes() == CLIP_BYTES


def test_save_video_footage_missing_file_fails_loudly(tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0",))
    os.remove(footage_map["f0"].clip_path)
    cb = task._make_save_video(_params("premise", footage_map=footage_map), str(tmp_path))

    with pytest.raises(FileNotFoundError):
        cb(footage_map["f0"].clip_path)


def test_save_video_non_footage_delegates_to_material():
    seen = {}

    def fake_save(video_url, save_dir=""):
        seen["args"] = (video_url, save_dir)
        return "/saved/x.mp4"

    with patch.object(task.material, "save_video", fake_save):
        cb = task._make_save_video(_params("stock", None), "/dir")
        assert cb("https://x/1.mp4") == "/saved/x.mp4"
    assert seen["args"] == ("https://x/1.mp4", "/dir")


# ---------------------------------------------------------------------------
# 候选 builder：thumb 本地直读
# ---------------------------------------------------------------------------


def test_premise_candidate_builder_reads_local_thumb(tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0",))
    item = task.MaterialInfo(
        provider=kb_footage.FOOTAGE_PROVIDER,
        url=footage_map["f0"].clip_path,
        duration=4,
        source_info={
            "asset_id": "f0",
            "thumb_path": footage_map["f0"].thumb_path,
        },
    )

    cand = video_match._premise_candidate_from_item(item, "")

    assert cand["url"] == footage_map["f0"].clip_path
    assert cand["data_uri"] == (
        "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode()
    )
    assert cand["item"] is item


def test_premise_candidate_builder_empty_when_thumb_missing(tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0",))
    os.remove(footage_map["f0"].thumb_path)
    item = task.MaterialInfo(
        provider=kb_footage.FOOTAGE_PROVIDER,
        url=footage_map["f0"].clip_path,
        source_info={"asset_id": "f0", "thumb_path": footage_map["f0"].thumb_path},
    )

    assert video_match._premise_candidate_from_item(item, "")["data_uri"] == ""


def test_vlm_judge_reads_footage_thumb_locally(tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0",))
    judge = vlm_judge.make_default_judge(embedding_gate=None)
    item = task.MaterialInfo(
        provider=kb_footage.FOOTAGE_PROVIDER,
        url=footage_map["f0"].clip_path,
        source_info={"asset_id": "f0", "thumb_path": footage_map["f0"].thumb_path},
    )
    expected = "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode()
    with (
        patch.object(vlm_judge, "download_thumbnail_bytes", side_effect=_no_http),
        patch.object(vlm_judge, "_download_candidate_to_temp", side_effect=_no_http),
        patch.object(vlm_judge, "extract_first_frame_jpeg", side_effect=_no_http),
        patch.object(
            vlm_judge, "judge_image", return_value=("relevant", "on topic", 1)
        ) as ji,
    ):
        record = judge(item, "seg text", "kw")
    assert ji.call_args.kwargs["image_data_uri"] == expected, "VLM 输入必须是本地 thumb 字节"
    assert record["image_source"] == "footage_thumb"
    assert record["verdict"] == "relevant"


def _no_http(*_a, **_k):
    raise AssertionError("BYOM 候选字节绝不走 HTTP（路径寻址，ADR-0023 决策 2）")


# ---------------------------------------------------------------------------
# 接线：pipeline 把 source + premise_pool 交给 match_segments
# ---------------------------------------------------------------------------


def test_pipeline_passes_source_and_premise_pool_to_match_segments(monkeypatch, tmp_path):
    footage_map = _footage_map(tmp_path, ids=("f0",))
    monkeypatch.setattr(
        kb_footage,
        "search_footage",
        lambda namespace, vector, group_keys, k: [
            {"group_key": "f0", "vector_id": "0", "score": 0.7}
        ],
    )
    monkeypatch.setattr(kb_footage, "embed_query", lambda text: [0.5])
    captured: dict = {}

    def fake_match(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(clips=["c.mp4"], holes=[])]

    def fake_state(*_a, **_k):
        return None

    with (
        patch.object(task.sm.state, "update_task", fake_state),
        patch.object(task.sm.state, "get_task", fake_state),
        patch.object(
            task.segmenter,
            "segment_script",
            lambda script: [SimpleNamespace(index=0, text="hello world", estimated_duration=3.0)],
        ),
        patch.object(task, "save_script_data", lambda *a, **k: None),
        patch.object(task.segment_material, "english_search_term", lambda text: "noodles"),
        patch.object(
            task.segment_audio,
            "prepare_segment_audio",
            lambda **kwargs: SimpleNamespace(
                ok=True, audio_file="a.mp3", total_duration_ms=1500, error=None,
                segments=[{"index": 0, "text": "hello", "audio_file": "a-0.mp3", "start_ms": 0, "duration_ms": 1500}],
            ),
        ),
        patch.object(task.image_embedding, "is_duplicate_gate_enabled", return_value=False),
        patch.object(task.vlm_judge, "is_enabled", return_value=False),
        patch.object(task.video_match, "match_segments", side_effect=fake_match),
        patch.object(task.segment_material, "persist_segment_material_sources", lambda *a, **k: None),
        patch.object(task.segment_material, "segments_to_records", lambda materials: []),
        patch.object(task.utils, "task_dir", lambda sub_dir="": "/tmp"),
    ):
        task._run_segment_first_pipeline(
            "t-wire", _params("premise", kb_scope=_scope(), footage_map=footage_map),
            "hello world", stop_at="materials",
        )

    assert captured.get("source") == "premise"
    premise_pool = captured.get("premise_pool")
    assert callable(premise_pool)
    items = premise_pool("a rainy street")
    assert [i.url for i in items] == [footage_map["f0"].clip_path]
    assert callable(captured.get("search_videos")), "stock 回调照常在场（mixed 兜底共用）"


# ---------------------------------------------------------------------------
# match_segments 单遍实现：premise 只用池内候选（零 stock、不打乱、本地首帧）
# ---------------------------------------------------------------------------


def _funnel_llm(prompt):
    return json.dumps({"terms": ["terma"], "coarse_query": "A broad scene.", "fine_query": "fp"})


def _accept_judge(*_a, **_kw):
    return {"verdict": "relevant", "reason": "ok"}


def _spy_coarse(captured):
    def spy(pool, coarse_query, vector_cache, embedding_gate):
        captured["pool"] = list(pool)
        captured["cache"] = vector_cache
        return list(pool), []

    return spy


def test_match_segments_premise_source_uses_pool_only(monkeypatch, tmp_path):
    """source="premise"：池即全部候选——零 stock 调用、不打乱、本地首帧直读。"""
    footage_map = _footage_map(tmp_path, ids=("f0", "f1"))
    monkeypatch.setattr(
        kb_footage,
        "search_footage",
        lambda namespace, vector, group_keys, k: [
            {"group_key": "f1", "vector_id": "0", "score": 0.9},
            {"group_key": "f0", "vector_id": "0", "score": 0.5},
        ],
    )
    monkeypatch.setattr(kb_footage, "embed_query", lambda text: [0.5])

    def fail_search(*_a, **_k):
        raise AssertionError("premise 单遍不得触发 stock 搜索")

    captured: dict = {}
    with (
        patch.object(video_match.llm, "generate_response", side_effect=_funnel_llm),
        patch.object(video_match, "fetch_image_bytes", side_effect=_no_http),
        patch.object(video_match, "to_data_uri", side_effect=_no_http),
        patch.object(video_match, "coarse_rank", side_effect=_spy_coarse(captured)),
        patch.object(video_match.material_rerank, "is_rerank_enabled", return_value=False),
        patch.object(video_match.material_rerank, "_walk_limit", return_value=5),
        patch.object(video_match, "random") as fake_random,
        patch.object(video_match, "logger"),
    ):
        results = video_match.match_segments(
            segments=[{"index": 0, "text": "A panda eats.", "duration": 3.0}],
            video_subject="panda",
            search_videos=fail_search,
            save_video=lambda video_url, save_dir="": f"/saved/{video_url.rsplit('/', 1)[-1]}",
            video_aspect="9:16",
            clip_duration=3,
            judge_candidate=_accept_judge,
            embedding_gate=None,
            generate_image=None,
            source="premise",
            premise_pool=lambda query: kb_footage.build_pool(
                _scope(), footage_map, query
            ),
        )
    fake_random.shuffle.assert_not_called()
    assert [c["url"] for c in captured["pool"]] == [
        footage_map["f1"].clip_path,
        footage_map["f0"].clip_path,
    ], "池序即 VPC 相似度序，不打乱、不翻页"
    assert all(c["data_uri"].startswith("data:image/jpeg;base64,") for c in captured["pool"])
    assert results[0].clips, "relevant 候选应被采纳"


def test_match_segments_premise_pool_failure_fails_open(monkeypatch, tmp_path):
    """premise 池读取失败：fail-open 空手而归（留 warning），不抛异常、不回退 stock。"""

    def broken_pool(_query):
        raise RuntimeError("index exploded")

    def fail_search(*_a, **_k):
        raise AssertionError("池失败后同样不得回退 stock 搜索")

    captured: dict = {}
    with (
        patch.object(video_match.llm, "generate_response", side_effect=_funnel_llm),
        patch.object(video_match, "fetch_image_bytes", side_effect=_no_http),
        patch.object(video_match, "to_data_uri", side_effect=_no_http),
        patch.object(video_match, "coarse_rank", side_effect=_spy_coarse(captured)),
        patch.object(video_match.material_rerank, "is_rerank_enabled", return_value=False),
        patch.object(video_match.material_rerank, "_walk_limit", return_value=5),
        patch.object(video_match, "logger") as fake_logger,
    ):
        results = video_match.match_segments(
            segments=[{"index": 0, "text": "A panda eats.", "duration": 3.0}],
            video_subject="panda",
            search_videos=fail_search,
            save_video=lambda video_url, save_dir="": "/saved/x.mp4",
            video_aspect="9:16",
            clip_duration=3,
            judge_candidate=_accept_judge,
            embedding_gate=None,
            generate_image=None,
            source="premise",
            premise_pool=broken_pool,
        )
    assert captured["pool"] == [], "fail-open：池失败不装填任何候选"
    assert results[0].clips == [], "fail-open：空手而归，不抛异常"
    assert any(
        "premise pool read failed" in str(call)
        for call in fake_logger.warning.call_args_list
    ), "池失败必须留下 warning 审计"


def test_match_segments_explicit_vector_cache_wins_over_gate_shared(monkeypatch, tmp_path):
    """显式 vector_cache 优先于 gate 的 _shared_cache（调用方契约）。"""
    footage_map = _footage_map(tmp_path, ids=("f0",))
    monkeypatch.setattr(
        kb_footage,
        "search_footage",
        lambda namespace, vector, group_keys, k: [
            {"group_key": "f0", "vector_id": "0", "score": 0.6}
        ],
    )
    monkeypatch.setattr(kb_footage, "embed_query", lambda text: [0.5])
    explicit_cache: dict = {}
    gate_shared: dict = {}

    class _Gate:
        _shared_cache = gate_shared

        @staticmethod
        def judge_candidate_embedding(*_a, **_k):
            return None

    captured: dict = {}
    with (
        patch.object(video_match.llm, "generate_response", side_effect=_funnel_llm),
        patch.object(video_match, "fetch_image_bytes", side_effect=_no_http),
        patch.object(video_match, "to_data_uri", side_effect=_no_http),
        patch.object(video_match, "coarse_rank", side_effect=_spy_coarse(captured)),
        patch.object(video_match.material_rerank, "is_rerank_enabled", return_value=False),
        patch.object(video_match.material_rerank, "_walk_limit", return_value=5),
        patch.object(video_match, "logger"),
    ):
        video_match.match_segments(
            segments=[{"index": 0, "text": "A panda eats.", "duration": 3.0}],
            video_subject="panda",
            search_videos=lambda *_a, **_k: [],
            save_video=lambda video_url, save_dir="": "/saved/x.mp4",
            video_aspect="9:16",
            clip_duration=3,
            judge_candidate=_accept_judge,
            embedding_gate=_Gate(),
            generate_image=None,
            vector_cache=explicit_cache,
            source="premise",
            premise_pool=lambda query: kb_footage.build_pool(
                _scope(), footage_map, query
            ),
        )
    assert captured["cache"] is explicit_cache, "显式传入的 vector_cache 不得被 gate 共享缓存改写"


# ---------------------------------------------------------------------------
# T6 线上形状：kb_scope + footage_map，绝无第三个字段
# ---------------------------------------------------------------------------


def test_task_video_request_wire_shape_matches_t6_contract():
    request = TaskVideoRequest(
        video_subject="Coffee",
        video_source="premise",
        owner_id="u42",
        kb_scope={"owner_id": "u42", "collection_ids": ["c1", "c2"]},
        footage_map={
            "f1": {
                "duration_s": 4.5,
                "t_start_ms": 0,
                "t_end_ms": 4500,
                "clip_path": "/shared/cache/m1/f1/clip.mp4",
                "thumb_path": "/shared/cache/m1/f1/thumb.jpg",
            }
        },
    )

    dumped = request.model_dump(warnings=False)

    assert dumped["kb_scope"] == {"owner_id": "u42", "collection_ids": ["c1", "c2"]}
    assert dumped["footage_map"] == {
        "f1": {
            "duration_s": 4.5,
            "t_start_ms": 0,
            "t_end_ms": 4500,
            "clip_path": "/shared/cache/m1/f1/clip.mp4",
            "thumb_path": "/shared/cache/m1/f1/thumb.jpg",
        }
    }
    assert "kb_collection_ids" not in dumped, "kb_collection_ids 是 VBL 侧 JobParams 字段，不是 wire 字段"
    assert set(dumped["footage_map"]["f1"]) == {
        "duration_s", "t_start_ms", "t_end_ms", "clip_path", "thumb_path"
    }, "footage_map 条目恰五个字段，不许多也不缺"


def test_task_video_request_defaults_have_no_premise_payload():
    request = TaskVideoRequest(video_subject="Coffee")

    assert request.kb_scope is None
    assert request.footage_map is None
