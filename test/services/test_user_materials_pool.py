"""todo 7 测试：owner_id + stock|premise|mixed 源与 premise 候选池（red-first）。

覆盖：pool -> MaterialInfo 映射（cap/空 owner/缺文件）、premise 与 mixed 缺
owner_id 失败、premise:// 缩略图与拷贝全本地（任何 HTTP 调用即炸）、
match_segments premise builder 路由、stock/legacy 路由不变、无 provider key
的 premise 不再 no-provider 硬失败。全部桩 + tmp_path registry。
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo, TaskVideoRequest
from app.services import task, user_materials, video_match, vlm_judge

OWNER = "owner-1"
CLIP_BYTES = b"clip-bytes-0"
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg"


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(user_materials, "_storage_root", lambda create=False: str(tmp_path))
    monkeypatch.setattr(user_materials, "_conn", None)
    monkeypatch.setattr(user_materials, "_last_sweep_monotonic", float("inf"))
    monkeypatch.setattr(config, "user_materials", {"max_age_days": 15, "max_candidates": 500}, raising=False)
    yield tmp_path
    if user_materials._conn is not None:
        user_materials._conn.close()


def _seed_materials(count=2, owner=OWNER, material_id="m1", rev=0):
    user_materials.begin_push(owner, material_id, rev)
    for idx in range(count):
        with open(user_materials.put_file(owner, material_id, rev, idx, "clip"), "wb") as f:
            f.write(CLIP_BYTES)
        with open(user_materials.put_file(owner, material_id, rev, idx, "thumb"), "wb") as f:
            f.write(JPEG_BYTES)
    manifest = [
        {"idx": idx, "t_start": idx * 5.0, "t_end": idx * 5.0 + 5.0, "duration": 5.0}
        for idx in range(count)
    ]
    user_materials.complete(owner, material_id, rev, manifest)


def _no_http(*_a, **_k):
    raise AssertionError("premise bytes must never go over HTTP (Metis #10)")


def _params(source, owner_id):
    return TaskVideoRequest(
        video_subject="s", video_source=source, owner_id=owner_id, video_clip_duration=3
    )


# ---------------------------------------------------------------------------
# pool -> MaterialInfo 映射
# ---------------------------------------------------------------------------

def test_premise_pool_maps_registry_rows_to_material_infos(registry):
    _seed_materials(count=3)
    items = task.premise_pool_materials(OWNER)
    assert [i.url for i in items] == [f"premise://{OWNER}/m1/{idx}" for idx in range(3)]
    assert all(i.provider == "user_material" for i in items)
    assert [i.duration for i in items] == [5, 5, 5]
    first = items[0].source_info
    assert first is not None
    assert first["asset_id"] == "m1/0"
    assert first["search_term"] == ""
    assert first["thumbnail_url"] == ""
    assert first["rendition"]["id"] == "m1/0"
    assert first["rendition"]["t_start"] == 0.0 and first["rendition"]["t_end"] == 5.0


def test_premise_pool_respects_max_candidates_cap(registry, monkeypatch):
    _seed_materials(count=4)
    monkeypatch.setattr(config, "user_materials", {"max_candidates": 2}, raising=False)
    items = task.premise_pool_materials(OWNER)
    assert [i.url for i in items] == [f"premise://{OWNER}/m1/0", f"premise://{OWNER}/m1/1"]


def test_premise_pool_empty_owner_returns_empty(registry):
    assert task.premise_pool_materials("nobody") == []


def test_premise_pool_survives_missing_files_on_disk(registry):
    _seed_materials(count=2)
    shutil.rmtree(os.path.join(str(registry), OWNER, "m1"))
    items = task.premise_pool_materials(OWNER)
    assert len(items) == 2, "registry 行照常返回，缺文件由 builder/save 各自降级"


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
# 搜索路由：premise 走 pool 且跳过 provider assert；stock/legacy 原样
# ---------------------------------------------------------------------------

def test_premise_search_bypasses_stock_providers_and_repeats_page1(registry):
    _seed_materials(count=2)
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
        out = search("ignored keyword", minimum_duration=5, video_aspect="9:16", page=1)
        assert [i.url for i in out] == [f"premise://{OWNER}/m1/0", f"premise://{OWNER}/m1/1"]
        assert search("anything", page=2) == [], "premise 一次返回全量，翻页为空"
        task._preflight_search_provider(_params("premise", OWNER))
    assert called == {"assert": 0, "multi": 0}, "premise 绝不触在线供应商路径"


def test_premise_pool_read_error_propagates_to_task(registry, monkeypatch):
    """registry DB 故障不做静默降级：异常上抛由任务失败路径处理（与 stock 搜索同构）。"""
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(user_materials, "pool", boom)
    search = task._make_search_videos(_params("premise", OWNER))
    with pytest.raises(RuntimeError, match="db down"):
        search("kw", page=1)


def test_stock_and_legacy_keep_multi_provider_search():
    calls = []

    def fake_multi(**kw):
        calls.append(kw)
        return [MaterialInfo(provider="pexels", url="https://x/1.mp4", duration=10)]

    with patch.object(task.material, "search_videos_multi_provider", fake_multi):
        for source in ("stock", "pexels", ""):
            search = task._make_search_videos(_params(source, None))
            out = search("term", minimum_duration=4, video_aspect="9:16", page=2)
            assert [i.url for i in out] == ["https://x/1.mp4"]
    assert [c["page"] for c in calls] == [2, 2, 2]


def test_preflight_asserts_for_stock_when_vlm_enabled():
    with (
        patch.object(task.vlm_judge, "is_enabled", lambda: True),
        patch.object(task.material, "assert_search_provider_available", side_effect=AssertionError("no provider")),
    ):
        with pytest.raises(AssertionError, match="no provider"):
            task._preflight_search_provider(_params("pexels", None))


def test_preflight_skipped_for_mixed_and_premise():
    with (
        patch.object(task.vlm_judge, "is_enabled", lambda: True),
        patch.object(task.material, "assert_search_provider_available", side_effect=AssertionError("no provider")),
    ):
        task._preflight_search_provider(_params("premise", "u1"))
        task._preflight_search_provider(_params("mixed", "u1"))


# ---------------------------------------------------------------------------
# save_video：premise:// 本地拷贝，其余委派 material.save_video
# ---------------------------------------------------------------------------

def test_save_video_premise_copies_local_file(registry, tmp_path):
    _seed_materials(count=1)
    save_dir = tmp_path / "task-materials"
    save_dir.mkdir()
    cb = task._make_save_video(_params("premise", OWNER), str(save_dir))
    out = cb(f"premise://{OWNER}/m1/0")
    assert out == str(save_dir / "premise-owner-1-m1-0.mp4")
    assert Path(out).read_bytes() == CLIP_BYTES


def test_save_video_premise_missing_file_raises(registry, tmp_path):
    _seed_materials(count=1)
    shutil.rmtree(os.path.join(str(registry), OWNER, "m1"))
    cb = task._make_save_video(_params("premise", OWNER), str(tmp_path))
    with pytest.raises(FileNotFoundError):
        cb(f"premise://{OWNER}/m1/0")


def test_save_video_non_premise_delegates_to_material():
    seen = {}

    def fake_save(video_url, save_dir=""):
        seen["args"] = (video_url, save_dir)
        return "/saved/x.mp4"

    with patch.object(task.material, "save_video", fake_save):
        cb = task._make_save_video(_params("stock", None), "/dir")
        assert cb("https://x/1.mp4") == "/saved/x.mp4"
    assert seen["args"] == ("https://x/1.mp4", "/dir")


# ---------------------------------------------------------------------------
# 候选 builder：data_uri 本地读 + url 身份
# ---------------------------------------------------------------------------

def _premise_item(idx=0):
    return MaterialInfo(
        provider="user_material",
        url=f"premise://{OWNER}/m1/{idx}",
        duration=5,
        source_info={"asset_id": f"m1/{idx}", "thumbnail_url": ""},
    )


def test_premise_candidate_builder_uses_local_thumb(registry):
    _seed_materials(count=1)
    item = _premise_item()
    with patch.object(video_match, "download_thumbnail_bytes", side_effect=_no_http):
        cand = video_match._premise_candidate_from_item(item, "")
    assert cand["url"] == f"premise://{OWNER}/m1/0"
    assert cand["data_uri"] == "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode()
    assert cand["item"] is item


def test_premise_candidate_builder_empty_when_thumb_missing(registry):
    _seed_materials(count=1)
    os.remove(os.path.join(str(registry), OWNER, "m1", "0", "0.jpg"))
    with patch.object(video_match, "download_thumbnail_bytes", side_effect=_no_http):
        cand = video_match._premise_candidate_from_item(_premise_item(), "")
    assert cand["data_uri"] == "", "缩略图缺失按既有空 URI 契约沉底"


def test_match_segments_builds_premise_candidates_locally(registry):
    _seed_materials(count=2)
    items = task.premise_pool_materials(OWNER)

    def fake_llm(prompt):
        return json.dumps({"terms": ["terma"], "coarse_query": "A broad scene.", "fine_query": "fp"})

    captured = {}

    def spy_coarse(pool, coarse_query, vector_cache, embedding_gate):
        captured["pool"] = list(pool)
        return [], []

    with (
        patch.object(video_match.llm, "generate_response", side_effect=fake_llm),
        patch.object(video_match, "download_thumbnail_bytes", side_effect=_no_http),
        patch.object(video_match, "to_data_uri", side_effect=_no_http),
        patch.object(video_match, "coarse_rank", spy_coarse),
        patch.object(video_match.material_rerank, "is_rerank_enabled", return_value=False),
        patch.object(video_match.material_rerank, "_walk_limit", return_value=0),
        patch.object(video_match, "logger"),
    ):
        video_match.match_segments(
            segments=[{"index": 0, "text": "A panda eats.", "duration": 3.0}],
            video_subject="panda",
            search_videos=lambda search_term, page=1, **_k: [] if page > 1 else list(items),
            save_video=lambda url, save_dir="": "/saved/x.mp4",
            video_aspect="9:16",
            clip_duration=3,
            judge_candidate=lambda **kw: _verdict(),
            embedding_gate=None,
            generate_image=None,
        )
    pool = captured["pool"]
    # match_segments 对池做 random.shuffle：断言按身份集合而非顺序（flake 纪律）。
    assert {c["url"] for c in pool} == {
        f"premise://{OWNER}/m1/0",
        f"premise://{OWNER}/m1/1",
    }
    assert all(c["data_uri"].startswith("data:image/jpeg;base64,") for c in pool)


def _verdict():
    return {"verdict": "irrelevant", "reason": "x", "image_source": "poster", "asset_id": "stub"}


# ---------------------------------------------------------------------------
# vlm_judge 首帧兜底：premise url 走本地 thumb 字节，绝不 requests.get
# ---------------------------------------------------------------------------

def test_vlm_judge_premise_first_frame_local_fallback(registry):
    _seed_materials(count=1)
    judge = vlm_judge.make_default_judge(embedding_gate=None)
    expected = "data:image/jpeg;base64," + base64.b64encode(JPEG_BYTES).decode()
    with (
        patch.object(vlm_judge, "download_thumbnail_bytes", side_effect=_no_http),
        patch.object(vlm_judge, "_download_candidate_to_temp", side_effect=_no_http),
        patch.object(vlm_judge, "extract_first_frame_jpeg", side_effect=_no_http),
        patch.object(vlm_judge, "judge_image", return_value=("relevant", "on topic", 1)) as ji,
    ):
        record = judge(_premise_item(), "seg text", "kw")
    assert ji.call_args.kwargs["image_data_uri"] == expected, "VLM 输入必须是本地 jpeg 字节 data URI"
    assert record["image_source"] == "premise_local"
    assert record["verdict"] == "relevant"


# ---------------------------------------------------------------------------
# todo 8 接线：pipeline 把 source + premise_pool 交给 match_segments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["premise", "mixed", "stock", "pexels"])
def test_make_premise_pool_only_for_premise_sources(source, registry):
    _seed_materials(count=2)
    pool_cb = task._make_premise_pool(_params(source, OWNER))
    if source in ("premise", "mixed"):
        assert callable(pool_cb)
        assert [i.url for i in pool_cb()] == [f"premise://{OWNER}/m1/0", f"premise://{OWNER}/m1/1"]
    else:
        assert pool_cb is None


def test_pipeline_passes_source_and_premise_pool_to_match_segments(registry):
    _seed_materials(count=2)
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
            "t-wire8", _params("premise", OWNER), "hello world", stop_at="materials"
        )

    assert captured.get("source") == "premise"
    premise_pool = captured.get("premise_pool")
    assert callable(premise_pool)
    assert [i.url for i in premise_pool()] == [f"premise://{OWNER}/m1/0", f"premise://{OWNER}/m1/1"]
    assert callable(captured.get("search_videos")), "stock 回调照常在场（pass B / stock 单遍共用）"
