"""
material_rerank 重排客户端的单元测试。

全部 mock，不发真实网络请求；fail-open 是硬契约，每条失败路径都要有
对应用例证明"原样返回 items"而不是抛异常。loguru 不走 stdlib logging 树，
caplog 看不到——挂临时 sink 收集原始消息文本（与 test_segment_material_quota
同款）。
"""

import base64
import sys
import tomllib as _tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger as loguru_logger

from app.config import config
from app.models.schema import MaterialInfo
from app.services import material_rerank
from app.services.material_rerank import (
    _credentials,
    _rerank_timeout,
    _walk_limit,
    is_rerank_enabled,
    rerank_candidates,
)


class _LogSink:
    """loguru 不走 stdlib logging 树，caplog 看不到——挂临时 sink 收集
    原始消息文本（与 test_segment_material_quota 同款）。"""

    def __init__(self):
        self.messages = []
        self._handler_id = None

    def __enter__(self):
        self._handler_id = loguru_logger.add(
            lambda message: self.messages.append(message.record["message"]),
            level="INFO",
        )
        return self

    def __exit__(self, *exc_info):
        loguru_logger.remove(self._handler_id)
        return False

    @property
    def text(self):
        return "\n".join(self.messages)


def _item(asset_id, thumbnail=None):
    source_info = None
    if asset_id is not None:
        source_info = {"asset_id": asset_id}
        if thumbnail is not None:
            source_info["thumbnail_url"] = thumbnail
    return MaterialInfo(
        provider="pexels",
        url=f"https://cdn.example.com/{asset_id}.mp4",
        duration=5,
        source_info=source_info,
    )


def _asset_ids(items):
    """按模块同样的防御式方式读回 asset_id，供断言比对顺序。"""
    return [str((item.source_info or {}).get("asset_id") or "") for item in items]


def _ok_response(scores):
    results = [{"index": index, "relevance_score": score} for index, score in scores]
    return SimpleNamespace(status_code=200, json=lambda: {"results": results})


def _status_response(status):
    return SimpleNamespace(status_code=status, json=lambda: {})


def _isolate_credentials(monkeypatch):
    """隔离 [vlm] 真实凭据与 [material_rerank] 覆盖项，测试结果不依赖
    开发机 config.toml；默认给一组测试专用凭据让请求路径可达。"""
    monkeypatch.setitem(config.material_rerank, "enabled", True)
    monkeypatch.setitem(config.material_rerank, "model", "Qwen/Qwen3-VL-Reranker-8B")
    monkeypatch.setitem(config.material_rerank, "timeout", 120)
    monkeypatch.setitem(config.material_rerank, "api_key", "test-rerank-key")
    monkeypatch.setitem(
        config.material_rerank, "base_url", "https://rerank.example.com/v1"
    )
    monkeypatch.setitem(config.vlm, "api_key", "")
    monkeypatch.setitem(config.vlm, "base_url", "")


# ---------------------------------------------------------------- 默认值与开关


def test_config_defaults():
    assert config.material_rerank["enabled"] is True
    assert config.material_rerank["model"] == "Qwen/Qwen3-VL-Reranker-8B"
    assert config.material_rerank["vlm_walk_limit"] == 10
    assert "top_n" not in config.material_rerank
    assert config.material_rerank["timeout"] == 120
    assert config.material_rerank["api_key"] == ""
    assert config.material_rerank["base_url"] == ""


def test_is_rerank_enabled_true_by_default(monkeypatch):
    _isolate_credentials(monkeypatch)
    assert is_rerank_enabled() is True


def test_invalid_walk_limit_falls_back_to_default(monkeypatch):
    monkeypatch.setitem(config.material_rerank, "vlm_walk_limit", "abc")
    assert _walk_limit() == 10
    monkeypatch.setitem(config.material_rerank, "vlm_walk_limit", 0)
    assert _walk_limit() == 10
    monkeypatch.setitem(config.material_rerank, "vlm_walk_limit", 3)
    assert _walk_limit() == 3


def test_rerank_timeout_from_config(monkeypatch):
    monkeypatch.setitem(config.material_rerank, "timeout", 42)
    assert _rerank_timeout() == 42.0
    monkeypatch.setitem(config.material_rerank, "timeout", "bogus")
    assert _rerank_timeout() == 120.0


# ---------------------------------------------------------------- 凭据


def test_credentials_prefer_material_rerank_override(monkeypatch):
    _isolate_credentials(monkeypatch)
    monkeypatch.setitem(config.material_rerank, "api_key", "override-key")
    monkeypatch.setitem(
        config.material_rerank, "base_url", "https://override.example.com/v1"
    )
    monkeypatch.setitem(config.vlm, "api_key", "vlm-key")
    monkeypatch.setitem(config.vlm, "base_url", "https://vlm.example.com/v1")
    assert _credentials() == ("override-key", "https://override.example.com/v1")


def test_credentials_fall_back_to_vlm(monkeypatch):
    _isolate_credentials(monkeypatch)
    monkeypatch.setitem(config.material_rerank, "api_key", "")
    monkeypatch.setitem(config.material_rerank, "base_url", "")
    monkeypatch.setitem(config.vlm, "api_key", "vlm-key")
    monkeypatch.setitem(config.vlm, "base_url", "https://vlm.example.com/v1")
    assert _credentials() == ("vlm-key", "https://vlm.example.com/v1")


def test_no_credentials_skip_call(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    monkeypatch.setitem(config.material_rerank, "api_key", "")
    monkeypatch.setitem(config.vlm, "api_key", "")
    with (
        _LogSink() as sink,
        patch.object(material_rerank.requests, "post") as post,
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    post.assert_not_called()
    assert "material rerank skipped, no credentials: query='panda'" in sink.text


# ---------------------------------------------------------------- 请求形状


def test_request_body_headers_and_timeout_shape(monkeypatch):
    items = [
        _item("a1", "https://img.example.com/a1.jpg"),
        _item("a2", "https://img.example.com/a2.jpg"),
    ]
    _isolate_credentials(monkeypatch)
    with patch.object(
        material_rerank.requests,
        "post",
        return_value=_ok_response([(0, 0.9), (1, 0.1)]),
    ) as post:
        rerank_candidates("panda", items)
    assert post.call_args.args[0] == "https://rerank.example.com/v1/rerank"
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-rerank-key"
    assert headers["Content-Type"] == "application/json"
    body = post.call_args.kwargs["json"]
    assert body == {
        "model": "Qwen/Qwen3-VL-Reranker-8B",
        "query": "panda",
        "documents": [
            {"image": "https://img.example.com/a1.jpg"},
            {"image": "https://img.example.com/a2.jpg"},
        ],
        "top_n": 2,
        "return_documents": False,
    }
    # 读超时来自 [material_rerank] timeout=120，连接超时固定 30s。
    assert post.call_args.kwargs["timeout"] == (30, 120)


# ---------------------------------------------------------------- 排序与全量返回


def test_orders_by_score_desc_with_stable_ties(monkeypatch):
    items = [
        _item("a1", "https://img.example.com/a1.jpg"),
        _item("a2", "https://img.example.com/a2.jpg"),
        _item("a3", "https://img.example.com/a3.jpg"),
        _item("a4", "https://img.example.com/a4.jpg"),
        _item("a5", "https://img.example.com/a5.jpg"),
    ]
    _isolate_credentials(monkeypatch)
    # 返回顺序故意打乱；a1/a4 同分 0.5，须保持原相对顺序（a1 在前）。
    response = _ok_response([(2, 0.9), (0, 0.5), (3, 0.5), (1, 0.1), (4, 0.3)])
    with patch.object(material_rerank.requests, "post", return_value=response):
        result = rerank_candidates("panda", items)
    # 全量返回：5 个文档打分 → 5 个文档按分数降序原样返回，不截断。
    assert len(result) == 5
    assert sorted(_asset_ids(result)) == ["a1", "a2", "a3", "a4", "a5"]
    assert _asset_ids(result) == ["a3", "a1", "a4", "a5", "a2"]


def test_unrankable_tail_after_full_ranked_order(monkeypatch):
    r1, u1, r2, u2, r3, r4 = (
        _item("r1", "https://img.example.com/r1.jpg"),
        _item("u1"),
        _item("r2", "https://img.example.com/r2.jpg"),
        _item("u2", None),
        _item("r3", "https://img.example.com/r3.jpg"),
        _item("r4", "https://img.example.com/r4.jpg"),
    )
    items = [r1, u1, r2, u2, r3, r4]
    _isolate_credentials(monkeypatch)
    response = _ok_response(
        [(3, 0.9), (0, 0.8), (2, 0.3), (1, 0.1)]  # rankable 的 index：r4, r1, r3, r2
    )
    with patch.object(material_rerank.requests, "post", return_value=response):
        result = rerank_candidates("panda", items)
    # 不再截断：全部可重排候选按降序在前，无缩略图候选按原序垫底；
    # 截断（walk_limit）是调用方的职责。
    assert _asset_ids(result) == ["r4", "r1", "r3", "r2", "u1", "u2"]


# ---------------------------------------------------------------- 成功路径审计日志


def test_success_log_lines_exact_format(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg"), _item("a2", "https://img.example.com/a2.jpg")]
    _isolate_credentials(monkeypatch)
    response = _ok_response([(1, 0.7), (0, 0.2)])
    with (
        _LogSink() as sink,
        patch.object(material_rerank.requests, "post", return_value=response),
    ):
        rerank_candidates("panda", items)
    assert "material rerank score: asset_id=a2, score=0.7, rank=1" in sink.text
    assert "material rerank score: asset_id=a1, score=0.2, rank=2" in sink.text
    assert (
        "material rerank selected: query='panda', ranked=2, fallback=False"
        in sink.text
    )


def test_no_secret_leak_in_logs(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    monkeypatch.setitem(config.material_rerank, "api_key", "sk-super-secret-123")
    with (
        _LogSink() as sink,
        patch.object(
            material_rerank.requests,
            "post",
            side_effect=material_rerank.requests.exceptions.ConnectionError("boom"),
        ),
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    assert "sk-super-secret-123" not in sink.text
    assert "material rerank failed, fail-open: query='panda'" in sink.text


# ---------------------------------------------------------------- 失败 fail-open


def test_fail_open_on_connection_error(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    with (
        _LogSink() as sink,
        patch.object(
            material_rerank.requests,
            "post",
            side_effect=material_rerank.requests.exceptions.ConnectionError("boom"),
        ) as post,
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    # 直连 + config.proxy 各一次，无更多重试。
    assert post.call_count == 2
    assert (
        "material rerank failed, fail-open: query='panda', error=ConnectionError"
        in sink.text
    )


def test_fail_open_on_429_exhausted(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    responses = [_status_response(429)] * 4
    with (
        _LogSink() as sink,
        patch.object(
            material_rerank.requests, "post", side_effect=responses
        ) as post,
        patch.object(material_rerank.time, "sleep") as sleep,
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    assert post.call_count == 4
    assert sleep.call_args_list == [call(1.5), call(3.0), call(6.0)]
    assert "material rerank failed, fail-open: query='panda'" in sink.text


def test_429_retry_then_succeeds(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg"), _item("a2", "https://img.example.com/a2.jpg")]
    _isolate_credentials(monkeypatch)
    responses = [_status_response(429), _ok_response([(1, 0.6), (0, 0.4)])]
    with (
        patch.object(
            material_rerank.requests, "post", side_effect=responses
        ) as post,
        patch.object(material_rerank.time, "sleep") as sleep,
    ):
        result = rerank_candidates("panda", items)
    assert _asset_ids(result) == ["a2", "a1"]
    assert post.call_count == 2
    sleep.assert_called_once_with(1.5)


def test_fail_open_on_400_then_400_again(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    stub_payload = b"\xff\xd8stub"
    responses = [_status_response(400), _status_response(400)]
    with (
        _LogSink() as sink,
        patch.object(
            material_rerank.requests, "post", side_effect=responses
        ) as post,
        patch.object(
            material_rerank,
            "download_thumbnail_bytes",
            return_value=(stub_payload, (16, 16)),
        ),
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    assert post.call_count == 2
    assert "material rerank failed, fail-open: query='panda'" in sink.text


def test_base64_fallback_after_400(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg"), _item("a2", "https://img.example.com/a2.jpg")]
    _isolate_credentials(monkeypatch)
    stub_payload = b"\xff\xd8stub"
    expected_uri = "data:image/jpeg;base64," + base64.b64encode(stub_payload).decode(
        "ascii"
    )
    responses = [_status_response(400), _ok_response([(1, 0.8), (0, 0.3)])]
    with (
        _LogSink() as sink,
        patch.object(
            material_rerank.requests, "post", side_effect=responses
        ) as post,
        patch.object(
            material_rerank,
            "download_thumbnail_bytes",
            return_value=(stub_payload, (16, 16)),
        ) as download,
    ):
        result = rerank_candidates("panda", items)
    # 首调用 URL 文档，400 后改发 base64 data URI 文档重试一次。
    assert post.call_count == 2
    assert download.call_count == 2
    first_body = post.call_args_list[0].kwargs["json"]
    second_body = post.call_args_list[1].kwargs["json"]
    assert first_body["documents"] == [
        {"image": "https://img.example.com/a1.jpg"},
        {"image": "https://img.example.com/a2.jpg"},
    ]
    assert second_body["documents"] == [{"image": expected_uri}] * 2
    assert all(
        document["image"].startswith("data:") for document in second_body["documents"]
    )
    assert _asset_ids(result) == ["a2", "a1"]
    assert "material rerank failed, fail-open" not in sink.text
    assert (
        "material rerank selected: query='panda', ranked=2, fallback=True"
        in sink.text
    )


def test_fail_open_on_malformed_json(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    broken = SimpleNamespace(status_code=200, json=lambda: (_ for _ in ()).throw(ValueError("bad json")))
    with (
        _LogSink() as sink,
        patch.object(material_rerank.requests, "post", return_value=broken),
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    assert (
        "material rerank failed, fail-open: query='panda', error=ValueError"
        in sink.text
    )


def test_fail_open_on_missing_results(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    response = SimpleNamespace(status_code=200, json=lambda: {"id": "resp-1"})
    with (
        _LogSink() as sink,
        patch.object(material_rerank.requests, "post", return_value=response),
    ):
        result = rerank_candidates("panda", items)
    assert result == items
    assert (
        "material rerank failed, fail-open: query='panda', error=KeyError" in sink.text
    )


# ---------------------------------------------------------------- 直通路径


def test_blank_term_passthrough(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    with patch.object(material_rerank.requests, "post") as post:
        result = rerank_candidates("   ", items)
    assert result == items
    post.assert_not_called()


def test_disabled_toggle_passthrough(monkeypatch):
    items = [_item("a1", "https://img.example.com/a1.jpg")]
    _isolate_credentials(monkeypatch)
    monkeypatch.setitem(config.material_rerank, "enabled", False)
    with patch.object(material_rerank.requests, "post") as post:
        result = rerank_candidates("panda", items)
    assert result == items
    post.assert_not_called()


def test_no_rankable_items_passthrough(monkeypatch):
    items = [_item("a1"), _item("a2", ""), _item("a3", "   ")]
    _isolate_credentials(monkeypatch)
    with patch.object(material_rerank.requests, "post") as post:
        result = rerank_candidates("panda", items)
    assert result == items
    post.assert_not_called()


# ---------------------------------------------------------------- config.example.toml 粗筛键清除验证


def _example_config():
    config_path = Path(__file__).resolve().parents[2] / "config.example.toml"
    return _tomllib.loads(config_path.read_text(encoding="utf-8"))


def test_example_config_image_embedding_no_coarse_keys():
    example = _example_config()
    ie = example["image_embedding"]
    assert "duplicate_gate" in ie
    assert "coarse_filter" not in ie
    assert "coarse_threshold" not in ie


def test_example_config_material_rerank_section():
    example = _example_config()
    mr = example["material_rerank"]
    for key in ("enabled", "model", "vlm_walk_limit", "timeout", "api_key", "base_url"):
        assert key in mr, f"[material_rerank] missing key: {key}"


def test_live_parser_defaults_no_coarse_in_image_embedding():
    raw = dict(config.image_embedding)
    user_only_keys = set(raw.keys()) - {
        "model", "api_key", "base_url", "duplicate_gate", "duplicate_threshold",
        "provider_name",
    }
    assert user_only_keys <= {"coarse_filter", "coarse_threshold"}, (
        f"unexpected extra keys in live config.image_embedding: {user_only_keys}"
    )
    config_path = Path(__file__).resolve().parents[2] / "app" / "config" / "config.py"
    src = config_path.read_text(encoding="utf-8")
    assert '"coarse_filter"' not in src
    assert '"coarse_threshold"' not in src
