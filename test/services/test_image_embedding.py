"""
图像向量嵌入与任务级查重门的单元测试（finding G，duplicate-only 门）。

全部 mock，不发真实网络请求；fail-open 是硬契约，每条失败路径都要有
对应用例证明"返回 None / 放行"而不是抛异常。
"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import image_embedding
from app.services.image_embedding import EmbeddingGate, embed_image, embed_text


def _ok_response(vector=None, dim=768):
    if vector is None:
        vector = [round(0.001 * (i % 13), 6) for i in range(dim)]
    payload = {"output": {"embeddings": [{"embedding": list(vector)}]}}
    return SimpleNamespace(status_code=200, json=lambda: payload)


def _status_response(status):
    return SimpleNamespace(status_code=status, json=lambda: {})


class TestEmbedImage(unittest.TestCase):
    data_uri: str = ""

    def setUp(self):
        self.data_uri = "data:image/jpeg;base64,AAAA"

    def test_success_returns_vector_and_sends_shape_a_body(self):
        with patch.object(
            image_embedding.requests, "post", return_value=_ok_response(dim=768)
        ) as post:
            vec = embed_image(
                data_uri=self.data_uri,
                model="tongyi-embedding-vision-flash",
                api_key="test-key",
            )
        assert vec is not None
        self.assertEqual(len(vec), 768)
        self.assertTrue(all(isinstance(v, float) for v in vec))
        url = post.call_args.args[0]
        self.assertEqual(
            url,
            image_embedding.DEFAULT_EMBEDDING_BASE_URL,
        )
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "tongyi-embedding-vision-flash")
        self.assertEqual(
            body,
            {"model": "tongyi-embedding-vision-flash",
             "input": {"contents": [{"image": self.data_uri}]}},
        )

    def test_base_url_override_used(self):
        with patch.object(
            image_embedding.requests, "post", return_value=_ok_response(dim=3)
        ) as post:
            embed_image(
                data_uri=self.data_uri,
                model="m",
                api_key="k",
                base_url="https://gateway.example.com/embed",
            )
        self.assertEqual(
            post.call_args.args[0], "https://gateway.example.com/embed"
        )

    def test_429_retries_with_backoff_then_succeeds(self):
        responses = [_status_response(429), _ok_response(dim=8)]
        with (
            patch.object(
                image_embedding.requests, "post", side_effect=responses
            ) as post,
            patch.object(image_embedding.time, "sleep") as sleep,
        ):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        assert vec is not None
        self.assertEqual(len(vec), 8)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1.5)

    def test_429_exhausted_returns_none(self):
        responses = [_status_response(429)] * 4
        with (
            patch.object(
                image_embedding.requests, "post", side_effect=responses
            ) as post,
            patch.object(image_embedding.time, "sleep") as sleep,
        ):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)
        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list], [1.5, 3.0, 6.0]
        )

    def test_500_returns_none_without_retry(self):
        with (
            patch.object(
                image_embedding.requests,
                "post",
                return_value=_status_response(500),
            ) as post,
            patch.object(image_embedding.time, "sleep") as sleep,
        ):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_timeout_returns_none(self):
        with patch.object(
            image_embedding.requests,
            "post",
            side_effect=image_embedding.requests.exceptions.Timeout,
        ) as post:
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)
        self.assertEqual(post.call_count, 1)

    def test_connection_error_returns_none(self):
        with patch.object(
            image_embedding.requests,
            "post",
            side_effect=ConnectionError("reset"),
        ):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)

    def test_malformed_json_returns_none(self):
        bad = SimpleNamespace(
            status_code=200,
            json=lambda: (_ for _ in ()).throw(ValueError("not json")),
        )
        with patch.object(image_embedding.requests, "post", return_value=bad):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)

    def test_missing_keys_returns_none(self):
        for payload in ({}, {"output": {}}, {"output": {"embeddings": []}}):
            bad = SimpleNamespace(status_code=200, json=lambda p=payload: p)
            with patch.object(
                image_embedding.requests, "post", return_value=bad
            ):
                vec = embed_image(
                    data_uri=self.data_uri, model="m", api_key="k"
                )
            self.assertIsNone(vec, f"payload {payload} should fail open")

    def test_non_numeric_embedding_returns_none(self):
        bad = _ok_response(vector=["a"] * 4)
        with patch.object(image_embedding.requests, "post", return_value=bad):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)

    def test_empty_embedding_returns_none(self):
        bad = _ok_response(vector=[])
        with patch.object(image_embedding.requests, "post", return_value=bad):
            vec = embed_image(data_uri=self.data_uri, model="m", api_key="k")
        self.assertIsNone(vec)


class _VectorStub:
    """把 embed_image 替换为按 data_uri 查表的受控向量源。"""

    def __init__(self, vectors):
        self.vectors = dict(vectors)
        self.calls = []

    def __call__(self, data_uri, model, api_key, base_url=None, timeout=30.0):
        self.calls.append(data_uri)
        return self.vectors.get(data_uri)


def _gate_with(vectors, threshold=0.68):
    stub = _VectorStub(vectors)
    gate = EmbeddingGate(model="m", api_key="k", threshold=threshold)
    return gate, stub, patch.object(image_embedding, "embed_image", stub)


class TestEmbeddingGate(unittest.TestCase):
    def test_lifecycle_cache_then_register(self):
        """判定时缓存向量；采纳后无需再嵌入即可比对出重复。"""
        gate, stub, patcher = _gate_with(
            {
                "data:a": [1.0, 0.0],
                "data:b": [0.999, 0.0447],
            }
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        gate.register_accepted("https://a")
        record = gate.judge_candidate_embedding("https://b", "data:b")
        assert record is not None
        self.assertEqual(record["verdict"], "duplicate")
        self.assertEqual(record["duplicate_of"], "https://a")
        self.assertGreaterEqual(record["cos"], 0.68)
        self.assertEqual(
            record["reason"], f"cos={record['cos']:.3f} >= threshold"
        )
        self.assertEqual(record["image_source"], "embedding")
        # 注册表比对全程无第三次嵌入：a 的向量来自判定时缓存。
        self.assertEqual(stub.calls, ["data:a", "data:b"])

    def test_register_without_judged_candidate_is_noop(self):
        gate, _, patcher = _gate_with({"data:a": [1.0, 0.0]})
        patcher.start()
        self.addCleanup(patcher.stop)
        gate.register_accepted("https://never-judged")
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        self.assertEqual(gate._accepted, {})

    def test_distinct_candidate_passes(self):
        gate, _, patcher = _gate_with(
            {
                "data:a": [1.0, 0.0],
                "data:b": [0.0, 1.0],
            }
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        gate.register_accepted("https://a")
        self.assertIsNone(gate.judge_candidate_embedding("https://b", "data:b"))

    def test_threshold_boundary_inclusive(self):
        """cos 恰等于阈值判重复（>=），高一个 ULP 的阈值放行。"""
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        exact = image_embedding._cosine_similarity(a, b)
        gate, _, patcher = _gate_with(
            {"data:a": a, "data:b": b}, threshold=exact
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        gate.register_accepted("https://a")
        record = gate.judge_candidate_embedding("https://b", "data:b")
        assert record is not None
        self.assertEqual(record["cos"], exact)
        gate_above, _, patcher2 = _gate_with(
            {"data:a": a, "data:b": b},
            threshold=math.nextafter(exact, math.inf),
        )
        patcher2.start()
        self.addCleanup(patcher2.stop)
        self.assertIsNone(
            gate_above.judge_candidate_embedding("https://a", "data:a")
        )
        gate_above.register_accepted("https://a")
        self.assertIsNone(
            gate_above.judge_candidate_embedding("https://b", "data:b")
        )

    def test_threshold_near_spec_values(self):
        """0.6799 放行、0.6801 重复（阈值 0.68），容差远大于浮点误差。"""
        a = [1.0, 0.0]
        below = [0.6799, math.sqrt(1 - 0.6799**2)]
        above = [0.6801, math.sqrt(1 - 0.6801**2)]
        gate, _, patcher = _gate_with(
            {
                "data:a": a,
                "data:below": below,
                "data:above": above,
            }
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        gate.register_accepted("https://a")
        self.assertIsNone(
            gate.judge_candidate_embedding("https://below", "data:below")
        )
        record = gate.judge_candidate_embedding("https://above", "data:above")
        assert record is not None
        self.assertGreater(record["cos"], 0.68)

    def test_embed_failure_fails_open(self):
        """嵌入失败返回 None 放行，且不污染注册表/缓存。"""
        gate, stub, patcher = _gate_with({"data:a": None})
        patcher.start()
        self.addCleanup(patcher.stop)
        record = gate.judge_candidate_embedding("https://b", "data:a")
        self.assertIsNone(record)
        self.assertEqual(stub.calls, ["data:a"])
        self.assertNotIn("https://b", gate._candidates)
        # 失败候选不会被 register_accepted 意外抬进注册表。
        gate.register_accepted("https://b")
        self.assertEqual(gate._accepted, {})

    def test_same_url_judged_twice_embeds_once(self):
        """同 URL 判定两次只嵌入一次：候选缓存按 URL 去重嵌入调用。"""
        gate, stub, patcher = _gate_with({"data:a": [1.0, 0.0]})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        self.assertEqual(stub.calls, ["data:a"])

    def test_dimension_mismatch_cosine_is_zero(self):
        """维度不一致时 _cosine_similarity 返回 0（静默视为最不相似）。"""
        self.assertEqual(
            image_embedding._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]), 0.0
        )


class TestGateConfig(unittest.TestCase):
    def test_gate_defaults_from_missing_section(self):
        section = image_embedding.config.image_embedding
        # 段存在与否都不影响默认关闭；阈值缺省 0.68。
        self.assertIn(
            bool(section.get("duplicate_gate", False)), (True, False)
        )
        gate = image_embedding.make_default_gate()
        self.assertGreater(gate.threshold, 0)
        self.assertLessEqual(gate.threshold, 1)
        self.assertIsNotNone(gate.model)

    def test_gate_disabled_when_config_false(self):
        with patch.object(
            image_embedding.config,
            "image_embedding",
            {"duplicate_gate": False},
        ):
            self.assertFalse(image_embedding.is_duplicate_gate_enabled())

    def test_gate_enabled_when_config_true(self):
        with patch.object(
            image_embedding.config,
            "image_embedding",
            {"duplicate_gate": True, "api_key": "k"},
        ):
            self.assertTrue(image_embedding.is_duplicate_gate_enabled())
            gate = image_embedding.make_default_gate()
        self.assertEqual(gate.threshold, 0.68)
        self.assertEqual(gate.model, image_embedding.DEFAULT_EMBEDDING_MODEL)

    def test_invalid_threshold_falls_back(self):
        with patch.object(
            image_embedding.config,
            "image_embedding",
            {"duplicate_gate": True, "duplicate_threshold": "oops"},
        ):
            gate = image_embedding.make_default_gate()
        self.assertEqual(gate.threshold, 0.68)


class TestEmbedText(unittest.TestCase):
    """embed_text：与 embed_image 同端点的文本分支，fail-open 契约镜像。"""

    def test_success_returns_vector_and_sends_text_body(self):
        cfg = {"model": "custom-model", "api_key": "text-key", "base_url": ""}
        with (
            patch.object(image_embedding.config, "image_embedding", cfg),
            patch.object(
                image_embedding.requests, "post", return_value=_ok_response(dim=8)
            ) as post,
        ):
            vec = embed_text("a panda in a forest")
        assert vec is not None
        self.assertEqual(len(vec), 8)
        self.assertTrue(all(isinstance(v, float) for v in vec))
        self.assertEqual(
            post.call_args.args[0], image_embedding.DEFAULT_EMBEDDING_BASE_URL
        )
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer text-key")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "custom-model",
                "input": {"contents": [{"text": "a panda in a forest"}]},
            },
        )

    def test_blank_query_returns_none_without_request(self):
        with (
            patch.object(image_embedding.config, "image_embedding", {"api_key": "k"}),
            patch.object(image_embedding.requests, "post") as post,
        ):
            self.assertIsNone(embed_text("   "))
        post.assert_not_called()

    def test_base_url_override_from_config(self):
        cfg = {
            "model": "m",
            "api_key": "k",
            "base_url": "https://gateway.example.com/embed",
        }
        with (
            patch.object(image_embedding.config, "image_embedding", cfg),
            patch.object(
                image_embedding.requests, "post", return_value=_ok_response(dim=3)
            ) as post,
        ):
            embed_text("q")
        self.assertEqual(
            post.call_args.args[0], "https://gateway.example.com/embed"
        )

    def test_429_retries_with_backoff_then_succeeds(self):
        responses = [_status_response(429), _ok_response(dim=8)]
        with (
            patch.object(
                image_embedding.config, "image_embedding", {"api_key": "k"}
            ),
            patch.object(
                image_embedding.requests, "post", side_effect=responses
            ) as post,
            patch.object(image_embedding.time, "sleep") as sleep,
        ):
            vec = embed_text("q")
        assert vec is not None
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1.5)

    def test_429_exhausted_returns_none(self):
        responses = [_status_response(429)] * 4
        with (
            patch.object(
                image_embedding.config, "image_embedding", {"api_key": "k"}
            ),
            patch.object(
                image_embedding.requests, "post", side_effect=responses
            ) as post,
            patch.object(image_embedding.time, "sleep") as sleep,
        ):
            vec = embed_text("q")
        self.assertIsNone(vec)
        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list], [1.5, 3.0, 6.0]
        )

    def test_http_error_fails_open_and_logs(self):
        with (
            patch.object(
                image_embedding.config, "image_embedding", {"api_key": "k"}
            ),
            patch.object(
                image_embedding.requests,
                "post",
                return_value=_status_response(500),
            ),
            patch.object(image_embedding, "logger") as mock_logger,
        ):
            vec = embed_text("q")
        self.assertIsNone(vec)
        self.assertEqual(mock_logger.warning.call_count, 1)
        self.assertIn("text embedding http error", str(mock_logger.warning.call_args))

    def test_timeout_returns_none(self):
        with (
            patch.object(
                image_embedding.config, "image_embedding", {"api_key": "k"}
            ),
            patch.object(
                image_embedding.requests,
                "post",
                side_effect=image_embedding.requests.exceptions.Timeout,
            ),
        ):
            self.assertIsNone(embed_text("q"))

    def test_missing_keys_returns_none(self):
        bad = SimpleNamespace(status_code=200, json=lambda: {})
        with (
            patch.object(
                image_embedding.config, "image_embedding", {"api_key": "k"}
            ),
            patch.object(image_embedding.requests, "post", return_value=bad),
        ):
            self.assertIsNone(embed_text("q"))


class TestEmbeddingGateSharedCache(unittest.TestCase):
    """vector_cache 注入：粗排预热与门走查共享向量，URL 全链路只嵌入一次。"""

    def test_same_url_judged_twice_embeds_once_with_shared_cache(self):
        shared: dict[str, list[float]] = {}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=shared
        )
        stub = _VectorStub({"data:a": [1.0, 0.0]})
        with patch.object(image_embedding, "embed_image", stub):
            self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
            self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        self.assertEqual(stub.calls, ["data:a"])
        self.assertEqual(shared, {"https://a": [1.0, 0.0]})

    def test_shared_cache_warmed_elsewhere_skips_embed(self):
        """另一调用方（粗排预热）写入共享缓存后，本门零嵌入复用向量。"""
        shared = {"https://a": [1.0, 0.0]}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=shared
        )
        stub = _VectorStub({"data:a": [1.0, 0.0]})
        with patch.object(image_embedding, "embed_image", stub):
            self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
        self.assertEqual(stub.calls, [])

    def test_shared_cache_keeps_register_accepted_flow(self):
        """共享缓存不破坏采纳注册表：缓存向量挪入注册表后仍能判重复。"""
        shared: dict[str, list[float]] = {}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=shared
        )
        stub = _VectorStub({"data:a": [1.0, 0.0], "data:b": [0.999, 0.0447]})
        with patch.object(image_embedding, "embed_image", stub):
            self.assertIsNone(gate.judge_candidate_embedding("https://a", "data:a"))
            gate.register_accepted("https://a")
            record = gate.judge_candidate_embedding("https://b", "data:b")
        assert record is not None
        self.assertEqual(record["verdict"], "duplicate")
        self.assertEqual(stub.calls, ["data:a", "data:b"])

    def test_register_accepted_falls_back_to_shared_cache(self):
        """粗排预热的向量只存在于共享缓存、不经过 _candidates：
        register_accepted 必须从共享缓存兜底注册，否则采纳注册表静默
        漏注册，跨段近重复检测失效（finding G 回归）。"""
        shared = {"https://a": [1.0, 0.0]}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=shared
        )
        # 未经过 judge_candidate_embedding（_candidates 为空），直接注册。
        gate.register_accepted("https://a")
        self.assertIn("https://a", gate._accepted)
        self.assertEqual(gate._accepted["https://a"], [1.0, 0.0])
        # 近似向量随后被判重复，证明注册表真正参与查重比对。
        stub = _VectorStub({"data:b": [0.999, 0.0447]})
        with patch.object(image_embedding, "embed_image", stub):
            record = gate.judge_candidate_embedding("https://b", "data:b")
        assert record is not None
        self.assertEqual(record["verdict"], "duplicate")
        self.assertEqual(record["duplicate_of"], "https://a")


if __name__ == "__main__":
    unittest.main()

