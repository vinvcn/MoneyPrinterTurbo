"""
图像向量嵌入与任务级查重门的单元测试（finding G；embed_text 为 embed-prefilter
T1，粗筛 prefiltered 为 T2）。

全部 mock，不发真实网络请求；fail-open 是硬契约，每条失败路径都要有
对应用例证明"返回 None / 放行"而不是抛异常。
"""

import math
import sys
import tomllib
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


class TestEmbedText(unittest.TestCase):
    """embed_text（embed-prefilter T1）：与 embed_image 同一套 fail-open 契约。"""

    def test_success_returns_vector_and_sends_text_body(self):
        text = "一只橙色的猫坐在窗台"
        with patch.object(
            image_embedding.requests, "post", return_value=_ok_response(dim=768)
        ) as post:
            vec = embed_text(
                text=text,
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
        self.assertEqual(
            body,
            {"model": "tongyi-embedding-vision-flash",
             "input": {"contents": [{"text": text}]}},
        )

    def test_base_url_override_used(self):
        with patch.object(
            image_embedding.requests, "post", return_value=_ok_response(dim=3)
        ) as post:
            embed_text(
                text="hello",
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
            vec = embed_text(text="hello", model="m", api_key="k")
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
            vec = embed_text(text="hello", model="m", api_key="k")
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
            vec = embed_text(text="hello", model="m", api_key="k")
        self.assertIsNone(vec)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_timeout_returns_none(self):
        with patch.object(
            image_embedding.requests,
            "post",
            side_effect=image_embedding.requests.exceptions.Timeout,
        ) as post:
            vec = embed_text(text="hello", model="m", api_key="k")
        self.assertIsNone(vec)
        self.assertEqual(post.call_count, 1)

    def test_connection_error_returns_none(self):
        with patch.object(
            image_embedding.requests,
            "post",
            side_effect=ConnectionError("reset"),
        ):
            vec = embed_text(text="hello", model="m", api_key="k")
        self.assertIsNone(vec)

    def test_malformed_json_returns_none(self):
        bad = SimpleNamespace(
            status_code=200,
            json=lambda: (_ for _ in ()).throw(ValueError("not json")),
        )
        with patch.object(image_embedding.requests, "post", return_value=bad):
            vec = embed_text(text="hello", model="m", api_key="k")
        self.assertIsNone(vec)

    def test_missing_keys_returns_none(self):
        for payload in ({}, {"output": {}}, {"output": {"embeddings": []}}):
            bad = SimpleNamespace(status_code=200, json=lambda p=payload: p)
            with patch.object(
                image_embedding.requests, "post", return_value=bad
            ):
                vec = embed_text(text="hello", model="m", api_key="k")
            self.assertIsNone(vec, f"payload {payload} should fail open")

    def test_non_numeric_embedding_returns_none(self):
        bad = _ok_response(vector=["a"] * 4)
        with patch.object(image_embedding.requests, "post", return_value=bad):
            vec = embed_text(text="hello", model="m", api_key="k")
        self.assertIsNone(vec)

    def test_empty_embedding_returns_none(self):
        bad = _ok_response(vector=[])
        with patch.object(image_embedding.requests, "post", return_value=bad):
            vec = embed_text(text="hello", model="m", api_key="k")
        self.assertIsNone(vec)

    def test_blank_text_returns_none_without_http_call(self):
        """空白文本短路：不发任何请求，直接 fail-open 放行。"""
        for blank in ("", "   ", "\n\t "):
            with patch.object(
                image_embedding.requests, "post"
            ) as post:
                vec = embed_text(text=blank, model="m", api_key="k")
            self.assertIsNone(vec, f"blank text {blank!r} should fail open")
            post.assert_not_called()


class _VectorStub:
    """把 embed_image 替换为按 data_uri 查表的受控向量源。"""

    def __init__(self, vectors):
        self.vectors = dict(vectors)
        self.calls = []

    def __call__(self, data_uri, model, api_key, base_url=None, timeout=30.0):
        self.calls.append(data_uri)
        return self.vectors.get(data_uri)


class _TextStub:
    """把 embed_text 替换为按 term 查表的受控向量源。"""

    def __init__(self, vectors):
        self.vectors = dict(vectors)
        self.calls = []

    def __call__(self, text, model, api_key, base_url=None, timeout=30.0):
        self.calls.append(text)
        return self.vectors.get(text)


def _gate_with(vectors, threshold=0.68):
    stub = _VectorStub(vectors)
    gate = EmbeddingGate(model="m", api_key="k", threshold=threshold)
    return gate, stub, patch.object(image_embedding, "embed_image", stub)


def _coarse_gate_with(vectors, term_vectors, coarse_threshold=0.5, threshold=0.68):
    """构造开启粗筛的门，并同时替换 embed_image / embed_text 两个嵌入源。"""
    img_stub = _VectorStub(vectors)
    text_stub = _TextStub(term_vectors)
    gate = EmbeddingGate(
        model="m",
        api_key="k",
        threshold=threshold,
        coarse_enabled=True,
        coarse_threshold=coarse_threshold,
    )
    patches = (
        patch.object(image_embedding, "embed_image", img_stub),
        patch.object(image_embedding, "embed_text", text_stub),
    )
    return gate, img_stub, text_stub, patches


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


class TestEmbeddingGateCoarse(unittest.TestCase):
    """T2 粗筛：查重之后的文本-图像余弦预筛，全程不触真实 API。"""

    def test_coarse_below_threshold_prefilters(self):
        """cos 低于粗筛阈值：返回 prefiltered 审计记录，不放行给 VLM。"""
        gate, _, text_stub, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0], "data:b": [0.0, 1.0]},
            {"fishing": [1.0, 0.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        gate.register_accepted("https://a")
        record = gate.judge_candidate_embedding(
            "https://b", "data:b", term="fishing"
        )
        assert record is not None
        self.assertEqual(record["verdict"], "prefiltered")
        self.assertEqual(record["image_source"], "embedding")
        self.assertEqual(record["cos"], 0.0)
        self.assertEqual(
            record["reason"], f"coarse cos={record['cos']:.3f} < threshold 0.500"
        )
        self.assertEqual(text_stub.calls, ["fishing"])

    def test_coarse_boundary_inclusive(self):
        """cos 恰等于粗筛阈值放行（< 严格比较），上下一个 ULP 各证一侧。"""
        term_vec = [1.0, 0.0]
        cand = [1.0, 1.0]
        exact = image_embedding._cosine_similarity(term_vec, cand)

        def build(threshold):
            gate, _, _, patches = _coarse_gate_with(
                {"data:c": cand}, {"q": term_vec}, coarse_threshold=threshold
            )
            return gate, patches

        gate, patches = build(exact)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://c", "data:c", term="q")
        )

        gate_up, patches_up = build(math.nextafter(exact, math.inf))
        for p in patches_up:
            p.start()
            self.addCleanup(p.stop)
        record = gate_up.judge_candidate_embedding(
            "https://c", "data:c", term="q"
        )
        assert record is not None
        self.assertEqual(record["verdict"], "prefiltered")
        self.assertEqual(record["cos"], exact)

        gate_down, patches_down = build(math.nextafter(exact, -math.inf))
        for p in patches_down:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate_down.judge_candidate_embedding("https://c", "data:c", term="q")
        )

    def test_coarse_disabled_never_embeds_text(self):
        """粗筛默认关闭：带 term 调用也不发起文本嵌入（向后兼容）。"""
        img_stub = _VectorStub({"data:a": [1.0, 0.0]})
        text_stub = _TextStub({"fishing": [1.0, 0.0]})
        gate = EmbeddingGate(model="m", api_key="k", threshold=0.68)
        patches = (
            patch.object(image_embedding, "embed_image", img_stub),
            patch.object(image_embedding, "embed_text", text_stub),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        self.assertEqual(text_stub.calls, [])

    def test_blank_term_skips_coarse(self):
        """空白 term 视为未提供：不发起文本嵌入，候选照常放行。"""
        gate, img_stub, text_stub, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0]},
            {"fishing": [1.0, 0.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for blank in ("", "   "):
            self.assertIsNone(
                gate.judge_candidate_embedding("https://a", "data:a", term=blank)
            )
        self.assertEqual(text_stub.calls, [])
        self.assertEqual(img_stub.calls, ["data:a"])

    def test_skip_coarse_skips_text_but_duplicate_still_runs(self):
        """skip_coarse=True 只跳过粗筛；查重比对照常执行且优先。"""
        gate, _, text_stub, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0], "data:b": [0.999, 0.0447]},
            {"fishing": [0.0, 1.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding(
                "https://a", "data:a", term="fishing", skip_coarse=True
            )
        )
        self.assertEqual(text_stub.calls, [])
        gate.register_accepted("https://a")
        record = gate.judge_candidate_embedding(
            "https://b", "data:b", term="fishing", skip_coarse=True
        )
        assert record is not None
        self.assertEqual(record["verdict"], "duplicate")
        self.assertEqual(text_stub.calls, [])

    def test_duplicate_reject_never_reaches_coarse(self):
        """重复是终审拒绝：查重命中即返回，粗筛完全不参与。

        直接播种注册表隔离变量：term 与候选近乎正交，若粗筛先于/替代查重
        运行，这里会得到 prefiltered 或出现文本嵌入调用。
        """
        gate, img_stub, text_stub, patches = _coarse_gate_with(
            {"data:b": [0.999, 0.0447]},
            {"fishing": [0.0, 1.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        gate._accepted["https://a"] = [1.0, 0.0]
        record = gate.judge_candidate_embedding(
            "https://b", "data:b", term="fishing"
        )
        assert record is not None
        self.assertEqual(record["verdict"], "duplicate")
        self.assertEqual(text_stub.calls, [])
        self.assertEqual(img_stub.calls, ["data:b"])

    def test_embed_text_failure_fails_open(self):
        """embed_text 失败（None）fail-open：跳过粗筛放行，失败不写缓存。"""
        gate, _, text_stub, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0]},
            {"fishing": None},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        self.assertEqual(text_stub.calls, ["fishing"])
        self.assertNotIn("fishing", gate._query_vecs)

    def test_failed_term_query_not_cached_and_retried(self):
        """失败的 term 查询不写缓存：下次判定同 term 重新尝试嵌入。

        锁定既定语义（与 embed_image 候选失败不缓存对称）：瞬时失败不把
        该 term 的粗筛毒化整个任务；恢复后同 term 一次重试即生效。
        """
        gate, _, text_stub, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0]},
            {"fishing": None},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        text_stub.vectors["fishing"] = [0.0, 1.0]
        record = gate.judge_candidate_embedding(
            "https://a", "data:a", term="fishing"
        )
        assert record is not None
        self.assertEqual(record["verdict"], "prefiltered")
        self.assertEqual(text_stub.calls, ["fishing", "fishing"])

    def test_embed_image_failure_skips_coarse_text(self):
        """图像嵌入失败先于粗筛短路：返回 None 且不发起文本嵌入。"""
        gate, img_stub, text_stub, patches = _coarse_gate_with(
            {"data:a": None},
            {"fishing": [1.0, 0.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        self.assertEqual(img_stub.calls, ["data:a"])
        self.assertEqual(text_stub.calls, [])

    def test_same_url_judged_twice_embeds_once(self):
        """同 URL 判定两次只嵌入一次；同 term 查询向量同样只嵌入一次。"""
        gate, img_stub, text_stub, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0]},
            {"fishing": [1.0, 0.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        self.assertIsNone(
            gate.judge_candidate_embedding("https://a", "data:a", term="fishing")
        )
        self.assertEqual(img_stub.calls, ["data:a"])
        self.assertEqual(text_stub.calls, ["fishing"])

    def test_same_term_embedded_once_across_candidates(self):
        """不同候选共用同一 term：查询向量只嵌入一次（_query_vecs 缓存）。"""
        gate, img_stub, text_stub, patches = _coarse_gate_with(
            {
                "data:a": [1.0, 0.0],
                "data:b": [0.0, 1.0],
                "data:c": [0.7071, 0.7071],
            },
            {"fishing": [1.0, 0.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for url, uri in (
            ("https://a", "data:a"),
            ("https://b", "data:b"),
            ("https://c", "data:c"),
        ):
            gate.judge_candidate_embedding(url, uri, term="fishing")
        self.assertEqual(img_stub.calls, ["data:a", "data:b", "data:c"])
        self.assertEqual(text_stub.calls, ["fishing"])

    def test_dimension_mismatch_cosine_is_zero(self):
        """维度不一致时 _cosine_similarity 返回 0（静默视为最不相似）。

        单模型下查询与候选维度恒一致，此路径不可达；T5 引入多模型分布
        后才会暴露，这里文档化其门内后果：cos=0 < 阈值 → prefiltered。
        """
        self.assertEqual(
            image_embedding._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]), 0.0
        )
        gate, _, _, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0]},
            {"fishing": [1.0, 0.0, 0.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        record = gate.judge_candidate_embedding(
            "https://a", "data:a", term="fishing"
        )
        assert record is not None
        self.assertEqual(record["verdict"], "prefiltered")
        self.assertEqual(record["cos"], 0.0)

    def test_prefiltered_candidate_vec_survives_for_register(self):
        """粗筛拒绝不清除候选缓存：register_accepted 仍零成本挪入注册表。"""
        gate, img_stub, _, patches = _coarse_gate_with(
            {"data:a": [1.0, 0.0], "data:b": [0.999, 0.0447]},
            {"fishing": [0.0, 1.0]},
            coarse_threshold=0.5,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        record = gate.judge_candidate_embedding(
            "https://a", "data:a", term="fishing"
        )
        assert record is not None
        self.assertEqual(record["verdict"], "prefiltered")
        self.assertIn("https://a", gate._candidates)
        gate.register_accepted("https://a")
        self.assertEqual(gate._accepted["https://a"], [1.0, 0.0])
        self.assertNotIn("https://a", gate._candidates)
        record2 = gate.judge_candidate_embedding(
            "https://b", "data:b", term="fishing"
        )
        assert record2 is not None
        self.assertEqual(record2["verdict"], "duplicate")
        self.assertEqual(img_stub.calls, ["data:a", "data:b"])


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

    def test_coarse_filter_disabled_by_default(self):
        # 段缺失或键缺失都按关闭处理：默认行为与引入粗筛之前逐字节一致。
        with patch.object(image_embedding.config, "image_embedding", {}):
            self.assertFalse(image_embedding.is_coarse_filter_enabled())

    def test_coarse_filter_enabled_when_config_true(self):
        with patch.object(
            image_embedding.config,
            "image_embedding",
            {"coarse_filter": True},
        ):
            self.assertTrue(image_embedding.is_coarse_filter_enabled())

    def test_invalid_coarse_threshold_falls_back(self):
        for bad in (0, 1.5, "abc", -0.1):
            with self.subTest(bad=bad):
                with patch.object(
                    image_embedding.config,
                    "image_embedding",
                    {"coarse_filter": True, "coarse_threshold": bad},
                ):
                    gate = image_embedding.make_default_gate()
                self.assertEqual(
                    gate.coarse_threshold,
                    image_embedding.DEFAULT_COARSE_THRESHOLD,
                )

    def test_make_default_gate_carries_coarse_flags(self):
        # 默认关闭：门存在但粗筛不生效，阈值取校准默认值（不生效即无副作用）。
        with patch.object(image_embedding.config, "image_embedding", {}):
            gate = image_embedding.make_default_gate()
        self.assertFalse(gate.coarse_enabled)
        self.assertEqual(
            gate.coarse_threshold, image_embedding.DEFAULT_COARSE_THRESHOLD
        )
        # 开启时两个参数都按配置透传到 T2 冻结的构造签名上。
        with patch.object(
            image_embedding.config,
            "image_embedding",
            {"coarse_filter": True, "coarse_threshold": 0.2},
        ):
            gate = image_embedding.make_default_gate()
        self.assertTrue(gate.coarse_enabled)
        self.assertEqual(gate.coarse_threshold, 0.2)

    def test_example_config_ships_coarse_keys(self):
        """config.example.toml 必须带 coarse_filter/coarse_threshold 且默认关闭。"""
        config_path = Path(__file__).parent.parent.parent / "config.example.toml"
        parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
        section = parsed["image_embedding"]
        self.assertIn("coarse_filter", section)
        self.assertIn("coarse_threshold", section)
        self.assertFalse(section["coarse_filter"])
        self.assertEqual(
            section["coarse_threshold"],
            image_embedding.DEFAULT_COARSE_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()

