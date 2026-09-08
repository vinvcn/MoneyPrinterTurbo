"""
逐段 LLM 查询生成（视频素材匹配三段漏斗的第一级）。

segment-first 流水线的素材匹配正在重构为三级漏斗：粗排（embedding 召回
排序）→ 精排（VL 模型重排）→ VLM 走查。本模块在分段之后为每个片段生成
一份查询包：terms（英文搜索词，主词在前，沿用 segment_terms 的既有语义）、
coarse_query（宽场景描述，供粗排 embedding 检索）、fine_query（精确视觉
时刻描述，供精排重排比对）。

与逐段搜索词提炼（segment_terms.py）的成本控制方式一致：一次
llm.generate_response 调用同时产出三类查询；响应无法解析出 JSON 对象时
按 segment_terms 的重试策略重试，重试耗尽后 fail-open 返回空查询包
（terms=[], coarse_query=None, fine_query=None），降级由调用方处理。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo
from app.services import image_embedding
from app.services import llm
from app.services import material_rerank
from app.services.segment_material import (
    CLIPS_PER_SEGMENT,
    MAX_SEARCH_PAGES,
    SegmentMaterials,
    _MAX_FILTER_RECORDS,
    english_search_term,
)
from app.services.video import segment_window_plan
from app.services.vlm_judge import download_thumbnail_bytes, to_data_uri

# 解析失败的最大尝试次数（与 segment_terms._MAX_RETRIES 同构，镜像其重试
# 语义：range(1, _MAX_RETRIES + 1) 即最多 2 次调用）。
_MAX_RETRIES = 2

# 每段最多提炼的搜索词数量；第一个词为主搜索词，其余为备用词（沿用
# segment_terms.TERMS_PER_SEGMENT 的语义）。
_MAX_TERMS = 3

# 字符级 CJK 判定：与 segment_terms/segment_material 保持同一正则——搜索
# API（Pexels/Pixabay/Coverr）仅接受英文查询，含中日韩字符的词召回极差。
_CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")


@dataclass
class SegmentQueries:
    """单个片段的素材匹配查询包（三段漏斗共用一份输入）。"""

    # 1..3 个英文搜索词，主词在前；全部词条被过滤时为空列表（降级由调用方处理）。
    terms: list[str]
    # 宽场景描述（英文一句话，粗排 embedding 检索）；LLM 失败/字段缺失时为 None。
    coarse_query: str | None
    # 精确视觉时刻描述（英文一句话，精排重排）；LLM 失败/字段缺失时为 None。
    fine_query: str | None


def _contains_cjk(text: str) -> bool:
    """判断文本是否包含中日韩字符（搜索 API 仅接受英文查询）。"""
    return bool(_CJK_PATTERN.search(text or ""))


def _build_prompt(subject: str, segment_text: str) -> str:
    """
    构造单段三查询提示词。

    保持与 generate_terms / segment_terms 相同的 Role/Constrains/Output
    Example 结构（模型对该格式已稳定），Context 为单段文本并绑定主题词。
    """
    output_example = json.dumps(
        {
            "terms": ["panda eating bamboo", "bamboo forest", "panda cub playing"],
            "coarse_query": (
                "A giant panda sits in a lush green bamboo forest "
                "munching on bamboo stalks."
            ),
            "fine_query": (
                "Close-up of a giant panda gripping a bamboo stalk with its "
                "paws and stripping the leaves with its teeth."
            ),
        },
        ensure_ascii=False,
    )

    return f"""
# Role: Video Material Query Generator

## Goals:
For one narration segment, generate stock-video search terms plus two English scene descriptions used to match material clips.

## Constrains:
1. return ONLY a json object with exactly three keys: "terms", "coarse_query", "fine_query". you must not return anything else. you must not return the script.
2. "terms" must be a json-array of 1-3 search terms: each term consists of 1-3 words, always translate the main subject of the video into English and append it, and the terms must describe DIFFERENT visual angles, scenes, or shot types of the segment's moment — never synonyms or minor rephrasings, because near-duplicate terms return the same stock-footage candidate pool.
3. "coarse_query" must be one English sentence broadly describing the segment's visual scene (environment, subjects, mood) for coarse embedding retrieval.
4. "fine_query" must be one English sentence precisely describing the segment's key visual moment (action, composition, details) for fine visual reranking.
5. every value must be pure English (A-Z letters, spaces and punctuation only). never include any Chinese or other non-English characters, even if the subject or the narration is written in Chinese.

## Output Example:
{output_example}

## Context:
### Video Subject
{subject}

### Narration Segment
{segment_text}

Please note that you must use English for generating search terms and queries; Chinese is not accepted. If the subject or a segment is written in Chinese, translate its meaning into English.
""".strip()


def _strip_code_fences(text: str) -> str:
    """剥掉 markdown 代码围栏（```json ... ```）；无围栏时原样返回。"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def _clean_query(value: object) -> str | None:
    """查询字段规整：非字符串或空白视为缺失，返回 None。"""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_terms(raw_terms: object, subject: str) -> list[str]:
    """
    词条规整，逐字沿用 segment_terms 的语义：截断、追加英文主题词、
    逐词 CJK 过滤；全部词条被过滤时返回空列表（降级由调用方处理）。
    """
    if not isinstance(raw_terms, list):
        return []
    cleaned = [str(term).strip() for term in raw_terms if str(term).strip()]
    # 超量词条截断；主词在前，顺序与模型输出一致。
    cleaned = cleaned[:_MAX_TERMS]

    # 主题词含 CJK 时直接放弃追加（没有可追加的英文主题），让词条保持
    # 纯英文，而不是发出必然低召回的混合查询。
    append_subject = bool(subject)
    if subject and _contains_cjk(subject):
        logger.warning(
            "video subject contains CJK, appending it would break "
            f"English-only search: subject={subject!r}"
        )
        append_subject = False

    terms: list[str] = []
    for term in cleaned:
        final_term = f"{term} {subject}".strip() if append_subject else term
        # 逐词过滤：单个词条含 CJK 只丢弃该词，分段靠剩余词条存活。
        if _contains_cjk(final_term):
            logger.warning(f"segment term contains CJK, discarding: term={final_term!r}")
            continue
        terms.append(final_term)
    return terms


def _parse_segment_queries(response: object, subject: str) -> SegmentQueries | None:
    """
    从模型回复解析查询包；无法得到 JSON 对象时返回 None 触发重试。

    防御性解析：容忍 markdown 围栏与 JSON 前后的说明文字；字段缺失或
    类型不符时逐字段降级（terms → 空列表，查询 → None），只有 JSON 本身
    不可解析（包括 llm 层的 "Error: " 失败串）才算解析失败。
    """
    text = str(response or "").strip()
    if not text or text.startswith("Error: "):
        return None

    candidate = _strip_code_fences(text)
    match = re.search(r"\{.*}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    return SegmentQueries(
        terms=_normalize_terms(data.get("terms"), subject),
        coarse_query=_clean_query(data.get("coarse_query")),
        fine_query=_clean_query(data.get("fine_query")),
    )


def generate_segment_queries(subject: str, segment_text: str) -> SegmentQueries:
    """
    为单个片段生成素材匹配查询包（terms + coarse_query + fine_query）。

    Args:
        subject: 视频主题，翻译成英文后附加进每个搜索词（含 CJK 时不追加）。
        segment_text: 该片段的旁白原文。

    Returns:
        SegmentQueries。解析失败按 segment_terms 的策略重试，重试耗尽后
        fail-open 返回空查询包（terms=[], coarse_query=None,
        fine_query=None），绝不抛异常；空词条等降级由调用方处理。
    """
    subject = str(subject or "").strip()
    segment_text = str(segment_text or "").strip()
    prompt = _build_prompt(subject, segment_text)

    for attempt in range(1, _MAX_RETRIES + 1):
        response = llm.generate_response(prompt)
        parsed = _parse_segment_queries(response, subject)
        if parsed is not None:
            return parsed
        logger.warning(
            f"segment queries response unusable: attempt={attempt}, "
            f"response={str(response)[:120]!r}"
        )

    logger.warning(
        "segment query generation failed after retries; returning empty "
        "queries and leaving degradation to the caller"
    )
    return SegmentQueries(terms=[], coarse_query=None, fine_query=None)


# ---------------------------------------------------------------------------
# 粗排：embedding 召回排序（三段漏斗第一段，消费 generate_segment_queries
# 产出的 coarse_query，输出 top-30 交给下游精排/走查）。
# ---------------------------------------------------------------------------

# 粗排最多放行的候选数：漏斗约定粗排产出 top-30，重复候选不计入配额。
_COARSE_TOP_K = 30


def coarse_rank(
    pool: list[dict],
    coarse_query: str | None,
    vector_cache: dict[str, list[float]] | None,
    embedding_gate: image_embedding.EmbeddingGate | None,
) -> tuple[list[dict], list[dict]]:
    """
    用 coarse_query 的文本向量对候选池做余弦排序，再过查重门取 top-30。

    pool 是调用方按 interleave 顺序给出的候选列表，每项为含 url /
    data_uri / asset_id 键的 dict。返回 (selected, duplicate_skips)：
    selected 是至多 30 个非重复候选（余弦降序、稳定），duplicate_skips
    是查重门拒收记录（asset_id/url/reason）供调用方写审计日志。

    fail-open 是硬契约：coarse_query 缺失或 embed_text 失败时原样返回
    pool[:30]（pool 切片即 interleave 路径，粗排完全退场）；单个候选
    嵌入失败得分 -1.0 沉底，不阻塞整体，交给查重门复判；查重门为
    None（duplicate_gate 关闭）时跳过查重直接按排序放行。

    vector_cache 是粗排与查重门共享的 url -> 向量缓存：候选向量优先读
    缓存，未命中时嵌入并写回；调用方把它注入 EmbeddingGate(vector_cache
    =...) 后，同一 URL 在粗排预热与门走查之间只嵌入一次。
    """
    if not coarse_query:
        logger.warning(
            "video match: coarse rank failed, fail-open: "
            "reason=missing coarse query"
        )
        return list(pool)[:_COARSE_TOP_K], []
    query_vec = image_embedding.embed_text(coarse_query)
    if query_vec is None:
        logger.warning(
            "video match: coarse rank failed, fail-open: "
            "reason=query embedding unavailable"
        )
        return list(pool)[:_COARSE_TOP_K], []

    # 嵌入凭据与端点直接读 [image_embedding] 配置（与查重门同一套解析），
    # 不依赖查重门是否注入。
    section = getattr(image_embedding.config, "image_embedding", None) or {}
    model = image_embedding._gate_setting(
        "model", image_embedding.DEFAULT_EMBEDDING_MODEL
    )
    api_key = str(section.get("api_key", "") or "")
    base_url = image_embedding._gate_setting("base_url", "") or None

    cache = vector_cache if vector_cache is not None else {}
    scored: list[tuple[float, dict]] = []
    for cand in pool:
        url = str(cand.get("url") or "")
        vec = cache.get(url)
        if vec is None:
            vec = image_embedding.embed_image(
                data_uri=str(cand.get("data_uri") or ""),
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            # 嵌入失败不写缓存（同 URL 之后仍可重试）；成功写回供查重
            # 门复用，避免同 URL 二次嵌入。
            if vec is not None:
                cache[url] = vec
        if vec is None:
            # 得分 -1.0 沉底而非剔除：保持 fail-open，候选仍参与走查，
            # 由查重门复判（门内嵌入重试成功仍可被放行）。
            logger.warning(
                "video match: coarse rank candidate embed failed, "
                f"sink to bottom: url={url}"
            )
            scored.append((-1.0, cand))
            continue
        scored.append((image_embedding._cosine_similarity(query_vec, vec), cand))

    # 稳定降序：sorted 的稳定性保证同分候选维持 pool（interleave）顺序。
    scored.sort(key=lambda pair: pair[0], reverse=True)

    selected: list[dict] = []
    duplicate_skips: list[dict] = []
    if embedding_gate is None:
        # 查重门关闭（调用方未注入）：跳过走查，排序前 30 直接放行。
        selected = [cand for _, cand in scored[:_COARSE_TOP_K]]
    else:
        for _, cand in scored:
            if len(selected) >= _COARSE_TOP_K:
                break
            url = str(cand.get("url") or "")
            record = embedding_gate.judge_candidate_embedding(
                url,
                str(cand.get("data_uri") or ""),
                coarse_query,
            )
            if record is not None:
                # 走查前向量已全部预热进共享缓存，此步不产生新的嵌入调用。
                duplicate_skips.append(
                    {
                        "asset_id": cand.get("asset_id"),
                        "url": url,
                        "reason": record.get("reason") or record.get("verdict"),
                    }
                )
                continue
            selected.append(cand)

    logger.info(
        f"video match: coarse rank query={coarse_query!r} pool={len(pool)} "
        f"selected={len(selected)} duplicates={len(duplicate_skips)}"
    )
    return selected, duplicate_skips


# ---------------------------------------------------------------------------
# 三段漏斗主编排（todo 5）：每段 查询包 → 搜索聚池（页优先 interleave）→
# 粗排 → 精排 → VLM 走查（配额满提前退出）→ image-gen 回填。替代
# prepare_segment_materials 的层级/翻页/延期机械（旧实现保留至 todo 6
# 与 task.py 接线一并删除，本模块只新增不改动）。
# ---------------------------------------------------------------------------


def _thumbnail_data_uri(item: MaterialInfo) -> str:
    """候选缩略图 → base64 data URI（粗排嵌入与查重门的判定输入）。

    缩略图缺失或下载失败返回空串：粗排把空 URI 候选按嵌入失败沉底，
    查重门 fail-open 放行，VLM 走查里由判定回调自行取首帧兜底——
    预览图层失败绝不阻塞流水线（与 vlm_judge 的降级哲学一致）。
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    thumbnail = str(source.get("thumbnail_url") or "").strip()
    if not thumbnail:
        return ""
    try:
        payload, _size = download_thumbnail_bytes(thumbnail)
    except Exception as exc:
        logger.warning(
            "video match: thumbnail download failed, continue without "
            f"preview: url={item.url}, error={type(exc).__name__}"
        )
        return ""
    return to_data_uri(payload)


def _candidate_from_item(item: MaterialInfo, term: str) -> dict:
    """MaterialInfo → 粗排候选 dict。

    url/data_uri/asset_id 供粗排与查重门读取（coarse_rank 的既有契约），
    item/term 供 VLM 走查把 MaterialInfo 原样交回判定回调并携带出处词条。
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    return {
        "asset_id": str(source.get("asset_id") or ""),
        "url": item.url,
        "data_uri": _thumbnail_data_uri(item),
        "term": term,
        "item": item,
    }


def _search_terms_for_queries(queries: SegmentQueries, subject: str) -> list[str]:
    """
    漏斗入口词条：queries.terms 为主，空词条时回落英文主题词。

    镜像 prepare_segment_materials 的词条处理：逐词过英文过滤（搜索
    API 仅接受英文）、去空串、按序去重。queries.terms 为空（LLM 失败
    或全部被 CJK 过滤）时以主题词兜底；主题词也含 CJK 时返回空列表——
    该段没有可用搜索词，候选池为空，直接落入 image-gen 回填。
    """
    raw_terms = list(queries.terms) or [subject]
    terms: list[str] = []
    for raw_term in raw_terms:
        english = english_search_term(str(raw_term or ""))
        if english and english not in terms:
            terms.append(english)
    return terms


def match_segments(
    segments: list[dict],
    video_subject: str,
    search_videos: Callable[..., list[MaterialInfo]],
    save_video: Callable[..., str],
    video_aspect: str,
    clip_duration: float,
    judge_candidate: Callable[..., dict] | None = None,
    embedding_gate: image_embedding.EmbeddingGate | None = None,
    generate_image: Callable[..., tuple[str, dict]] | None = None,
) -> list[SegmentMaterials]:
    """
    三段漏斗素材匹配主编排：粗排（embedding 召回）→ 精排（VL 重排）→
    VLM 走查（配额满提前退出），未满名额（含 VLM 关闭的整段情形）一律
    image-gen 回填。字段与旧 prepare_segment_materials 逐字段一致。

    Args:
        segments: 片段 dict 列表，至少含 {"index", "text", "duration"}。
        video_subject: 视频主题；queries.terms 为空时回落为主题搜索词。
        search_videos: 搜索回调。优先按旧契约调用
            search_videos(search_term=..., minimum_duration=...,
            video_aspect=..., page=...)；旧签名（无 page）与两参新契约
            search_videos(search_term, page) 依次 TypeError 回退，三种
            形态的接线都能工作。本函数再做 (词条, 页) 级备忘，同一
            (词条, 页) 只透传一次。
        save_video: 下载回调，按旧调用形态
            save_video(video_url=..., save_dir=...) -> 本地路径（失败返回
            ""）。save_dir 取 [app] material_directory 的模块级兜底解析
            （与旧实现一致）；task.py 接线如需任务级目录，把目录绑定进
            回调（lambda/partial 包一层）即可。
        video_aspect: 目标画幅，原样透传给 search_videos。
        clip_duration: 请求供应商的最短素材时长，同时是窗口宽度 W——
            每段配额 = max(CLIPS_PER_SEGMENT, len(segment_window_plan(D, W)))，
            与装配层共用同一窗口计划（B3 契约）。
        judge_candidate: VLM 判定回调，契约与旧实现逐字一致：
            judge_candidate(item=MaterialInfo, segment_text=..., search_term=...)
            -> 判定记录 dict（verdict ∈ relevant/irrelevant/uncertain）。
            None = VLM 关闭：跳过搜索与排序，整段 image-gen（VLM 对
            搜索素材是强制立场，plan video-match）。
        embedding_gate: 任务级查重门（粗排边界 + 走查复判共用）。其
            _shared_cache 即任务级 url->向量缓存：task.py 构造 gate 时
            注入 vector_cache 即可，本函数取回同一 dict 供粗排预热，
            同一 URL 全链路只嵌入一次（plan finding G）。
        generate_image: image-gen 回填回调，契约
            generate_image(segment, duration) -> (clip_path, audit_record)。
            segment 是完整片段 dict（接线层自行取旁白原文与主题词），
            duration 是待回填窗口的时长和（秒）——image_gen.make_subject_clip
            已支持 duration 覆盖参数，接线层用 partial/lambda 绑定
            video_aspect 与 save_dir 后注入。名额未满时调用一次，覆盖
            全部剩余窗口；回调缺失或失败时段保持视频短缺，不阻塞。

    Returns:
        与输入等长的 SegmentMaterials 列表（字段与旧实现一致）。
    """
    subject = english_search_term(str(video_subject or ""))
    # (词条, 页) 搜索备忘：镜像 prepare_segment_materials 的语义——同一
    # 关键词组合跨段、跨词条只打一次供应商 API。
    search_cache: dict[tuple[str, int], list[MaterialInfo]] = {}

    def search_page_cached(term: str, page: int) -> list[MaterialInfo]:
        normalized = (term or "").strip()
        if not normalized:
            return []
        cache_key = (normalized, page)
        if cache_key not in search_cache:
            # 三种搜索回调形态依次回退（见 docstring）：旧四参（带页码）、
            # 旧三参（无页码，TypeError 退回）、两参新契约（按位传词条
            # 与页码）。回退调用自身的 TypeError（如返回 None 不可迭代）
            # 按空结果处理，绝不阻塞流水线。
            found: list[MaterialInfo]
            try:
                found = list(
                    search_videos(
                        search_term=normalized,
                        minimum_duration=clip_duration,
                        video_aspect=video_aspect,
                        page=page,
                    )
                )
            except TypeError:
                try:
                    found = list(
                        search_videos(
                            search_term=normalized,
                            minimum_duration=clip_duration,
                            video_aspect=video_aspect,
                        )
                    )
                except TypeError:
                    try:
                        found = list(search_videos(normalized, page))
                    except TypeError:
                        found = []
            # 逐条打印搜索返回的候选，供运行审计核对"搜到了什么"。
            logger.info(
                f"segment search returned {len(found)} candidates for "
                f"term={normalized!r}, page={page}"
            )
            for item in found:
                logger.info(
                    f"  candidate: provider={item.provider}, "
                    f"duration={item.duration}s, url={item.url}"
                )
            search_cache[cache_key] = found
        return search_cache[cache_key]

    # 任务级已用素材 URL：前面 segment 已采纳的候选不再进入后续 segment
    # 的候选池（镜像 used_urls_across_segments 机制）；注册无条件——每次
    # 下载采纳都入册，查重门注册同步进行。
    used_urls: set[str] = set()

    # 任务级 url->向量缓存：粗排与查重门共享同一份 dict。task.py 构造
    # gate 时注入 vector_cache，这里取回同一份（私有访问先例与
    # material_rerank._walk_limit 相同）；gate 缺席时退化为局部缓存。
    vector_cache: dict[str, list[float]] = {}
    if embedding_gate is not None:
        shared = getattr(embedding_gate, "_shared_cache", None)
        if isinstance(shared, dict):
            vector_cache = shared

    # VLM 走查预算：与旧实现同一先例——直接读 material_rerank._walk_limit
    # （私有访问先例已在 segment_material.py 建立），每次运行只读一次。
    walk_limit = material_rerank._walk_limit()

    material_directory = str(config.app.get("material_directory", "")).strip()
    if material_directory == "task":
        material_directory = ""

    results: list[SegmentMaterials] = []
    for position, segment in enumerate(segments):
        segment_index = segment.get("index", position)
        segment_text = str(segment.get("text") or "")
        segment_duration = float(segment.get("duration") or 0)
        windows = segment_window_plan(segment_duration, clip_duration)
        needed_clips = max(CLIPS_PER_SEGMENT, len(windows))

        # 1. 每段一次 LLM 查询包（terms + coarse_query + fine_query）；
        #    失败 fail-open 为空包，降级由下方各阶段自行处理。
        queries = generate_segment_queries(str(video_subject or ""), segment_text)
        terms = _search_terms_for_queries(queries, subject)

        clips: list[str] = []
        clip_sources: list[dict] = []
        vlm_filter_records: list[dict] = []
        image_gen_records: list[dict] = []
        search_attempts: list[dict] = []
        resolved_term = ""
        fallback_level = ""

        if judge_candidate is None:
            # VLM 关闭（强制立场）：跳过搜索与排序，整段交给 image-gen
            # 回填（下方 remaining 分支覆盖全部窗口）。
            logger.info(
                f"segment {segment_index}: video match: vlm disabled, image-gen only"
            )
        else:
            # 2. 搜索聚池：页优先 interleave（page1-term1, page1-term2, ...,
            #    page2-*），按 URL 去重保首个，跨段已用 URL 不入池。
            pool: list[dict] = []
            pool_urls: set[str] = set()
            active_terms = list(terms)
            for page in range(1, MAX_SEARCH_PAGES + 1):
                if not active_terms:
                    break
                still_active: list[str] = []
                for term in active_terms:
                    page_items = search_page_cached(term, page)
                    if not page_items:
                        # 第 1 页即空的词条不参与翻页（镜像旧分页语义）。
                        continue
                    still_active.append(term)
                    for item in page_items:
                        url = str(item.url or "")
                        if not url or url in used_urls or url in pool_urls:
                            continue
                        pool_urls.add(url)
                        pool.append(_candidate_from_item(item, term))
                active_terms = still_active

            # 3. 粗排：coarse_query 余弦排序 + 查重门走查 → top-30 非重复。
            #    重复计数由 coarse_rank 的既有汇总行落日志（grep 锚点
            #    "video match: coarse rank ... duplicates=N"），不重复打点。
            top30, _dup_skips = coarse_rank(
                pool, queries.coarse_query, vector_cache, embedding_gate
            )

            # 4. 精排：fine_query 全量降序重排；开关关闭 → 粗排序 + 独立
            #    日志行；调用本身抛异常 → 粗排序兜底（重排模块内部已
            #    fail-open，这里只兜"调用被替换/抛异常"的情况，镜像旧实现）。
            #    rerank_candidates 消费 MaterialInfo 列表（读 source_info
            #    缩略图），这里把 top-30 候选还原为素材对象送重排，再按
            #    URL 映射回候选 dict（池内 URL 唯一，映射无损）。
            if material_rerank.is_rerank_enabled():
                try:
                    reranked_items = material_rerank.rerank_candidates(
                        queries.fine_query,
                        [c["item"] for c in top30],
                    )
                    by_url = {c["url"]: c for c in top30}
                    ranked = [by_url[str(item.url)] for item in reranked_items]
                except Exception as exc:
                    logger.warning(
                        "video match: fine rerank failed, falling back to "
                        f"coarse order: error={type(exc).__name__}, detail={exc}"
                    )
                    ranked = top30
            else:
                logger.info("video match: fine rerank disabled, using coarse order")
                ranked = top30

            # 5. VLM 走查：fine 序前 walk_limit 个，配额满即提前退出。
            accepted_terms: set[str] = set()
            for candidate in ranked[:walk_limit]:
                if len(clips) >= needed_clips:
                    break
                url = str(candidate.get("url") or "")
                asset_id = str(candidate.get("asset_id") or "")
                term = str(candidate.get("term") or "")
                # 查重门复判（cache-hit 安全）：粗排边界已判过一次，这里
                # 兜"粗排之后才被采纳注册"的近重复；向量全部命中共享缓存，
                # 零新增嵌入。门自身异常 fail-open 放行，交给 VLM 终审。
                try:
                    duplicate = (
                        embedding_gate.judge_candidate_embedding(
                            url,
                            str(candidate.get("data_uri") or ""),
                            term,
                        )
                        if embedding_gate is not None
                        else None
                    )
                except Exception as exc:
                    logger.warning(
                        "embedding gate failed, fail-open: "
                        f"asset_id={asset_id}, error={type(exc).__name__}"
                    )
                    duplicate = None
                if duplicate is not None:
                    # 重复是终审拒绝（plan T5）：不下载、不采纳、无兜底；
                    # 审计记录保留，让审计链显示门在走查层再次生效。
                    logger.info(
                        "vlm filter rejected candidate: "
                        f"asset_id={asset_id}, verdict=duplicate, "
                        f"reason={duplicate.get('reason')!r}, "
                        f"image_source=embedding, "
                        f"duplicate_of={duplicate.get('duplicate_of')}, "
                        f"cos={duplicate.get('cos')}"
                    )
                    vlm_filter_records.append(
                        {
                            "term": term,
                            "asset_id": asset_id,
                            "verdict": duplicate.get("verdict", "duplicate"),
                            "reason": duplicate.get("reason", ""),
                            "image_source": "embedding",
                            "attempts": 0,
                            "duplicate_of": duplicate.get("duplicate_of"),
                            "cos": duplicate.get("cos"),
                        }
                    )
                    continue
                try:
                    verdict = judge_candidate(
                        item=candidate.get("item"),
                        segment_text=segment_text,
                        search_term=term,
                    )
                except Exception as exc:
                    # 单候选判定异常 = 跳过该候选继续走查，绝不阻塞
                    # （判定是质量增强，不是硬闸；生产判定回调自身
                    # fail-open，这里兜第三方/测试替身实现抛异常）。
                    logger.warning(
                        "vlm judge failed, fail-open: "
                        f"asset_id={asset_id}, error={type(exc).__name__}, "
                        f"detail={exc}"
                    )
                    continue
                if not isinstance(verdict, dict):
                    logger.warning(
                        "vlm judge returned unusable verdict, skip candidate: "
                        f"asset_id={asset_id}, verdict={verdict!r}"
                    )
                    continue
                v = str(verdict.get("verdict") or "")
                vlm_filter_records.append(verdict)
                if v in ("irrelevant", "duplicate"):
                    logger.info(
                        "vlm filter rejected candidate: "
                        f"asset_id={verdict.get('asset_id')}, "
                        f"verdict={v}, "
                        f"reason={verdict.get('reason')!r}, "
                        f"image_source={verdict.get('image_source')}"
                    )
                    continue
                if v == "uncertain":
                    # uncertain 不再兜底采纳（旧 last-resort 已被 image-gen
                    # 回填取代）：跳过并留独立日志行（plan video-match：
                    # uncertain = skip + distinct log）。
                    logger.info(
                        "vlm filter uncertain candidate skipped: "
                        f"asset_id={verdict.get('asset_id')}, "
                        f"reason={verdict.get('reason')!r}, "
                        f"image_source={verdict.get('image_source')}"
                    )
                    continue
                logger.info(
                    "vlm filter accepted candidate: "
                    f"asset_id={verdict.get('asset_id')}, "
                    f"verdict={v}, "
                    f"image_source={verdict.get('image_source')}"
                )
                saved = ""
                try:
                    saved = save_video(video_url=url, save_dir=material_directory)
                except Exception as exc:
                    logger.warning(
                        "failed to download segment clip: "
                        f"provider={getattr(candidate.get('item'), 'provider', '')}, "
                        f"error={type(exc).__name__}, detail={exc}"
                    )
                if saved and saved not in clips:
                    logger.info(f"segment clip saved: {saved}")
                    clips.append(saved)
                    clip_sources.append(
                        {"url": url, "local_file": Path(saved).name}
                    )
                    used_urls.add(url)
                    accepted_terms.add(term)
                    if not resolved_term:
                        resolved_term = term
                        fallback_level = "self"
                    if embedding_gate is not None:
                        try:
                            embedding_gate.register_accepted(url)
                        except Exception as exc:
                            # 注册失败只降级为告警：查重注册绝不中断素材链路
                            #（镜像旧 on_clip_accepted 的容错语义）。
                            logger.warning(
                                "embedding gate register failed: "
                                f"url={url}, error={type(exc).__name__}, detail={exc}"
                            )
                    if len(clips) >= needed_clips:
                        # 配额打满即提前退出走查（VLM 预算的核心约束）。
                        break
                # 下载失败继续看下一个候选（与旧实现一致）。

            # 走查审计：合并池之后逐词条的"贡献了候选"可从池归因，
            # "产出成片"按采纳候选的出处词条归因（镜像旧 found 语义）。
            search_attempts = [
                {"level": "self", "term": term, "found": term in accepted_terms}
                for term in terms
            ]

        # 6. image-gen 回填：未满名额的窗口尾部交给生成概念图（VLM 关闭
        #    时整段回填走同一分支）。单次调用覆盖全部剩余窗口；回填时长
        #    = 未填充尾部窗口的时长和（多样性下限超出窗口数的短段按最后
        #    一窗兜底，避免 0 时长 clip）。
        remaining = needed_clips - len(clips)
        if remaining > 0 and generate_image is not None:
            tail = windows[len(clips):] or windows[-1:]
            backfill_duration = sum(tail) if tail else float(clip_duration or 0.0)
            logger.info(
                f"segment {segment_index}: video match: image-gen backfill "
                f"windows={len(tail)} duration={backfill_duration:.3f}"
            )
            image_clip = ""
            image_record: dict | None = None
            try:
                image_clip, image_record = generate_image(segment, backfill_duration)
            except Exception as exc:
                logger.warning(
                    "video match: image-gen backfill failed: "
                    f"segment={segment_index}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
            if image_clip:
                clips.append(image_clip)
                clip_sources.append(
                    {"url": "", "local_file": Path(image_clip).name}
                )
                if not resolved_term:
                    resolved_term = subject
                    fallback_level = "subject"
                # 审计缺口 G4（适配）：回填 clip 成功时标明该段画面（部分）
                # 来自生成概念图；只记文件名，不记 prompt/图像内容。
                logger.info(
                    f"segment {segment_index}: "
                    f"image-gen fallback engaged: clip={Path(image_clip).name}"
                )
            if image_record:
                image_gen_records.append(image_record)

        # 审计缺口 G1：每段一行汇总（成功/部分/空手都触发），格式与旧实现
        # 一致（grep 锚点 "material resolution summary:" 供下游日志断言）；
        # vlm_judged 取截断前的真实判定条数。
        logger.info(
            f"segment {segment_index}: "
            f"material resolution summary: clips={len(clips)}/{needed_clips}, "
            f"resolved_term={resolved_term!r}, fallback_level={fallback_level}, "
            f"levels_tried={len(search_attempts)}, "
            f"vlm_judged={len(vlm_filter_records)}, "
            f"image_gen={len(image_gen_records)}"
        )

        results.append(
            SegmentMaterials(
                index=int(segment.get("index", position)),
                search_term=terms[0] if terms else "",
                clips=clips,
                resolved_term=resolved_term,
                fallback_level=fallback_level,
                search_attempts=search_attempts,
                clip_sources=clip_sources,
                vlm_filter=vlm_filter_records[:_MAX_FILTER_RECORDS],
                image_gen=image_gen_records,
            )
        )
        if not clips:
            logger.warning(
                f"no materials found for segment {segment_index} "
                f"(subject={subject!r})"
            )

    return results
