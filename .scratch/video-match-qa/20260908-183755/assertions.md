# QA assertions — repo-wide QA for the three-stage video-match refactor

Date: 2026-09-08 · Branch: `feat/segment-first-pipeline` (HEAD `64849cb`) · Refactor range `543a17f..64849cb`

Image: `mpt-video-match:qa` (rebuilt from HEAD this run, `sha256:15cc812b3408…`), prior image `mpt-segment-first:uat` (`sha256:6fb9fd9a8fad…`, built ~7 h before HEAD). Containers: `mpt-vm-qa` (API host 8092, happy path) and `mpt-vm-qa-vlmoff` (API host 8093, VLM-off), both removed after the runs.

Evidence files in this directory (all untracked by design except `assertions.md`):
- `run-api-happy.log` — full de-ANSI'd in-container API log, contains BOTH submissions (defect task `0df39803` then happy task `c40ec665`)
- `run-api-defect-task.log` — snapshot of the log at the moment the defect task failed (62 lines)
- `run-api-vlmoff.log`, `run-docker-*.log`, `pytest-baseline.log`, `ruff-check.log`, `image-build.log`
- `eval-ab.log`, `eval-ab-results.json` — eval A/B console + JSON (sibling script `.scratch/rerank-eval/caption_ab_eval.py`, which imports the existing `qwen3vl_rerank_eval.py` harness)
- `request-happy.json`, `submit-response-*.txt`, `status-happy.json` / `status-vlmoff.json` (final task states), `poll-*.log`
- `storage-happy/`, `storage-vlmoff/` — task dirs incl. `final-1.mp4` videos (kept as evidence)

All line numbers below refer to the de-ANSI'd content of the named file in this directory.

---

## 0. CODE DEFECT — CJK early-failure guard orphaned by the wiring refactor

**BLOCKED (defect reported, not fixed — QA constraint).** The brief's happy-path subject 为什么大熊猫是国宝 (CJK) **cannot** complete on HEAD.

Citation chain:
- Commit `18e6c8d` added an early-failure guard to `app/services/task.py:1153-1173`: it fails the task at stage=terms unless `english_subject` is non-empty or any segment record carries an English candidate (`record.get("search_terms")` / `record.get("search_term")` / `record["text"]`).
- Commit `64849cb` ("task pipeline wired to video match process") deleted `segment_terms.py` and removed the pre-population that the guard depends on (`git show 64849cb -- app/services/task.py`: `- segment_term_map = segment_terms.extract_terms_for_segments(`, `- record["search_terms"] = terms`) — **the guard was kept**.
- At HEAD, `segment_records` are built at `app/services/task.py:1138-1140` as `{"index", "text"}` only; `english_search_term` returns `""` for any CJK text (`app/services/segment_material.py:38-48`). So for a CJK subject + CJK narration, `has_usable_search_source` is always False → **every Chinese-subject segment-first task hard-fails at progress=10, before TTS/search/VLM**.
- Live proof (defect task `0df39803-fdc6-4b9a-9558-9e1ad9ae831c`): script generated OK (8 segments, `run-api-happy.log:57-61`), then `run-api-happy.log:62`:
  `task failed, task_id: 0df39803…, stage: terms, error: 无可用英文搜索词：LLM 词条提炼失败或为空，且旁白文本与主题词均为中文…`
- Regression proof: at `625781d` the same guard passed for Chinese subject 大熊猫的野外日常 (prior QA task `3129367d` produced a video) because `extract_terms_for_segments` populated `search_terms` BEFORE the guard (`git show 625781d:app/services/task.py` lines 1148-1150). The refactor removed the pre-population without updating the guard.

**Assertion A0 (happy path with CJK subject completes): FAIL — code defect** (task `0df39803`, state=-1, progress=10). All remaining pipeline-mechanics assertions below were executed with a documented deviation: an English-subject panda task ("Why giant pandas are China's national treasure"), which passes the guard via `english_subject` and exercises the three-stage pipeline exactly as designed. No app/ or test/ code was modified.

## 1. Stale sweep (tracked files only) — PASS

- `git grep -n -E "rerank_page|extract_terms_for_segments|deferred_uncertain|prefiltered|skip_coarse" -- app config.example.toml docs` → **0 hits**. PASS.
- `git grep -n "top_n" -- app config.example.toml docs` → exactly **1 hit**: `app/services/material_rerank.py:227: "top_n": len(rankable),` — the rerank request-body key (provider API contract, `top_n = len(documents)` so every document is scored). **Justified**. PASS.
- `config.example.toml` coherence: `[material_rerank]` (lines 417-424) = `enabled=true`, `model="Qwen/Qwen3-VL-Reranker-8B"`, **`vlm_walk_limit = 10` present**, `timeout=120`, credentials empty — and **no `top_n`**. `[image_embedding]` (398-405) = model/key/base_url/`duplicate_gate=false`/`duplicate_threshold=0.68`. `[vlm]` (348-354) = `enabled=false` + comments. All coherent with the new key set. PASS.
- Note (user's gitignored `config.toml`, not part of the sweep scope): `[material_rerank]` still carries a legacy `top_n: 5` key (tolerated by the parser) and lacks `vlm_walk_limit` → runtime `_walk_limit()` correctly falls back to 10 (`app/services/material_rerank.py:78-86`); only the strict defaults TEST asserts on it (see §2).

## 2. Suite baseline record — PASS

Command: `.venv/bin/python -m pytest test/ -q` (single run, 159.95s, full log `pytest-baseline.log`).

Exact summary line: `27 failed, 742 passed, 11 skipped, 20 warnings, 4071 subtests passed in 159.95s (0:02:39)`

Expected 27 = 26 documented baseline + 1. Breakdown (grep-verified over `pytest-baseline.log:1441+`):

| File | Count | Detail |
|---|---|---|
| `test/services/test_llm.py` | 16 | all `TestLiteLLMProvider::*` (documented stale assertions) |
| `test/services/test_task.py` | 9 | 5 top-level FAILED + 4 SUBFAILED (2× `test_start_marks_pipeline_failures`, 2× `test_start_returns_each_intermediate_result`) — documented |
| `test/services/test_webui_voice_preview.py` | 1 | `test_non_default_volume_regenerates_audio_without_double_gain` — documented |
| `test/services/test_material_rerank.py::test_config_defaults` | 1 | **`KeyError: 'vlm_walk_limit'`** at `pytest-baseline.log:699-701` — proven caused by the user's gitignored `config.toml` whose `[material_rerank]` section lacks the key (in-container config print shows keys `enabled/model/top_n/timeout/base_url` only); tracked code correct (defaults + `_walk_limit` fallback, `material_rerank.py:78-86`); baseline-worktree proof already in ledger. User config intentionally NOT touched. |

The known slow test `test_image_gen::TestStillToClip::test_returns_empty_on_bad_image` is **not in the failure list** (passed; full-suite ordering quirk documented, not chased).

Command: `.venv/bin/python -m ruff check app/ test/` → `All checks passed!` (0 findings). PASS.

## 3. Eval A/B (caption-style fine query vs term query + coarse sanity) — PASS

Sibling script `.scratch/rerank-eval/caption_ab_eval.py` (imports the existing harness; no rewrite). Real API calls; credentials read via app config loading from the user `config.toml`. Console `eval-ab.log`, JSON `eval-ab-results.json`.

**Probe 1 — caption vs term, over the exact saved eval candidates (5 terms, SiliconFlow Qwen3-VL-Reranker-8B):**

| term | top-10 overlap | kendall-tau |
|---|---|---|
| giant panda walking giant panda | (per-term values in eval-ab.log / results JSON) | |
| bamboo forest China giant panda | " | |
| panda eating bamboo giant panda | " | |
| panda chewing closeup giant panda | 8/10 | +0.813 |
| fresh bamboo stalks giant panda | 7/10 | +0.569 |

Summary (verbatim from `eval-ab.log`): `SUMMARY: mean top-10 overlap = 7.4/10, mean kendall-tau = +0.589 over 5 terms`. Caption queries are hand-written in the `fine_query` style of `video_match._build_prompt` (one English sentence, precise visual moment) — deterministic and reproducible. The reranker ranks real pandas (e.g. asset 38863023) top under BOTH query styles; caption queries reshuffle habitat-filler orderings moderately. PASS.

**Probe 2 — coarse cosine sanity via `app.services.image_embedding.embed_text` / `embed_image` (real DashScope calls, dim=768):**

```
30757215  cos=+0.1869  panda (expected top: real giant panda walking)
28154058  cos=+0.1355  bamboo forest (expected middle: habitat filler)
36698122  cos=+0.0444  giraffe (expected bottom: irrelevant)
ordering: 30757215 > 28154058 > 36698122  ->  MATCH
```

Expected ordering reproduced exactly. PASS.

## 4. Live happy path (English-subject deviation per §0) — PASS (mechanics)

Submit: `POST /api/v1/videos` → task **`c40ec665-668e-4f11-9b53-14938f5cdd0f`** (`submit-response-happy.txt`; request `request-happy.json`, 12 segments, portrait, pexels). Completed state=1 progress=100; `run-api-happy.log:1892`:
`segment-first task c40ec665-668e-4f11-9b53-14938f5cdd0f finished, generated 1 videos.`
Video: `storage-happy/tasks/c40ec665-…/final-1.mp4` (39,210,063 bytes).

- **Stale state**: image rebuilt from HEAD (`image-build.log` ends `#12 DONE`); new image ID `15cc812b3408` ≠ prior `6fb9fd9a8fad`; in-container `grep -n vlm_walk_limit app/services/material_rerank.py` → lines 79, 81 (today's code runs). PASS.
- **Gates wired**: `run-api-happy.log:158` `image embedding gate enabled: model=tongyi-embedding-vision-flash, threshold=0.68, duplicate=True`; `:159` `vlm pre-download material filter enabled`. PASS.
- **Per-segment query generation (terms + coarse + fine)**: e.g. segment 0 — terms visible in search lines `:162` `"giant panda Why giant pandas are China's national treasure"`, `:183` `"misty bamboo forest …"`, `:204` `"panda walking …"`; coarse query at `:288` `coarse_rank - video match: coarse rank query='A giant panda lives deep in the misty bamboo forests of central China…' pool=84 selected=30 duplicates=0`; fine (caption) query at `:319` `material rerank selected: query='A giant panda slowly walks through dense fog between tall bamboo stalks…', ranked=30, fallback=True`. PASS.
- **6 searches per segment (3 terms × pages 1-2)**: segment 0 logs exactly 6 fresh searches `:162,:183,:204` (page 1) + `:226,:245,:266` (page 2). Whole run: 52 fresh `segment search returned` lines, page distribution 26×page=1 + 26×page=2 (nothing outside 1-2); the 12 segments issue 72 (term,page) lookups, 20 of which were (term,page)-memoized cross-segment (designed memo behavior). PASS.
- **Coarse embed activity + dup gate at the coarse boundary**: 10 `video match: coarse rank query=… pool=77-97 selected=30 duplicates=0` lines (`:288,:420,:590,:731,:876,:1061,:1343,:1523,:1616,:1794`); every line carries the dup-gate `duplicates=` counter (0 near-duplicates encountered this run; walk-level gate active too, 0 `embedding gate flagged duplicate` lines). 2 additional `coarse rank failed, fail-open: reason=missing coarse query` lines (`:1162,:1193`) are the designed fail-open for the two segments whose LLM query generation hiccuped (see below). PASS.
- **Fine rerank, ranked ≤ 30**: 10 `material rerank selected` lines, every one `ranked=30` (`:319,:451,:621,:762,:907,:1092,:1374,:1554,:1647,:1825`); 300 `material rerank score` lines (10×30, first `:289` `asset_id=36798463, score=0.0182605516165494…`, real float spread); 2 blocks `fallback=True` (`:319,:1374`) = the designed HTTP-400 → base64 data-URI retry, **0** `material rerank failed` / `no credentials` lines (real rerank, not fail-open). PASS.
- **VLM walk ≤ 10 with early exit**: 117 `vlm filter verdict` lines = seg0's 7 + 11 segments × 10 (walk cap); per-segment summaries (`:337,:475,:647,:788,:931,:1116,:1187,:1217,:1398,:1578,:1671,:1849`) show `vlm_judged` ∈ {7, 10} — never above the limit. Early-exit evidence: segment 0 filled its quota at 3 accepted (`:325,:330,:335` `vlm filter accepted candidate: … verdict=relevant`) after only 7 judgments → loop exited before reaching 10 (`:337` `clips=3/3 … vlm_judged=7`). Accepted-total cross-foot: 6 accepted lines ↔ 6 `relevant` verdicts; 5 `uncertain candidate skipped` lines (new semantics: uncertain = skip, no last-resort adoption); everything else rejected. PASS.
- **Image-gen backfill for unfilled windows**: 11 `segment N: video match: image-gen backfill windows=… duration=…` lines (first `:472`, last `:1846`) + 11 `image-gen fallback engaged: clip=imgclip-….mp4` lines; 11/12 segments carried ≥1 generated clip (segment 0 filled 3/3 from search). Observed edge note: most backfills log `windows=0 duration=5.000` — segments whose window plan is empty/short fall back to the configured clip duration (code path `windows[-1:] or clip_duration`), by design. PASS.
- **Legacy machinery: zero matches** in the entire happy log for `prefiltered|skip_coarse|promoting|deferred_uncertain|rerank_page|extract_terms_for_segments` — each greps to **0**. PASS.
- **Designed degradation observed (not a defect)**: segments 6 and 7 — per-segment LLM query generation failed after retries (`:1121,:1192` `segment query generation failed after retries; returning empty queries…`) → subject-term search only (`levels_tried=1` in summaries `:1187,:1217`), coarse fail-open (`:1162,:1193`), no fine rerank, image-gen backfill; task completed. Mirrors the transient LLM hiccups documented in the prior QA run. PASS.
- **Zero Tracebacks** in the happy log. PASS.

## 5. VLM-off sub-run — PASS

Scratch config copy `config-vlm-off.toml` (line-for-line copy of the user config; ONLY `[vlm] enabled` flipped to false at line 131; user's real config untouched, mounted `:ro` in the happy container and never in this one). Second container `mpt-vm-qa-vlmoff` (host 8093), same image `mpt-video-match:qa`; in-container probe printed `vlm.enabled = False` before submission.

Submit → task **`86579c78-f43d-4971-b797-6f57e140c0e7`** (`submit-response-vlmoff.txt`). Completed state=1 progress=100 in ~14 min (11:46:53 → 12:00:52, `run-api-vlmoff.log:194`); video `storage-vlmoff/tasks/86579c78-…/final-1.mp4` (10,146,853 bytes).

- **`video match: vlm disabled, image-gen only` present per segment**: 9 lines (`run-api-vlmoff.log:93,:99,…`, emitted at `video_match.py:561`), one per segment (9 segments). PASS.
- **Every segment image-gen only**: 9 `image-gen backfill` + 9 `image-gen fallback engaged` lines; 9/9 summaries `clips=1/3, fallback_level=subject, image_gen=1`. PASS.
- **ZERO search calls**: `segment search returned` = **0**; also 0 coarse-rank, 0 rerank, 0 VLM lines in the whole log. PASS.
- Legacy machinery grep = 0 for all six patterns. PASS.

## 6. Adversarial classes

- **misleading_success_output**: every assertion above cites file + line numbers + short verbatim excerpts; counts cross-foot (117 verdicts = 7 + 11×10; 300 score lines = 10×30; 52 = 26+26 pages; 9 disabled-lines = 9 segments).
- **stale_state**: rebuild from HEAD proven (new image ID; in-container code grep; container recreated).
- **hung/long commands**: docker build (~2 min) and both generations run under poll loops with per-iteration snapshots (`poll-*.log`); happy run ~42 min (submitted 10:55:28, finished ~11:37:40 — exceeds the ~20 min note, reported here as expected for 12 segments × full funnel), VLM-off ~14 min.
- **cancel/resume**: first submission of the happy lane (task `0df39803`) failed fast at stage=terms — captured, root-caused to the §0 defect (NOT retried blindly); the retry was a documented-deviation English-subject run, not a silent retry. VLM-off ran once, clean.
- **malformed input**: N/A this lane (fail-open behavior separately evidenced by segments 6/7 LLM hiccup + rerank 400→base64 fallbacks).
- **flaky tests**: single suite run; failure set byte-matches the documented baseline + the config-dependent KeyError.

## 7. Cleanup receipts (f)

- Containers `mpt-vm-qa`, `mpt-vm-qa-vlmoff` → `docker rm -f` after the runs ✓ (docker ps shows no mpt-vm-* containers)
- Stale `mpt-uat` container (pre-refactor image, held port 8092) → removed; recreatable any time via `.scratch/uat-run.sh` ✓
- `config-vlm-off.toml` (contained the user's API keys) → shredded (`shred -u`), never staged ✓
- Kept: image `mpt-video-match:qa` (`15cc812b3408`), prior image `mpt-segment-first:uat` (`6fb9fd9a8fad`), storage dirs `storage-happy/` + `storage-vlmoff/` (task evidence incl. videos), all raw logs in this directory — all untracked
- tmux helper sessions (`qa-build`, `qa-pytest`, `qa-eval`, `qa-poll*`) → killed ✓
- The user's live `config.toml` was never modified (mounted `:ro` in the happy container; never mounted writable anywhere)

## 8. Risks

- **§0 defect is ship-blocking for CJK subjects**: the repo's primary demo flow (Chinese subject + Chinese script) fails at progress=10 on HEAD. Suggested fix direction (for the owner, not applied): drop or rework the `18e6c8d` guard — in the new architecture English terms are generated per-segment inside `match_segments`, and empty-term segments degrade to image-gen instead of being unrunnable; at minimum the guard must not fire when the segment-first pipeline no longer pre-computes terms.
- Happy run: 11/12 segments needed image-gen backfill (panda terms on pexels remain junk-heavy — consistent with the prior QA run's finding); VLM savings are partial for junk-heavy terms.
- 2/12 segments lost their per-segment query package to transient LLM errors (retried once each, then designed fail-open) — cost-control retry budget (2 attempts) may be tight for flaky endpoints.
- The user config's stale `top_n: 5` key under `[material_rerank]` is silently ignored; a config migration note (or parser warning) would prevent user confusion.
