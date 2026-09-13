# QA assertions — todo 7 live segment-first run with material reranker

Date: 2026-09-08 · Branch: `feat/segment-first-pipeline` (HEAD 625781d) · Image: `mpt-segment-first:uat` (rebuilt this run) · Container: `mpt-uat` (API on host 8092)

Evidence files in this directory:
- `run.log` — happy-path run (docker logs + in-container API log `/tmp/api.log`, both streams)
- `manifest-3129367d.txt` — md5sum manifest of the happy-path task dir (54 files; `script.json` unreadable: root-owned inside container, same permission quirk as prior runs)
- `run-failopen.log` — fail-open probe run (in-container API log)
- `request.json` / `submit-response.txt`, `request-failopen.json` / `submit-response-failopen.txt` — exact curl bodies and API responses
- `run-docker.log` — streamlit (webui) docker-logs stream (noise only; API logs live in run.log)

All line numbers below refer to the de-ANSI'd content of the named log file.

## 0. Full pytest suite (gate)

Command: `.venv/bin/python -m pytest test/ -q` (single run, 264.69s)

Exact summary line:

```
26 failed, 760 passed, 11 skipped, 20 warnings, 4067 subtests passed in 264.69s (0:04:24)
```

Failure breakdown vs AMENDED BASELINE (ledger `baseline-amended`: ~10 test_task + 16 test_llm + 1 test_webui_voice_preview = 26):

| File | Count | Detail |
|---|---|---|
| `test/services/test_llm.py` | 16 | all `TestLiteLLMProvider::*` (stale newline assertions, pre-existing) |
| `test/services/test_task.py` | 9 | 5 top-level (`test_start_completes_video_without_cross_posting`, `test_start_generates_youtube_metadata_for_each_cross_post`, `test_start_returns_before_cross_post_worker_runs`, `test_start_returns_cross_post_scheduling_failure`, `test_start_stops_before_materials_when_term_provider_fails`) + 4 SUBFAILED (`test_start_marks_pipeline_failures` stage=materials/video, `test_start_returns_each_intermediate_result` stop_at=audio/materials) |
| `test/services/test_webui_voice_preview.py` | 1 | `test_non_default_volume_regenerates_audio_without_double_gain` |

**GATE: PASS** — exactly the 26 pre-existing failures, zero new failures. (Passed count 760 vs baseline 764 is explained by todo 4–6 test deletions — 15 coarse tests + G3 audit tests + 1 wiring test removed — partially offset by the new `test_material_rerank.py`; failure set is byte-identical to baseline.)

## 1. Stale-state probe (rebuilt image runs today's code)

Image rebuilt: `docker build -t mpt-segment-first:uat .` → exit 0. Container recreated from it (`docker rm -f mpt-uat` + run per `.scratch/uat-run.sh` recipe, ports 8092→8080 / 8091→8501).

- `docker exec mpt-uat grep -c "material_rerank" /MoneyPrinterTurbo/app/services/segment_material.py` → **3** (> 0 ✓)
- `docker exec mpt-uat python -c "from app.config import config; print(dict(config.material_rerank))"` →
  `{'enabled': True, 'model': 'Qwen/Qwen3-VL-Reranker-8B', 'top_n': 5, 'timeout': 120, 'api_key': '', 'base_url': ''}` — exactly the parser defaults ✓

Note: the live `config.toml` DOES contain a `[material_rerank]` section (line 168) whose values are identical to the defaults — the orchestrator brief's "no section" note was stale but behaviorally equivalent. Live config was never edited.

**STALE_STATE: PASS** — container provably runs today's code.

## 2. Happy-path run

- Submit: `curl -s -i -X POST http://127.0.0.1:8092/api/v1/videos -H "Content-Type: application/json" -d @request.json` → HTTP 200, task_id **`3129367d-58d1-4bcf-b9ef-e657df5f483b`** (`submit-response.txt`)
- Subject `大熊猫的野外日常`, 9 segments, portrait, pexels source, zh-CN-XiaoxiaoNeural voice
- Completed: state=1 (success), progress=100 at 04:24:12 — run.log:2288 `segment-first task 3129367d-58d1-4bcf-b9ef-e657df5f483b finished, generated 1 videos.`
- Video: `.scratch/uat-storage/tasks/3129367d-58d1-4bcf-b9ef-e657df5f483b/final-1.mp4` (67,077,548 bytes) ✓

### (a) Rerank score + selection lines — PASS

- 425 `material rerank score: asset_id=..., score=..., rank=...` lines (first :61, last :2236), 31 `material rerank selected: term=..., ranked=..., top=..., fallback=...` summary lines (run.log :78, :138, :206, :287, :386, :481, :572, :665, :739, :788, :886, :946, :996, :1053, :1096, :1179, :1227, :1284, :1332, :1427, :1483, :1531, :1580, :1635, :1739, :1794, :1895, :1992, :2088, :2142, :2237) across 20 distinct terms (page-2 reranks included, e.g. 'bamboo forest dawn' :78/:138, 'panda scent marking' :886/:946).
- Every one of the 31 blocks has ≥ 1 score line (0 blocks with ranked=0). At least one rerank per processed (term, page): ✓
- Scores are plausible SiliconFlow floats: min 0.000735, max 0.466592, 413 distinct values across 425 lines — a real spread, not dummy constants (e.g. :61 score=0.1925002634525299 rank=1; :2236 score=0.058212775737047195 rank=14).

### (b) Zero prefilter/promotion lines — PASS

- `grep -c "prefiltered"` → **0**
- `grep -c "promoting prefiltered"` → **0**
- `grep -c "skip_coarse"` → **0**

### (c) Top-5 boundary — PASS with documented designed tail-walk

Cross-check results (scripted over run.log):

- **407/407 `vlm filter verdict` asset_ids map back to a `material rerank score` line of the same run** — 0 orphans, 0 no-thumbnail stragglers (all pexels candidates had thumbnails). The VLM never judged an unranked candidate.
- **Judging follows rerank order strictly.** Per (term, page) block, the first-encounter judged sequence is an exact prefix of the rerank-ordered list in 28/31 blocks; the 3 exceptions are fully explained by the KEPT duplicate gate: 6 `embedding gate flagged duplicate` lines (run.log :1813 asset 6318894, :2011 asset 26845207, :2091 asset 36980298, :2111 asset 6318886, :2178 asset 6318886, :2180 asset 6318894) short-circuit BEFORE the VLM and therefore emit no `vlm filter verdict` line by design. Counting duplicate-gated assets as processed, **31/31 blocks are exact prefixes — 0 ordering violations.**
- **Strict "rank ≤ 5 only" does not hold verbatim, by design:** 250/407 verdicts have rank > 5. This is the plan's fixed interface contract (plan line 53): `rerank_page` hands onward `[top-5] + [no-thumbnail tail] + [remaining rankable in rerank order]`; the judge walks that list in order and only reaches the tail when the top-5 did not fill the clip quota (unit test `test_judge_receives_reranked_top5_in_ranked_order`, test_segment_material_quota.py:335: "名额打满后尾部候选不再判定"). Observed: 10/31 blocks resolved entirely within the top-5 (tail never judged); 21/31 blocks walked into the ranked tail because panda-term pexels pages are full of hikers-in-bamboo junk that the VLM rejected (e.g. 'bamboo forest dawn' page 1: top-5 yielded 0 relevant → judged all 17 in rank order → page 2). Top-5 PRIORITY is proven in every block: judging always starts at rank 1 and proceeds in rank order.
- Internal consistency: per-segment `vlm_judged` counts (68+49+20+58+23+58+42+54+41 = 413) = 407 verdict lines + 6 duplicate-gate lines. ✓

### (d) Clips downloaded only from judged candidates — PASS

- 15 `downloading segment clip` events; **15/15 map to an asset with a preceding `vlm filter accepted candidate` line** (accepted verdict events: 13 `relevant` + 15 `uncertain` — uncertain accepted as designed last-resort deferral). 0 downloads without an accepted verdict.
- 3 `image-gen fallback engaged` lines (run.log :624 segment 1, :1058 segment 3, :2050 segment 7) — starved segments fell to the subject concept-image fallback, documented and by design (their `material resolution summary` shows `fallback_level=subject, image_gen=1`).

### (e) Pipeline completes with clips — PASS

- 9/9 `material resolution summary` lines with non-zero clips (run.log :348 clips=1/3, :625 1/3+image_gen, :748 3/3, :1059 1/3+image_gen, :1188 3/3, :1498 2/3, :1752 3/3, :2051 1/3+image_gen, :2260 3/3).
- Task state success (state=1, progress=100); `final-1.mp4` present (67 MB); assembler concatenated 19 segment clips (run.log :2245 area, `concatenating 19 segment clips with ffmpeg`).

### (f) REAL rerank happened (not fail-open) — PASS

- 425 score lines with plausible float spread from SiliconFlow (see (a)); **0** `material rerank failed, fail-open` lines; **0** `material rerank request failed` warnings; **0** `material rerank skipped, no credentials`; **0** `material rerank page selection failed` (belt-and-braces); **0** Tracebacks in the whole run.
- `via=direct`→proxy failover warnings: **0 observed** — the direct-first requests succeeded (container proxy env present but not needed by the reranker). N/A this run.
- 2 blocks report `fallback=True` (run.log :78 'bamboo forest dawn', :1227 'panda climbing tree') = the DESIGNED HTTP-400 → base64 data-URI retry path succeeded (SiliconFlow server-side thumbnail fetch hiccup), NOT fail-open. 29/31 blocks reranked on the first URL attempt (`fallback=False`).

## 3. Fail-open probe run

- Scratch config copy `config-failopen.toml` (copy of live config; ONLY the `[material_rerank] model` line changed to `nonexistent/wrong-model-qa`; **file deleted after the run, never committed**). Note: live config already had a `[material_rerank]` section, so the edit replaced the model line in place (an appended duplicate section would have broken TOML parsing — caught and fixed pre-run).
- Second container `mpt-uat-failopen` (ports 8093/8094), separate storage `.scratch/uat-storage-failopen`. In-container probe printed `{'enabled': True, 'model': 'nonexistent/wrong-model-qa', ...}` before submission.
- Submit: task **`f5510ed4-f61e-41e0-9154-a6a921af0286`** (subject `城市清晨的咖啡香气`, 6 segments) → state=1 success, `final-1.mp4` present (26,909,536 bytes).

Assertions:

- **`material rerank failed, fail-open: term=..., error=_RerankUnavailableError` lines appear: 7** (run-failopen.log :223 'pre dawn cityscape', :257 'cafe door opening', :316 'coffee grinder beans', :348 'commuters morning rush', :416 'sunrise city skyline', :451 'tree shadow sidewalk', :499 'empty road morning') — one per (term, page) rerank attempt.
- **0** `material rerank score` lines and **0** `material rerank selected` lines — the reranker never succeeded, so no score audit lines (correct: score lines are only emitted on success).
- **Pipeline still completed with clips: all 6 segments 3/3** (run-failopen.log resolution summaries at :348/:625-equivalents — segments 0–5 all `clips=3/3`), VLM judged candidates in provider order (per-page `vlm_judged` = 3, 20, 19, 3, 9, 7 — full pages judged, no top-5 cut), video produced.
- Environmental note: two earlier submissions to this container (`a1c66342-313a-4c35-a79e-1f2fd17130a5`, `6a672bb8-a27a-42ec-a26a-ec916ca640bf`) failed at the SCRIPT stage with transient LLM `Connection error.` before any material work — endpoint hiccup, unrelated to the reranker (LLM verified healthy immediately after via a direct `llm.generate_script` call inside the same container). Third submission ran clean.

## 4. Adversarial classes

- **stale_state**: PASS — rebuilt image probes in §1 (grep count 3; defaults printed in-container).
- **misleading_success_output**: PASS — score values verified as plausible floats with real spread (413 distinct / 425, 0.000735–0.466592), not constants; assertion (c)/(d) are structural cross-checks (prefix property, download↔verdict mapping), not grep-happy; vlm_judged sum cross-foots with verdict+dup-gate line counts.
- **hung_commands**: PASS — docker build and pytest run with 30-min timeouts; generation polled at 2–8 min intervals; no unbounded waits. Log followers ran in tmux sessions and were killed after each run.
- **dirty_worktree**: handled at commit step — only assertions.md staged; run.log / run-failopen.log / manifests / request files stay untracked; config-failopen.toml (contained API keys) deleted before commit.
- **flaky_tests**: single full-suite run; failure set byte-identical to amended baseline (deterministic stale assertions, reproduced in prior ledger entries).
- **cancel/resume**: N/A — no cancellable background tasks used; long waits were sleep-polls in foreground commands with timeouts.
- **repeated interrupts**: N/A — single-session execution; no interruptions occurred.
- **malformed input**: covered end-to-end by the fail-open probe (wrong model name = malformed provider config → fail-open lines, pipeline completes).
- **prompt injection**: N/A — no untrusted external content processed in this QA lane.

## 5. Cleanup receipts

- `docker rm -f mpt-uat-failopen` → removed ✓
- `config-failopen.toml` (contained API keys) → deleted from this directory ✓ (not in dir listing; never staged)
- `.scratch/uat-storage-failopen/` → KEPT as fail-open evidence (contains task dirs for f5510ed4 + the 2 aborted script-stage tasks; bgm/cache_material_search/tasks)
- `mpt-uat` (rebuilt image) → left running as the established UAT env; `docker ps` at end: `mpt-uat  Up About an hour  127.0.0.1:8092->8080/tcp, 127.0.0.1:8091->8501/tcp` ✓
- Untracked-by-design in this dir: `run.log`, `run-docker.log`, `run-failopen.log`, `manifest-3129367d.txt`, `request*.json`, `submit-response*.txt` (large logs; no secrets in request/response files, but kept untracked per plan commit strategy)

## 6. Risks

- The strict "only top-5 reach the VLM" reading does not hold when a page's top-5 yields fewer relevant clips than the segment quota: the ranked tail is then judged as a same-page backstop (plan-fixed interface, 21/31 blocks). VLM call savings therefore depend on term quality — for junk-heavy pexels panda terms the saving was partial, not the advertised ~70-75%. This is the plan's documented accepted consequence, not a defect.
- 3/9 happy-path segments fell to image-gen fallback (panda terms on pexels are junk-heavy); pipeline completed regardless.
- Transient LLM endpoint connection errors were observed twice during the fail-open lane (script stage); retry succeeded. Unrelated to the reranker but worth watching in CI-like environments.
