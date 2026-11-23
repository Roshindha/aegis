# Decision log

Format: decision. Reason.

- Committed `corpus/chunks.jsonl` (built chunks, MIT-derived) outside `data/`. CI, Docker and HF Spaces need the corpus but `data/` is gitignored and the raw repo is large.
- Added `db/schema.sql` introspected from the live table. Makes the setup reproducible.
- Benchmark has 124 questions, not ~100: no_level covers all 10 level-less controls plus 3 missing-class questions (13); out_of_scope raised from 15 to 26. Coverage and a usable negative class for threshold calibration beat round numbers (15 negatives is too few for cross-validation).
- Added `crosswalk_reverse` (5 questions, control -> external refs) so Step 11's reverse direction is evaluated.
- Crosswalk gold comes from the same `references` data a crosswalk index would use, so post-fix scores measure query routing, not independent truth. Stated as a limitation.
- Template generator is seeded (`random.Random(7)`) with sorted candidates, so qids are stable for a given corpus.
- Validator skips unknown-ID checks on question text when the question names NIST/MVSP/CIS/SLSA/CCM (NIST IDs like AC-2 collide with IM8 ID format) or is flagged `unknown_id_probe`.
- Curated questions: 61 questions were drafted with LLM assistance; gold labels pending manual verification. (`source: curated`, `verified: false`).
- Retrieval has two backends with the same model and data: `local` (default; reads `corpus/chunks.jsonl`, vectors in memory) and `db` (`AEGIS_BACKEND=db`, Supabase). Needed so tests, CI, Docker and HF Spaces run without a database. Default flipped from db to local.
- `rrf_fuse` now copies hits instead of mutating the caller's lists (needed for the per-stage Pipeline Explorer; test added).
- Guardrail = regex pre-filter (injection, other frameworks/laws, unknown control IDs) + rerank-score threshold. Refusal is decided in code before the LLM is called; the LLM is never asked to refuse.
- Lookup-style questions (names a real control ID, a group title, or a crosswalk reference) bypass the threshold: the cross-encoder scores relevance of prose, and these questions get low scores although they are answerable (seen: group_list and crosswalk questions at -6 to -8).
- Threshold calibrated by 5-fold stratified CV (seed 13) on non-bypassed questions; reported F1 is held-out. Result is weak on its own (CV F1 ~0.49): the MS-MARCO cross-encoder cannot separate off-topic from terse in-scope queries well. Prefilter patterns were written after seeing the benchmark negatives, so prefilter numbers are in-sample and optimistic; ablations report each layer separately.
- Refusal feature chosen by data, not spec: the rerank logit (spec) had held-out F1 0.49 vs 0.85 for top BM25 score (stable threshold across folds). calibrate.py picks the better of the two by CV; the rerank-only variant stays in the ablation table (`threshold_rerank_only_heldout`). Caveat: choosing between two features on the same questions is a mild selection effect.
- API: sync FastAPI endpoints (run in threadpool), SSE via StreamingResponse with one event per pipeline stage; LLM failure (e.g. no GROQ_API_KEY) becomes an `error` event so retrieval stages still render. Explorer page is one static HTML file, no build step.
- Crosswalk (Step 11): chunk text has no references, so prose search cannot answer reference questions. `aegis/crosswalk.py` builds an inverted index (framework, ref_id) -> controls from `parse_reference`, detects "framework + id" in the query (framework name must precede the id, so IM8 `AC-2` and NIST `AC-2` are not confused), and for "does AS-12 map to ..." detects reverse intent. Matched controls go first in the context with an "External references:" line appended; context grows past k=5 (cap 12) so multi-gold answers fit. Detected crosswalk queries bypass the refusal threshold.
- Baseline (pre-crosswalk) eval was run first and kept as `eval/results/baseline-pre-crosswalk.*`; Step 11 numbers are a re-run with the same questions. The crosswalk gold is derived from the same `references` data the index uses, so the post-crosswalk retrieval score measures detection/routing, not independent truth.
- LLM cache key = model + question + context; the system prompt is not in the key, so it was not edited after baseline runs.
- RAGAS ran in a separate venv (its dependency tree is large) on a 15-answer sample against the Groq judge: only 6 scored, 9 failed (judge JSON validation errors, rate limits). Reported as such; not retried to avoid inflating the daily Groq budget.
- Fixed a broken .gitignore line (an append without newline merged `data/processed/` with `eval/cache/`); `data/` is now ignored whole.
- Docker image and GitHub Actions were not executed here (Docker daemon not running; no remote). CI gate floors were checked locally with `python -m eval.gate`.

## Deployment pass (free-tier hosting)
- Serving uses ONNX via fastembed (BAAI/bge-small-en-v1.5; Xenova/ms-marco-MiniLM-L-6-v2 = ONNX export of the same cross-encoder), torch only in requirements-eval.txt. Reason: 512 MB RAM target; torch alone is >300 MB.
- Equivalence check (`python -m scripts.equivalence`, committed as eval/results/onnx-equivalence.json), all 124 benchmark questions: query-embedding cosine min 0.999998, max |rerank score diff| 8.6e-6 over the 20 candidates, top-5 ranking identical on 124/124 (top-1 124/124). Rankings are identical, so the README numbers and CI floors were NOT regenerated; the retrieval-only gate on the ONNX path gives the same values (MRR 0.7698 / 0.9372).
- Document vectors are precomputed with torch into corpus/embeddings.npz (192 KB, guarded by a content digest + test) instead of embedding 137 chunks at start-up; fallback to ONNX embedding if the file is stale.
- ORT memory: fastembed forces graph optimisation ON, which keeps a second copy of the weights (~110 MB for bge-small). `aegis.embed.low_memory_onnx` turns it off, plus 1 thread, no CPU arena, rerank batch size 4. Measured in the container (python:3.12-slim, `--memory 512m`, db backend): uvicorn RSS 334 MB after start-up, 412 MB after 5 queries (peak 414 MB).
- DB resilience: `backend()` returns the active backend; a failed Supabase call (connect timeout 5 s) marks the DB down for 60 s and the snapshot is used; BM25 always indexes the snapshot. /health pings the DB (keeps Supabase active) and reports the active backend.
- Rate limiting is per client IP using the rightmost X-Forwarded-For entry (the one Render's proxy appends; the leftmost is client-controlled). In-memory, single process: limits reset on restart.
- Answer cache stores only error-free runs; Groq 429 or missing key become a "demo quota reached" error event while retrieval stages and the result still stream.
- Explorer moved from app/static to docs/ (GitHub Pages); FastAPI serves the same files at / for local use. The API base URL is the single line in docs/config.js.
- constraints.txt is now the closure of the serving+dev requirements (no torch); constraints-eval.txt is the full eval environment.
- The README's run header still shows commit `02729f5+dirty`: that was the working-tree commit when the run was made; history was later rewritten, so the hash no longer exists. The run itself is unchanged.
