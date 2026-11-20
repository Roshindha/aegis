# Decision log

Format: decision. Reason.

- Committed `corpus/chunks.jsonl` (built chunks, MIT-derived) outside `data/`. CI, Docker and HF Spaces need the corpus but `data/` is gitignored and the raw repo is large.
- Added `db/schema.sql` introspected from the live table. Makes the setup reproducible.
- Benchmark has 124 questions, not ~100: no_level covers all 10 level-less controls plus 3 missing-class questions (13); out_of_scope raised from 15 to 26. Coverage and a usable negative class for threshold calibration beat round numbers (15 negatives is too few for cross-validation).
- Added `crosswalk_reverse` (5 questions, control -> external refs) so Step 11's reverse direction is evaluated.
- Crosswalk gold comes from the same `references` data a crosswalk index would use, so post-fix scores measure query routing, not independent truth. Stated as a limitation.
- Template generator is seeded (`random.Random(7)`) with sorted candidates, so qids are stable for a given corpus.
- Validator skips unknown-ID checks on question text when the question names NIST/MVSP/CIS/SLSA/CCM (NIST IDs like AC-2 collide with IM8 ID format) or is flagged `unknown_id_probe`.
- Curated questions (61) are AI-drafted from the control text, `source: curated`, `verified: false`. User verifies later.
- Retrieval has two backends with the same model and data: `local` (default; reads `corpus/chunks.jsonl`, vectors in memory) and `db` (`AEGIS_BACKEND=db`, Supabase). Needed so tests, CI, Docker and HF Spaces run without a database. Default flipped from db to local.
- `rrf_fuse` now copies hits instead of mutating the caller's lists (needed for the per-stage Pipeline Explorer; test added).
- Guardrail = regex pre-filter (injection, other frameworks/laws, unknown control IDs) + rerank-score threshold. Refusal is decided in code before the LLM is called; the LLM is never asked to refuse.
- Lookup-style questions (names a real control ID, a group title, or a crosswalk reference) bypass the threshold: the cross-encoder scores relevance of prose, and these questions get low scores although they are answerable (seen: group_list and crosswalk questions at -6 to -8).
- Threshold calibrated by 5-fold stratified CV (seed 13) on non-bypassed questions; reported F1 is held-out. Result is weak on its own (CV F1 ~0.49): the MS-MARCO cross-encoder cannot separate off-topic from terse in-scope queries well. Prefilter patterns were written after seeing the benchmark negatives, so prefilter numbers are in-sample and optimistic; ablations report each layer separately.
