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
