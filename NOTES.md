# Decision log

Format: decision. Reason.

- Committed `corpus/chunks.jsonl` (built chunks, MIT-derived) outside `data/`. CI, Docker and HF Spaces need the corpus but `data/` is gitignored and the raw repo is large.
- Added `db/schema.sql` introspected from the live table. Makes the setup reproducible.
