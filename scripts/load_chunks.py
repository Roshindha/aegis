"""Step 3: embed every chunk and upsert it into Supabase."""
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from aegis.db import connect
from aegis.embed import EMBEDDING_DIM
from aegis.torch_ref import embed_documents   # torch: offline loading only

CHUNKS = Path("data/processed/chunks.jsonl")

UPSERT_SQL = """
    insert into chunks (chunk_id, control_id, title, group_id, group_title, statement,
                        guidance, risk_statement, levels, refs, published, last_modified,
                        catalog_version, source_url, content, embedding)
    values (%(chunk_id)s, %(control_id)s, %(title)s, %(group_id)s, %(group_title)s,
            %(statement)s, %(guidance)s, %(risk_statement)s, %(levels)s, %(refs)s,
            %(published)s, %(last_modified)s, %(catalog_version)s, %(source_url)s,
            %(content)s, %(embedding)s)
    on conflict (chunk_id) do update set
        content = excluded.content, embedding = excluded.embedding,
        levels = excluded.levels, refs = excluded.refs
"""


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS.read_text().splitlines()]
    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_documents([c["text"] for c in chunks])
    assert vectors.shape == (len(chunks), EMBEDDING_DIM), vectors.shape

    rows = [
        {
            **{k: c[k] for k in ("chunk_id", "control_id", "title", "group_id", "group_title",
                                 "statement", "guidance", "risk_statement", "published",
                                 "last_modified", "catalog_version", "source_url")},
            "levels": Jsonb(c["levels"]),
            "refs": Jsonb(c["references"]),
            "content": c["text"],
            "embedding": vec,
        }
        for c, vec in zip(chunks, vectors)
    ]

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
        count = conn.execute("select count(*) from chunks").fetchone()[0]
    print(f"Done. Rows in table: {count}")


if __name__ == "__main__":
    main()
