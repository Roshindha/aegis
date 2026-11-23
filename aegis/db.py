"""Database connection helper."""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    conn = psycopg.connect(url, connect_timeout=5)
    # Supabase installs pgvector in the "extensions" schema. Put it on the search path,
    # otherwise Postgres can't find the vector type or the <=> operator.
    conn.execute("set search_path to public, extensions")
    conn.commit()
    register_vector(conn)   # lets us pass numpy arrays straight into vector columns
    return conn
