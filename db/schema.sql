-- Aegis: schema for the `chunks` table (mirrors the live Supabase table).
-- Run in the Supabase SQL editor. Safe to re-run.

-- Supabase keeps extensions in their own schema; aegis/db.py puts it on search_path.
create extension if not exists vector with schema extensions;

create table if not exists public.chunks (
    chunk_id        text primary key,                 -- e.g. im8-2025.05.13-as-5
    control_id      text  not null,                   -- e.g. AS-5
    title           text  not null,
    group_id        text  not null,
    group_title     text  not null,
    statement       text  not null,
    guidance        text  not null,
    risk_statement  text  not null,
    levels          jsonb not null default '{}'::jsonb, -- {"low-risk": 1, "medium-risk": 0}; {} = no level
    refs            jsonb not null default '[]'::jsonb, -- [{framework, ref_id, raw}, ...]
    published       text,
    last_modified   text,
    catalog_version text  not null,
    source_url      text  not null,
    content         text  not null,                   -- what is embedded and keyword-indexed
    embedding       extensions.vector(384) not null   -- BAAI/bge-small-en-v1.5
);

-- RLS on with NO policies: the anon/authenticated API roles see nothing.
-- The app connects with the database owner's connection string, which bypasses RLS.
alter table public.chunks enable row level security;
