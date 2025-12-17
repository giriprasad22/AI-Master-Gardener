# Supabase Knowledge Base (KB)

This folder sets up a semantic KB in Supabase using pgvector and local Ollama embeddings.

## What you get
- SQL to enable pgvector and create `documents` and `chunks` tables
- A Postgres function `match_chunks` for vector search
- `ingest.py` to chunk local docs and insert with embeddings
- `search.py` to query the KB via embeddings + RPC
- `.env.example` with required settings

## Prerequisites
- Supabase project (URL + keys)
- pgvector extension enabled (see SQL below)
- Local Ollama running with an embedding model (default `nomic-embed-text`)

## 1) Configure Supabase
Run these in Supabase SQL editor (Project > SQL):

- `sql/001_enable_pgvector.sql`
- `sql/010_schema.sql`

Note: Schema uses `vector(768)` for `embedding` (matching `nomic-embed-text`). If you use a different embedding model with another dimension, adjust both SQL files accordingly and re-create the table/function.

## 2) Environment variables
Copy `.env.example` to `.env` and fill in values:

- `SUPABASE_URL` – your project URL (https://xxx.supabase.co)
- `SUPABASE_SERVICE_ROLE_KEY` – service role key (server-side writes); keep it secret
- `SUPABASE_ANON_KEY` – optional for search; service key also works
- `EMBEDDING_MODEL` – e.g., `nomic-embed-text`

Optional:
- `CHUNK_SIZE` (default 1000), `CHUNK_OVERLAP` (default 200)
- `SOURCE_DIR` (default `docs`)

## 3) Install dependencies (Windows PowerShell)
```
& "..\.venv311\Scripts\python.exe" -m pip install -r ".\kb\requirements.txt"
```

## 4) Ingest documents
Place files in `kb/docs` (create the folder if missing). Supported: `.txt`, `.md`, `.html`, `.pdf`.

```
& "..\.venv311\Scripts\python.exe" ".\kb\ingest.py"
```

This will:
- Insert a row in `documents` per source file
- Chunk the file and create `chunks` with embeddings

## 5) Search
```
& "..\.venv311\Scripts\python.exe" ".\kb\search.py" "what grows well with tomatoes?"
```

Returns the top matches with similarity and content.

## Notes
- If vector inserts fail via PostgREST, confirm pgvector is enabled and the `embedding` column uses the right dimension.
- For large ingestions, consider limiting batch sizes or adding basic rate limits on embedding requests to Ollama.
