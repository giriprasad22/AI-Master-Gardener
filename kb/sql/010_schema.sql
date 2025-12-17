-- Documents and chunks schema for KB

-- Documents table stores source documents and optional metadata
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  source text not null,           -- e.g., file path or URL
  title text,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

-- Chunks table holds text segments and their embeddings
-- Default vector dimension set to 768 (nomic-embed-text); adjust if you use a different model
create table if not exists public.chunks (
  id bigserial primary key,
  doc_id uuid references public.documents(id) on delete cascade,
  chunk_index int not null,
  content text not null,
  embedding vector(768) not null,
  created_at timestamptz default now()
);

create index if not exists idx_chunks_doc_id on public.chunks(doc_id);
create index if not exists idx_chunks_embedding on public.chunks using ivfflat (embedding vector_l2_ops) with (lists = 100);

-- RPC function to perform semantic search
create or replace function public.match_chunks(
  query_embedding vector(768),
  match_count int default 5,
  min_content_length int default 10
)
returns table (
  id bigserial,
  doc_id uuid,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select c.id, c.doc_id, c.content,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  where length(c.content) >= min_content_length
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;
