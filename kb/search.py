"""
Query the Supabase KB using local embeddings + pgvector match function.
"""

from __future__ import annotations

import os
from typing import List, Dict
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client
from ollama import Client as OllamaClient
import numpy as np
import hashlib


def _fake_embed(text: str, dim: int = 768) -> list[float]:
    seed = int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def embed_query(ollama: OllamaClient, model: str, text: str) -> list[float]:
    if os.getenv("FAKE_EMBEDDINGS", "0") == "1":
        return _fake_embed(text)
    resp = ollama.embeddings(model=model, prompt=text)
    emb = resp.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError("Ollama embeddings response missing 'embedding' list")
    return emb


def search(query: str, k: int = 5) -> List[Dict]:
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    # Default to gemma3:latest since it's installed locally; override via EMBEDDING_MODEL if desired
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemma3:latest")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_*_KEY in environment.")

    supa = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    # Use local Ollama without specifying a URL
    ollama = OllamaClient()

    q_emb = embed_query(ollama, EMBEDDING_MODEL, query)
    rpc = supa.rpc("match_chunks", {"query_embedding": q_emb, "match_count": k}).execute()
    return rpc.data or []


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what grows well with tomatoes?"
    results = search(q, k=5)
    for i, r in enumerate(results, 1):
        sim = r.get('similarity')
        doc_id = r.get('doc_id')
        content = (r.get('content') or '')[:300]
        sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else "n/a"
        print(f"[{i}] sim={sim_str} doc={doc_id}\n{content}\n---")
