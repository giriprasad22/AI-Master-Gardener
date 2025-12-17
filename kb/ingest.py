"""
Ingest local documents into Supabase KB with pgvector embeddings.

Sources supported: .txt, .md, .html, .pdf (basic), plus simple fallback for others.
Embeddings: uses local Ollama (default nomic-embed-text) to generate 768-d vectors.
Chunking: simple character-based chunks with overlap.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client
from ollama import Client as OllamaClient

import numpy as np


SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".pdf"}


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(" ")
    if ext == ".pdf":
        try:
            import PyPDF2
        except Exception:
            raise RuntimeError("PyPDF2 not installed; install kb/requirements.txt dependencies.")
        text = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    # Fallback: try text read
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    if chunk_size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # if we've reached the end, stop to avoid repeating the last window
        if end >= n:
            break
        # advance with overlap
        start = max(0, end - overlap)
        # safety: if no progress, break
        if start >= end:
            break
    return chunks


def _fake_embed(text: str, dim: int = 768) -> List[float]:
    import hashlib
    seed = int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def embed_texts(ollama: OllamaClient, model: str, texts: List[str]) -> List[List[float]]:
    if os.getenv("FAKE_EMBEDDINGS", "0") == "1":
        return [_fake_embed(t) for t in texts]
    embeddings: List[List[float]] = []
    for t in texts:
        try:
            resp = ollama.embeddings(model=model, prompt=t)
            vec = resp.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError("Ollama embeddings response missing 'embedding' list")
            embeddings.append(vec)
        except Exception as e:
            raise RuntimeError(
                f"Embedding failed. Ensure the embedding model '{model}' is pulled in Ollama (e.g., 'ollama pull {model}'). Original error: {e}"
            )
    return embeddings


def upsert_document(supa: Client, source: str, title: str | None, metadata: dict) -> str:
    data = {"source": source, "title": title, "metadata": metadata or {}}
    res = supa.table("documents").insert(data).execute()
    if not res.data:
        # Try selecting existing
        sel = supa.table("documents").select("id").eq("source", source).limit(1).execute()
        if not sel.data:
            raise RuntimeError(f"Failed to insert/select document for source={source}")
        return sel.data[0]["id"]
    return res.data[0]["id"]


def upsert_chunks(supa: Client, doc_id: str, chunks: List[str], embeddings: List[List[float]]):
    rows = []
    for i, (c, e) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "content": c,
            "embedding": e,
        })
    # Insert in batches to avoid payload limits
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        supa.table("chunks").insert(rows[i:i+BATCH]).execute()


def main():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    # Use gemma3:latest by default for embeddings to align with simplified model set
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemma3:latest")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    SOURCE_DIR = os.getenv("SOURCE_DIR", "docs")

    if not SUPABASE_URL or not SERVICE_ROLE:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment.")

    supa = create_client(SUPABASE_URL, SERVICE_ROLE)
    # Use local Ollama without specifying a URL; relies on default local socket
    ollama = OllamaClient()

    src_dir = Path(SOURCE_DIR)
    if not src_dir.exists():
        print(f"Source directory '{src_dir}' does not exist. Create it and add .txt/.md/.html/.pdf docs.")
        return

    files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        print(f"No supported documents found in '{src_dir}'. Supported: {sorted(SUPPORTED_EXTS)}")
        return

    for path in files:
        print(f"Ingesting {path}")
        text = read_text(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        embs = embed_texts(ollama, EMBEDDING_MODEL, chunks)
        doc_id = upsert_document(supa, source=str(path), title=path.stem, metadata={"ext": path.suffix})
        upsert_chunks(supa, doc_id, chunks, embs)
        print(f"Inserted {len(chunks)} chunks for {path}")

    print("Done.")


if __name__ == "__main__":
    main()
