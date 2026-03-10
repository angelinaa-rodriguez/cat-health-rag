from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb

from petmed_rag.embeddings import get_embedder  # your wrapper
# If you don't have a chunker yet, this is a tiny one:
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if i >= n:
            break
    return [c.strip() for c in chunks if c.strip()]

def ingest_folder(
    input_dir: str,
    persist_dir: str,
    collection: str,
    embedder: str = "openai",
    openai_model: str = "text-embedding-3-large",
    st_model: str = "all-MiniLM-L6-v2",
    glob_pattern: str = "**/*",
    chunk_size: int = 1200,
    overlap: int = 150,
) -> int:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    embedding_fn = get_embedder(
        embedder=embedder,
        openai_model=openai_model,
        st_model=st_model,
    )

    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(name=collection, embedding_function=embedding_fn)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for p in base.glob(glob_pattern):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for idx, ch in enumerate(chunks):
            doc_id = f"{p.as_posix()}::chunk{idx}"
            ids.append(doc_id)
            docs.append(ch)
            metas.append({"source_path": str(p), "chunk_index": idx, "file_name": p.name})

    if not docs:
        return 0

    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)
