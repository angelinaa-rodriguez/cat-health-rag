from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import chromadb

from petmed_rag.embeddings import get_embedder


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])

        if j == n:
            break

        i = j - overlap

    return chunks


def ingest_folder(
    input_dir: str,
    persist_dir: str,
    collection: str,
    embedder: str = "openai",
    openai_model: str = "text-embedding-3-small",
    st_model: str = "all-MiniLM-L6-v2",
    glob_pattern: str = "**/*",
    chunk_size: int = 1200,
    overlap: int = 150,
    batch_size: int = 100,
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
    col = client.get_or_create_collection(name=collection)

    total_added = 0
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    def flush_batch():
        nonlocal ids, docs, metas, total_added
        if not docs:
            return
        embeddings = embedding_fn(docs)
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total_added += len(docs)
        ids, docs, metas = [], [], []

    print(f"ingesting from: {base.resolve()}")

    for p in base.glob(glob_pattern):
        if p.is_dir():
            continue
        print(f"found file: {p} | suffix={p.suffix.lower()}")

        if p.suffix.lower() not in {".txt", ".md"}:
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for idx, ch in enumerate(chunks):
            doc_id = f"{p.as_posix()}::chunk{idx}"
            ids.append(doc_id)
            docs.append(ch)
            metas.append({
                "source_path": str(p),
                "chunk_index": idx,
                "file_name": p.name,
            })

            if len(docs) >= batch_size:
                flush_batch()

    flush_batch()
    return total_added