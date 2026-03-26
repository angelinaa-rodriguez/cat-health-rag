from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb

from petmed_rag.config import PROCESSED_DIR, CHROMA_DIR, settings
from petmed_rag.embeddings import get_embedder
from petmed_rag.ingestion.chunk import chunk_text


def ingest_processed_documents(batch_size: int = 100) -> int:
    base = PROCESSED_DIR
    meta_dir = base / "_meta"

    if not base.exists():
        raise FileNotFoundError(f"Processed directory not found: {base}")

    if not meta_dir.exists():
        raise FileNotFoundError(f"Processed metadata directory not found: {meta_dir}")

    embedding_fn = get_embedder(
        embedder="openai",
        openai_model=settings.embedding_model,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(name=settings.collection_name)

    total_added = 0
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal ids, docs, metas, total_added

        if not docs:
            return

        embeddings = embedding_fn(docs)
        col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

        total_added += len(docs)
        ids, docs, metas = [], [], []

    txt_files = sorted([p for p in base.glob("*.txt") if p.is_file()])
    if not txt_files:
        raise FileNotFoundError(f"No processed .txt files found in {base}")

    print(f"Ingesting from: {base.resolve()}")

    for txt_path in txt_files:
        meta_path = meta_dir / f"{txt_path.stem}.json"
        if not meta_path.exists():
            print(f"Missing metadata for {txt_path.name}, skipping.")
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        base_metadata = {
            "doc_id": meta.get("id", txt_path.stem),
            "source_id": meta.get("id", txt_path.stem),
            "source_path": str(txt_path),
            "file_name": txt_path.name,
            "title": meta.get("title") or meta.get("extracted_title"),
            "publisher": meta.get("publisher"),
            "url": meta.get("final_url") or meta.get("url"),
        }

        chunks = chunk_text(
            text=text,
            base_metadata=base_metadata,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for ch in chunks:
            ids.append(ch.chunk_id)
            docs.append(ch.text)
            metas.append(ch.metadata)

            if len(docs) >= batch_size:
                flush_batch()

    flush_batch()

    print(f"Ingested {total_added} chunks into collection '{settings.collection_name}'")
    print(f"Chroma persisted at: {CHROMA_DIR}")

    return total_added