from __future__ import annotations
import yaml

from pathlib import Path
from typing import List, Dict, Any
import chromadb

from petmed_rag.ingestion.chunk import chunk_text
from petmed_rag.embeddings import get_embedder


def load_source_metadata() -> dict[str, dict]:
    sources_file = Path("./data/sources.yaml")
    if not sources_file.exists():
        return {}

    with sources_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    out = {}
    for src in config.get("sources", []):
        src_id = src.get("id")
        if src_id:
            out[src_id] = src
    return out

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

    source_metadata_map = load_source_metadata()

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
        source_id = p.stem.split("__")[0]
        source_info = source_metadata_map.get(source_id, {})

        base_metadata = {
            "doc_id": p.stem,
            "source_id": source_id,
            "source_path": str(p),
            "file_name": p.name,
            "title": source_info.get("title"),
            "publisher": source_info.get("publisher"),
            "url": source_info.get("url"),
        }

        chunks = chunk_text(text, base_metadata, chunk_size, overlap)

        for ch in chunks:
            ids.append(ch.chunk_id)
            docs.append(ch.text)
            metas.append(ch.metadata)

            if len(docs) >= batch_size:
                flush_batch()

    flush_batch()
    return total_added