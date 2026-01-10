from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from petmed_rag.config import settings
from petmed_rag.ingestion.chunk import chunk_text
from petmed_rag.retrieval.vectorstore import get_collection

PROCESSED_DIR = Path("data/processed")
PROCESSED_META_DIR = Path("data/processed/_meta")

client = OpenAI(api_key=settings.openai_api_key)

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [d.embedding for d in resp.data]

def main():
    txt_files = sorted([p for p in PROCESSED_DIR.glob("*.txt") if p.is_file()])
    if not txt_files:
        raise SystemExit("No processed .txt files found in data/processed. Run scripts/parse_html.py first.")

    col = get_collection()

    all_chunks = []
    for txt_path in txt_files:
        meta_path = PROCESSED_META_DIR / txt_path.name.replace(".txt", ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        base_md = {
            "doc_id": meta["id"],
            "url": meta.get("final_url") or meta.get("url"),
            "publisher": meta.get("publisher"),
            "title": meta.get("title") or meta.get("extracted_title"),
        }

        chunks = chunk_text(
            text=text,
            base_metadata=base_md,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Batch upserts (keeps memory sane)
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embs = embed_texts(texts)

        col.upsert(
            ids=[c.chunk_id for c in batch],
            documents=texts,
            metadatas=[c.metadata for c in batch],
            embeddings=embs,
        )
        print(f"Upserted {i+len(batch)}/{len(all_chunks)}")

    print("\n✅ Index build complete.")
    print(f"Chroma persisted at: {settings.chroma_dir}")
    print(f"Collection: {settings.collection_name}")

if __name__ == "__main__":
    main()
