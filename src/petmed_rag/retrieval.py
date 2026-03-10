from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float  # higher is better (display-friendly)
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def _distance_to_score(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance))


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedder,  # callable: (list[str]) -> list[list[float]]
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)

        # IMPORTANT: do NOT pass embedding_function here.
        self.collection = self.client.get_collection(name=collection_name)

        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.embedder([query])[0]  # single vector

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        out: List[RetrievedChunk] = []
        for i in range(len(docs)):
            out.append(
                RetrievedChunk(
                    rank=i + 1,
                    score=_distance_to_score(dists[i] if i < len(dists) else None),
                    doc_id=str(ids[i]),
                    text=str(docs[i]),
                    metadata=dict(metas[i] or {}),
                )
            )
        return out
