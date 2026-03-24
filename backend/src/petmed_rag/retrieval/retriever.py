from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from petmed_rag.retrieval.vectorstore import get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def _distance_to_score(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance)) # returns score, not distance (better for UX)


class ChromaRetriever:
    def __init__(
        self,
        embedder,  # callable: (list[str]) -> list[list[float]]
    ) -> None:
        self.collection = get_collection()
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.embedder([query])[0] # single vector

        # returns documents, metadatas, distances, ids (nested lists)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # returns [[]] if empty, prevents crash
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

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
