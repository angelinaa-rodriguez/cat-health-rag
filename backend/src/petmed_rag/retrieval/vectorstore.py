import chromadb
from chromadb.config import Settings as ChromaSettings
from petmed_rag.config import settings

_client = None
_collection = None


def get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    _collection = _client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return _collection