from petmed_rag.config import settings
from petmed_rag.embeddings import embed_texts
from petmed_rag.retrieval.retriever import ChromaRetriever, RetrievedChunk


retriever = ChromaRetriever(
    persist_dir=settings.chroma_dir,
    collection_name=settings.collection_name,
    embedder=embed_texts,
)


def retrieve_context(question: str, k: int = 5) -> list[RetrievedChunk]:
    return retriever.retrieve(question, k=k)