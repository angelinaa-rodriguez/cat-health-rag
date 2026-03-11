from openai import OpenAI

from petmed_rag.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def get_embedder(
    embedder: str = "openai",
    openai_model: str = "text-embedding-3-small",
    st_model: str = "all-MiniLM-L6-v2",
):
    if embedder != "openai":
        raise ValueError("Only openai embedder is supported right now.")

    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            model=openai_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    return _embed