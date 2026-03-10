from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple


EmbedderName = Literal["openai", "st"]


@dataclass(frozen=True)
class EmbedderConfig:
    """
    Configuration for selecting an embedding backend.

    embedder:
      - "openai" -> OpenAI embeddings (recommended for your 3072-dim collection)
      - "st"     -> sentence-transformers embeddings (often 384-dim MiniLM)
    """
    embedder: EmbedderName = "openai"
    openai_model: str = "text-embedding-3-large"
    st_model: str = "all-MiniLM-L6-v2"
    openai_api_key_env: str = "OPENAI_API_KEY"

    def normalized_embedder(self) -> EmbedderName:
        e = (self.embedder or "").strip().lower()
        if e in ("openai",):
            return "openai"
        if e in ("st", "sentence-transformers", "sbert", "minilm"):
            return "st"
        # Type narrowing: raise for unknowns
        raise ValueError(f"Unknown embedder '{self.embedder}'. Use 'openai' or 'st'.")


def get_openai_api_key(cfg: EmbedderConfig) -> Optional[str]:
    return os.getenv(cfg.openai_api_key_env)


def ensure_openai_key_if_needed(cfg: EmbedderConfig) -> None:
    """
    Call this early in ingest/retrieve if you default to OpenAI.
    """
    if cfg.normalized_embedder() == "openai":
        key = get_openai_api_key(cfg)
        if not key:
            raise RuntimeError(
                f"{cfg.openai_api_key_env} is not set, but embedder='openai' was requested. "
                "Set it in your PowerShell session (e.g., $env:OPENAI_API_KEY='...') "
                "or use --embedder st explicitly (not recommended if your collection is 3072-dim)."
            )


def build_embedding_function(cfg: EmbedderConfig):
    """
    Returns a Chroma-compatible embedding_function.
    """
    embedder = cfg.normalized_embedder()

    if embedder == "openai":
        # Chroma's built-in embedding function for OpenAI
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        key = get_openai_api_key(cfg)
        if not key:
            # keep error message consistent
            raise RuntimeError(
                f"{cfg.openai_api_key_env} is not set, but embedder='openai' was requested."
            )

        return OpenAIEmbeddingFunction(api_key=key, model_name=cfg.openai_model)

    # Sentence-transformers
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=cfg.st_model)


def describe_embedder(cfg: EmbedderConfig) -> Dict[str, str]:
    """
    Helpful debug info for printing in CLI.
    """
    embedder = cfg.normalized_embedder()
    info: Dict[str, str] = {"embedder": embedder}

    if embedder == "openai":
        info["model"] = cfg.openai_model
        info["openai_key_env"] = cfg.openai_api_key_env
        info["openai_key_present"] = str(bool(get_openai_api_key(cfg)))
    else:
        info["model"] = cfg.st_model

    return info


def try_probe_dim(embedding_function, text: str = "dim probe") -> Tuple[Optional[int], Optional[str]]:
    """
    Best-effort probe to determine embedding dimension by embedding one string.

    Returns:
        (dim, error_message)

    Notes:
      - For sentence-transformers, this should always work offline.
      - For OpenAI, this requires OPENAI_API_KEY and network access; it may fail.
      - This is meant for debug output, not for core functionality.
    """
    try:
        vecs = embedding_function([text])
        if not vecs or not vecs[0]:
            return None, "no vectors returned"
        return len(vecs[0]), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    
def get_embedder(
    embedder: str = "openai",
    openai_model: str = "text-embedding-3-large",
    st_model: str = "all-MiniLM-L6-v2",
    openai_api_key_env: str = "OPENAI_API_KEY",
    ):
        """
        Backwards-compatible wrapper so existing code that imports get_embedder keeps working.

        Returns a Chroma embedding_function.
        """
        cfg = EmbedderConfig(
            embedder=embedder, openai_model=openai_model, st_model=st_model, openai_api_key_env=openai_api_key_env
        )
        # If openai requested but key missing, fail loudly (prevents silent ST fallback)
        ensure_openai_key_if_needed(cfg)
        return build_embedding_function(cfg)

