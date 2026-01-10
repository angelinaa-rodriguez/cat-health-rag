from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"

    chroma_dir: str = "data/chroma"
    collection_name: str = "petmed_cats"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6

settings = Settings()
