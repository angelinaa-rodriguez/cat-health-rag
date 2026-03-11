from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "./data/chroma"
    collection_name: str = "cat-health"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()