from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
PROCESSED_DIR = DATA_DIR / "processed"

class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "./data/chroma"
    collection_name: str = "cat-health"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()