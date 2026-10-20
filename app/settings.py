from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = ""
    llm_temperature: float = Field(default=0.0, ge=0, le=1)
    llm_max_tokens: int = Field(default=2000, gt=0)
    llm_timeout_segundos: float = Field(default=90, gt=0)
    llm_max_reintentos: int = Field(default=2, ge=0)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "ecoruta_conocimiento"
    directorio_conocimiento: Path = RAIZ / "data" / "conocimiento"
    chunk_tokens: int = Field(default=500, ge=100)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    rag_top_k: int = Field(default=5, ge=1, le=10)
    rag_peso_vectorial: float = Field(default=0.6, ge=0, le=1)

    redis_url: str = "redis://localhost:6379/0"
    checkpointer: Literal["redis", "sqlite", "memory"] = "redis"
    sqlite_checkpoints: Path = RAIZ / "checkpoints.sqlite"
    checkpoint_ttl_minutos: int = Field(default=10080, gt=0)
    job_ttl_segundos: int = Field(default=604800, gt=0)

    max_jobs: int = Field(default=5, ge=1)
    job_timeout_segundos: int = Field(default=300, ge=30)

    observabilidad: Literal["phoenix", "ninguna"] = "phoenix"
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "intelligence-system"

    max_pasos_supervisor: int = Field(default=10, ge=3)
    max_intentos_por_tarea: int = Field(default=2, ge=1)
    max_revisiones: int = Field(default=2, ge=1)
    puntaje_minimo_rubrica: int = Field(default=4, ge=1, le=5)
    recursion_limit_grafo: int = Field(default=60, ge=10)
    recursion_limit_especialista: int = Field(default=16, ge=5)
    turnos_de_contexto: int = Field(default=3, ge=0)
    umbral_tokens_aprobacion: int = Field(default=80000, gt=0)

    @model_validator(mode="after")
    def completar_modelo(self) -> "Settings":
        if not self.llm_model:
            self.llm_model = {"openai": "gpt-4.1-mini", "anthropic": "claude-haiku-4-5"}[self.llm_provider]
        return self

    def verificar_credenciales(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en .env")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError("Falta ANTHROPIC_API_KEY en .env")
        if not self.openai_api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en .env: se usa para los embeddings del RAG")


@lru_cache
def get_settings() -> Settings:
    return Settings()
