from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.settings import Settings, get_settings


def crear_modelo_chat(settings: Settings | None = None, api_key: str | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    comunes = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout_segundos,
        "max_retries": settings.llm_max_reintentos,
    }
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**comunes, api_key=api_key or settings.anthropic_api_key)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**comunes, api_key=api_key or settings.openai_api_key)


def crear_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key, max_retries=settings.llm_max_reintentos)
