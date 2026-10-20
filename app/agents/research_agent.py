from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.agents.base import ejecutar_especialista
from app.agents.contratos import InformeInvestigacion
from app.agents.state import EstadoOrquestador

PROMPT_INVESTIGADOR = """Sos el Agente de Investigación de un equipo de análisis. Obtenés datos de la base de conocimiento con recuperación híbrida (RAG). No calculás ni opinás.

Procedimiento:
1. Buscá con buscar_conocimiento usando términos concretos. Si los fragmentos no alcanzan, reformulá la búsqueda o filtrá por tipo de documento.
2. Antes de reportar una cifra, confirmala con leer_documento sobre el doc_id del fragmento. Nunca reportes cifras que no figuren en un documento.
3. Copiá cada valor exactamente como figura, con su doc_id, fecha y tipo de fuente.
4. Si dos documentos dan cifras distintas para lo mismo, registralo en conflictos indicando cuál es la fuente oficial más reciente.
5. Si la tarea pide comentarios o reclamos, transcribilos textualmente en textos, uno por elemento.
6. Si la base de conocimiento no tiene la información pedida, decilo en faltantes. No inventes datos."""


def crear_agente_investigador(llm: BaseChatModel, herramientas: list[BaseTool]) -> CompiledStateGraph:
    return create_react_agent(llm, herramientas, prompt=PROMPT_INVESTIGADOR, response_format=InformeInvestigacion, name="investigador")


def crear_nodo_investigador(llm: BaseChatModel, herramientas: list[BaseTool]) -> Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]:
    agente = crear_agente_investigador(llm, herramientas)

    async def investigador(state: EstadoOrquestador) -> dict[str, Any]:
        return await ejecutar_especialista(agente, state, contexto="")

    return investigador
