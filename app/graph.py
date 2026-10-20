import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite

from app.agents.analyst_agent import crear_nodo_analista
from app.agents.research_agent import crear_nodo_investigador
from app.agents.reviewer_agent import crear_nodo_revisor, enrutar_revision
from app.agents.state import Turno
from app.agents.supervisor import crear_nodo_supervisor
from app.agents.synthesizer_agent import crear_nodo_sintetizador
from app.hitl import NODO_CRITICO, EstadoProduccion, Publicador, crear_nodo_accion_critica, enrutar_criticidad, nodo_control_criticidad
from app.llm import crear_modelo_chat
from app.settings import Settings

logger = logging.getLogger("grafo")


def enrutar_supervisor(state: EstadoProduccion) -> Literal["investigador", "analista", "sintetizador"]:
    return state["siguiente"]


def enrutar_tras_revision(state: EstadoProduccion) -> Literal["supervisor", "control_criticidad"]:
    return "supervisor" if enrutar_revision(state) == "supervisor" else "control_criticidad"


async def cerrar_turno(state: EstadoProduccion) -> dict[str, Any]:
    aprobacion = state.get("aprobacion")
    estado = "rechazado" if aprobacion and not aprobacion["aprobado"] else state.get("estado_final", "aprobado")
    turno: Turno = {
        "numero": len(state.get("turnos", [])) + 1,
        "job_id": state.get("job_actual", ""),
        "consulta": state["consulta"],
        "respuesta": state.get("respuesta_final", ""),
        "estado": estado,
    }
    return {"turnos": [turno]}


def construir_grafo(
    checkpointer: BaseCheckpointSaver,
    herramientas_busqueda: list[BaseTool],
    publicar: Publicador,
    llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    llm = llm or crear_modelo_chat()
    grafo = StateGraph(EstadoProduccion)
    grafo.add_node("supervisor", crear_nodo_supervisor(llm))
    grafo.add_node("investigador", crear_nodo_investigador(llm, herramientas_busqueda))
    grafo.add_node("analista", crear_nodo_analista(llm))
    grafo.add_node("sintetizador", crear_nodo_sintetizador(llm))
    grafo.add_node("revisor", crear_nodo_revisor(llm))
    grafo.add_node("control_criticidad", nodo_control_criticidad)
    grafo.add_node(NODO_CRITICO, crear_nodo_accion_critica(publicar))
    grafo.add_node("cerrar_turno", cerrar_turno)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar_supervisor)
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_edge("sintetizador", "revisor")
    grafo.add_conditional_edges("revisor", enrutar_tras_revision)
    grafo.add_conditional_edges("control_criticidad", enrutar_criticidad)
    grafo.add_edge(NODO_CRITICO, "cerrar_turno")
    grafo.add_edge("cerrar_turno", END)
    return grafo.compile(checkpointer=checkpointer, interrupt_before=[NODO_CRITICO])


def entrada_de_turno(consulta: str, publicar_para: list[str], job_id: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(consulta)],
        "consulta": consulta,
        "plan": Overwrite([]),
        "aportes": Overwrite([]),
        "decisiones": Overwrite([]),
        "revisiones": Overwrite([]),
        "pasos": Overwrite(0),
        "siguiente": "investigador",
        "tarea_actual": "",
        "observaciones_revisor": "",
        "informe": None,
        "respuesta_final": "",
        "estado_final": "en_progreso",
        "publicar_para": publicar_para,
        "motivos_criticidad": [],
        "aprobacion": None,
        "publicacion_id": "",
        "job_actual": job_id,
    }


@asynccontextmanager
async def abrir_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    if settings.checkpointer == "memory":
        logger.warning("Checkpointer en memoria: las conversaciones se pierden al reiniciar")
        yield InMemorySaver()
        return
    if settings.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(str(settings.sqlite_checkpoints)) as checkpointer:
            await checkpointer.setup()
            logger.info("Checkpoints en SQLite (%s)", settings.sqlite_checkpoints)
            yield checkpointer
        return

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    ttl = {"default_ttl": settings.checkpoint_ttl_minutos, "refresh_on_read": True}
    contexto = AsyncRedisSaver.from_conn_string(settings.redis_url, ttl=ttl)
    try:
        checkpointer = await contexto.__aenter__()
        await checkpointer.asetup()
    except Exception as error:
        raise RuntimeError(
            f"No se pudo iniciar AsyncRedisSaver en {settings.redis_url}: {error}. "
            "Se necesita Redis 8 (incluye los módulos JSON y Search): docker compose up -d redis"
        ) from error
    logger.info("Checkpoints en Redis (%s)", settings.redis_url)
    try:
        yield checkpointer
    finally:
        await contexto.__aexit__(None, None, None)
