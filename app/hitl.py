import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.metrics import medidor_actual
from app.observability import trazar_chain
from app.agents.state import EstadoOrquestador
from app.settings import get_settings

logger = logging.getLogger("hitl")

NODO_CRITICO = "accion_critica"
Publicador = Callable[[dict[str, Any]], Awaitable[str]]


class EstadoProduccion(EstadoOrquestador):
    publicar_para: list[str]
    motivos_criticidad: list[str]
    aprobacion: dict[str, Any] | None
    publicacion_id: str
    job_actual: str


@trazar_chain("hitl.evaluar_criticidad")
def evaluar_criticidad(publicar_para: list[str], tokens_consumidos: int, umbral_tokens: int) -> list[str]:
    motivos = []
    if publicar_para:
        motivos.append(f"Efecto secundario irreversible: publicar el informe a {', '.join(publicar_para)}.")
    if tokens_consumidos > umbral_tokens:
        motivos.append(f"Costo alto: la ejecución consumió {tokens_consumidos} tokens (umbral {umbral_tokens}).")
    return motivos


async def nodo_control_criticidad(state: EstadoProduccion) -> dict[str, Any]:
    medidor = medidor_actual.get()
    tokens = medidor.tokens_totales if medidor else 0
    motivos = evaluar_criticidad(state.get("publicar_para") or [], tokens, get_settings().umbral_tokens_aprobacion)
    if motivos:
        logger.info("Tarea crítica, se pausará antes de %s: %s", NODO_CRITICO, " | ".join(motivos))
    return {"motivos_criticidad": motivos}


def enrutar_criticidad(state: EstadoProduccion) -> Literal["accion_critica", "cerrar_turno"]:
    return NODO_CRITICO if state.get("motivos_criticidad") else "cerrar_turno"


def crear_herramienta_publicacion(publicar: Publicador) -> BaseTool:
    @tool
    async def publicar_informe(destinatarios: list[str], informe: str, job_id: str) -> str:
        """Publica el informe aprobado y lo envía a los destinatarios. Tiene efectos secundarios
        que no se pueden deshacer: solo se ejecuta después de una aprobación humana explícita.

        Args:
            destinatarios: direcciones o canales que reciben el informe.
            informe: texto final del informe.
            job_id: identificador de la tarea que lo generó.

        Returns:
            Identificador de la publicación.
        """
        return await publicar({"destinatarios": destinatarios, "informe": informe, "job_id": job_id})

    return publicar_informe


def crear_nodo_accion_critica(publicar: Publicador):
    herramienta = crear_herramienta_publicacion(publicar)

    async def accion_critica(state: EstadoProduccion, config: RunnableConfig) -> dict[str, Any]:
        aprobacion = state.get("aprobacion") or {}
        if not aprobacion.get("aprobado"):
            raise RuntimeError("Se intentó ejecutar la acción crítica sin una aprobación registrada")
        if not state.get("publicar_para"):
            return {"messages": [AIMessage(content="Entrega aprobada por revisión humana.", name=NODO_CRITICO)]}
        publicacion_id = await herramienta.ainvoke({
            "destinatarios": state["publicar_para"],
            "informe": state["respuesta_final"],
            "job_id": state["job_actual"],
        })
        return {"publicacion_id": publicacion_id, "messages": [AIMessage(content=f"Informe publicado ({publicacion_id}).", name=NODO_CRITICO)]}

    return accion_critica


def solicitud_de_aprobacion(valores: dict[str, Any]) -> dict[str, Any]:
    return {
        "motivos": valores.get("motivos_criticidad", []),
        "accion": f"Publicar el informe a {', '.join(valores['publicar_para'])}" if valores.get("publicar_para") else "Entregar la respuesta",
        "informe_propuesto": valores.get("respuesta_final", ""),
        "estado_revision": valores.get("estado_final"),
    }


async def aplicar_decision_humana(grafo: CompiledStateGraph, config: RunnableConfig, decision: dict[str, Any]) -> None:
    instantanea = await grafo.aget_state(config)
    if instantanea.next != (NODO_CRITICO,):
        raise RuntimeError(f"El grafo no está detenido en {NODO_CRITICO} (próximo: {instantanea.next})")
    if decision["aprobado"]:
        await grafo.aupdate_state(config, {"aprobacion": decision})
    else:
        await grafo.aupdate_state(config, {"aprobacion": decision}, as_node=NODO_CRITICO)
