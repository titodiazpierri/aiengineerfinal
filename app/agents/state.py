import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

Agente = Literal["investigador", "analista"]
EstadoTarea = Literal["pendiente", "en_curso", "a_refinar", "aprobada", "agotada"]
Siguiente = Literal["investigador", "analista", "sintetizador"]


class Tarea(TypedDict):
    id: str
    agente: Agente
    objetivo: str
    depende_de: list[str]
    estado: EstadoTarea
    intentos: int
    observacion: str


class Aporte(TypedDict):
    tarea_id: str
    agente: Agente
    intento: int
    salida: dict[str, Any]
    herramientas: list[str]
    evidencia: list[str]


class RegistroDecision(TypedDict):
    paso: int
    veredicto: str
    evaluacion: str
    siguiente: Siguiente
    tarea_id: str
    razonamiento: str
    ajustes: list[str]


class Turno(TypedDict):
    numero: int
    job_id: str
    consulta: str
    respuesta: str
    estado: str


class Revision(TypedDict):
    intento: int
    aprobada: bool
    puntajes: dict[str, int]
    problemas: list[str]


def fusionar_plan(actual: list[Tarea], cambios: list[Tarea]) -> list[Tarea]:
    por_id = {tarea["id"]: dict(tarea) for tarea in actual}
    orden = [tarea["id"] for tarea in actual]
    for cambio in cambios:
        if cambio["id"] in por_id:
            por_id[cambio["id"]].update(cambio)
        else:
            por_id[cambio["id"]] = dict(cambio)
            orden.append(cambio["id"])
    return [por_id[i] for i in orden]


class EstadoOrquestador(MessagesState):
    consulta: str
    plan: Annotated[list[Tarea], fusionar_plan]
    aportes: Annotated[list[Aporte], operator.add]
    decisiones: Annotated[list[RegistroDecision], operator.add]
    revisiones: Annotated[list[Revision], operator.add]
    pasos: Annotated[int, operator.add]
    siguiente: Siguiente
    tarea_actual: str
    observaciones_revisor: str
    informe: dict[str, Any] | None
    respuesta_final: str
    estado_final: Literal["en_progreso", "aprobado", "aprobado_con_advertencias"]
    turnos: Annotated[list[Turno], operator.add]


def estado_inicial(consulta: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(consulta)],
        "consulta": consulta,
        "plan": [],
        "aportes": [],
        "decisiones": [],
        "revisiones": [],
        "pasos": 0,
        "siguiente": "investigador",
        "tarea_actual": "",
        "observaciones_revisor": "",
        "informe": None,
        "respuesta_final": "",
        "estado_final": "en_progreso",
        "turnos": [],
    }


def buscar_tarea(state: EstadoOrquestador, tarea_id: str) -> Tarea | None:
    return next((t for t in state.get("plan", []) if t["id"] == tarea_id), None)


def aportes_de_tarea(state: EstadoOrquestador, tarea_id: str) -> list[Aporte]:
    return [a for a in state.get("aportes", []) if a["tarea_id"] == tarea_id]


def tareas_listas(state: EstadoOrquestador) -> list[Tarea]:
    aprobadas = {t["id"] for t in state.get("plan", []) if t["estado"] in ("aprobada", "agotada")}
    return [
        t for t in state.get("plan", [])
        if t["estado"] in ("pendiente", "a_refinar") and set(t["depende_de"]) <= aprobadas
    ]


def tareas_abiertas(state: EstadoOrquestador) -> list[Tarea]:
    return [t for t in state.get("plan", []) if t["estado"] in ("pendiente", "a_refinar", "en_curso")]


def contexto_conversacion(state: EstadoOrquestador, cantidad: int) -> str:
    turnos = state.get("turnos", [])[-cantidad:] if cantidad else []
    if not turnos:
        return "Es el primer turno de la conversación."
    return "\n\n".join(f"Turno {t['numero']} - Usuario: {t['consulta']}\nRespuesta: {t['respuesta'][:700]}" for t in turnos)
