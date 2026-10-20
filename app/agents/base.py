import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.settings import get_settings
from app.agents.state import Aporte, EstadoOrquestador, Tarea, buscar_tarea

MAX_CARACTERES_EVIDENCIA = 3000


def tarea_en_curso(state: EstadoOrquestador) -> Tarea:
    tarea = buscar_tarea(state, state["tarea_actual"])
    if tarea is None:
        raise ValueError(f"El supervisor asignó la tarea inexistente '{state['tarea_actual']}'")
    return tarea


def armar_instruccion(tarea: Tarea, contexto: str) -> str:
    partes = [f"TAREA {tarea['id']}: {tarea['objetivo']}"]
    if tarea["estado"] == "a_refinar" and tarea["observacion"]:
        partes.append(f"CORRECCIÓN PEDIDA POR EL SUPERVISOR (intento anterior rechazado): {tarea['observacion']}")
    if contexto:
        partes.append(f"INSUMOS DISPONIBLES:\n{contexto}")
    return "\n\n".join(partes)


async def ejecutar_especialista(agente: CompiledStateGraph, state: EstadoOrquestador, contexto: str) -> dict[str, Any]:
    tarea = tarea_en_curso(state)
    intento = tarea["intentos"] + 1
    resultado = await agente.ainvoke(
        {"messages": [HumanMessage(armar_instruccion(tarea, contexto))]},
        {"recursion_limit": get_settings().recursion_limit_especialista},
    )
    mensajes = resultado["messages"]
    estructurada: BaseModel | None = resultado.get("structured_response")
    salida = estructurada.model_dump() if estructurada is not None else {"error": "El especialista no produjo una salida estructurada."}

    aporte: Aporte = {
        "tarea_id": tarea["id"],
        "agente": tarea["agente"],
        "intento": intento,
        "salida": salida,
        "herramientas": [llamada["name"] for m in mensajes if isinstance(m, AIMessage) for llamada in m.tool_calls],
        "evidencia": [f"[{m.name}] {str(m.content)[:MAX_CARACTERES_EVIDENCIA]}" for m in mensajes if isinstance(m, ToolMessage)],
    }
    resumen = json.dumps(salida, ensure_ascii=False)
    return {
        "aportes": [aporte],
        "plan": [{**tarea, "estado": "en_curso", "intentos": intento}],
        "messages": [AIMessage(content=f"[{tarea['id']}] {resumen}", name=tarea["agente"])],
    }
