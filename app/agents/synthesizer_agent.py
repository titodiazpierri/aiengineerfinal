import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.contratos import InformeFinal
from app.agents.state import EstadoOrquestador, aportes_de_tarea, contexto_conversacion
from app.settings import get_settings

PROMPT_SINTETIZADOR = """Sos el Agente Sintetizador. Redactás el informe final a partir de los resultados aprobados de las tareas.

Reglas:
1. Usá solo información presente en los RESULTADOS. No agregues datos ni hagas cálculos nuevos.
2. En cada hallazgo, copiá el valor exactamente como figura en los resultados y poné como fuente el doc_id del documento o el id de la tarea que lo calculó.
3. Si hubo cifras contradictorias entre fuentes, explicá en conflictos_resueltos cuál se usó y por qué.
4. La recomendación debe desprenderse de los hallazgos y, si existe, del criterio de decisión evaluado.
5. En limitaciones indicá datos faltantes o tareas que no pudieron completarse.
6. Escribí en español."""

PROMPT_DATOS = """CONVERSACIÓN PREVIA (solo para interpretar la consulta; no reutilices cifras que no estén en los RESULTADOS):
{conversacion}

CONSULTA ACTUAL:
{consulta}

RESULTADOS DE LAS TAREAS:
{resultados}

{observaciones}"""

prompt_sintetizador = ChatPromptTemplate.from_messages([("system", PROMPT_SINTETIZADOR), ("human", PROMPT_DATOS)])


def resultados_para_sintesis(state: EstadoOrquestador) -> str:
    bloques = []
    for tarea in state.get("plan", []):
        aportes = aportes_de_tarea(state, tarea["id"])
        if not aportes:
            continue
        estado = "aprobada" if tarea["estado"] == "aprobada" else f"{tarea['estado']} (usar con cautela)"
        bloques.append(f"[{tarea['id']} - {tarea['agente']} - {estado}] {tarea['objetivo']}\n{json.dumps(aportes[-1]['salida'], ensure_ascii=False)}")
    return "\n\n".join(bloques)


def a_markdown(informe: InformeFinal) -> str:
    lineas = [informe.resumen_ejecutivo, "", "**Hallazgos**"]
    for h in informe.hallazgos:
        valor = f" ({h.valor})" if h.valor else ""
        lineas.append(f"- {h.afirmacion}{valor} [{h.fuente}]")
    if informe.conflictos_resueltos:
        lineas += ["", "**Conflictos entre fuentes**", *[f"- {c}" for c in informe.conflictos_resueltos]]
    lineas += ["", f"**Recomendación:** {informe.recomendacion}"]
    if informe.limitaciones:
        lineas += ["", "**Limitaciones**", *[f"- {l}" for l in informe.limitaciones]]
    return "\n".join(lineas)


def crear_nodo_sintetizador(llm: BaseChatModel) -> Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]:
    cadena = prompt_sintetizador | llm.with_structured_output(InformeFinal, method="function_calling")

    async def sintetizador(state: EstadoOrquestador) -> dict[str, Any]:
        observaciones = ""
        if state.get("observaciones_revisor"):
            observaciones = f"OBSERVACIONES DEL REVISOR A CORREGIR:\n{state['observaciones_revisor']}"
        informe: InformeFinal = await cadena.ainvoke({
            "consulta": state["consulta"],
            "conversacion": contexto_conversacion(state, get_settings().turnos_de_contexto),
            "resultados": resultados_para_sintesis(state),
            "observaciones": observaciones,
        })
        texto = a_markdown(informe)
        return {"informe": informe.model_dump(), "respuesta_final": texto, "messages": [AIMessage(content=texto, name="sintetizador")]}

    return sintetizador
