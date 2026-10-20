import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.agents.base import ejecutar_especialista, tarea_en_curso
from app.agents.contratos import InformeAnalisis
from app.agents.state import EstadoOrquestador, aportes_de_tarea
from app.tools.analisis import HERRAMIENTAS_ANALISIS

MAX_CARACTERES_INSUMOS = 8000

PROMPT_ANALISTA = """Sos el Agente de Análisis de un equipo de análisis. Procesás los datos que te entregan como insumos usando tus herramientas de cómputo. No buscás información nueva.

Reglas:
1. Antes de calcular sobre una serie temporal, validala con validar_serie_temporal. Si no es válida, no calcules: informalo en faltantes.
2. Todo número que reportes tiene que salir de una herramienta. Nunca hagas cuentas de memoria.
3. Usá exactamente las cifras de los insumos. Si hay cifras contradictorias, usá la que viene de la fuente oficial más reciente.
4. Para evaluar un criterio de decisión numérico, usá comparar_con_umbral.
5. Para comentarios de usuarios, usá sentimiento_por_aspecto pasando cada comentario textual.
6. En insumos de cada resultado indicá los valores de entrada y su doc_id."""


def crear_agente_analista(llm: BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(
        llm,
        HERRAMIENTAS_ANALISIS,
        prompt=PROMPT_ANALISTA,
        response_format=InformeAnalisis,
        name="analista",
    )


def insumos_para(state: EstadoOrquestador) -> str:
    tarea = tarea_en_curso(state)
    origenes = tarea["depende_de"] or [t["id"] for t in state.get("plan", []) if t["agente"] == "investigador" and t["estado"] == "aprobada"]
    bloques = []
    for tarea_id in origenes:
        aportes = aportes_de_tarea(state, tarea_id)
        if aportes:
            bloques.append(f"Resultado de {tarea_id}:\n{json.dumps(aportes[-1]['salida'], ensure_ascii=False, indent=1)}")
    return "\n\n".join(bloques)[:MAX_CARACTERES_INSUMOS]


def crear_nodo_analista(llm: BaseChatModel) -> Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]:
    agente = crear_agente_analista(llm)

    async def analista(state: EstadoOrquestador) -> dict[str, Any]:
        return await ejecutar_especialista(agente, state, contexto=insumos_para(state))

    return analista
