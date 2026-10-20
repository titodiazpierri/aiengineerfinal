import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.settings import get_settings
from app.agents.contratos import DecisionSupervisor
from app.agents.state import EstadoOrquestador, RegistroDecision, contexto_conversacion, Siguiente, Tarea, aportes_de_tarea, buscar_tarea, fusionar_plan, tareas_listas

logger = logging.getLogger("supervisor")

MAX_CARACTERES_APORTE = 3500

PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo multi-agente de análisis e investigación. Planificás el trabajo, revisás cada aporte y decidís quién interviene. No resolvés la tarea vos.

Especialistas:
- investigador: busca y lee documentos de la base documental. Devuelve datos con su fuente, textos transcriptos y conflictos entre fuentes. No calcula.
- analista: valida series, calcula crecimientos, estadísticas y criterios de decisión, y analiza el sentimiento de comentarios, siempre con herramientas y solo sobre los insumos que recibe. No busca.

Cómo trabajar:
1. Primera decisión (plan vacío): proponé en nuevas_tareas un plan completo de 2 a 5 tareas con ids T1, T2, ... Cada objetivo debe ser autocontenido porque el especialista NO ve la consulta original. Las tareas del analista declaran en depende_de las tareas del investigador que necesitan.
2. Decisiones siguientes: evaluá el último aporte con la rúbrica y poné veredicto "aprobar" o "refinar". Si refinás, escribí en observacion qué corregir exactamente.
3. Elegí en siguiente y tarea_id la próxima tarea pendiente cuyas dependencias estén aprobadas. Si todas están aprobadas, elegí "sintetizar".
4. Si el revisor final dejó observaciones, podés agregar tareas nuevas para resolverlas o volver a sintetizar.
5. Si la consulta hace referencia a la conversación previa (por ejemplo "¿y en el trimestre anterior?"), resolvé la referencia con el historial y escribí objetivos autocontenidos con los datos concretos.

RÚBRICA para aprobar un aporte:
- Completo: responde todo lo que pedía el objetivo de la tarea, o declara en faltantes lo que no existe.
- Trazable: cada dato o resultado indica su doc_id o la herramienta que lo calculó.
- Fiel a la fuente: ante cifras contradictorias usa la fuente oficial más reciente y lo explicita.
- Verificable: ningún cálculo hecho de memoria; todos salen de herramientas.

Presupuesto: van {pasos} decisiones de un máximo de {maximo}."""

PROMPT_ESTADO = """CONVERSACIÓN PREVIA:
{conversacion}

CONSULTA ACTUAL DEL USUARIO:
{consulta}

PLAN ACTUAL:
{plan}

ÚLTIMO APORTE RECIBIDO:
{ultimo_aporte}

OBSERVACIONES DEL REVISOR FINAL:
{observaciones}"""

prompt_supervisor = ChatPromptTemplate.from_messages([("system", PROMPT_SUPERVISOR), ("human", PROMPT_ESTADO)])


def describir_plan(plan: list[Tarea]) -> str:
    if not plan:
        return "Vacío: todavía no hay plan."
    return "\n".join(
        f"- {t['id']} [{t['agente']}] estado={t['estado']} intentos={t['intentos']} depende_de={t['depende_de'] or '-'}: {t['objetivo']}"
        for t in plan
    )


def describir_ultimo_aporte(state: EstadoOrquestador) -> str:
    if not state.get("tarea_actual"):
        return "Ninguno."
    aportes = aportes_de_tarea(state, state["tarea_actual"])
    if not aportes:
        return "Ninguno."
    aporte = aportes[-1]
    salida = json.dumps(aporte["salida"], ensure_ascii=False)[:MAX_CARACTERES_APORTE]
    return f"Tarea {aporte['tarea_id']} ({aporte['agente']}, intento {aporte['intento']}). Herramientas: {', '.join(aporte['herramientas']) or 'ninguna'}.\n{salida}"


def plan_por_defecto(consulta: str) -> list[Tarea]:
    return [
        {"id": "T1", "agente": "investigador", "objetivo": f"Reunir de la base documental todos los datos y textos necesarios para responder: {consulta}", "depende_de": [], "estado": "pendiente", "intentos": 0, "observacion": ""},
        {"id": "T2", "agente": "analista", "objetivo": f"Procesar con herramientas los datos de T1 para responder: {consulta}", "depende_de": ["T1"], "estado": "pendiente", "intentos": 0, "observacion": ""},
    ]


def aplicar_veredicto(state: EstadoOrquestador, decision: DecisionSupervisor, ajustes: list[str]) -> list[Tarea]:
    tarea = buscar_tarea(state, state.get("tarea_actual", ""))
    if tarea is None or tarea["estado"] != "en_curso":
        return []
    if decision.veredicto == "refinar":
        if tarea["intentos"] >= get_settings().max_intentos_por_tarea:
            ajustes.append(f"{tarea['id']} alcanzó {get_settings().max_intentos_por_tarea} intentos: se cierra como agotada con lo obtenido.")
            return [{**tarea, "estado": "agotada", "observacion": decision.observacion}]
        return [{**tarea, "estado": "a_refinar", "observacion": decision.observacion}]
    return [{**tarea, "estado": "aprobada", "observacion": ""}]


def incorporar_tareas(state: EstadoOrquestador, decision: DecisionSupervisor, ajustes: list[str]) -> list[Tarea]:
    existentes = {t["id"] for t in state.get("plan", [])}
    nuevas: list[Tarea] = []
    for t in decision.nuevas_tareas:
        if t.id in existentes or any(n["id"] == t.id for n in nuevas):
            continue
        nuevas.append({"id": t.id, "agente": t.agente, "objetivo": t.objetivo, "depende_de": t.depende_de, "estado": "pendiente", "intentos": 0, "observacion": ""})
    if not existentes and not nuevas:
        ajustes.append("El supervisor no propuso un plan: se usa el plan por defecto investigador → analista.")
        nuevas = plan_por_defecto(state["consulta"])
    return nuevas


def elegir_siguiente(plan: list[Tarea], decision: DecisionSupervisor, ajustes: list[str]) -> tuple[Siguiente, str]:
    listas = tareas_listas({"plan": plan})
    if decision.siguiente == "sintetizar":
        if listas:
            ajustes.append(f"Criterio de suficiencia: no se puede sintetizar con tareas pendientes; se ejecuta {listas[0]['id']}.")
            return listas[0]["agente"], listas[0]["id"]
        return "sintetizador", ""
    elegida = next((t for t in listas if t["id"] == decision.tarea_id), None)
    if elegida is None and listas:
        elegida = next((t for t in listas if t["agente"] == decision.siguiente), listas[0])
        ajustes.append(f"La tarea '{decision.tarea_id}' no está lista o no existe; se ejecuta {elegida['id']}.")
    if elegida is None:
        ajustes.append("No quedan tareas ejecutables: se pasa a la síntesis.")
        return "sintetizador", ""
    return elegida["agente"], elegida["id"]


def crear_nodo_supervisor(llm: BaseChatModel) -> Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]:
    cadena = prompt_supervisor | llm.with_structured_output(DecisionSupervisor, method="function_calling")

    async def supervisor(state: EstadoOrquestador) -> dict[str, Any]:
        paso = state.get("pasos", 0) + 1
        ajustes: list[str] = []

        maximo = get_settings().max_pasos_supervisor
        if paso > maximo:
            ajustes.append(f"Se alcanzó el máximo de {maximo} decisiones: se sintetiza con lo disponible.")
            cambios_plan: list[Tarea] = []
            siguiente, tarea_id = "sintetizador", ""
            decision = DecisionSupervisor(evaluacion="No evaluado por límite de pasos.", veredicto="sin_aporte", siguiente="sintetizar", razonamiento="Presupuesto agotado.")
        else:
            decision = await cadena.ainvoke({
                "consulta": state["consulta"],
                "conversacion": contexto_conversacion(state, get_settings().turnos_de_contexto),
                "plan": describir_plan(state.get("plan", [])),
                "ultimo_aporte": describir_ultimo_aporte(state),
                "observaciones": state.get("observaciones_revisor") or "Ninguna.",
                "pasos": paso - 1,
                "maximo": maximo,
            })
            cambios_plan = aplicar_veredicto(state, decision, ajustes) + incorporar_tareas(state, decision, ajustes)
            siguiente, tarea_id = elegir_siguiente(fusionar_plan(state.get("plan", []), cambios_plan), decision, ajustes)

        registro: RegistroDecision = {
            "paso": paso,
            "veredicto": decision.veredicto,
            "evaluacion": decision.evaluacion,
            "siguiente": siguiente,
            "tarea_id": tarea_id,
            "razonamiento": decision.razonamiento,
            "ajustes": ajustes,
        }
        logger.info("Paso %d: %s → %s %s", paso, decision.veredicto, siguiente, tarea_id)
        destino = f"{siguiente} ({tarea_id})" if tarea_id else siguiente
        return {
            "plan": cambios_plan,
            "decisiones": [registro],
            "pasos": 1,
            "siguiente": siguiente,
            "tarea_actual": tarea_id,
            "messages": [AIMessage(content=f"Veredicto: {decision.veredicto}. Siguiente: {destino}. {decision.razonamiento}", name="supervisor")],
        }

    return supervisor
