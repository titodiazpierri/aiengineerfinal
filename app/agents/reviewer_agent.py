import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from app.settings import get_settings
from app.agents.contratos import EvaluacionRubrica
from app.agents.state import EstadoOrquestador, Revision
from app.agents.validation import controles_deterministas, resumen_para_revisor

logger = logging.getLogger("revisor")

PROMPT_REVISOR = """Sos el Revisor de calidad. Evaluás el informe final contra los resultados de los especialistas antes de entregarlo al usuario. No reescribís el informe: lo puntuás de 1 a 5 en cada criterio y listás problemas concretos.

- cobertura: ¿responde todas las partes de la consulta?
- fidelidad: ¿cada cifra coincide exactamente con los resultados de los especialistas?
- trazabilidad: ¿cada hallazgo cita la fuente o la tarea que lo respalda?
- manejo_conflictos: si hubo cifras contradictorias, ¿se usó la fuente oficial más reciente y se explicó? Si no hubo conflictos, 5.

Un 5 es impecable, un 3 tiene errores que el usuario notaría, un 1 es inutilizable."""

PROMPT_DATOS = """CONSULTA:
{consulta}

RESULTADOS DE LOS ESPECIALISTAS:
{resultados}

INFORME FINAL A REVISAR:
{informe}"""

prompt_revisor = ChatPromptTemplate.from_messages([("system", PROMPT_REVISOR), ("human", PROMPT_DATOS)])


def crear_nodo_revisor(llm: BaseChatModel) -> Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]:
    cadena = prompt_revisor | llm.with_structured_output(EvaluacionRubrica, method="function_calling")

    async def revisor(state: EstadoOrquestador) -> dict[str, Any]:
        intento = len(state.get("revisiones", [])) + 1
        problemas = controles_deterministas(state)
        evaluacion: EvaluacionRubrica = await cadena.ainvoke(resumen_para_revisor(state))
        puntajes = {
            "cobertura": evaluacion.cobertura,
            "fidelidad": evaluacion.fidelidad,
            "trazabilidad": evaluacion.trazabilidad,
            "manejo_conflictos": evaluacion.manejo_conflictos,
        }
        minimo = get_settings().puntaje_minimo_rubrica
        bajos = [f"{criterio}={valor}" for criterio, valor in puntajes.items() if valor < minimo]
        if bajos:
            problemas.append(f"Rúbrica por debajo de {minimo}: {', '.join(bajos)}. " + " ".join(evaluacion.problemas))
        aprobada = not problemas

        revision: Revision = {"intento": intento, "aprobada": aprobada, "puntajes": puntajes, "problemas": problemas}
        actualizacion: dict[str, Any] = {"revisiones": [revision]}
        if aprobada:
            logger.info("Revisión %d aprobada: %s", intento, puntajes)
            actualizacion |= {"estado_final": "aprobado", "observaciones_revisor": ""}
            texto = f"Informe aprobado. Puntajes: {puntajes}."
        elif intento >= get_settings().max_revisiones:
            logger.warning("Revisión %d rechazada y sin reintentos: se entrega con advertencias", intento)
            advertencia = "\n\n**Advertencias de la revisión:**\n" + "\n".join(f"- {p}" for p in problemas)
            actualizacion |= {"estado_final": "aprobado_con_advertencias", "respuesta_final": state.get("respuesta_final", "") + advertencia}
            texto = "Informe entregado con advertencias tras agotar las revisiones."
        else:
            logger.warning("Revisión %d rechazada: %s", intento, problemas)
            actualizacion |= {"observaciones_revisor": " ".join(problemas)}
            texto = "Informe rechazado: " + " ".join(problemas)
        actualizacion["messages"] = [AIMessage(content=texto, name="revisor")]
        return actualizacion

    return revisor


def enrutar_revision(state: EstadoOrquestador) -> Literal["supervisor", "__end__"]:
    return END if state.get("estado_final") != "en_progreso" else "supervisor"
