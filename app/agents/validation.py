import json
import re
from typing import Any

from app.agents.contratos import InformeFinal
from app.agents.state import EstadoOrquestador, tareas_abiertas
from app.rag.documentos import cargar_documentos
from app.settings import get_settings

NUMERO = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")
TOLERANCIA_ABSOLUTA = 0.051
TOLERANCIA_RELATIVA = 0.005


def interpretar_numero(token: str) -> float | None:
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", token):
        token = token.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", token):
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def numeros_en(texto: str) -> set[float]:
    return {n for n in (interpretar_numero(t) for t in NUMERO.findall(texto)) if n is not None}


def es_cifra_relevante(numero: float) -> bool:
    return abs(numero) >= 10 and not (numero.is_integer() and 2000 <= numero <= 2100)


def esta_respaldado(numero: float, respaldo: set[float]) -> bool:
    return any(abs(numero - r) <= max(TOLERANCIA_ABSOLUTA, TOLERANCIA_RELATIVA * abs(r)) for r in respaldo)


def texto_de_respaldo(state: EstadoOrquestador) -> str:
    partes = [state["consulta"]]
    for aporte in state.get("aportes", []):
        partes.append(json.dumps(aporte["salida"], ensure_ascii=False))
        partes.extend(aporte["evidencia"])
    return "\n".join(partes)


def controles_deterministas(state: EstadoOrquestador) -> list[str]:
    problemas: list[str] = []
    if not state.get("informe"):
        return ["No hay informe final para validar."]
    informe = InformeFinal.model_validate(state["informe"])

    for agente in ("investigador", "analista"):
        if not any(a["agente"] == agente and a["herramientas"] for a in state.get("aportes", [])):
            problemas.append(f"Ningún aporte del {agente} usó herramientas.")

    respaldo = texto_de_respaldo(state)
    ids_tareas = {t["id"] for t in state.get("plan", [])}
    doc_ids = set(cargar_documentos(get_settings().directorio_conocimiento))
    for hallazgo in informe.hallazgos:
        fuente = hallazgo.fuente.strip()
        if fuente not in ids_tareas and fuente not in doc_ids:
            problemas.append(f"El hallazgo '{hallazgo.afirmacion[:60]}' cita una fuente inexistente: '{fuente}'.")
        elif fuente in doc_ids and fuente not in respaldo:
            problemas.append(f"El hallazgo '{hallazgo.afirmacion[:60]}' cita '{fuente}', que ningún especialista consultó.")

    textos_informe = [informe.resumen_ejecutivo, informe.recomendacion] + [f"{h.afirmacion} {h.valor}" for h in informe.hallazgos]
    numeros_respaldo = numeros_en(respaldo)
    sin_respaldo = sorted({n for t in textos_informe for n in numeros_en(t) if es_cifra_relevante(n) and not esta_respaldado(n, numeros_respaldo)})
    if sin_respaldo:
        problemas.append("Cifras del informe que no aparecen en ningún aporte ni resultado de herramienta: " + ", ".join(f"{n:g}" for n in sin_respaldo) + ".")

    pendientes = [t["id"] for t in tareas_abiertas(state)]
    if pendientes:
        problemas.append(f"Quedaron tareas sin completar: {', '.join(pendientes)}.")
    return problemas


def resumen_para_revisor(state: EstadoOrquestador) -> dict[str, Any]:
    return {
        "consulta": state["consulta"],
        "resultados": "\n\n".join(
            f"[{a['tarea_id']} - {a['agente']} - intento {a['intento']}] {json.dumps(a['salida'], ensure_ascii=False)[:3000]}"
            for a in state.get("aportes", [])
        ),
        "informe": json.dumps(state.get("informe"), ensure_ascii=False, indent=1),
    }
