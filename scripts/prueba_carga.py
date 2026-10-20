import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = RAIZ / "resultados"
README = RAIZ / "README.md"
INICIO, FIN = "<!-- RESULTADOS_CARGA -->", "<!-- /RESULTADOS_CARGA -->"
FINALES = {"DONE", "REJECTED", "FAILED"}

CONSULTAS = [
    "¿Cómo evolucionaron los viajes mensuales de EcoRuta en el tercer trimestre de 2026? Calculá el crecimiento total y la tasa mensual compuesta.",
    "¿Qué temas y qué sentimiento predominan en los reclamos de soporte de EcoRuta de septiembre de 2026?",
    "Según el criterio del plan de expansión 2027, ¿conviene ampliar la flota de EcoRuta con los datos del tercer trimestre?",
    "¿Cuál fue el promedio mensual de viajes de EcoRuta entre julio y septiembre de 2026 y cuánto difiere septiembre de la proyección municipal?",
    "Analizá el tercer trimestre de EcoRuta: crecimiento de viajes, sentimiento de los reclamos de septiembre y si conviene ampliar la flota.",
]


def percentil(valores: list[float], p: float) -> float:
    if len(valores) == 1:
        return valores[0]
    return statistics.quantiles(valores, n=100, method="inclusive")[int(p) - 1]


async def lanzar(cliente: httpx.AsyncClient, consulta: str, destinatarios: list[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = await cliente.post("/tasks", json={"consulta": consulta, "publicar_para": destinatarios})
    r.raise_for_status()
    return {"job_id": r.json()["job_id"], "t0": t0, "encolado_ms": round((time.perf_counter() - t0) * 1000, 1)}


async def esperar(cliente: httpx.AsyncClient, tarea: dict[str, Any], aprobar: bool, limite: float) -> dict[str, Any]:
    while time.perf_counter() - tarea["t0"] < limite:
        job = (await cliente.get(f"/tasks/{tarea['job_id']}")).json()
        if job["estado"] == "WAITING_APPROVAL":
            if not aprobar:
                return {**tarea, "job": job}
            await cliente.post(f"/tasks/{tarea['job_id']}/approve", json={"aprobado": True, "revisor": "prueba-de-carga"})
        elif job["estado"] in FINALES:
            return {**tarea, "job": job}
        await asyncio.sleep(0.25)
    return {**tarea, "job": {"estado": "SIN_RESPUESTA", "consumos": []}}


def consolidar(ejecuciones: list[dict[str, Any]]) -> dict[str, Any]:
    segundos = [sum(c["segundos"] for c in e["job"].get("consumos", [])) for e in ejecuciones]
    tokens = [sum(c["tokens"] for c in e["job"].get("consumos", [])) for e in ejecuciones]
    encolado = [e["encolado_ms"] for e in ejecuciones]
    por_nodo: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e in ejecuciones:
        for consumo in e["job"].get("consumos", []):
            for nodo, datos in consumo["por_nodo"].items():
                for campo, valor in datos.items():
                    por_nodo[nodo][campo] += valor
    por_nodo_plano = {n: {k: round(v, 2) for k, v in d.items()} for n, d in por_nodo.items()}
    estados: dict[str, int] = defaultdict(int)
    for e in ejecuciones:
        estados[e["job"]["estado"]] += 1
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "peticiones": len(ejecuciones),
        "estados": dict(estados),
        "encolado_ms": {"p50": round(percentil(encolado, 50), 1), "p95": round(percentil(encolado, 95), 1)},
        "ejecucion_s": {"p50": round(percentil(segundos, 50), 2), "p95": round(percentil(segundos, 95), 2), "max": round(max(segundos), 2)},
        "tokens": {"total": int(sum(tokens)), "promedio_por_ejecucion": round(sum(tokens) / len(tokens))},
        "por_nodo": por_nodo_plano,
        "nodo_mas_tokens": max(por_nodo_plano, key=lambda n: por_nodo_plano[n]["tokens_entrada"] + por_nodo_plano[n]["tokens_salida"], default="-"),
        "nodo_mas_lento": max(por_nodo_plano, key=lambda n: por_nodo_plano[n]["segundos_nodo"], default="-"),
    }


def mostrar(ejecuciones: list[dict[str, Any]], r: dict[str, Any]) -> None:
    print(f"\n{'job_id':<34}{'estado':<18}{'encolado ms':>12}{'ejecución s':>13}{'tokens':>9}")
    for e in ejecuciones:
        consumos = e["job"].get("consumos", [])
        print(f"{e['job_id']:<34}{e['job']['estado']:<18}{e['encolado_ms']:>12}{sum(c['segundos'] for c in consumos):>13.2f}{sum(c['tokens'] for c in consumos):>9}")
    print(f"\nEncolado POST /tasks  p50 {r['encolado_ms']['p50']} ms · p95 {r['encolado_ms']['p95']} ms")
    print(f"Ejecución en el worker p50 {r['ejecucion_s']['p50']} s · p95 {r['ejecucion_s']['p95']} s · máx {r['ejecucion_s']['max']} s")
    print(f"Tokens: {r['tokens']['total']} en total · {r['tokens']['promedio_por_ejecucion']} por ejecución")
    print(f"\n{'nodo':<20}{'llamadas LLM':>13}{'tokens entrada':>16}{'tokens salida':>15}{'seg. nodo':>11}")
    for nodo, d in sorted(r["por_nodo"].items(), key=lambda x: -(x[1]["tokens_entrada"] + x[1]["tokens_salida"])):
        print(f"{nodo:<20}{int(d['llamadas_llm']):>13}{int(d['tokens_entrada']):>16}{int(d['tokens_salida']):>15}{d['segundos_nodo']:>11}")
    print(f"\nMás tokens: {r['nodo_mas_tokens']} · Más tiempo: {r['nodo_mas_lento']}")


def escribir_readme(r: dict[str, Any]) -> None:
    texto = README.read_text(encoding="utf-8") if README.exists() else ""
    if INICIO not in texto or FIN not in texto:
        return
    filas = [
        INICIO,
        f"Corrida del {r['fecha']}: {r['peticiones']} peticiones concurrentes ({', '.join(f'{k}: {v}' for k, v in r['estados'].items())}). Medido por `scripts/prueba_carga.py`; el costo y el p95 de Phoenix están en `screenshots/`.",
        "",
        "| Métrica | p50 | p95 |",
        "|---|---|---|",
        f"| Encolado `POST /tasks` | {r['encolado_ms']['p50']} ms | {r['encolado_ms']['p95']} ms |",
        f"| Ejecución en el worker | {r['ejecucion_s']['p50']} s | {r['ejecucion_s']['p95']} s |",
        "",
        f"Tokens de la corrida: {r['tokens']['total']} ({r['tokens']['promedio_por_ejecucion']} por ejecución).",
        "",
        "| Nodo | Llamadas LLM | Tokens entrada | Tokens salida | Segundos en el nodo |",
        "|---|---|---|---|---|",
        *[f"| {n} | {int(d['llamadas_llm'])} | {int(d['tokens_entrada'])} | {int(d['tokens_salida'])} | {d['segundos_nodo']} |"
          for n, d in sorted(r["por_nodo"].items(), key=lambda x: -(x[1]["tokens_entrada"] + x[1]["tokens_salida"]))],
        "",
        f"Nodo con más tokens: **{r['nodo_mas_tokens']}**. Nodo donde se concentra la latencia: **{r['nodo_mas_lento']}**.",
        FIN,
    ]
    antes, resto = texto.split(INICIO, 1)
    README.write_text(antes + "\n".join(filas) + resto.split(FIN, 1)[1], encoding="utf-8")
    print("README.md actualizado con los resultados")


async def main(url: str, cantidad: int, hitl: bool, limite: float, guardar: bool) -> None:
    destinatarios = ["directorio@ecoruta.example"] if hitl else []
    async with httpx.AsyncClient(base_url=url, timeout=30) as cliente:
        print("Estado del servicio:", (await cliente.get("/health")).json())
        tareas = await asyncio.gather(*(lanzar(cliente, CONSULTAS[i % len(CONSULTAS)], destinatarios) for i in range(cantidad)))
        print(f"{cantidad} tareas encoladas en paralelo")
        ejecuciones = await asyncio.gather(*(esperar(cliente, t, hitl, limite) for t in tareas))
    resumen = consolidar(ejecuciones)
    mostrar(ejecuciones, resumen)
    if guardar:
        CARPETA_RESULTADOS.mkdir(exist_ok=True)
        ruta = CARPETA_RESULTADOS / f"carga_{datetime.now():%Y%m%d_%H%M%S}.json"
        ruta.write_text(json.dumps({"resumen": resumen, "ejecuciones": ejecuciones}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"Resultados en {ruta.relative_to(RAIZ)}")
        escribir_readme(resumen)
    print("Capturá en Phoenix (http://localhost:6006) el costo por traza y la latencia P95 de esta corrida.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de peticiones concurrentes contra la API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", "--cantidad", type=int, default=5)
    parser.add_argument("--hitl", action="store_true", help="Pide publicar cada informe y aprueba automáticamente la pausa humana")
    parser.add_argument("--limite", type=float, default=900, help="Segundos máximos de espera por tarea")
    parser.add_argument("--no-guardar", action="store_true")
    a = parser.parse_args()
    asyncio.run(main(a.url, a.cantidad, a.hitl, a.limite, not a.no_guardar))
