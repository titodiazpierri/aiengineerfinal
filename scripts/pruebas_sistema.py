import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

RAIZ = Path(__file__).resolve().parent.parent
README = RAIZ / "README.md"
INICIO, FIN = "<!-- PRUEBAS -->", "<!-- /PRUEBAS -->"
FINALES = {"DONE", "REJECTED", "FAILED"}


class ResultadoPrueba(BaseModel):
    numero: int
    nombre: str
    objetivo: str
    aprobada: bool
    detalle: str
    job_ids: list[str]
    conversacion_id: str
    segundos: float


class Cliente:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http

    async def enviar(self, consulta: str, conversacion_id: str | None = None, publicar_para: list[str] | None = None) -> dict[str, Any]:
        cuerpo = {"consulta": consulta, "conversacion_id": conversacion_id, "publicar_para": publicar_para or []}
        respuesta = await self.http.post("/tasks", json=cuerpo)
        respuesta.raise_for_status()
        return respuesta.json()

    async def esperar(self, job_id: str, estados: set[str], limite: float = 600) -> dict[str, Any]:
        inicio = time.perf_counter()
        while time.perf_counter() - inicio < limite:
            job = (await self.http.get(f"/tasks/{job_id}")).json()
            if job["estado"] in estados:
                return job
            await asyncio.sleep(0.5)
        raise TimeoutError(f"El job {job_id} no llegó a {estados} en {limite} s")

    async def decidir(self, job_id: str, aprobado: bool) -> None:
        cuerpo = {"aprobado": aprobado, "revisor": "pruebas-del-sistema", "comentario": "Decisión de la prueba automatizada"}
        (await self.http.post(f"/tasks/{job_id}/approve", json=cuerpo)).raise_for_status()

    async def conversacion(self, conversacion_id: str) -> dict[str, Any]:
        return (await self.http.get(f"/conversations/{conversacion_id}")).json()


async def prueba_analisis_completo(c: Cliente) -> tuple[bool, str, list[str], str]:
    t = await c.enviar("Analizá el tercer trimestre de 2026 de EcoRuta: evolución de los viajes mensuales (crecimiento total, tasa mensual compuesta y promedio), sentimiento de los reclamos de soporte de septiembre y si conviene ampliar la flota según el plan de expansión.")
    job = await c.esperar(t["job_id"], FINALES)
    r = job.get("resultado") or {}
    herramientas = {h for consumo in job.get("consumos", []) for h in consumo["por_nodo"]}
    ok = job["estado"] == "DONE" and r.get("estado_revision") in ("aprobado", "aprobado_con_advertencias") and {"investigador", "analista"} <= herramientas
    return ok, f"estado={job['estado']}, revisión={r.get('estado_revision')}, plan={[p['id'] + ':' + p['estado'] for p in r.get('plan', [])]}", [t["job_id"]], t["conversacion_id"]


async def prueba_conversacion_multiturno(c: Cliente) -> tuple[bool, str, list[str], str]:
    t1 = await c.enviar("¿Cómo evolucionaron los viajes mensuales de EcoRuta en el tercer trimestre de 2026?")
    await c.esperar(t1["job_id"], FINALES)
    t2 = await c.enviar("¿Y cómo se compara con el segundo trimestre de 2026?", conversacion_id=t1["conversacion_id"])
    await c.esperar(t2["job_id"], FINALES)
    conversacion = await c.conversacion(t1["conversacion_id"])
    turnos = conversacion.get("turnos", [])
    ok = len(turnos) == 2 and turnos[1]["consulta"].startswith("¿Y cómo se compara")
    return ok, f"turnos en el checkpointer={len(turnos)}: {[t['consulta'][:45] for t in turnos]}", [t1["job_id"], t2["job_id"]], t1["conversacion_id"]


async def prueba_fuera_de_conocimiento(c: Cliente) -> tuple[bool, str, list[str], str]:
    t = await c.enviar("¿Cuántos monopatines tiene EcoRuta operando en la ciudad de Rosario y cuántos viajes hicieron en septiembre de 2026?")
    job = await c.esperar(t["job_id"], FINALES)
    informe = (job.get("resultado") or {}).get("informe") or {}
    respuesta = (job.get("resultado") or {}).get("respuesta_final", "")
    ok = job["estado"] in ("DONE", "FAILED") and "Rosario" not in json.dumps(informe.get("hallazgos", []), ensure_ascii=False)
    return ok, f"estado={job['estado']}, limitaciones={informe.get('limitaciones', [])[:2]}, respuesta={respuesta[:120]!r}", [t["job_id"]], t["conversacion_id"]


async def prueba_hitl_aprobada(c: Cliente) -> tuple[bool, str, list[str], str]:
    t = await c.enviar("Prepará un informe sobre si conviene ampliar la flota de EcoRuta según el plan de expansión 2027.", publicar_para=["directorio@ecoruta.example"])
    job = await c.esperar(t["job_id"], FINALES | {"WAITING_APPROVAL"})
    pausa = job["estado"] == "WAITING_APPROVAL"
    await c.decidir(t["job_id"], True)
    job = await c.esperar(t["job_id"], FINALES)
    publicacion = (job.get("resultado") or {}).get("publicacion_id")
    ok = pausa and job["estado"] == "DONE" and bool(publicacion)
    return ok, f"pausó en WAITING_APPROVAL={pausa}, final={job['estado']}, publicación={publicacion}", [t["job_id"]], t["conversacion_id"]


async def prueba_hitl_rechazada(c: Cliente) -> tuple[bool, str, list[str], str]:
    t = await c.enviar("Prepará un resumen del sentimiento de los reclamos de soporte de septiembre de 2026 para enviarlo al equipo de producto.", publicar_para=["producto@ecoruta.example"])
    job = await c.esperar(t["job_id"], FINALES | {"WAITING_APPROVAL"})
    pausa = job["estado"] == "WAITING_APPROVAL"
    await c.decidir(t["job_id"], False)
    job = await c.esperar(t["job_id"], FINALES)
    publicacion = (job.get("resultado") or {}).get("publicacion_id")
    ok = pausa and job["estado"] == "REJECTED" and not publicacion
    return ok, f"pausó en WAITING_APPROVAL={pausa}, final={job['estado']}, publicación={publicacion}", [t["job_id"]], t["conversacion_id"]


PRUEBAS: list[tuple[str, str, Callable[[Cliente], Awaitable[tuple[bool, str, list[str], str]]]]] = [
    ("Análisis completo RAG + agentes", "El supervisor planifica, el investigador recupera del RAG, el analista calcula y el revisor aprueba", prueba_analisis_completo),
    ("Conversación de dos turnos", "El segundo turno usa el historial y ambos turnos quedan en el checkpointer", prueba_conversacion_multiturno),
    ("Pregunta fuera del conocimiento", "El sistema no inventa datos de una ciudad donde EcoRuta no opera", prueba_fuera_de_conocimiento),
    ("Tarea crítica aprobada", "La publicación pausa el grafo y se ejecuta solo tras la aprobación", prueba_hitl_aprobada),
    ("Tarea crítica rechazada", "Al rechazar, el grafo termina sin ejecutar la publicación", prueba_hitl_rechazada),
]


def actualizar_readme(resultados: list[ResultadoPrueba], fecha: str) -> None:
    texto = README.read_text(encoding="utf-8") if README.exists() else ""
    if INICIO not in texto or FIN not in texto:
        return
    filas = [INICIO, f"Ejecución del {fecha} (`python scripts/pruebas_sistema.py`). Cada fila es una sesión en Phoenix con el id de conversación indicado.", "",
             "| # | Prueba | Resultado | Detalle | Conversación |", "|---|---|---|---|---|"]
    for r in resultados:
        filas.append(f"| {r.numero} | {r.nombre} | {'✅' if r.aprobada else '❌'} | {r.detalle.replace('|', '/')} | `{r.conversacion_id[:12]}` |")
    filas.append(FIN)
    antes, resto = texto.split(INICIO, 1)
    README.write_text(antes + "\n".join(filas) + resto.split(FIN, 1)[1], encoding="utf-8")


async def main(url: str, guardar: bool) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=30) as http:
        print("Servicio:", (await http.get("/health")).json())
        cliente = Cliente(http)
        resultados = []
        for numero, (nombre, objetivo, prueba) in enumerate(PRUEBAS, start=1):
            inicio = time.perf_counter()
            try:
                ok, detalle, jobs, conversacion = await prueba(cliente)
            except Exception as error:
                ok, detalle, jobs, conversacion = False, f"{type(error).__name__}: {error}", [], "-"
            r = ResultadoPrueba(numero=numero, nombre=nombre, objetivo=objetivo, aprobada=ok, detalle=detalle, job_ids=jobs, conversacion_id=conversacion, segundos=round(time.perf_counter() - inicio, 2))
            resultados.append(r)
            print(f"{'✅' if ok else '❌'} {numero}. {nombre} ({r.segundos} s): {detalle}")

    aprobadas = sum(r.aprobada for r in resultados)
    print(f"\n{aprobadas} de {len(resultados)} pruebas aprobadas. Revisá las trazas en Phoenix: http://localhost:6006")
    if guardar:
        fecha = datetime.now().isoformat(timespec="seconds")
        (RAIZ / "resultados").mkdir(exist_ok=True)
        ruta = RAIZ / "resultados" / f"pruebas_{datetime.now():%Y%m%d_%H%M%S}.json"
        ruta.write_text(json.dumps([r.model_dump() for r in resultados], ensure_ascii=False, indent=2), encoding="utf-8")
        actualizar_readme(resultados, fecha)
        print(f"Resultados en {ruta.relative_to(RAIZ)} y en el README")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinco pruebas de punta a punta del sistema")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--no-guardar", action="store_true")
    a = parser.parse_args()
    asyncio.run(main(a.url, not a.no_guardar))
