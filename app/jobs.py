import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

PREFIJO = "orquestador:job:"
INDICE = "orquestador:jobs"
PUBLICACIONES = "orquestador:publicaciones"
BLOQUEO_CONVERSACION = "orquestador:conversacion:{}:activo"


class EstadoJob(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


FINALES = {EstadoJob.DONE, EstadoJob.REJECTED, EstadoJob.FAILED}


class TransicionInvalida(Exception):
    def __init__(self, job_id: str, actual: str, esperados: set[EstadoJob]) -> None:
        super().__init__(f"El job {job_id} está en {actual}; se esperaba {', '.join(sorted(esperados))}")
        self.actual = actual


class JobNoEncontrado(Exception):
    pass


class ConversacionOcupada(Exception):
    def __init__(self, conversacion_id: str, job_id: str) -> None:
        super().__init__(f"La conversación {conversacion_id} ya tiene un turno en curso ({job_id})")
        self.job_id = job_id


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RepositorioJobs:
    def __init__(self, redis: Redis, ttl_segundos: int) -> None:
        self.redis = redis
        self.ttl = ttl_segundos

    @staticmethod
    def clave(job_id: str) -> str:
        return PREFIJO + job_id

    async def crear(self, consulta: str, publicar_para: list[str], conversacion_id: str | None) -> dict[str, Any]:
        momento = ahora()
        job_id = uuid.uuid4().hex
        conversacion_id = conversacion_id or uuid.uuid4().hex
        bloqueo = BLOQUEO_CONVERSACION.format(conversacion_id)
        if not await self.redis.set(bloqueo, job_id, nx=True, ex=self.ttl):
            raise ConversacionOcupada(conversacion_id, await self.redis.get(bloqueo) or "desconocido")
        job = {
            "job_id": job_id,
            "conversacion_id": conversacion_id,
            "estado": EstadoJob.PENDING,
            "consulta": consulta,
            "publicar_para": publicar_para,
            "creado": momento,
            "actualizado": momento,
            "progreso": {"ultimo_nodo": None, "pasos": 0},
            "historial": [{"estado": EstadoJob.PENDING, "momento": momento}],
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(self.clave(job["job_id"]), json.dumps(job, ensure_ascii=False), ex=self.ttl)
            pipe.zadd(INDICE, {job["job_id"]: datetime.now(timezone.utc).timestamp()})
            await pipe.execute()
        return job

    async def obtener(self, job_id: str) -> dict[str, Any]:
        crudo = await self.redis.get(self.clave(job_id))
        if crudo is None:
            raise JobNoEncontrado(job_id)
        return json.loads(crudo)

    async def modificar(self, job_id: str, cambio: Callable[[dict[str, Any]], dict[str, Any]], reintentos: int = 10) -> dict[str, Any]:
        clave = self.clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            for _ in range(reintentos):
                try:
                    await pipe.watch(clave)
                    crudo = await pipe.get(clave)
                    if crudo is None:
                        raise JobNoEncontrado(job_id)
                    job = cambio(json.loads(crudo))
                    job["actualizado"] = ahora()
                    pipe.multi()
                    pipe.set(clave, json.dumps(job, ensure_ascii=False, default=str), keepttl=True)
                    await pipe.execute()
                    return job
                except WatchError:
                    continue
        raise RuntimeError(f"No se pudo actualizar el job {job_id} por escrituras concurrentes")

    async def transicionar(self, job_id: str, desde: set[EstadoJob], hacia: EstadoJob, **campos: Any) -> dict[str, Any]:
        def cambio(job: dict[str, Any]) -> dict[str, Any]:
            if job["estado"] not in desde:
                raise TransicionInvalida(job_id, job["estado"], desde)
            job.update(campos)
            job["estado"] = hacia
            job["historial"].append({"estado": hacia, "momento": ahora()})
            return job

        job = await self.modificar(job_id, cambio)
        if hacia in FINALES:
            await self.liberar_conversacion(job)
        return job

    async def liberar_conversacion(self, job: dict[str, Any]) -> None:
        bloqueo = BLOQUEO_CONVERSACION.format(job["conversacion_id"])
        if await self.redis.get(bloqueo) == job["job_id"]:
            await self.redis.delete(bloqueo)

    async def registrar_progreso(self, job_id: str, nodo: str) -> None:
        def cambio(job: dict[str, Any]) -> dict[str, Any]:
            job["progreso"] = {"ultimo_nodo": nodo, "pasos": job["progreso"]["pasos"] + 1}
            return job

        await self.modificar(job_id, cambio)

    async def listar(self, limite: int, estado: EstadoJob | None = None) -> list[dict[str, Any]]:
        resultado = []
        for job_id in await self.redis.zrevrange(INDICE, 0, -1):
            try:
                job = await self.obtener(job_id)
            except JobNoEncontrado:
                await self.redis.zrem(INDICE, job_id)
                continue
            if estado is None or job["estado"] == estado:
                resultado.append(job)
            if len(resultado) >= limite:
                break
        return resultado

    async def publicar(self, registro: dict[str, Any]) -> str:
        publicacion_id = "pub-" + uuid.uuid4().hex[:10]
        await self.redis.lpush(PUBLICACIONES, json.dumps({"publicacion_id": publicacion_id, "fecha": ahora(), **registro}, ensure_ascii=False))
        return publicacion_id
