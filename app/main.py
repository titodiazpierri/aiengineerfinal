import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from redis.asyncio import Redis

from app.api_schemas import Conversacion, DecisionHumana, NuevaTarea, Salud, TareaEncolada, TurnoConversacion, VistaTarea
from app.graph import abrir_checkpointer
from app.jobs import BLOQUEO_CONVERSACION, ConversacionOcupada, EstadoJob, JobNoEncontrado, RepositorioJobs, TransicionInvalida
from app.settings import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    async with AsyncExitStack() as pila:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        pila.push_async_callback(redis.aclose)
        await redis.ping()
        cola = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        pila.push_async_callback(cola.aclose)
        app.state.redis = redis
        app.state.cola = cola
        app.state.repositorio = RepositorioJobs(redis, settings.job_ttl_segundos)
        app.state.checkpointer = None if settings.checkpointer == "memory" else await pila.enter_async_context(abrir_checkpointer(settings))
        logger.info("API lista: Redis %s, checkpointer %s", settings.redis_url, settings.checkpointer)
        yield


def repositorio(request: Request) -> RepositorioJobs:
    return request.app.state.repositorio


def cola(request: Request) -> ArqRedis:
    return request.app.state.cola


Repositorio = Annotated[RepositorioJobs, Depends(repositorio)]
Cola = Annotated[ArqRedis, Depends(cola)]
tareas = APIRouter(prefix="/tasks", tags=["tareas"])
conversaciones = APIRouter(prefix="/conversations", tags=["conversaciones"])


async def job_o_404(repo: RepositorioJobs, job_id: str) -> dict[str, Any]:
    try:
        return await repo.obtener(job_id)
    except JobNoEncontrado:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la tarea {job_id}") from None


@tareas.post("", response_model=TareaEncolada, status_code=status.HTTP_202_ACCEPTED)
async def encolar_tarea(cuerpo: NuevaTarea, repo: Repositorio, arq: Cola) -> TareaEncolada:
    try:
        job = await repo.crear(cuerpo.consulta, cuerpo.publicar_para, cuerpo.conversacion_id)
    except ConversacionOcupada as error:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{error}. Esperá a que termine o resolvé su aprobación.") from None
    try:
        await arq.enqueue_job("ejecutar_tarea", job["job_id"], _job_id=f"{job['job_id']}:ejecucion")
    except Exception as error:
        await repo.transicionar(job["job_id"], {EstadoJob.PENDING}, EstadoJob.FAILED, error=f"No se pudo encolar: {error}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola de tareas no está disponible") from error
    return TareaEncolada(
        job_id=job["job_id"],
        conversacion_id=job["conversacion_id"],
        estado=job["estado"],
        consultar_en=f"/tasks/{job['job_id']}",
        aprobar_en=f"/tasks/{job['job_id']}/approve",
        conversacion_en=f"/conversations/{job['conversacion_id']}",
    )


@tareas.get("/{job_id}", response_model=VistaTarea)
async def consultar_tarea(job_id: str, repo: Repositorio) -> VistaTarea:
    return VistaTarea.desde_job(await job_o_404(repo, job_id))


@tareas.post("/{job_id}/approve", response_model=VistaTarea, status_code=status.HTTP_202_ACCEPTED, tags=["aprobación humana"])
async def decidir_tarea(job_id: str, decision: DecisionHumana, repo: Repositorio, arq: Cola) -> VistaTarea:
    await job_o_404(repo, job_id)
    try:
        job = await repo.transicionar(job_id, {EstadoJob.WAITING_APPROVAL}, EstadoJob.PENDING, decision_humana=decision.model_dump())
    except TransicionInvalida as error:
        raise HTTPException(status.HTTP_409_CONFLICT, f"La tarea está en {error.actual} y no espera aprobación") from None
    await arq.enqueue_job("reanudar_tarea", job_id, _job_id=f"{job_id}:reanudacion")
    logger.info("Tarea %s %s por %s", job_id, "aprobada" if decision.aprobado else "rechazada", decision.revisor)
    return VistaTarea.desde_job(job)


@tareas.get("", response_model=list[VistaTarea])
async def listar_tareas(repo: Repositorio, estado: EstadoJob | None = None, limite: int = Query(default=20, ge=1, le=100)) -> list[VistaTarea]:
    return [VistaTarea.desde_job(job) for job in await repo.listar(limite, estado)]


@conversaciones.get("/{conversacion_id}", response_model=Conversacion)
async def consultar_conversacion(conversacion_id: str, request: Request) -> Conversacion:
    checkpointer: BaseCheckpointSaver | None = request.app.state.checkpointer
    if checkpointer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Con CHECKPOINTER=memory el historial vive solo en el worker")
    tupla = await checkpointer.aget_tuple({"configurable": {"thread_id": conversacion_id}})
    if tupla is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {conversacion_id}")
    turnos = tupla.checkpoint.get("channel_values", {}).get("turnos", [])
    en_curso = await request.app.state.redis.get(BLOQUEO_CONVERSACION.format(conversacion_id))
    return Conversacion(conversacion_id=conversacion_id, turnos=[TurnoConversacion(**t) for t in turnos], turno_en_curso=en_curso)


app = FastAPI(
    title="The Intelligence System",
    description="Sistema de IA de grado producción: RAG híbrido sobre ChromaDB, orquestador multi-agente con supervisor planificador, workers Arq, estado y checkpoints en Redis, aprobación humana y trazas en Arize Phoenix.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(tareas)
app.include_router(conversaciones)


@app.get("/health", response_model=Salud, tags=["operación"])
async def salud(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> Salud:
    arq: ArqRedis = request.app.state.cola
    latido = await arq.get(f"{arq.default_queue_name}:health-check")
    return Salud(
        redis=await request.app.state.redis.ping(),
        tareas_en_cola=await arq.zcard(arq.default_queue_name),
        worker=latido.decode() if latido else "sin worker activo",
        checkpointer=settings.checkpointer,
        observabilidad=settings.observabilidad,
        modelo=f"{settings.llm_provider}/{settings.llm_model}",
    )
