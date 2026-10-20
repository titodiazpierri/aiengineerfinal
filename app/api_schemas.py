from typing import Any, Literal

from pydantic import BaseModel, Field

from app.jobs import EstadoJob


class NuevaTarea(BaseModel):
    consulta: str = Field(min_length=10, max_length=2000, description="Consulta de análisis para el sistema multi-agente.")
    conversacion_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{6,64}$",
        description="Id de una conversación existente para continuarla. Si se omite, se crea una conversación nueva.",
    )
    publicar_para: list[str] = Field(default_factory=list, max_length=10, description="Destinatarios a los que publicar el informe. Vuelve crítica la tarea.")


class TareaEncolada(BaseModel):
    job_id: str
    conversacion_id: str
    estado: EstadoJob
    consultar_en: str
    aprobar_en: str
    conversacion_en: str


class DecisionHumana(BaseModel):
    aprobado: bool
    revisor: str = Field(min_length=1, max_length=100)
    comentario: str = Field(default="", max_length=1000)


class VistaTarea(BaseModel):
    job_id: str
    conversacion_id: str
    estado: EstadoJob
    consulta: str
    publicar_para: list[str]
    creado: str
    actualizado: str
    progreso: dict[str, Any]
    historial: list[dict[str, Any]]
    solicitud_aprobacion: dict[str, Any] | None = None
    decision_humana: dict[str, Any] | None = None
    resultado: dict[str, Any] | None = None
    consumos: list[dict[str, Any]] = []
    error: str | None = None

    @classmethod
    def desde_job(cls, job: dict[str, Any]) -> "VistaTarea":
        return cls(**{k: v for k, v in job.items() if k in cls.model_fields})


class TurnoConversacion(BaseModel):
    numero: int
    job_id: str
    consulta: str
    respuesta: str
    estado: str


class Conversacion(BaseModel):
    conversacion_id: str
    turnos: list[TurnoConversacion]
    turno_en_curso: str | None = Field(description="job_id del turno que se está ejecutando o espera aprobación, si hay uno.")
    fuente: Literal["checkpointer"] = "checkpointer"


class Salud(BaseModel):
    redis: bool
    tareas_en_cola: int
    worker: str
    checkpointer: str
    observabilidad: str
    modelo: str
