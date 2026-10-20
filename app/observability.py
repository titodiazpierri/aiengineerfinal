import functools
import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from openinference.instrumentation import OITracer, TraceConfig, using_attributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.settings import Settings

logger = logging.getLogger("observabilidad")

F = TypeVar("F", bound=Callable[..., Any])


class Observabilidad:
    def __init__(self) -> None:
        self.proveedor: Any = None
        self.tracer: OITracer = OITracer(trace.get_tracer("intelligence-system"), config=TraceConfig())

    def configurar(self, settings: Settings) -> str:
        if settings.observabilidad != "phoenix":
            logger.warning("Observabilidad desactivada")
            return "ninguna"
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        self.proveedor = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint,
            batch=True,
            set_global_tracer_provider=False,
            verbose=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=self.proveedor)
        self.tracer = self.proveedor.get_tracer("intelligence-system")
        logger.info("Trazas hacia Arize Phoenix: %s (proyecto %s)", settings.phoenix_collector_endpoint, settings.phoenix_project_name)
        return "phoenix"

    def cerrar(self) -> None:
        if self.proveedor is not None:
            self.proveedor.force_flush()
            self.proveedor.shutdown()


observabilidad = Observabilidad()


def trazar_chain(nombre: str) -> Callable[[F], F]:
    def decorador(funcion: F) -> F:
        if inspect.iscoroutinefunction(funcion):
            @functools.wraps(funcion)
            async def envoltorio_async(*args: Any, **kwargs: Any) -> Any:
                return await observabilidad.tracer.chain(name=nombre)(funcion)(*args, **kwargs)

            return envoltorio_async

        @functools.wraps(funcion)
        def envoltorio(*args: Any, **kwargs: Any) -> Any:
            return observabilidad.tracer.chain(name=nombre)(funcion)(*args, **kwargs)

        return envoltorio

    return decorador


@contextmanager
def traza_de_tarea(conversacion_id: str, job_id: str, tipo: str, consulta: str) -> Iterator[Any]:
    with using_attributes(session_id=conversacion_id, metadata={"job_id": job_id, "conversacion_id": conversacion_id, "tipo": tipo}, tags=["intelligence-system", tipo]):
        with observabilidad.tracer.start_as_current_span(f"tarea.{tipo}", openinference_span_kind="chain") as span:
            span.set_input(consulta)
            span.set_attribute("job.id", job_id)
            yield span


def cerrar_span(span: Any, estado: str, salida: str, tokens: int) -> None:
    span.set_output(salida)
    span.set_attribute("job.estado", estado)
    span.set_attribute("job.tokens", tokens)
    span.set_status(Status(StatusCode.OK if estado != "FAILED" else StatusCode.ERROR))
