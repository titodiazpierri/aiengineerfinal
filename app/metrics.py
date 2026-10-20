import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class ConsumoNodo:
    llamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    segundos_llm: float = 0.0
    segundos_nodo: float = 0.0


@dataclass
class MedidorDeConsumo(AsyncCallbackHandler):
    por_nodo: dict[str, ConsumoNodo] = field(default_factory=dict)
    _en_curso: dict[UUID, tuple[str, float]] = field(default_factory=dict)

    @staticmethod
    def nodo_raiz(metadata: dict[str, Any] | None) -> str:
        metadata = metadata or {}
        espacio = metadata.get("langgraph_checkpoint_ns") or ""
        if espacio:
            return espacio.split("|")[0].split(":")[0]
        return metadata.get("langgraph_node") or "fuera_del_grafo"

    def _nodo(self, nombre: str) -> ConsumoNodo:
        return self.por_nodo.setdefault(nombre, ConsumoNodo())

    async def on_chat_model_start(self, serialized: dict[str, Any], messages: list, *, run_id: UUID, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._en_curso[run_id] = (self.nodo_raiz(metadata), time.perf_counter())

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        nodo, inicio = self._en_curso.pop(run_id, ("fuera_del_grafo", time.perf_counter()))
        consumo = self._nodo(nodo)
        consumo.llamadas_llm += 1
        consumo.segundos_llm += time.perf_counter() - inicio
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None) or {}
                consumo.tokens_entrada += uso.get("input_tokens", 0)
                consumo.tokens_salida += uso.get("output_tokens", 0)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._en_curso.pop(run_id, None)

    def registrar_duracion(self, nodo: str, segundos: float) -> None:
        self._nodo(nodo).segundos_nodo += segundos

    @property
    def tokens_totales(self) -> int:
        return sum(c.tokens_entrada + c.tokens_salida for c in self.por_nodo.values())

    def resumen(self) -> dict[str, dict[str, float]]:
        return {
            nodo: {
                "llamadas_llm": c.llamadas_llm,
                "tokens_entrada": c.tokens_entrada,
                "tokens_salida": c.tokens_salida,
                "segundos_llm": round(c.segundos_llm, 3),
                "segundos_nodo": round(c.segundos_nodo, 3),
            }
            for nodo, c in sorted(self.por_nodo.items())
        }


medidor_actual: ContextVar[MedidorDeConsumo | None] = ContextVar("medidor_actual", default=None)
