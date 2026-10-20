from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.rag.modelos import Documento, ResultadoBusqueda, TipoDocumento
from app.rag.recuperador import RecuperadorHibrido


class EntradaBusqueda(BaseModel):
    consulta: str = Field(min_length=3, max_length=300, description="Pregunta o palabras clave en español, por ejemplo 'viajes por mes del tercer trimestre'.")
    cantidad: int = Field(default=5, ge=1, le=8, description="Cantidad de fragmentos a devolver.")
    tipo: TipoDocumento | None = Field(default=None, description="Filtrar por tipo de documento, por ejemplo 'informe oficial'. Omitir para buscar en todos.")


class EntradaLectura(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9-]+$", description="doc_id exacto devuelto por buscar_conocimiento, por ejemplo 'informe-operativo-q3-2026'.")


class DocumentoNoEncontrado(BaseModel):
    error: str
    doc_ids_validos: list[str]


DESCRIPCION_BUSQUEDA = """Busca en la base de conocimiento de EcoRuta (informes operativos oficiales, notas de prensa, tickets de soporte,
encuestas, manuales, políticas y documentos internos) con recuperación híbrida: similitud de embeddings en ChromaDB
más ranking léxico BM25, fusionados con Reciprocal Rank Fusion.

Usar primero para descubrir qué documentos y fragmentos responden la pregunta. Cada fragmento trae doc_id, título,
tipo de fuente, fecha y sección. Si un fragmento no alcanza para tener la cifra exacta, usar leer_documento."""

DESCRIPCION_LECTURA = """Devuelve un documento completo de la base de conocimiento con su tipo, fecha y, si los tiene, la lista completa de
textos (por ejemplo, los reclamos de soporte transcriptos). Usar después de buscar_conocimiento para obtener cifras
exactas o transcribir textos."""


def crear_herramientas_busqueda(recuperador: RecuperadorHibrido, documentos: dict[str, Documento]) -> list[BaseTool]:
    async def buscar_conocimiento(consulta: str, cantidad: int = 5, tipo: str | None = None) -> dict[str, Any]:
        resultado: ResultadoBusqueda = await recuperador.buscar(consulta, k=cantidad, tipo=tipo)
        return resultado.model_dump()

    async def leer_documento(doc_id: str) -> dict[str, Any]:
        documento = documentos.get(doc_id)
        if documento is None:
            return DocumentoNoEncontrado(error=f"No existe el documento '{doc_id}'", doc_ids_validos=sorted(documentos)).model_dump()
        return documento.model_dump()

    return [
        StructuredTool.from_function(coroutine=buscar_conocimiento, name="buscar_conocimiento", description=DESCRIPCION_BUSQUEDA, args_schema=EntradaBusqueda),
        StructuredTool.from_function(coroutine=leer_documento, name="leer_documento", description=DESCRIPCION_LECTURA, args_schema=EntradaLectura),
    ]
