from typing import Literal

from pydantic import BaseModel, Field

TipoDocumento = Literal[
    "informe oficial",
    "nota de prensa preliminar",
    "tickets de soporte",
    "encuesta",
    "documento interno",
    "manual técnico",
    "política",
]


class MetadatosDocumento(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9-]+$")
    titulo: str
    tipo: TipoDocumento
    fecha: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    categoria: str


class Documento(MetadatosDocumento):
    contenido: str
    textos: list[str] = Field(default_factory=list, description="Elementos de lista del documento, por ejemplo reclamos transcriptos")


class Fragmento(BaseModel):
    chunk_id: str
    doc_id: str
    titulo: str
    tipo: str
    fecha: str
    seccion: str
    texto: str


class FragmentoRecuperado(Fragmento):
    puntaje: float = Field(description="Puntaje de fusión Reciprocal Rank Fusion")
    recuperado_por: list[Literal["vectorial", "lexica"]]


class ResultadoBusqueda(BaseModel):
    consulta: str
    fragmentos: list[FragmentoRecuperado]
    total: int


class EstadoIndice(BaseModel):
    coleccion: str
    fragmentos: int
    documentos: int
    huella: str
    reindexado: bool
