from typing import Literal

from pydantic import BaseModel, Field


class DatoVerificado(BaseModel):
    descripcion: str = Field(description="Qué mide el dato, por ejemplo 'viajes de agosto 2026'.")
    valor: str = Field(description="Valor exacto tal como figura en el documento, por ejemplo '53.900'.")
    unidad: str = Field(description="Unidad del valor, por ejemplo 'viajes', 'USD', 'proporción'.")
    doc_id: str = Field(description="doc_id del documento de donde sale el dato.")
    fecha_fuente: str = Field(description="Fecha del documento fuente.")
    tipo_fuente: str = Field(description="Tipo del documento: informe oficial, nota de prensa preliminar, etc.")


class TextoFuente(BaseModel):
    doc_id: str
    texto: str = Field(description="Texto transcripto literalmente, por ejemplo un reclamo de soporte.")


class InformeInvestigacion(BaseModel):
    datos: list[DatoVerificado] = Field(description="Datos encontrados con su fuente.")
    textos: list[TextoFuente] = Field(default_factory=list, description="Textos transcriptos literalmente si la tarea los pide.")
    conflictos: list[str] = Field(default_factory=list, description="Cifras contradictorias entre documentos y cuál es la fuente oficial más reciente.")
    faltantes: list[str] = Field(default_factory=list, description="Información pedida que no se encontró.")


class ResultadoCalculo(BaseModel):
    metrica: str = Field(description="Nombre de la métrica calculada.")
    valor: str = Field(description="Valor exacto devuelto por la herramienta.")
    unidad: str = Field(description="Unidad del resultado, por ejemplo '%', 'viajes', 'cumple'.")
    herramienta: str = Field(description="Nombre de la herramienta que lo calculó.")
    insumos: str = Field(description="Datos de entrada usados y su doc_id.")


class InformeAnalisis(BaseModel):
    resultados: list[ResultadoCalculo] = Field(description="Resultados calculados con herramientas.")
    conclusiones: list[str] = Field(default_factory=list, description="Conclusiones que se desprenden directamente de los resultados.")
    faltantes: list[str] = Field(default_factory=list, description="Datos necesarios que no estaban disponibles.")


class TareaPlan(BaseModel):
    id: str = Field(description="Identificador corto de la tarea, por ejemplo T1, T2.")
    agente: Literal["investigador", "analista"]
    objetivo: str = Field(description="Instrucción autocontenida para el especialista.")
    depende_de: list[str] = Field(default_factory=list, description="ids de las tareas cuyos resultados necesita.")


class DecisionSupervisor(BaseModel):
    evaluacion: str = Field(description="Evaluación del último aporte según la rúbrica. 'Sin aportes' si no hay ninguno.")
    veredicto: Literal["aprobar", "refinar", "sin_aporte"] = Field(description="Veredicto sobre el último aporte recibido.")
    observacion: str = Field(default="", description="Si el veredicto es 'refinar', qué tiene que corregir el especialista.")
    nuevas_tareas: list[TareaPlan] = Field(default_factory=list, description="Tareas a agregar al plan. En la primera decisión, el plan completo.")
    siguiente: Literal["investigador", "analista", "sintetizar"] = Field(description="Quién interviene ahora.")
    tarea_id: str = Field(default="", description="id de la tarea a ejecutar. Vacío si siguiente es 'sintetizar'.")
    razonamiento: str = Field(description="Justificación breve de la decisión.")


class Hallazgo(BaseModel):
    afirmacion: str = Field(description="Hallazgo en una oración.")
    valor: str = Field(description="Cifra que respalda el hallazgo, copiada exactamente de los aportes. Vacío si no aplica.")
    fuente: str = Field(description="doc_id o id de tarea que respalda el hallazgo.")


class InformeFinal(BaseModel):
    resumen_ejecutivo: str = Field(description="Respuesta directa a la consulta en dos o tres oraciones.")
    hallazgos: list[Hallazgo] = Field(min_length=1)
    conflictos_resueltos: list[str] = Field(default_factory=list, description="Contradicciones entre fuentes y cómo se resolvieron.")
    recomendacion: str = Field(min_length=10)
    limitaciones: list[str] = Field(default_factory=list)


class EvaluacionRubrica(BaseModel):
    cobertura: int = Field(ge=1, le=5, description="¿Responde todas las partes de la consulta?")
    fidelidad: int = Field(ge=1, le=5, description="¿Cada cifra coincide con los aportes de los especialistas?")
    trazabilidad: int = Field(ge=1, le=5, description="¿Cada hallazgo cita una fuente verificable?")
    manejo_conflictos: int = Field(ge=1, le=5, description="¿Se resolvieron las contradicciones entre fuentes priorizando la oficial más reciente?")
    problemas: list[str] = Field(default_factory=list, description="Problemas concretos encontrados.")
