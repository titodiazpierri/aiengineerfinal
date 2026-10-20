import operator
import re
import statistics
import unicodedata
from collections import defaultdict
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

POSITIVAS = ("excelent", "buen", "rapid", "barat", "encant", "comod", "facil", "mejor", "reintegr", "recomiend", "util", "disponib", "respond", "solucion")
NEGATIVAS = ("agot", "frustr", "defectuos", "apag", "cierr", "lent", "fall", "error", "rot")
NEGATIVAS_EXACTAS = {"malo", "mala", "malos", "malas", "caro", "cara", "caros", "caras", "peor"}
NEGADORES = {"no", "nunca", "nadie", "ni", "tampoco", "sin"}
INTENSIFICADORES = {"muy": 1.5, "mucho": 1.5, "re": 1.5, "super": 1.5, "otra": 1.3}
ASPECTOS = {
    "bateria": ("bateri", "carga", "apag"),
    "aplicacion": ("app", "aplicacion", "desbloque"),
    "precio": ("tarifa", "precio", "caro", "barat", "cobr"),
    "atencion": ("soporte", "atencion", "reclam", "respond"),
    "disponibilidad": ("disponib", "estacion"),
}
COMPARADORES = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}


class EntradaSerie(BaseModel):
    periodos: list[str] = Field(description="Períodos en formato AAAA-MM, por ejemplo ['2026-07', '2026-08', '2026-09'].")
    valores: list[float] = Field(description="Valores sin separadores de miles, por ejemplo [48200, 53900, 61700].")


class EntradaValores(BaseModel):
    valores: list[float] = Field(min_length=1, description="Valores numéricos sin separadores de miles, en orden cronológico si es una serie.")


class EntradaUmbral(BaseModel):
    metrica: str = Field(min_length=1, description="Nombre de la métrica evaluada.")
    valor: float = Field(description="Valor calculado de la métrica.")
    operador: Literal[">", ">=", "<", "<=", "=="] = Field(description="Comparación a aplicar.")
    umbral: float = Field(description="Valor de referencia del criterio.")


class EntradaTextos(BaseModel):
    textos: list[str] = Field(min_length=1, max_length=200, description="Comentarios textuales, uno por elemento.")


class SerieTemporal(BaseModel):
    periodos: list[str] = Field(min_length=2)
    valores: list[float] = Field(min_length=2)

    @field_validator("periodos")
    @classmethod
    def formato_periodos(cls, periodos: list[str]) -> list[str]:
        invalidos = [p for p in periodos if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", p)]
        if invalidos:
            raise ValueError(f"Períodos con formato inválido (se espera AAAA-MM): {invalidos}")
        if len(set(periodos)) != len(periodos):
            raise ValueError("Hay períodos repetidos")
        if periodos != sorted(periodos):
            raise ValueError("Los períodos no están en orden cronológico")
        return periodos

    @field_validator("valores")
    @classmethod
    def valores_positivos(cls, valores: list[float]) -> list[float]:
        if any(v <= 0 for v in valores):
            raise ValueError("Todos los valores deben ser mayores a 0")
        return valores

    @model_validator(mode="after")
    def mismo_largo(self) -> "SerieTemporal":
        if len(self.periodos) != len(self.valores):
            raise ValueError(f"Hay {len(self.periodos)} períodos y {len(self.valores)} valores")
        return self


class ResultadoValidacionSerie(BaseModel):
    valida: bool
    errores: list[str] = []
    serie: SerieTemporal | None = None


class Estadisticas(BaseModel):
    cantidad: int
    total: float
    promedio: float
    mediana: float
    minimo: float
    maximo: float
    desvio_estandar: float


class Crecimiento(BaseModel):
    valor_inicial: float
    valor_final: float
    diferencia_absoluta: float
    variacion_total_porcentual: float
    variaciones_por_periodo: list[float]
    tasa_crecimiento_compuesta_porcentual: float


class EvaluacionUmbral(BaseModel):
    metrica: str
    valor: float
    criterio: str
    cumple: bool


class SentimientoTexto(BaseModel):
    texto: str
    puntaje: float
    sentimiento: Literal["positivo", "negativo", "neutral"]
    aspectos: list[str]
    senales: list[str]


class ResumenAspecto(BaseModel):
    menciones: int
    positivos: int
    negativos: int
    puntaje_promedio: float


class ResumenGlobal(BaseModel):
    total: int
    positivos: int
    negativos: int
    neutrales: int
    porcentaje_negativo: float
    sentimiento_predominante: Literal["positivo", "negativo", "mixto"]
    aspecto_mas_critico: str


class AnalisisSentimiento(BaseModel):
    detalle: list[SentimientoTexto]
    por_aspecto: dict[str, ResumenAspecto]
    global_: ResumenGlobal = Field(alias="global", serialization_alias="global")


class ErrorCalculo(BaseModel):
    error: str


@tool(args_schema=EntradaSerie)
def validar_serie_temporal(periodos: list[str], valores: list[float]) -> dict[str, Any]:
    """Valida con un esquema Pydantic que una serie temporal esté bien formada antes de calcular sobre ella:
    al menos dos puntos, períodos AAAA-MM sin repetir y en orden cronológico, valores positivos y la misma
    cantidad de períodos que de valores. Usar SIEMPRE antes de calcular crecimientos o estadísticas de una serie."""
    try:
        serie = SerieTemporal(periodos=periodos, valores=valores)
    except ValidationError as error:
        return ResultadoValidacionSerie(valida=False, errores=[e["msg"] for e in error.errors()]).model_dump()
    return ResultadoValidacionSerie(valida=True, serie=serie).model_dump()


@tool(args_schema=EntradaValores)
def estadisticas_serie(valores: list[float]) -> dict[str, Any]:
    """Calcula estadísticas descriptivas: cantidad, total, promedio, mediana, mínimo, máximo y desvío estándar."""
    return Estadisticas(
        cantidad=len(valores),
        total=round(sum(valores), 2),
        promedio=round(statistics.mean(valores), 2),
        mediana=round(statistics.median(valores), 2),
        minimo=min(valores),
        maximo=max(valores),
        desvio_estandar=round(statistics.stdev(valores), 2) if len(valores) > 1 else 0.0,
    ).model_dump()


@tool(args_schema=EntradaValores)
def crecimiento_de_serie(valores: list[float]) -> dict[str, Any]:
    """Calcula el crecimiento de una serie en orden cronológico: variación total entre el primer y el último valor,
    variación de cada período respecto del anterior y tasa de crecimiento compuesta por período."""
    if len(valores) < 2 or any(v <= 0 for v in valores):
        return ErrorCalculo(error="Se necesitan al menos dos valores positivos.").model_dump()
    periodos = len(valores) - 1
    return Crecimiento(
        valor_inicial=valores[0],
        valor_final=valores[-1],
        diferencia_absoluta=round(valores[-1] - valores[0], 2),
        variacion_total_porcentual=round((valores[-1] / valores[0] - 1) * 100, 2),
        variaciones_por_periodo=[round((b / a - 1) * 100, 2) for a, b in zip(valores, valores[1:])],
        tasa_crecimiento_compuesta_porcentual=round(((valores[-1] / valores[0]) ** (1 / periodos) - 1) * 100, 2),
    ).model_dump()


@tool(args_schema=EntradaUmbral)
def comparar_con_umbral(metrica: str, valor: float, operador: str, umbral: float) -> dict[str, Any]:
    """Evalúa si una métrica cumple un criterio de decisión numérico, por ejemplo 'tasa de crecimiento > 10'."""
    return EvaluacionUmbral(metrica=metrica, valor=valor, criterio=f"{operador} {umbral}", cumple=COMPARADORES[operador](valor, umbral)).model_dump()


def _tokens(texto: str) -> list[str]:
    plano = "".join(c for c in unicodedata.normalize("NFKD", texto.lower()) if not unicodedata.combining(c))
    return re.findall(r"[a-z]+", plano)


def _polaridad(palabra: str) -> int:
    if palabra in NEGATIVAS_EXACTAS or palabra.startswith(NEGATIVAS):
        return -1
    if palabra.startswith(POSITIVAS):
        return 1
    return 0


def puntuar(texto: str) -> SentimientoTexto:
    palabras = _tokens(texto)
    puntaje = 0.0
    senales = []
    for i, palabra in enumerate(palabras):
        polaridad = _polaridad(palabra)
        if not polaridad:
            continue
        ventana = palabras[max(0, i - 3):i]
        if NEGADORES & set(ventana):
            polaridad = -polaridad
        puntaje += polaridad * max((INTENSIFICADORES.get(p, 1.0) for p in ventana), default=1.0)
        senales.append(("+" if polaridad > 0 else "-") + palabra)
    aspectos = sorted({a for a, raices in ASPECTOS.items() if any(p.startswith(raices) for p in palabras)}) or ["general"]
    sentimiento = "positivo" if puntaje > 0 else "negativo" if puntaje < 0 else "neutral"
    return SentimientoTexto(texto=texto, puntaje=round(puntaje, 2), sentimiento=sentimiento, aspectos=aspectos, senales=senales)


@tool(args_schema=EntradaTextos)
def sentimiento_por_aspecto(textos: list[str]) -> dict[str, Any]:
    """Analiza el sentimiento de comentarios en español y lo agrupa por aspecto del servicio (batería, aplicación,
    precio, atención, disponibilidad o general). Contempla negaciones ('nunca hay disponibles', 'nadie responde')
    e intensificadores ('muy malo'). Pasar cada comentario textual como un elemento separado."""
    detalle = [puntuar(t) for t in textos]
    grupos: dict[str, list[SentimientoTexto]] = defaultdict(list)
    for d in detalle:
        for aspecto in d.aspectos:
            grupos[aspecto].append(d)
    por_aspecto = {
        a: ResumenAspecto(
            menciones=len(g),
            positivos=sum(x.sentimiento == "positivo" for x in g),
            negativos=sum(x.sentimiento == "negativo" for x in g),
            puntaje_promedio=round(statistics.mean(x.puntaje for x in g), 2),
        )
        for a, g in sorted(grupos.items())
    }
    positivos = sum(d.sentimiento == "positivo" for d in detalle)
    negativos = sum(d.sentimiento == "negativo" for d in detalle)
    peor = max(por_aspecto, key=lambda a: (por_aspecto[a].negativos, -por_aspecto[a].puntaje_promedio))
    resumen = ResumenGlobal(
        total=len(detalle),
        positivos=positivos,
        negativos=negativos,
        neutrales=len(detalle) - positivos - negativos,
        porcentaje_negativo=round(negativos / len(detalle) * 100, 2),
        sentimiento_predominante="negativo" if negativos > positivos else "positivo" if positivos > negativos else "mixto",
        aspecto_mas_critico=peor,
    )
    return AnalisisSentimiento(detalle=detalle, por_aspecto=por_aspecto, **{"global": resumen}).model_dump(by_alias=True)


HERRAMIENTAS_ANALISIS = [validar_serie_temporal, estadisticas_serie, crecimiento_de_serie, comparar_con_umbral, sentimiento_por_aspecto]
