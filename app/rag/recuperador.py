import logging
import re
import unicodedata

from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from app.rag.modelos import Fragmento, FragmentoRecuperado, ResultadoBusqueda
from app.settings import Settings

logger = logging.getLogger("rag.recuperador")

CONSTANTE_RRF = 60
PALABRAS_VACIAS = {
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo", "los", "para", "por", "que",
    "se", "su", "sus", "un", "una", "y", "o", "cual", "cuales", "cuanto", "cuantos", "como", "mes",
}


def tokenizar(texto: str) -> list[str]:
    plano = "".join(c for c in unicodedata.normalize("NFKD", texto.lower()) if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", plano) if t not in PALABRAS_VACIAS]


class RecuperadorHibrido:
    def __init__(self, coleccion: AsyncCollection, embeddings: Embeddings, fragmentos: list[Fragmento], settings: Settings) -> None:
        self.coleccion = coleccion
        self.embeddings = embeddings
        self.fragmentos = {f.chunk_id: f for f in fragmentos}
        self.orden = list(self.fragmentos)
        self.bm25 = BM25Okapi([tokenizar(f.texto) for f in fragmentos])
        self.k = settings.rag_top_k
        self.peso_vectorial = settings.rag_peso_vectorial

    @classmethod
    async def crear(cls, cliente: AsyncClientAPI, embeddings: Embeddings, settings: Settings) -> "RecuperadorHibrido":
        coleccion = await cliente.get_collection(settings.chroma_collection)
        datos = await coleccion.get(include=["documents", "metadatas"])
        fragmentos = [Fragmento(chunk_id=i, texto=d, **m) for i, d, m in zip(datos["ids"], datos["documents"], datos["metadatas"])]
        logger.info("Recuperador híbrido listo: %d fragmentos en BM25 y en Chroma", len(fragmentos))
        return cls(coleccion, embeddings, fragmentos, settings)

    async def _ranking_vectorial(self, consulta: str, candidatos: int, tipo: str | None) -> list[str]:
        vector = await self.embeddings.aembed_query(consulta)
        resultado = await self.coleccion.query(query_embeddings=[vector], n_results=candidatos, where={"tipo": tipo} if tipo else None)
        return resultado["ids"][0]

    def _ranking_lexico(self, consulta: str, candidatos: int, tipo: str | None) -> list[str]:
        puntajes = self.bm25.get_scores(tokenizar(consulta))
        ranking = sorted(zip(self.orden, puntajes), key=lambda par: par[1], reverse=True)
        return [i for i, p in ranking if p > 0 and (tipo is None or self.fragmentos[i].tipo == tipo)][:candidatos]

    async def buscar(self, consulta: str, k: int | None = None, tipo: str | None = None) -> ResultadoBusqueda:
        k = k or self.k
        candidatos = min(len(self.orden), k * 3)
        vectorial = await self._ranking_vectorial(consulta, candidatos, tipo)
        lexica = self._ranking_lexico(consulta, candidatos, tipo)

        puntajes: dict[str, float] = {}
        origenes: dict[str, list[str]] = {}
        for nombre, ranking, peso in (("vectorial", vectorial, self.peso_vectorial), ("lexica", lexica, 1 - self.peso_vectorial)):
            for posicion, chunk_id in enumerate(ranking, start=1):
                puntajes[chunk_id] = puntajes.get(chunk_id, 0.0) + peso / (CONSTANTE_RRF + posicion)
                origenes.setdefault(chunk_id, []).append(nombre)

        mejores = sorted(puntajes, key=puntajes.get, reverse=True)[:k]
        fragmentos = [
            FragmentoRecuperado(**self.fragmentos[i].model_dump(), puntaje=round(puntajes[i], 5), recuperado_por=origenes[i])
            for i in mejores
        ]
        return ResultadoBusqueda(consulta=consulta, fragmentos=fragmentos, total=len(fragmentos))
