import hashlib
import json
import logging
import re
from functools import lru_cache

from chromadb.api import AsyncClientAPI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.documentos import cargar_documentos
from app.rag.modelos import Documento, EstadoIndice, Fragmento
from app.settings import Settings

logger = logging.getLogger("rag.ingesta")

SEPARADORES = ["\n## ", "\n### ", "\n\n", "\n- ", "\n", ". ", " ", ""]
TITULO = re.compile(r"^#{1,3} (.+)$", re.MULTILINE)


@lru_cache
def _codificador():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("tiktoken no disponible: se estiman los tokens como caracteres / 4")
        return None


def contar_tokens(texto: str) -> int:
    codificador = _codificador()
    return len(codificador.encode(texto)) if codificador else max(1, len(texto) // 4)


def fragmentar(documentos: dict[str, Documento], settings: Settings) -> list[Fragmento]:
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=contar_tokens,
        separators=SEPARADORES,
        add_start_index=True,
    )
    fragmentos = []
    for documento in documentos.values():
        for numero, parte in enumerate(divisor.create_documents([documento.contenido]), start=1):
            inicio = parte.metadata["start_index"]
            titulos = [m.group(1) for m in TITULO.finditer(documento.contenido) if m.start() <= inicio + 1]
            seccion = titulos[-1] if titulos else documento.titulo
            fragmentos.append(Fragmento(
                chunk_id=f"{documento.doc_id}#c{numero}",
                doc_id=documento.doc_id,
                titulo=documento.titulo,
                tipo=documento.tipo,
                fecha=documento.fecha,
                seccion=seccion,
                texto=f"[{documento.titulo} | {seccion}]\n{parte.page_content.strip()}",
            ))
    return fragmentos


def calcular_huella(fragmentos: list[Fragmento], settings: Settings) -> str:
    base = json.dumps([f.model_dump() for f in fragmentos], ensure_ascii=False) + settings.embedding_model
    return hashlib.sha256(base.encode()).hexdigest()[:16]


async def asegurar_indice(cliente: AsyncClientAPI, embeddings: Embeddings, settings: Settings) -> EstadoIndice:
    documentos = cargar_documentos(settings.directorio_conocimiento)
    fragmentos = fragmentar(documentos, settings)
    huella = calcular_huella(fragmentos, settings)

    coleccion = await cliente.get_or_create_collection(settings.chroma_collection, metadata={"hnsw:space": "cosine"})
    actual = (coleccion.metadata or {}).get("huella")
    if actual == huella and await coleccion.count() == len(fragmentos):
        logger.info("Índice al día (%d fragmentos, huella %s): no se reindexa", len(fragmentos), huella)
        return EstadoIndice(coleccion=settings.chroma_collection, fragmentos=len(fragmentos), documentos=len(documentos), huella=huella, reindexado=False)

    logger.info("Indexando %d fragmentos de %d documentos con %s", len(fragmentos), len(documentos), settings.embedding_model)
    await cliente.delete_collection(settings.chroma_collection)
    coleccion = await cliente.create_collection(
        settings.chroma_collection,
        metadata={"hnsw:space": "cosine", "huella": huella, "embedding_model": settings.embedding_model},
    )
    vectores = await embeddings.aembed_documents([f.texto for f in fragmentos])
    await coleccion.add(
        ids=[f.chunk_id for f in fragmentos],
        documents=[f.texto for f in fragmentos],
        embeddings=vectores,
        metadatas=[f.model_dump(exclude={"chunk_id", "texto"}) for f in fragmentos],
    )
    return EstadoIndice(coleccion=settings.chroma_collection, fragmentos=len(fragmentos), documentos=len(documentos), huella=huella, reindexado=True)
