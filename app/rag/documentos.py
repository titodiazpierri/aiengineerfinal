import re
from functools import lru_cache
from pathlib import Path

from app.rag.modelos import Documento


def leer_markdown(ruta: Path) -> Documento:
    texto = ruta.read_text(encoding="utf-8")
    if not texto.startswith("---"):
        raise ValueError(f"{ruta.name} no tiene encabezado de metadatos")
    _, cabecera, cuerpo = texto.split("---", 2)
    metadatos = {}
    for linea in cabecera.strip().splitlines():
        clave, _, valor = linea.partition(":")
        metadatos[clave.strip()] = valor.strip()
    textos = [m.strip() for m in re.findall(r"^- (.+)$", cuerpo, flags=re.MULTILINE)]
    return Documento(**metadatos, contenido=cuerpo.strip(), textos=textos)


@lru_cache
def cargar_documentos(directorio: Path) -> dict[str, Documento]:
    documentos = [leer_markdown(ruta) for ruta in sorted(directorio.glob("*.md"))]
    if not documentos:
        raise FileNotFoundError(f"No hay documentos .md en {directorio}")
    return {d.doc_id: d for d in documentos}
