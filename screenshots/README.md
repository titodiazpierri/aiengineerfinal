# Evidencia de observabilidad (Arize Phoenix)

Capturas de http://localhost:6006, proyecto `intelligence-system`, tomadas después de `python scripts/pruebas_sistema.py`. Poner el selector de tiempo en **Last 1 Hour** antes de capturar.

| Archivo | Pantalla | Qué tiene que verse |
|---|---|---|
| `01_trazas.png` | Pestaña **Traces** | Las trazas `tarea.ejecucion` y `tarea.reanudacion` de las 5 pruebas con `latency`, `total tokens` y `total cost` (costo por ejecución) |
| `02_arbol_de_spans.png` | Click en la traza de la prueba 1 | Árbol con supervisor, investigador (`buscar_conocimiento`, `leer_documento`), analista (herramientas Pydantic), sintetizador, revisor, `control_criticidad` y `cerrar_turno` |
| `03_sesion_multiturno.png` | Pestaña **Sessions** → sesión de la prueba 2 | Dos trazas en la misma sesión (`session.id` = `conversacion_id`) |
| `04_contexto_turno_anterior.png` | Span LLM del supervisor en el turno 2 de la prueba 2 | El prompt con el bloque `CONVERSACIÓN PREVIA` que viene del checkpointer |
| `05_sesion_hitl.png` | Pestaña **Sessions** → sesión de la prueba 4 | `tarea.ejecucion` (termina en la pausa) y `tarea.reanudacion` (aprobación y `publicar_informe`) |
| `06_latencia_p95.png` | Pestaña **Metrics** → **Trace latency** | Percentiles de latencia con el valor de P95 |
| `07_costo.png` | Pestaña **Metrics** → **Cost** | Costo en USD separado en tokens de entrada y de salida |
| `08_conversacion_tras_reinicio.png` | http://localhost:8000/docs → `GET /conversations/{id}` | Los turnos de una conversación leídos después de `docker compose restart api worker` |
La captura 08 es de Swagger (http://localhost:8000/docs), no de Phoenix.
