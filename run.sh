#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example. Completá OPENAI_API_KEY y volvé a ejecutar ./run.sh"
  exit 1
fi
if ! grep -qE '^OPENAI_API_KEY=.+' .env; then
  echo "Falta OPENAI_API_KEY en .env (los embeddings del RAG la necesitan)"
  exit 1
fi

docker compose up --build -d

echo -n "Esperando a la API y al worker"
for _ in $(seq 1 90); do
  salud=$(curl -fs http://localhost:8000/health || true)
  if [[ "$salud" == *'"redis":true'* && "$salud" != *"sin worker activo"* ]]; then
    echo
    echo "$salud"
    echo "API:     http://localhost:8000/docs"
    echo "Phoenix: http://localhost:6006"
    echo "Pruebas: python scripts/pruebas_sistema.py"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo
echo "El sistema no respondió a tiempo. Revisá: docker compose logs worker api"
exit 1
