#!/bin/bash
set -euo pipefail
export GOWORK=off
root=$(cd "$(dirname "$0")/.." && pwd)
image=pgvector/pgvector@sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d
docker info >/dev/null || { echo 'PostgreSQL integration requires a running Docker daemon' >&2; exit 1; }
container=$(docker run -d --rm --label ragy.test=postgres -e POSTGRES_PASSWORD=ragy-test -e POSTGRES_DB=ragy "$image")
cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((attempt=0; attempt<60; attempt++)); do
  if docker exec "$container" pg_isready -U postgres -d ragy >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if (( ready == 0 )); then docker logs "$container" >&2; echo 'PostgreSQL readiness deadline exceeded' >&2; exit 1; fi
docker exec "$container" psql -U postgres -d ragy -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION IF NOT EXISTS vector' >/dev/null
cd "$root/adapters/pgvector"
RAGY_PG_TEST_CONTAINER="$container" "${GO:-go}" test -race -count=1 -tags=integration_pg -timeout=10m ./...
