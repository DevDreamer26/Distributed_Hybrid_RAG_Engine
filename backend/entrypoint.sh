#!/usr/bin/env bash
set -e

PORT="${PORT:-10000}"

echo "Starting Celery worker in background..."
celery -A app.workers.celery_worker.celery_app worker --loglevel=info --pool=solo --without-gossip --without-mingle &

echo "Binding Uvicorn to 0.0.0.0:${PORT}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1 --no-access-log