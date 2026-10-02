#!/usr/bin/env bash
set -e

# Default to port 8000 if PORT is not set by Render
PORT="${PORT:-8000}"

echo "Starting Celery worker with solo pool to stay under 512MB RAM..."
# --pool=solo avoids preforking multiple Python processes, saving ~200MB RAM
celery -A app.workers.celery_worker.celery_app worker --loglevel=info --pool=solo &

echo "Starting Uvicorn on 0.0.0.0:${PORT}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"