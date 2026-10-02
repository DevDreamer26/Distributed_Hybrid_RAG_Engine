FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (build-essential, curl, supervisor)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt supervisor

# Copy application source code
COPY . .

# Default port for Render/Railway
ENV PORT=8000

EXPOSE 8000

CMD ["supervisord", "-c", "/app/supervisord.conf"]