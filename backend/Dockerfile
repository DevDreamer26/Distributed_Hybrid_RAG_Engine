FROM python:3.12-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install lightweight CPU-ONLY PyTorch (~180MB instead of 2.5GB CUDA runtime)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies (pip will see torch is already satisfied and skip CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and set permissions
COPY . .
RUN chmod +x entrypoint.sh

ENV PORT=8000
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]