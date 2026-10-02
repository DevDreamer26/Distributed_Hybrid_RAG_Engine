FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Render sets the PORT environment variable automatically (e.g. 10000)
ENV PORT=8000

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]