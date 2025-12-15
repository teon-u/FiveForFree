# Multi-stage build for NASDAQ Predictor

# Stage 1: Frontend build
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy backend code
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create data directories
RUN mkdir -p data/raw data/processed data/models logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_GPU=true

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/api/health')"

# Run the application
CMD ["python3", "scripts/run_system.py", "--host", "0.0.0.0", "--port", "8000"]
