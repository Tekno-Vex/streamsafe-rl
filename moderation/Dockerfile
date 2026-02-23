# Build stage
FROM python:3.13-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies (libgomp for uvloop)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use local pip packages
ENV PATH=/root/.local/bin/:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY ./app /app/app

COPY ./models/ppo_policy.onnx /app/models/ppo_policy.onnx
COPY ./models/ppo_policy.onnx.data /app/models/ppo_policy.onnx.data

# Create logs directory for Parquet files
RUN mkdir -p /app/logs/parquet

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)"

# Expose port
EXPOSE 8000

# Start service
CMD ["python3", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]