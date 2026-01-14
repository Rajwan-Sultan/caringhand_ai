# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU-only first (much smaller than GPU version)
RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

# Install remaining dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Clean up pip cache
RUN rm -rf /root/.cache/pip

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code only (not the entire directory)
COPY app ./app
COPY main.py .
COPY .env .

# Create directory for ChromaDB with proper permissions
RUN mkdir -p /app/chromadb_data && chmod 777 /app/chromadb_data

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV CHROMA_PERSIST_DIR=/app/chromadb_data

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]