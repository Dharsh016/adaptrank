# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]