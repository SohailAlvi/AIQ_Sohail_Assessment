# Use official Python image
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=.:..


# Copy project code
COPY . ./


# Set work directory
WORKDIR ./

# Install system packages needed for OpenCV and others
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run with Gunicorn + Uvicorn workers
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers=2", "--timeout=120"]

#docker build -t object-detection-api .
#
#docker run -p 8000:8000 \
#  --env MONGO_URI=mongodb://host.docker.internal:27017 \
#  --env MINIO_ENDPOINT=host.docker.internal:9000 \
#  --env MONGO_USER=admin \
#  --env MONGO_PASS=admin123 \
#  --env MONGO_HOST=localhost:27017 \
#  --env MINIO_ENDPOINT=localhost:9000 \
#  --env MINIO_ACCESS_KEY=minioadmin \
#  --env MINIO_SECRET_KEY=minioadmin \
#  --env MINIO_BUCKET=object-detection \
#  object-detection-api

