
# 📸 Depth-based Frame Stitching API

A FastAPI microservice that serves depth-filtered stitched images from frame data stored in MinIO and metadata stored in MongoDB. Supports Redis caching and an optional NGINX cache layer to simulate a CDN.

---

## 🚀 Features

- ✅ Depth-based filtering of frames
- ✅ Frame storage in MinIO (object storage)
- ✅ Metadata storage in MongoDB
- ✅ Redis-based caching for faster image retrieval
- ✅ NGINX caching for CDN-like simulation
- ✅ Gunicorn with Uvicorn workers for production-grade FastAPI deployment
- ✅ Fully Dockerized (using Docker Compose)

---

## 🛠️ Tech Stack

- **FastAPI** (Python backend)
- **MongoDB** (Metadata storage)
- **MinIO** (Frame storage)
- **Redis** (Application cache)
- **NGINX** (Reverse proxy & cache layer)
- **Docker Compose**

---

## 📂 Directory Structure

```
challenge_2
│   .env
│   api_server.py
│   config.py
│   database.py
│   docker-compose.yaml
│   Dockerfile
│   image_processor.py
│   process_and_persist_image_notebook.ipynb
│   README.md
│   requirements.txt
│                 
├───tests
│      test_api.py <-- ✅ Unit Tests
│
├───infra
│   ├───local
│   │       init_minio.sh
│   │       init_mongo.sh
│   │       init_redis.sh
│   │
│   └───nginx
│           nginx.conf

```

---

## ⚙️ Environment Variables

| Variable          | Description             | Example                |
|-------------------|-------------------------|------------------------|
| MONGO_USER        | MongoDB username        | `admin`                |
| MONGO_PASS        | MongoDB password        | `admin123`             |
| MONGO_HOST        | MongoDB host + port     | `mongo:27017`          |
| MINIO_ENDPOINT    | MinIO host + port       | `minio:9000`           |
| MINIO_ACCESS_KEY  | MinIO access key        | `minioadmin`           |
| MINIO_SECRET_KEY  | MinIO secret key        | `minioadmin`           |
| MINIO_BUCKET      | MinIO bucket name       | `challenge-2`          |
| REDIS_HOST        | Redis host              | `redis`                |
| REDIS_PORT        | Redis port              | `6379`                 |
| REDIS_PASSWORD    | Redis password          | `your_redis_password`  |

---

## 🐳 Running Locally with Docker Compose

```bash
# Build FastAPI Docker image
docker-compose build

# Start all services (FastAPI, MongoDB, MinIO, Redis, NGINX)
docker-compose up
```

Access services:

- **FastAPI API** → http://localhost:8001
- **NGINX Cached API (CDN Simulation)** → http://localhost:8080
- **MinIO Console** → http://localhost:9001
- **MongoDB** → localhost:27017
- **Redis** → localhost:6379

---

## 📡 API Usage Example

### Get Stitched Image for Depth Range

**Endpoint:**  
`GET /frames/?depth_min=5000&depth_max=10000`

**Returns:**  
A PNG image stitched from frames within the given depth range.

---

## 🧱 Building the FastAPI Image Separately (Optional)

```bash
cd api_server
docker build -t fastapi_app .
```

---

## 📝 NGINX Cache (Local CDN Layer)

NGINX is configured to cache responses from FastAPI `/frames/` route. This helps you simulate how a CDN edge cache would behave.

- Config file: `infra/nginx/nginx.conf`
- Persistent cache path: Docker volume: `nginx_cache`

---

## ✅ Redis Caching Logic (In API)

The FastAPI server first checks Redis for a pre-cached stitched image for the requested depth range before querying MongoDB/MinIO.

---

## 🧹 Cleaning Docker Resources

```bash
docker-compose down -v
```

(To also delete attached volumes and caches)

---

## ✅ Future Improvements (Ideas)

- Add Prometheus & Grafana for monitoring
- Implement JWT-based authentication
- Use AWS S3 instead of MinIO for production
- Deploy on Kubernetes (optional)

---

## 📌 Author

Built for Challenge-2: Depth-based Image Serving API  
Maintained by: **[Syed Sohail Ali Alvi]**
