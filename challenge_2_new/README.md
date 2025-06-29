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
├───infra
│   ├───local
│   │       init_minio.sh
│   │       init_mongo.sh
│   │       init_redis.sh
│   │
│   └───nginx
│           nginx.conf

