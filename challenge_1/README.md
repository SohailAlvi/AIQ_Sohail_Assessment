
# 🪙 Coin Detection & Segmentation API 🚀

This project focuses on **fine-tuning a Mask R-CNN model** for coin detection & segmentation using a COCO-style dataset, and **deploying it via a FastAPI backend with Docker Compose orchestration**.

---

## 📂 Project Directory Structure

```
challenge_1/
├── .env
├── fine_tune_notebook.ipynb                 # Jupyter notebook for fine-tuning Mask R-CNN
├── app/                                     # FastAPI application for inference
│   ├── docker-compose.yaml                  # Docker Compose file for multi-container setup
│   ├── Dockerfile                           # API container Dockerfile
│   ├── main.py                              # FastAPI entrypoint
│   ├── models.py                            # Pydantic models for API input/output
│   ├── requirements.txt                     # Python dependencies
│   ├── model/                               # Trained model artifacts (.pth files)
│   ├── utils/                               # Helper modules
│   │   ├── ml/detection.py                  # Core detection logic (inference)
│   │   ├── mongo_client.py                  # MongoDB connector
│       └── minio_client.py                  # MinIO (object storage) connector
├── infra/
│   ├── local/                               # Local infra bootstrap scripts
│   │   ├── init_minio.sh
│   │   ├── init_mongo.sh
│   │   └── init_redis.sh
│   └── torchserve/                          # TorchServe deployment scripts and MAR files
│       ├── create_mar.sh
│       ├── Dockerfile
│       ├── init_torchserve.sh
│       ├── maskrcnn_handler.py
│       └── model-store/
│           ├── dummy.mar
│           └── maskrcnn_v2.mar
```

---

## 🏋️‍♂️ Model Training (Fine-Tuning)

- Located in: `fine_tune_notebook.ipynb`
- Uses `maskrcnn_resnet50_fpn_v2` from torchvision
- Trained on COCO-formatted coin dataset with 70:30 data split
- Outputs `.pth` models saved in `/app/model/`

---

## 🧪 API Inference (FastAPI App)

- Located in: `/app/main.py`
- Loads fine-tuned `.pth` models
- Performs object detection and segmentation
- Persists detection metadata to **MongoDB**
- Stores detection result images in **MinIO Object Storage**
- Optional caching layer using **Redis** (TTL = 1 hour)

---

## 🐳 Docker Compose Setup

This project uses **MongoDB**, **MinIO**, **Redis**, and the **FastAPI app**. Launch all containers:

```bash
cd app
docker-compose up --build
```

Services:
- MongoDB → `localhost:27017`
- MinIO → `localhost:9000` (console: `localhost:9001`)
- Redis → `localhost:6379`
- FastAPI → `localhost:8000`

---

## 🚫 Ignore Files (Optional .gitignore Additions)

```
__pycache__/
.ipynb_checkpoints/
*.pth
.env
```

---

## ✅ TODO / Next Steps

- [ ] Improve model evaluation metrics (e.g., mAP, IoU) on validation split
- [ ] Add health check endpoints for services
- [ ] Integrate TorchServe API deployment

---
