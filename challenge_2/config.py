import os

class Config:
    MONGO_USER = os.getenv('MONGO_USER', 'admin')
    MONGO_PASS = os.getenv('MONGO_PASS', 'admin123')
    MONGO_HOST = os.getenv('MONGO_HOST', 'localhost:27017')

    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'challenge-2')
