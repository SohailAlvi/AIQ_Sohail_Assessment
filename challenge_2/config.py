import os

class Config:
    # MongoDB Configuration
    MONGO_USER = os.getenv('MONGO_USER', 'admin')
    MONGO_PASS = os.getenv('MONGO_PASS', 'admin123')
    MONGO_HOST = os.getenv('MONGO_HOST', 'localhost:27017')

    # MinIO Configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'challenge-2')

    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'your_redis_password')
