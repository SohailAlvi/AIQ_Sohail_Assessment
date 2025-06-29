from pymongo import MongoClient
from minio import Minio, error
import redis
from config import Config


class MongoDBClient:
    def __init__(self):
        uri = f"mongodb://{Config.MONGO_USER}:{Config.MONGO_PASS}@{Config.MONGO_HOST}/"
        self.client = MongoClient(uri)
        self.db = self.client['challenge_db']
        self.collection = self.db['frames']


class MinioClient:
    def __init__(self):
        self.client = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            secure=False
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(Config.MINIO_BUCKET):
                self.client.make_bucket(Config.MINIO_BUCKET)
                print(f"Bucket '{Config.MINIO_BUCKET}' created in MinIO.")
            else:
                print(f"Bucket '{Config.MINIO_BUCKET}' already exists.")
        except error.S3Error as e:
            print(f"MinIO bucket error: {e}")


class RedisClient:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD,
                decode_responses=False
            )
            # Test connection
            self.client.ping()
            print(f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except redis.RedisError as e:
            print(f"Redis connection error: {e}")
            raise
