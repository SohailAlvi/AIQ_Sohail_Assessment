from pymongo import MongoClient
from minio import Minio, error
import redis
from config import Config
from logging import setup_logger

# Setup logger for this module
logger = setup_logger(__name__)

class MongoDBClient:
    def __init__(self):
        try:
            uri = f"mongodb://{Config.MONGO_USER}:{Config.MONGO_PASS}@{Config.MONGO_HOST}/"
            self.client = MongoClient(uri)
            self.db = self.client['challenge_db']
            self.collection = self.db['frames']
            logger.info(f"Connected to MongoDB at {Config.MONGO_HOST}")
        except Exception as e:
            logger.exception(f"Error connecting to MongoDB: {e}")
            raise


class MinioClient:
    def __init__(self):
        try:
            self.client = Minio(
                Config.MINIO_ENDPOINT,
                access_key=Config.MINIO_ACCESS_KEY,
                secret_key=Config.MINIO_SECRET_KEY,
                secure=False
            )
            logger.info(f"Initialized MinIO client for endpoint {Config.MINIO_ENDPOINT}")
            self._ensure_bucket_exists()
        except Exception as e:
            logger.exception(f"Error initializing MinIO client: {e}")
            raise

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(Config.MINIO_BUCKET):
                self.client.make_bucket(Config.MINIO_BUCKET)
                logger.info(f"Bucket '{Config.MINIO_BUCKET}' created in MinIO.")
            else:
                logger.info(f"Bucket '{Config.MINIO_BUCKET}' already exists in MinIO.")
        except error.S3Error as e:
            logger.error(f"MinIO bucket error: {e}")
            raise


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
            logger.info(f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except redis.RedisError as e:
            logger.error(f"Redis connection error: {e}")
            raise
