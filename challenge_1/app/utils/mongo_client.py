from pymongo import MongoClient, errors
from datetime import datetime
import os
import json
import redis
from utils.logger import setup_logger

logger = setup_logger(__name__)

MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASS = os.getenv("MONGO_PASS", "admin123")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost:27017")
MONGO_URI = os.getenv("MONGO_URI", f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}")

client = MongoClient(MONGO_URI)
db = client["object_detection"]
image_collection = db["image_metadata"]

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "your_redis_password")
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

def save_uploaded_image_metadata(image_name, minio_key, uploaded_by=None):
    try:
        metadata = {
            "image_name": image_name,
            "minio_key": minio_key,
            "uploaded_by": uploaded_by,
            "upload_time": datetime.utcnow(),
            "detection_done": False,
            "detection_in_progress": False,
            "object_count": 0,
            "masked_image_key": None,
            "objects": []
        }
        image_collection.insert_one(metadata)
        _cache_metadata(image_name, metadata)
        logger.info(f"Uploaded image metadata saved for: {image_name}")
    except errors.PyMongoError as e:
        logger.error(f"Failed to save uploaded image metadata: {e}")

def save_detection_result(image_name, object_count, objects, masked_image_key=None):
    try:
        update_fields = {
            "detection_done": True,
            "detection_in_progress": False,
            "object_count": object_count,
            "objects": objects,
            "masked_image_key": masked_image_key
        }
        result = image_collection.update_one({"image_name": image_name}, {"$set": update_fields})

        if result.matched_count == 0:
            logger.warning(f"No document found to update for image: {image_name}")
        else:
            logger.info(f"Detection results saved for image: {image_name}")

        metadata = get_image_metadata(image_name, skip_cache=True)
        if metadata:
            _cache_metadata(image_name, metadata)
    except errors.PyMongoError as e:
        logger.error(f"Failed to save detection result: {e}")

def get_image_metadata(image_name, skip_cache=False):
    cache_key = f"image_metadata:{image_name}"

    if not skip_cache:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Redis cache hit for: {image_name}")
            return json.loads(cached)

    try:
        metadata = image_collection.find_one({"image_name": image_name}, {"_id": 0})
        if metadata:
            _cache_metadata(image_name, metadata)
        return metadata
    except errors.PyMongoError as e:
        logger.error(f"Failed to fetch metadata for image: {image_name} -> {e}")
        return None

def set_detection_in_progress(image_name, in_progress: bool):
    try:
        result = image_collection.update_one(
            {"image_name": image_name},
            {"$set": {"detection_in_progress": in_progress}}
        )
        if result.matched_count == 0:
            logger.warning(f"No document found for image: {image_name} when setting in_progress={in_progress}")
        else:
            logger.info(f"Set detection_in_progress={in_progress} for image: {image_name}")

        metadata = get_image_metadata(image_name, skip_cache=True)
        if metadata:
            _cache_metadata(image_name, metadata)
    except errors.PyMongoError as e:
        logger.error(f"Failed to set detection_in_progress for {image_name}: {e}")

def clear_detection_in_progress(image_name):
    try:
        image_collection.update_one(
            {"image_name": image_name},
            {"$set": {"detection_in_progress": False, "detection_done": True}}
        )
        logger.info(f"Cleared detection_in_progress and set detection_done=True for image: {image_name}")

        metadata = get_image_metadata(image_name, skip_cache=True)
        if metadata:
            _cache_metadata(image_name, metadata)
    except errors.PyMongoError as e:
        logger.error(f"Failed to clear detection_in_progress for {image_name}: {e}")

def _cache_metadata(image_name, metadata):
    try:
        redis_client.setex(f"image_metadata:{image_name}", REDIS_TTL_SECONDS, json.dumps(metadata, default=str))
        logger.info(f"Cached metadata for: {image_name}")
    except Exception as e:
        logger.error(f"Failed to cache metadata for {image_name}: {e}")
