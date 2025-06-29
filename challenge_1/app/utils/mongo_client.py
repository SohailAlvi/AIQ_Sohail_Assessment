from pymongo import MongoClient, errors
from datetime import datetime
import os

MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASS = os.getenv("MONGO_PASS", "admin123")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost:27017")
MONGO_URI = os.getenv("MONGO_URI", f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}")

client = MongoClient(MONGO_URI)
db = client["object_detection"]
image_collection = db["image_metadata"]

def save_uploaded_image_metadata(image_name, minio_key, uploaded_by=None):
    try:
        image_collection.insert_one({
            "image_name": image_name,
            "minio_key": minio_key,
            "uploaded_by": uploaded_by,
            "upload_time": datetime.utcnow(),
            "detection_done": False,
            "detection_in_progress": False,   # New field
            "object_count": 0,
            "masked_image_key": None,
            "objects": []
        })
        print(f"[MongoDB] Uploaded image metadata saved for: {image_name}")
    except errors.PyMongoError as e:
        print(f"[MongoDB Error] Failed to save uploaded image metadata: {e}")

def save_detection_result(image_name, object_count, objects, masked_image_key=None):
    try:
        result = image_collection.update_one(
            {"image_name": image_name},
            {
                "$set": {
                    "detection_done": True,
                    "detection_in_progress": False,  # Clear in-progress flag
                    "object_count": object_count,
                    "objects": objects,
                    "masked_image_key": masked_image_key
                }
            }
        )
        if result.matched_count == 0:
            print(f"[MongoDB Warning] No document found to update for image: {image_name}")
        else:
            print(f"[MongoDB] Detection results saved for image: {image_name}")
    except errors.PyMongoError as e:
        print(f"[MongoDB Error] Failed to save detection result: {e}")

def get_image_metadata(image_name):
    try:
        return image_collection.find_one({"image_name": image_name})
    except errors.PyMongoError as e:
        print(f"[MongoDB Error] Failed to fetch metadata for image: {image_name} -> {e}")
        return None

def set_detection_in_progress(image_name, in_progress: bool):
    """ Atomically set detection_in_progress flag """
    try:
        result = image_collection.update_one(
            {"image_name": image_name},
            {"$set": {"detection_in_progress": in_progress}}
        )
        if result.matched_count == 0:
            print(f"[MongoDB Warning] No document found for image: {image_name} when setting in_progress={in_progress}")
        else:
            print(f"[MongoDB] Set detection_in_progress={in_progress} for image: {image_name}")
    except errors.PyMongoError as e:
        print(f"[MongoDB Error] Failed to set detection_in_progress for {image_name}: {e}")

def clear_detection_in_progress(image_name):
    """ Clear detection_in_progress flag and mark detection as done (even on failure cases if needed) """
    try:
        image_collection.update_one(
            {"image_name": image_name},
            {"$set": {"detection_in_progress": False, "detection_done": True}}
        )
        print(f"[MongoDB] Cleared detection_in_progress and set detection_done=True for image: {image_name}")
    except errors.PyMongoError as e:
        print(f"[MongoDB Error] Failed to clear detection_in_progress for {image_name}: {e}")
