import io
import pickle
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from database import MongoDBClient, MinioClient, RedisClient
from config import Config
from logging import setup_logger

# Setup logger for this module
logger = setup_logger(__name__)

app = FastAPI()

mongo_client = MongoDBClient()
minio_client = MinioClient()
redis_client = RedisClient()

@app.get("/frames/")
def get_frames(depth_min: float, depth_max: float):
    logger.info(f"Received request for frames with depth_min={depth_min}, depth_max={depth_max}")

    cache_key = f"frames:{depth_min}:{depth_max}"

    try:
        # Check Redis cache
        cached_image = redis_client.client.get(cache_key)
        if cached_image:
            logger.info(f"Cache hit for key: {cache_key}")
            return StreamingResponse(io.BytesIO(pickle.loads(cached_image)), media_type="image/png")
        logger.info(f"Cache miss for key: {cache_key}")

        query = {
            "depth": {"$gte": depth_min, "$lte": depth_max}
        }
        frames_cursor = mongo_client.collection.find(query).sort("depth", 1)
        frame_count = mongo_client.collection.count_documents(query)
        logger.info(f"Found {frame_count} frames matching query")

        if frame_count == 0:
            logger.warning(f"No frames found for depth range: {depth_min} - {depth_max}")
            raise HTTPException(status_code=404, detail="No frames found for given depth range")

        image_rows = []
        for frame in frames_cursor:
            logger.debug(f"Fetching image from MinIO for object: {frame['minio_object']}")
            obj = minio_client.client.get_object(Config.MINIO_BUCKET, frame["minio_object"])
            image_bytes = io.BytesIO(obj.read())
            obj.close()
            img_row = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            image_rows.append(img_row)

        full_image = cv2.vconcat(image_rows)
        _, img_encoded = cv2.imencode('.png', full_image)
        final_bytes = img_encoded.tobytes()

        # Cache the final image in Redis
        redis_client.client.setex(cache_key, 300, pickle.dumps(final_bytes))  # TTL: 5 minutes
        logger.info(f"Cached image for key: {cache_key} with TTL=300s")

        return StreamingResponse(io.BytesIO(final_bytes), media_type="image/png")

    except Exception as e:
        logger.exception(f"Error while processing frames for depth range {depth_min}-{depth_max}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
