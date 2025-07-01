import io
import pickle
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from database import MongoDBClient, MinioClient, RedisClient
from config import Config
from logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

app = FastAPI()

mongo_client = MongoDBClient()
minio_client = MinioClient()
redis_client = RedisClient()

CHUNK_SIZE = 10  # Depth chunk size for caching (e.g., 10 depths per chunk)

def generate_depth_chunks(start, end, chunk_size):
    current = start
    while current < end:
        next_limit = min(current + chunk_size, end)
        yield (current, next_limit)
        current = next_limit

def fetch_and_cache_chunk(chunk_min, chunk_max):
    cache_key = f"frames:{chunk_min}:{chunk_max}"
    cached_chunk = redis_client.client.get(cache_key)
    if cached_chunk:
        logger.info(f"Cache hit for chunk {chunk_min}-{chunk_max}")
        return pickle.loads(cached_chunk)

    logger.info(f"Cache miss for chunk {chunk_min}-{chunk_max}. Fetching from MinIO/Mongo.")

    query = {"depth": {"$gte": chunk_min, "$lt": chunk_max}}
    frames_cursor = mongo_client.collection.find(query).sort("depth", 1)
    frames = list(frames_cursor)

    if not frames:
        logger.warning(f"No frames found in chunk range {chunk_min}-{chunk_max}")
        return None  # We'll skip empty chunks when stitching

    image_rows = []
    for frame in frames:
        try:
            obj = minio_client.client.get_object(Config.MINIO_BUCKET, frame["minio_object"])
            image_bytes = io.BytesIO(obj.read())
            obj.close()
            img_row = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            image_rows.append(img_row)
        except Exception as e:
            logger.error(f"Error fetching object {frame['minio_object']}: {e}")

    if image_rows:
        stitched_chunk = cv2.vconcat(image_rows)
        _, img_encoded = cv2.imencode('.png', stitched_chunk)
        chunk_bytes = img_encoded.tobytes()
        redis_client.client.setex(cache_key, 300, pickle.dumps(chunk_bytes))  # TTL: 5 mins
        logger.info(f"Cached chunk {chunk_min}-{chunk_max}")
        return chunk_bytes

    return None

@app.get("/frames/")
def get_frames(depth_min: float, depth_max: float):
    logger.info(f"Received /frames/ request for range: {depth_min}-{depth_max}")

    try:
        all_chunks = []
        for chunk_min, chunk_max in generate_depth_chunks(depth_min, depth_max, CHUNK_SIZE):
            chunk_bytes = fetch_and_cache_chunk(chunk_min, chunk_max)
            if chunk_bytes:
                chunk_img = cv2.imdecode(np.frombuffer(chunk_bytes, np.uint8), cv2.IMREAD_COLOR)
                all_chunks.append(chunk_img)

        if not all_chunks:
            logger.warning(f"No frames found in total range {depth_min}-{depth_max}")
            raise HTTPException(status_code=404, detail="No frames found for given depth range")

        # Final vertical stitching of all cached+fetched chunks
        final_image = cv2.vconcat(all_chunks)
        _, img_encoded = cv2.imencode('.png', final_image)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

    except Exception as e:
        logger.exception(f"Error processing /frames/ request for {depth_min}-{depth_max}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)