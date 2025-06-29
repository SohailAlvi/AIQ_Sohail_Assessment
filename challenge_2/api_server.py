from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from database import MongoDBClient, MinioClient, RedisClient
from config import Config
import io
import cv2
import numpy as np
import pickle

app = FastAPI()

mongo_client = MongoDBClient()
minio_client = MinioClient()
redis_client = RedisClient()

@app.get("/frames/")
def get_frames(depth_min: float, depth_max: float):
    cache_key = f"frames:{depth_min}:{depth_max}"

    # Check Redis cache
    cached_image = redis_client.client.get(cache_key)
    if cached_image:
        return StreamingResponse(io.BytesIO(pickle.loads(cached_image)), media_type="image/png")

    query = {
        "depth": {"$gte": depth_min, "$lte": depth_max}
    }
    frames_cursor = mongo_client.collection.find(query).sort("depth", 1)

    if mongo_client.collection.count_documents(query) == 0:
        raise HTTPException(status_code=404, detail="No frames found for given depth range")

    image_rows = []
    for frame in frames_cursor:
        obj = minio_client.client.get_object(Config.MINIO_BUCKET, frame["minio_object"])
        image_bytes = io.BytesIO(obj.read())
        obj.close()
        img_row = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        image_rows.append(img_row)

    full_image = cv2.vconcat(image_rows)
    _, img_encoded = cv2.imencode('.png', full_image)
    final_bytes = img_encoded.tobytes()

    # Cache the final image in Redis for future requests
    redis_client.client.setex(cache_key, 300, pickle.dumps(final_bytes))  # TTL: 5 minutes

    return StreamingResponse(io.BytesIO(final_bytes), media_type="image/png")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8001)
