from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from database import MongoDBClient, MinioClient
from config import Config
import io
import cv2
import numpy as np


app = FastAPI()

mongo_client = MongoDBClient()
minio_client = MinioClient()

@app.get("/frames/")
def get_frames(depth_min: float, depth_max: float):
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

    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)