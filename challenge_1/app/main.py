from fastapi import FastAPI, UploadFile, File, HTTPException, Path, BackgroundTasks
from models import InferenceResponse
from utils.ml.detection import ModelConfig, ObjectDetectionModel
from utils.minio_client import MinioClientWrapper
from utils.mongo_client import save_uploaded_image_metadata, save_detection_result, get_image_metadata
from uuid import uuid4
from typing import Optional
from fastapi.responses import StreamingResponse
import io
import numpy as np
import cv2

app = FastAPI(title="Object Detection API")

model_config = ModelConfig()
detection_model = ObjectDetectionModel(model_config)
minio_client = MinioClientWrapper()


# ✅ Centralized Detection Logic - Avoids Duplication
def run_object_detection_and_save(image_name: str, threshold: float = 0.5):
    # --- Check and Lock the detection ---
    metadata = get_image_metadata(image_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in MongoDB.")

    if metadata.get("detection_done"):
        return  # Already done, skip

    if metadata.get("detection_in_progress"):
        return  # Detection is already in progress, skip starting again

    # --- Atomically set detection_in_progress = True ---
    from utils.mongo_client import set_detection_in_progress
    set_detection_in_progress(image_name, True)

    try:
        # --- Run detection as before ---
        image_bytes = minio_client.download_file(metadata["minio_key"])
        if image_bytes is None:
            raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in MinIO.")

        result = detection_model.evaluate_image(image_bytes, confidence_threshold=threshold)

        masked_image_key = None
        if result.get("visualization_image"):
            masked_image_key = f"{image_name}_masked.png"
            minio_client.upload_file(result["visualization_image"], masked_image_key, content_type="image/png")

        save_detection_result(
            image_name=image_name,
            object_count=result["object_count"],
            objects=result["objects"],
            masked_image_key=masked_image_key
        )

    finally:
        # --- Always clear in-progress flag, even if detection failed ---
        from utils.mongo_client import clear_detection_in_progress
        clear_detection_in_progress(image_name)

@app.post("/upload-image")
async def upload_image_to_minio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG or PNG allowed")

    file_bytes = await file.read()
    file_ext = file.filename.split('.')[-1]
    file_name = f"{str(uuid4())}.{file_ext}"

    upload_url = minio_client.upload_file(file_bytes, file_name, file.content_type)
    if not upload_url:
        raise HTTPException(status_code=500, detail="Failed to upload to MinIO")

    save_uploaded_image_metadata(image_name=file_name, minio_key=file_name)

    # ✅ Start detection in background using centralized logic
    def detection_background_task():
        try:
            run_object_detection_and_save(file_name, threshold=0.5)
        except Exception as e:
            print(f"Background detection failed for {file_name}: {e}")

    background_tasks.add_task(detection_background_task)

    return {"message": "File uploaded successfully", "url": upload_url, "file_name": file_name}

@app.get("/objects/{image_name}/{object_id}/overlay-mask")
def get_object_mask_overlay(image_name: str, object_id: int):
    metadata = get_image_metadata(image_name)
    if not metadata or "objects" not in metadata:
        raise HTTPException(status_code=404, detail=f"No detection result for image '{image_name}'.")

    target_object = next((obj for obj in metadata["objects"] if obj["id"] == object_id), None)
    if not target_object:
        raise HTTPException(status_code=404, detail=f"Object ID {object_id} not found for image '{image_name}'.")

    image_bytes = minio_client.download_file(metadata["minio_key"])
    if image_bytes is None:
        raise HTTPException(status_code=500, detail=f"Original image not found for '{image_name}'.")

    img_array = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    height, width, _ = original_img.shape

    x_min, y_min, x_max, y_max = target_object["bbox"]
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    radius_x = (x_max - x_min) / 2
    radius_y = (y_max - y_min) / 2
    radius = int(min(radius_x, radius_y))

    if radius <= 0:
        raise HTTPException(status_code=400, detail="Invalid bounding box size for circular mask.")

    circular_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circular_mask, (center_x, center_y), radius, 255, thickness=-1)

    overlay = np.zeros_like(original_img, dtype=np.uint8)
    overlay[circular_mask == 255] = (0, 255, 0)

    alpha = 0.5
    overlay_img = cv2.addWeighted(original_img, 1 - alpha, overlay, alpha, 0)

    success, buffer = cv2.imencode('.png', overlay_img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode overlay image.")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

@app.get("/objects/{image_name}")
async def get_or_predict_objects(image_name: str, threshold: float = 0.5):
    metadata = get_image_metadata(image_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")

    if metadata.get("detection_done", False):
        response = {
            "object_count": metadata["object_count"],
            "objects": metadata["objects"],
            "visualization_url": None
        }
        if metadata.get("masked_image_key"):
            response["visualization_url"] = minio_client.get_presigned_url(metadata["masked_image_key"])
        return response

    # ✅ Run detection now if not done yet
    return run_object_detection_and_save(image_name, threshold)

@app.get("/objects/{image_name}/{object_id}")
def get_object_details(image_name: str, object_id: int = Path(..., description="ID of the object")):
    metadata = get_image_metadata(image_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")

    target_object = next((obj for obj in metadata["objects"] if obj["id"] == object_id), None)
    if not target_object:
        raise HTTPException(status_code=404, detail=f"Object ID {object_id} not found for image '{image_name}'.")

    visualization_url = None
    if metadata.get("masked_image_key"):
        visualization_url = minio_client.get_presigned_url(metadata["masked_image_key"])

    return {
        "object_id": target_object["id"],
        "bounding_box": target_object["bbox"],
        "centroid": target_object["centroid"],
        "radius": target_object["radius"],
        "visualization_url": visualization_url
    }

@app.get("/objects/{image_name}/{object_id}/crop")
def get_object_crop(image_name: str, object_id: int = Path(..., description="ID of the circular object within the image")):
    metadata = get_image_metadata(image_name)
    if not metadata or "objects" not in metadata:
        raise HTTPException(status_code=404, detail=f"No detection result found for image '{image_name}'.")

    target_object = next((obj for obj in metadata["objects"] if obj["id"] == object_id), None)
    if not target_object:
        raise HTTPException(status_code=404, detail=f"Object ID {object_id} not found for image '{image_name}'.")

    visualization_key = metadata.get("masked_image_key")
    if not visualization_key:
        raise HTTPException(status_code=404, detail="Visualization image not found in MinIO.")

    visualization_bytes = minio_client.download_file(visualization_key)
    if visualization_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to load visualization image from MinIO.")

    img_array = np.frombuffer(visualization_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    x_min, y_min, x_max, y_max = target_object["bbox"]
    cropped_img = img[y_min:y_max, x_min:x_max]

    success, buffer = cv2.imencode('.png', cropped_img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode cropped image.")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)
