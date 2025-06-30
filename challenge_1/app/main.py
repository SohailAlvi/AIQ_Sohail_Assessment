from fastapi import FastAPI, UploadFile, File, HTTPException, Path, BackgroundTasks
from models import InferenceResponse
from utils.ml.detection import ModelConfig, ObjectDetectionModel
from utils.minio_client import MinioClientWrapper
from utils.mongo_client import save_uploaded_image_metadata, save_detection_result, get_image_metadata
from utils.logger import setup_logger
from uuid import uuid4
from typing import Optional
from fastapi.responses import StreamingResponse
import io
import numpy as np
import cv2

app = FastAPI(title="Object Detection API")

logger = setup_logger(__name__)

model_config = ModelConfig()
detection_model = ObjectDetectionModel(model_config)
minio_client = MinioClientWrapper()


# ✅ Centralized Detection Logic - Avoids Duplication
def run_object_detection_and_save(image_name: str, threshold: float = 0.5):
    logger.info(f"Starting object detection for image: {image_name} with threshold: {threshold}")

    # --- Check and Lock the detection ---
    metadata = get_image_metadata(image_name)
    if not metadata:
        logger.error(f"Image '{image_name}' not found in MongoDB.")
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in MongoDB.")

    if metadata.get("detection_done"):
        logger.info(f"Detection already done for image: {image_name}, skipping.")
        return  # Already done, skip

    if metadata.get("detection_in_progress"):
        logger.info(f"Detection already in progress for image: {image_name}, skipping.")
        return  # Detection is already in progress, skip starting again

    # --- Atomically set detection_in_progress = True ---
    from utils.mongo_client import set_detection_in_progress
    set_detection_in_progress(image_name, True)
    logger.info(f"Set detection_in_progress flag for image: {image_name}")

    try:
        # --- Run detection as before ---
        image_bytes = minio_client.download_file(metadata["minio_key"])
        if image_bytes is None:
            logger.error(f"Image '{image_name}' not found in MinIO.")
            raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in MinIO.")

        result = detection_model.evaluate_image(image_bytes, confidence_threshold=threshold)
        logger.info(f"Detection completed for image: {image_name}. Objects found: {result.get('object_count', 0)}")

        masked_image_key = None
        if result.get("visualization_image"):
            masked_image_key = f"{image_name}_masked.png"
            minio_client.upload_file(result["visualization_image"], masked_image_key, content_type="image/png")
            logger.info(f"Uploaded visualization image to MinIO as {masked_image_key}")

        save_detection_result(
            image_name=image_name,
            object_count=result["object_count"],
            objects=result["objects"],
            masked_image_key=masked_image_key
        )
        logger.info(f"Saved detection result to MongoDB for image: {image_name}")

    except Exception as e:
        logger.error(f"Error during detection for image {image_name}: {e}", exc_info=True)
        raise e

    finally:
        # --- Always clear in-progress flag, even if detection failed ---
        from utils.mongo_client import clear_detection_in_progress
        clear_detection_in_progress(image_name)
        logger.info(f"Cleared detection_in_progress flag for image: {image_name}")


@app.post("/upload-image")
async def upload_image_to_minio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    logger.info(f"Received upload request for file: {file.filename}")

    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Unsupported file type: {file.content_type} for file: {file.filename}")
        raise HTTPException(status_code=400, detail="Only JPG or PNG allowed")

    file_bytes = await file.read()
    file_ext = file.filename.split('.')[-1]
    file_name = f"{str(uuid4())}.{file_ext}"

    upload_url = minio_client.upload_file(file_bytes, file_name, file.content_type)
    if not upload_url:
        logger.error(f"Failed to upload file {file_name} to MinIO")
        raise HTTPException(status_code=500, detail="Failed to upload to MinIO")

    save_uploaded_image_metadata(image_name=file_name, minio_key=file_name)
    logger.info(f"File {file_name} uploaded successfully to MinIO")

    # ✅ Start detection in background using centralized logic
    def detection_background_task():
        try:
            logger.info(f"Starting background detection for file: {file_name}")
            run_object_detection_and_save(file_name, threshold=0.5)
            logger.info(f"Background detection completed for file: {file_name}")
        except Exception as e:
            logger.error(f"Background detection failed for {file_name}: {e}", exc_info=True)

    background_tasks.add_task(detection_background_task)

    return {"message": "File uploaded successfully", "url": upload_url, "file_name": file_name}


@app.get("/objects/{image_name}/{object_id}/overlay-mask")
def get_object_mask_overlay(image_name: str, object_id: int):
    logger.info(f"Request for mask overlay of object {object_id} in image {image_name}")

    metadata = get_image_metadata(image_name)
    if not metadata or "objects" not in metadata:
        logger.error(f"No detection result for image '{image_name}'.")
        raise HTTPException(status_code=404, detail=f"No detection result for image '{image_name}'.")

    target_object = next((obj for obj in metadata["objects"] if obj["id"] == object_id), None)
    if not target_object:
        logger.error(f"Object ID {object_id} not found for image '{image_name}'.")
        raise HTTPException(status_code=404, detail=f"Object ID {object_id} not found for image '{image_name}'.")

    image_bytes = minio_client.download_file(metadata["minio_key"])
    if image_bytes is None:
        logger.error(f"Original image not found for '{image_name}'.")
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
        logger.warning(f"Invalid bounding box size for mask on object {object_id} in image {image_name}.")
        raise HTTPException(status_code=400, detail="Invalid bounding box size for mask.")

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(binary_mask, (center_x, center_y), radius, 255, thickness=-1)

    masked_object_img = cv2.bitwise_and(original_img, original_img, mask=binary_mask)
    background_with_patch = original_img.copy()
    background_with_patch[binary_mask == 255] = (255, 255, 255)
    combined_image = np.hstack((masked_object_img, background_with_patch))

    success, buffer = cv2.imencode('.png', combined_image)
    if not success:
        logger.error(f"Failed to encode combined image for object {object_id} in image {image_name}.")
        raise HTTPException(status_code=500, detail="Failed to encode combined image.")

    logger.info(f"Successfully generated mask overlay for object {object_id} in image {image_name}.")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@app.get("/objects/{image_name}")
async def get_or_predict_objects(image_name: str, threshold: float = 0.5):
    logger.info(f"Get or predict objects for image: {image_name} with threshold: {threshold}")

    metadata = get_image_metadata(image_name)
    if not metadata:
        logger.error(f"Image '{image_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")

    if metadata.get("detection_done", False):
        response = {
            "object_count": metadata["object_count"],
            "objects": metadata["objects"],
            "visualization_url": None
        }
        if metadata.get("masked_image_key"):
            response["visualization_url"] = minio_client.get_presigned_url(metadata["masked_image_key"])
        logger.info(f"Returning existing detection result for image: {image_name}")
        return response

    logger.info(f"Detection not done yet for image: {image_name}, running detection now.")
    run_object_detection_and_save(image_name, threshold)

    # After detection is done, fetch metadata again to respond
    metadata = get_image_metadata(image_name)
    response = {
        "object_count": metadata.get("object_count", 0),
        "objects": metadata.get("objects", []),
        "visualization_url": None
    }
    if metadata.get("masked_image_key"):
        response["visualization_url"] = minio_client.get_presigned_url(metadata["masked_image_key"])

    return response


@app.get("/objects/{image_name}/{object_id}")
def get_object_details(image_name: str, object_id: int = Path(..., description="ID of the object")):
    logger.info(f"Requesting details for object {object_id} in image {image_name}")

    metadata = get_image_metadata(image_name)
    if not metadata:
        logger.error(f"Image '{image_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")

    target_object = next((obj for obj in metadata["objects"] if obj["id"] == object_id), None)
    if not target_object:
        logger.error(f"Object ID {object_id} not found for image '{image_name}'.")
        raise HTTPException(status_code=404, detail=f"Object ID {object_id} not found for image '{image_name}'.")

    visualization_url = None
    if metadata.get("masked_image_key"):
        visualization_url = minio_client.get_presigned_url(metadata["masked_image_key"])

    logger.info(f"Returning details for object {object_id} in image {image_name}")
    return {
        "object_id": target_object["id"],
        "bounding_box": target_object["bbox"],
        "centroid": target_object["centroid_from_box"],
        "radius": target_object["radius_from_box"],
        "visualization_url": visualization_url
    }


@app.get("/objects/{image_name}/{object_id}/crop")
def get_object_crop(image_name: str, object_id: int = Path(..., description="ID of the circular object within the image")):
    logger.info(f"Requesting cropped image for object {object_id} in image {image_name}")

    metadata = get_image_metadata(image_name)
    if not metadata or "objects" not in metadata:
        logger.error(f"No detection result found for image '{image_name}'.")
        raise HTTPException(status_code=404, detail=f"No detection result found for image '{image_name}'.")

    target_object = next((obj for obj in metadata["objects"] if obj["id"] == object_id), None)
    if not target_object:
        logger.error(f"Object ID {object_id} not found for image '{image_name}'.")
        raise HTTPException(status_code=404, detail=f"Object ID {object_id} not found for image '{image_name}'.")

    visualization_key = metadata.get("masked_image_key")
    if not visualization_key:
        logger.error(f"Visualization image not found in MinIO for image '{image_name}'.")
        raise HTTPException(status_code=404, detail="Visualization image not found in MinIO.")

    visualization_bytes = minio_client.download_file(visualization_key)
    if visualization_bytes is None:
        logger.error(f"Failed to load visualization image from MinIO for image '{image_name}'.")
        raise HTTPException(status_code=500, detail="Failed to load visualization image from MinIO.")

    img_array = np.frombuffer(visualization_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    x_min, y_min, x_max, y_max = target_object["bbox"]
    cropped_img = img[y_min:y_max, x_min:x_max]

    success, buffer = cv2.imencode('.png', cropped_img)
    if not success:
        logger.error(f"Failed to encode cropped image for object {object_id} in image {image_name}.")
        raise HTTPException(status_code=500, detail="Failed to encode cropped image.")

    logger.info(f"Successfully generated cropped image for object {object_id} in image {image_name}.")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)
