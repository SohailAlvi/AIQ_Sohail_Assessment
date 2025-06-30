import torch
import numpy as np
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import io
import uuid
from typing import Tuple, Optional
import os
import requests
import json
import base64

from utils.minio_client import MinioClientWrapper
from utils.logger import setup_logger

logger = setup_logger(__name__)
minio_client = MinioClientWrapper()

class ModelConfig:
    num_classes: int = 3
    weight_path: str = os.getenv("PT_MODEL_PATH", "model/best_model.pth")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torchserve_api_url: Optional[str] = os.getenv("TORCHSERVE_API_URL")

class ObjectDetectionModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        self.use_torchserve = config.torchserve_api_url is not None

        if not self.use_torchserve:
            self.model = self._load_local_model()

    def _load_local_model(self):
        try:
            model = maskrcnn_resnet50_fpn_v2(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.config.num_classes)

            model.load_state_dict(torch.load(self.config.weight_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info(f"Local PyTorch model loaded from {self.config.weight_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise

    def evaluate_image(self, image_bytes: bytes, confidence_threshold: float = 0.95) -> dict:
        if self.use_torchserve:
            return self._evaluate_via_torchserve(image_bytes)
        else:
            return self._evaluate_locally(image_bytes, confidence_threshold)

    def _evaluate_via_torchserve(self, image_bytes: bytes) -> dict:
        url = self.config.torchserve_api_url
        files = {'data': ('image.png', image_bytes, 'application/octet-stream')}

        try:
            response = requests.post(url, files=files, timeout=30)
            response.raise_for_status()
            result = response.json()

            if 'visualization_image' in result and result['visualization_image']:
                result['visualization_image'] = base64.b64decode(result['visualization_image'])

            logger.info(f"Inference via TorchServe successful.")
            return result

        except Exception as e:
            logger.error(f"TorchServe Error: {e}")
            return {"object_count": 0, "objects": [], "visualization_image": None, "error": str(e)}

    def _evaluate_locally(self, image_bytes: bytes, confidence_threshold: float = 0.95) -> dict:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img)
            img_tensor = F.to_tensor(img).to(self.device)

            with torch.no_grad():
                prediction = self.model([img_tensor])[0]

            results = []
            visualization_img = img_np.copy()

            for idx, score in enumerate(prediction['scores']):
                if score >= confidence_threshold:
                    box = prediction['boxes'][idx].cpu().numpy().astype(int)
                    mask = prediction['masks'][idx, 0].cpu().numpy() > 0.5

                    centroid_mask, radius_mask = self.get_centroid_and_radius(mask)
                    centroid_box, radius_box = self.get_centroid_and_radius_from_box(box)

                    mask_uint8 = (mask.astype(np.uint8)) * 255
                    _, mask_buffer = cv2.imencode('.png', mask_uint8)
                    mask_bytes = mask_buffer.tobytes()

                    mask_filename = f"object_masks/{str(uuid.uuid4())}_mask.png"
                    upload_success = minio_client.upload_file(mask_bytes, mask_filename, content_type="image/png")

                    if not upload_success:
                        logger.warning(f"Failed to upload mask for object {idx + 1}.")

                    colored_mask = np.zeros_like(visualization_img, dtype=np.uint8)
                    color = (0, 255, 0)
                    colored_mask[mask] = color
                    visualization_img = cv2.addWeighted(visualization_img, 1, colored_mask, 0.5, 0)

                    cv2.rectangle(visualization_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    cv2.putText(visualization_img, f"Obj {idx + 1}", (box[0], max(box[1] - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    result = {
                        "id": idx + 1,
                        "ref_id": str(uuid.uuid4()),
                        "score": float(score),
                        "bbox": box.tolist(),
                        "centroid_from_mask": centroid_mask,
                        "radius_from_mask": float(radius_mask) if radius_mask else None,
                        "centroid_from_box": centroid_box,
                        "radius_from_box": float(radius_box),
                        "mask_key": mask_filename if upload_success else None
                    }
                    results.append(result)

            _, buffer = cv2.imencode(".png", cv2.cvtColor(visualization_img, cv2.COLOR_RGB2BGR))
            visualization_bytes = buffer.tobytes()

            logger.info(f"Local inference complete. Total objects: {len(results)}")
            return {
                "object_count": len(results),
                "objects": results,
                "visualization_image": visualization_bytes
            }
        except Exception as e:
            logger.error(f"Local inference error: {e}")
            return {"object_count": 0, "objects": [], "visualization_image": None, "error": str(e)}

    @staticmethod
    def get_centroid_and_radius(binary_mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        mask_uint8 = (binary_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] == 0:
            return None, None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        (_, _), radius = cv2.minEnclosingCircle(largest_contour)

        return (cx, cy), radius

    @staticmethod
    def get_centroid_and_radius_from_box(box: np.ndarray) -> Tuple[Tuple[float, float], float]:
        x_min, y_min, x_max, y_max = box
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        radius = max(width, height) / 2.0
        return (cx, cy), radius
