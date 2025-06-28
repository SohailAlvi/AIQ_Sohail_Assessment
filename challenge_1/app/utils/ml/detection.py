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


class ModelConfig:
    num_classes: int = 3  # Background + your 2 classes
    weight_path: str = "model/maskrcnn_finetuned_v2.pth"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ObjectDetectionModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        self.model = self._load_model()

    def _load_model(self):
        model = maskrcnn_resnet50_fpn_v2(weights=None)

        # Update box head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

        # Update mask head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.config.num_classes)

        model.load_state_dict(torch.load(self.config.weight_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def evaluate_image(self, image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
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
                centroid, radius = self.get_centroid_and_radius(mask)

                # ---- Draw Mask ----
                colored_mask = np.zeros_like(visualization_img, dtype=np.uint8)
                color = (0, 255, 0)  # Green mask
                colored_mask[mask] = color
                alpha = 0.5
                visualization_img = cv2.addWeighted(visualization_img, 1, colored_mask, alpha, 0)

                # ---- Draw Bounding Box ----
                cv2.rectangle(
                    visualization_img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color=(255, 0, 0),
                    thickness=2
                )

                # ---- Draw Object ID ----
                cv2.putText(
                    visualization_img,
                    f"Obj {idx + 1}",
                    (box[0], max(box[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                result = {
                    "id": idx + 1,
                    "ref_id": str(uuid.uuid4()),
                    "score": float(score),
                    "bbox": box.tolist(),
                    "centroid": centroid,
                    "radius": float(radius) if radius is not None else None
                }
                results.append(result)

        # ---- Encode Visualization Image to PNG Bytes ----
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(visualization_img, cv2.COLOR_RGB2BGR))
        visualization_bytes = buffer.tobytes() if is_success else None

        return {
            "object_count": len(results),
            "objects": results,
            "visualization_image": visualization_bytes
        }

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
