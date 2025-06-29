from ts.torch_handler.base_handler import BaseHandler
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import io
from PIL import Image
import numpy as np
import cv2
import base64


class MaskRCNNHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        num_classes = 3  # Background + 2 classes
        model = maskrcnn_resnet50_fpn_v2(weights=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        serialized_file = self.manifest['model']['serializedFile']
        model.load_state_dict(torch.load(serialized_file, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model

        self.transform = T.Compose([T.ToTensor()])

    def preprocess(self, data):
        image_bytes = data[0].get('body')
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(image).to(self.device)
        return img_tensor.unsqueeze(0)

    def postprocess(self, outputs):
        prediction = outputs[0]
        result = []
        masks = []

        for idx, score in enumerate(prediction['scores']):
            if score >= 0.5:
                box = prediction['boxes'][idx].detach().cpu().numpy()
                mask = prediction['masks'][idx, 0].detach().cpu().numpy() > 0.5

                centroid_mask, radius_mask = self.get_centroid_and_radius(mask)
                centroid_box, radius_box = self.get_centroid_and_radius_from_box(box)

                obj = {
                    "id": idx + 1,
                    "bbox": box.tolist(),
                    "score": float(score),
                    "centroid_from_mask": centroid_mask,
                    "radius_from_mask": float(radius_mask) if radius_mask is not None else None,
                    "centroid_from_box": centroid_box,
                    "radius_from_box": float(radius_box)
                }
                result.append(obj)
                masks.append(mask.astype(np.uint8) * 255)

        # Visualization
        height, width = prediction['masks'].shape[-2:]
        visualization = np.zeros((height, width, 3), dtype=np.uint8)

        for mask in masks:
            color = (0, 0, 255)
            colored_mask = np.stack([mask if c > 0 else np.zeros_like(mask) for c in color], axis=-1)
            visualization = cv2.addWeighted(visualization, 1.0, colored_mask, 0.5, 0)

        # Encode visualization to base64
        _, buffer = cv2.imencode('.png', visualization)
        visualization_base64 = base64.b64encode(buffer).decode('utf-8')

        return [{
            "object_count": len(result),
            "objects": result,
            "visualization_base64": visualization_base64
        }]

    @staticmethod
    def get_centroid_and_radius(binary_mask):
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
    def get_centroid_and_radius_from_box(box):
        x_min, y_min, x_max, y_max = box
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        radius = min(width, height) / 2.0
        return (cx, cy), radius
