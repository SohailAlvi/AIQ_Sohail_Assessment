import io
import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from ts.torch_handler.base_handler import BaseHandler

class MaskRCNNHandler(BaseHandler):
    def __init__(self):
        super(MaskRCNNHandler, self).__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")
        model_path = f"{model_dir}/maskrcnn_finetuned_v2.pth"

        num_classes = 3  # Background + your 2 object classes
        model = maskrcnn_resnet50_fpn_v2(weights=None)

        # Replace box head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace mask head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        self.model = model
        self.initialized = True

    def preprocess(self, data):
        image_bytes = None
        for row in data:
            if "body" in row:
                image_bytes = row["body"]
            elif "data" in row:
                image_bytes = row["data"]

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = F.to_tensor(img).to(self.device)
        img_np = np.array(img)
        return img_tensor.unsqueeze(0), img_np

    def inference(self, inputs, raw_image_np):
        with torch.no_grad():
            outputs = self.model(inputs)[0]

        confidence_threshold = 0.95
        results = []
        visualization_img = raw_image_np.copy()

        for idx, score in enumerate(outputs['scores']):
            if score >= confidence_threshold:
                box = outputs['boxes'][idx].cpu().numpy().astype(int)
                mask = outputs['masks'][idx, 0].cpu().numpy() > 0.5

                centroid_mask, radius_mask = self.get_centroid_and_radius(mask)
                centroid_box, radius_box = self.get_centroid_and_radius_from_box(box)

                # Draw mask overlay
                colored_mask = np.zeros_like(visualization_img, dtype=np.uint8)
                color = (0, 255, 0)
                colored_mask[mask] = color
                alpha = 0.5
                visualization_img = cv2.addWeighted(visualization_img, 1, colored_mask, alpha, 0)

                # Draw bounding box
                cv2.rectangle(visualization_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(visualization_img, f"Obj {idx + 1}", (box[0], max(box[1] - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                result = {
                    "id": idx + 1,
                    "score": float(score),
                    "bbox": box.tolist(),
                    "centroid_from_mask": centroid_mask,
                    "radius_from_mask": float(radius_mask) if radius_mask else None,
                    "centroid_from_box": centroid_box,
                    "radius_from_box": float(radius_box),
                }
                results.append(result)

        _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization_img, cv2.COLOR_RGB2BGR))
        viz_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        output = {
            "object_count": len(results),
            "objects": results,
            "visualization_image": viz_base64
        }
        return output

    def postprocess(self, inference_output):
        return [json.dumps(inference_output)]

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
        radius = max(width, height) / 2.0
        return (cx, cy), radius
