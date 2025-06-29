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
import json


class MaskRCNNHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model (same way as you already do)
        num_classes = 3  # Background + 2 classes
        model = maskrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        model.load_state_dict(torch.load(self.manifest['model']['serializedFile'], map_location=self.device))
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
                obj = {
                    "id": idx + 1,
                    "bbox": prediction['boxes'][idx].detach().cpu().numpy().tolist(),
                    "score": float(score),
                }
                result.append(obj)

                mask = prediction['masks'][idx, 0].detach().cpu().numpy() > 0.5
                masks.append(mask.astype(np.uint8) * 255)

        # Optional: Create visualization image with masks and boxes
        visualization = np.zeros((prediction['masks'].shape[-2], prediction['masks'].shape[-1], 3), dtype=np.uint8)
        for mask in masks:
            color = (0, 0, 255)
            colored_mask = np.stack([mask if c > 0 else np.zeros_like(mask) for c in color], axis=-1)
            visualization = cv2.addWeighted(visualization, 1.0, colored_mask, 0.5, 0)

        _, buffer = cv2.imencode('.png', visualization)
        visualization_base64 = base64.b64encode(buffer).decode('utf-8')

        return [{
            "object_count": len(result),
            "objects": result,
            "visualization_base64": visualization_base64
        }]
