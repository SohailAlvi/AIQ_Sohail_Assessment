import unittest
import numpy as np
import torch
import cv2
from unittest.mock import patch, MagicMock
from utils.minio_client import MinioClientWrapper

MinioClientWrapper.__init__ = lambda self: None
MinioClientWrapper.client = MagicMock()


from utils.ml.detection import ObjectDetectionModel, ModelConfig


class TestObjectDetectionModel(unittest.TestCase):

    def setUp(self):
        config = ModelConfig()
        # Skip real model loading by patching _load_local_model
        with patch.object(ObjectDetectionModel, "_load_local_model", return_value=MagicMock()):
            self.model = ObjectDetectionModel(config)

    def test_centroid_and_radius_from_box(self):
        box = np.array([10, 20, 50, 60])  # [x_min, y_min, x_max, y_max]
        centroid, radius = self.model.get_centroid_and_radius_from_box(box)

        expected_centroid = (30.0, 40.0)
        expected_radius = max(40, 40) / 2.0  # 20.0

        self.assertAlmostEqual(centroid[0], expected_centroid[0], places=2)
        self.assertAlmostEqual(centroid[1], expected_centroid[1], places=2)
        self.assertAlmostEqual(radius, expected_radius, places=2)

    def test_centroid_and_radius_from_mask(self):
        # Create a simple binary mask with a filled circle at center (50, 50) radius 10
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2 = __import__('cv2')
        cv2.circle(mask, (50, 50), 10, 1, thickness=-1)

        centroid, radius = self.model.get_centroid_and_radius(mask)

        self.assertIsNotNone(centroid)
        self.assertIsNotNone(radius)
        self.assertAlmostEqual(centroid[0], 50, delta=1)
        self.assertAlmostEqual(centroid[1], 50, delta=1)
        self.assertAlmostEqual(radius, 10, delta=1)

    @patch('utils.ml.detection.minio_client.upload_file', return_value=True)
    @patch('utils.ml.detection.maskrcnn_resnet50_fpn_v2')
    def test_evaluate_locally(self, mock_model_loader, mock_minio_upload):
        # Mock model output
        mock_model = MagicMock()
        mock_model.return_value = [{
            'boxes': torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            'scores': torch.tensor([0.99]),
            'masks': torch.ones((1, 1, 100, 100))
        }]
        self.model.model = mock_model

        # Dummy image bytes (small black image)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', dummy_img)
        image_bytes = buffer.tobytes()

        result = self.model._evaluate_locally(image_bytes, confidence_threshold=0.5)

        self.assertIn('object_count', result)
        self.assertGreaterEqual(result['object_count'], 1)
        self.assertIn('visualization_image', result)

        first_object = result['objects'][0]
        self.assertIn('centroid_from_box', first_object)
        self.assertIn('radius_from_box', first_object)
        self.assertIn('mask_key', first_object)


if __name__ == '__main__':
    unittest.main()
