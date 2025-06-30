import pytest
from utils.ml.detection import ObjectDetectionModel, ModelConfig

def test_model_detection_on_dummy_image():
    config = ModelConfig()
    model = ObjectDetectionModel(config)

    # Create a dummy image (e.g., a black 224x224 RGB image)
    import numpy as np
    import cv2

    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    success, img_bytes = cv2.imencode('.jpg', dummy_img)
    assert success, "Failed to encode dummy image."

    detection_result = model.evaluate_image(img_bytes.tobytes(), confidence_threshold=0.1)

    assert isinstance(detection_result, dict)
    assert "objects" in detection_result
    assert isinstance(detection_result["objects"], list)
