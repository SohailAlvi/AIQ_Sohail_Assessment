import sys
import os

sys.path.append(os.path.join(os.getcwd(), "app"))
os.environ["PT_MODEL_PATH"] = os.path.join("app", "model", "maskrcnn_finetuned_v2.pth")


import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)
