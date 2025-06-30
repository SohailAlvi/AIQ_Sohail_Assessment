import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
import numpy as np
import io
import cv2

# ✅ Mock database clients BEFORE importing app
from database import MongoDBClient, MinioClient, RedisClient

# Mock MongoDBClient
MongoDBClient.__init__ = lambda self: None
MongoDBClient.collection = MagicMock()

# Mock MinioClient
MinioClient.__init__ = lambda self: None
MinioClient.client = MagicMock()

# Mock RedisClient
RedisClient.__init__ = lambda self: None
RedisClient.client = MagicMock()

# ✅ Now import FastAPI app and client instances
from api_server import app, mongo_client, minio_client, redis_client


@pytest.mark.asyncio
async def test_frames_success():
    # ✅ Mock MongoDB .find().sort() and .count_documents()
    mock_frames = [
        {"depth": 1.0, "minio_object": "frame1.png"},
        {"depth": 2.0, "minio_object": "frame2.png"},
    ]

    # Mock cursor that supports .sort()
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter(mock_frames)
    mongo_client.collection.find.return_value = mock_cursor
    mongo_client.collection.count_documents.return_value = 2

    # ✅ Mock Minio .get_object()
    def mock_get_object(bucket, object_name):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', img)
        return io.BytesIO(buffer.tobytes())

    minio_client.client.get_object.side_effect = mock_get_object

    # ✅ Mock Redis .get() and .setex()
    redis_client.client.get.return_value = None
    redis_client.client.setex.return_value = None

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/frames/?depth_min=0.0&depth_max=3.0")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_frames_not_found():
    # ✅ Mock MongoDB .count_documents() returning 0
    mongo_client.collection.count_documents.return_value = 0

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/frames/?depth_min=0.0&depth_max=3.0")

    assert response.status_code == 404
    assert response.json()["detail"] == "No frames found for given depth range"
