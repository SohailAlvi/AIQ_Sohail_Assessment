import pytest
from httpx import AsyncClient
from api_server import app


@pytest.mark.asyncio
async def test_frames_success(monkeypatch):
    # Mock MongoDB and Minio
    from database import mongo_client, minio_client
    import numpy as np
    import io

    # Mock MongoDB query result
    mock_frames = [
        {"depth": 1.0, "minio_object": "frame1.png"},
        {"depth": 2.0, "minio_object": "frame2.png"},
    ]

    monkeypatch.setattr(mongo_client.collection, "find", lambda query: iter(mock_frames))
    monkeypatch.setattr(mongo_client.collection, "count_documents", lambda query: 2)

    # Mock Minio get_object
    def mock_get_object(bucket, object_name):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', img)
        return io.BytesIO(buffer.tobytes())

    monkeypatch.setattr(minio_client.client, "get_object", lambda bucket, key: mock_get_object(bucket, key))

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/frames/?depth_min=0.0&depth_max=3.0")

    assert response.status_code == 200
    assert response.headers['content-type'] == "image/png"


@pytest.mark.asyncio
async def test_frames_not_found(monkeypatch):
    from database import mongo_client

    monkeypatch.setattr(mongo_client.collection, "count_documents", lambda query: 0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/frames/?depth_min=0.0&depth_max=3.0")

    assert response.status_code == 404
    assert response.json()["detail"] == "No frames found for given depth range"
