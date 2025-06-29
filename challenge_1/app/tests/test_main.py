def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_image(client):
    # NOTE: Adjust path to your sample image for testing
    with open("tests/sample.jpg", "rb") as img_file:
        files = {"file": ("sample.jpg", img_file, "image/jpeg")}
        response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "message" in response.json()

def test_object_detection(client):
    image_name = "sample.jpg"
    response = client.get(f"/detect/{image_name}")
    assert response.status_code == 200
    json_data = response.json()
    assert "objects" in json_data
    assert isinstance(json_data["objects"], list)

def test_get_object_mask(client):
    image_name = "sample.jpg"
    object_id = 1
    response = client.get(f"/objects/{image_name}/{object_id}/mask")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_overlay_mask(client):
    image_name = "sample.jpg"
    object_id = 1
    response = client.get(f"/objects/{image_name}/{object_id}/overlay-mask")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_invalid_object_id(client):
    image_name = "sample.jpg"
    invalid_object_id = 999
    response = client.get(f"/objects/{image_name}/{invalid_object_id}/mask")
    assert response.status_code == 404
