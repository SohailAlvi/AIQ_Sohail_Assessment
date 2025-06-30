import unittest
from unittest.mock import MagicMock, patch
from utils import minio_client

class TestMinioClientWrapper(unittest.TestCase):
    def setUp(self):
        # Prevent running real __init__
        self.wrapper = minio_client.MinioClientWrapper.__new__(minio_client.MinioClientWrapper)
        # Mock attributes manually
        self.wrapper.client = MagicMock()
        self.wrapper.bucket_name = "object-detection"

    def test_create_bucket_if_not_exists(self):
        self.wrapper.client.bucket_exists.return_value = False
        self.wrapper._create_bucket_if_not_exists()
        self.wrapper.client.make_bucket.assert_called_once_with("object-detection")

    def test_upload_file_success(self):
        dummy_bytes = b"test data"
        result = self.wrapper.upload_file(dummy_bytes, "test_file.png", "image/png")
        self.wrapper.client.put_object.assert_called_once()
        self.assertTrue(result.startswith("http://"))

    def test_upload_file_failure(self):
        self.wrapper.client.put_object.side_effect = Exception("Upload failed")
        result = self.wrapper.upload_file(b"data", "test.png", "image/png")
        self.assertIsNone(result)

    def test_download_file_success(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b"file content"
        self.wrapper.client.get_object.return_value = mock_response

        result = self.wrapper.download_file("test_file.png")
        self.assertEqual(result, b"file content")

    def test_download_file_failure(self):
        self.wrapper.client.get_object.side_effect = Exception("Download failed")
        result = self.wrapper.download_file("missing_file.png")
        self.assertIsNone(result)

    def test_get_presigned_url_success(self):
        self.wrapper.client.presigned_get_object.return_value = "http://fake-url"
        result = self.wrapper.get_presigned_url("test_file.png")
        self.assertEqual(result, "http://fake-url")

    def test_get_presigned_url_failure(self):
        self.wrapper.client.presigned_get_object.side_effect = Exception("URL generation failed")
        result = self.wrapper.get_presigned_url("test_file.png")
        self.assertIsNone(result)
