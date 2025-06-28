from minio import Minio
from minio.error import S3Error
import io
from typing import Optional
from datetime import timedelta
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "object-detection")


class MinioClientWrapper:
    def __init__(self):
        self.client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        self.bucket_name = MINIO_BUCKET

        # Ensure bucket exists
        self._create_bucket_if_not_exists()

    def _create_bucket_if_not_exists(self):
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)

    def download_file(self, object_name: str) -> Optional[bytes]:
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.read()
        except Exception as e:
            print(f"Error downloading {object_name} from MinIO: {e}")
            return None

    def get_presigned_url(self, object_name: str, expiry: int = 3600) -> Optional[str]:
        try:
            url = self.client.presigned_get_object(self.bucket_name, object_name, expires=timedelta(hours=1))
            return url
        except Exception as e:
            print(f"Error generating presigned URL for {object_name}: {e}")
            return None

    def upload_file(self, file_data: bytes, file_name: str, content_type: str = "application/octet-stream"):
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=file_name,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type=content_type
            )
            url = f"http://localhost:9000/{self.bucket_name}/{file_name}"
            return url
        except S3Error as err:
            print(f"MinIO Upload Error: {err}")
            return None
