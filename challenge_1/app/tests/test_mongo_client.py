import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from utils import mongo_client


class TestMongoClient(unittest.TestCase):

    def setUp(self):
        self.image_name = "test_image.jpg"
        self.minio_key = "minio/test_image.jpg"
        self.sample_metadata = {
            "image_name": self.image_name,
            "minio_key": self.minio_key,
            "uploaded_by": "tester",
            "upload_time": datetime.utcnow(),
            "detection_done": False,
            "detection_in_progress": False,
            "object_count": 0,
            "masked_image_key": None,
            "objects": []
        }

    @patch.object(mongo_client, 'image_collection')
    @patch.object(mongo_client, '_cache_metadata')
    def test_save_uploaded_image_metadata(self, mock_cache, mock_collection):
        mongo_client.save_uploaded_image_metadata(
            image_name=self.image_name,
            minio_key=self.minio_key,
            uploaded_by="tester"
        )
        mock_collection.insert_one.assert_called_once()
        mock_cache.assert_called_once()

    @patch.object(mongo_client, 'image_collection')
    @patch.object(mongo_client, '_cache_metadata')
    @patch.object(mongo_client, 'get_image_metadata')
    def test_save_detection_result(self, mock_get_meta, mock_cache, mock_collection):
        mock_collection.update_one.return_value.matched_count = 1
        mock_get_meta.return_value = self.sample_metadata

        mongo_client.save_detection_result(
            image_name=self.image_name,
            object_count=2,
            objects=[{"id": 1}, {"id": 2}],
            masked_image_key="masked.png"
        )
        mock_collection.update_one.assert_called_once()
        mock_get_meta.assert_called_once_with(self.image_name, skip_cache=True)
        mock_cache.assert_called_once()

    @patch.object(mongo_client, 'redis_client')
    def test_get_image_metadata_cache_hit(self, mock_redis):
        mock_redis.get.return_value = '{"image_name": "test_image.jpg"}'
        result = mongo_client.get_image_metadata(self.image_name)
        self.assertEqual(result["image_name"], self.image_name)
        mock_redis.get.assert_called_once()

    @patch.object(mongo_client, 'image_collection')
    @patch.object(mongo_client, '_cache_metadata')
    @patch.object(mongo_client, 'redis_client')
    def test_get_image_metadata_mongo_fallback(self, mock_redis, mock_cache, mock_collection):
        mock_redis.get.return_value = None
        mock_collection.find_one.return_value = self.sample_metadata

        result = mongo_client.get_image_metadata(self.image_name)
        self.assertEqual(result["image_name"], self.image_name)
        mock_collection.find_one.assert_called_once()
        mock_cache.assert_called_once()

    @patch.object(mongo_client, 'image_collection')
    @patch.object(mongo_client, '_cache_metadata')
    @patch.object(mongo_client, 'get_image_metadata')
    def test_set_detection_in_progress(self, mock_get_meta, mock_cache, mock_collection):
        mock_collection.update_one.return_value.matched_count = 1
        mock_get_meta.return_value = self.sample_metadata

        mongo_client.set_detection_in_progress(self.image_name, True)
        mock_collection.update_one.assert_called_once()
        mock_cache.assert_called_once()

    @patch.object(mongo_client, 'image_collection')
    @patch.object(mongo_client, '_cache_metadata')
    @patch.object(mongo_client, 'get_image_metadata')
    def test_clear_detection_in_progress(self, mock_get_meta, mock_cache, mock_collection):
        mock_get_meta.return_value = self.sample_metadata

        mongo_client.clear_detection_in_progress(self.image_name)
        mock_collection.update_one.assert_called_once()
        mock_cache.assert_called_once()

    @patch.object(mongo_client, 'redis_client')
    def test__cache_metadata(self, mock_redis):
        mongo_client._cache_metadata(self.image_name, self.sample_metadata)
        mock_redis.setex.assert_called_once()


if __name__ == "__main__":
    unittest.main()
