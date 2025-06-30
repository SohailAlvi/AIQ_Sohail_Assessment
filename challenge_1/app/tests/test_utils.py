def test_minio_bucket_exists():
    from utils.minio_client import MinioClientWrapper
    bucket_name = "object-detection"
    assert MinioClientWrapper().client.bucket_exists(bucket_name) == True

def test_mongo_connection():
    from utils.mongo_client import client
    db = client["object_detection"]
    assert db is not None

def test_redis_connection():
    from utils.mongo_client import redis_client
    redis_client.set('test_key', 'test_value', ex=60)
    value = redis_client.get('test_key')
    assert value == 'test_value'
