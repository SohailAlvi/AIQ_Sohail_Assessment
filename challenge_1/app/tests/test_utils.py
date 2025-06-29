def test_minio_bucket_exists():
    from app.utils.minio_client import minio_client
    bucket_name = "object-detection"
    assert minio_client.bucket_exists(bucket_name) == True

def test_mongo_connection():
    from app.utils.mongo_client import mongo_client
    db = mongo_client["object_detection"]
    assert db is not None

def test_redis_connection():
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r.set('test_key', 'test_value', ex=60)
    value = r.get('test_key')
    assert value == 'test_value'
