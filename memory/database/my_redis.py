import redis

my_redis = redis.Redis(host="192.168.1.119", port=6380, decode_responses=True)
