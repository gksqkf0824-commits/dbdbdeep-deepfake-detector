import redis

# 로컬에 설치된 Redis에 연결 (Docker 등으로 띄웠을 때 기준)
redis_db = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)