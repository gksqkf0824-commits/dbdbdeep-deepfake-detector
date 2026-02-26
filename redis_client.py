"""
redis_client.py — Redis connection singleton

Analysis results are cached here for 1 hour after inference,
so the frontend can retrieve them via /get-result/{token}.

Make sure Redis is running before starting the server:
  docker run -p 6379:6379 -d redis
"""

import os
import redis

redis_db = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)
