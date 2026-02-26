"""
redis_client.py — Redis connection singleton (Standalone / Local Dev)

Used by the root-level main.py for local testing.
For production (Docker), the service-name default is `redis`.
See backend/services/redis_client.py for the production version.

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
