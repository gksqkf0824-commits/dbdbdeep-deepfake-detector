"""Storage/hash utilities."""

import hashlib
import json
from typing import Any, Dict, Optional


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def redis_set_json(redis_db, key: str, payload: Dict[str, Any], ex: int) -> None:
    redis_db.set(key, json.dumps(payload), ex=ex)


def redis_get_json(redis_db, key: str) -> Optional[Dict[str, Any]]:
    value = redis_db.get(key)
    if value is None:
        return None
    return json.loads(value)


__all__ = [
    "sha256_bytes",
    "redis_set_json",
    "redis_get_json",
]
