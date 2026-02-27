import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

_LOCK = threading.Lock()
_CONFIGURED = False


def _parse_level(value: str) -> int:
    raw = str(value or "").strip().upper()
    if not raw:
        return logging.INFO
    if hasattr(logging, raw):
        level = getattr(logging, raw)
        if isinstance(level, int):
            return level
    return logging.INFO


def _parse_int(value: str, default: int, minimum: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except Exception:
        return default
    return max(parsed, minimum)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    with _LOCK:
        if _CONFIGURED:
            return

        level = _parse_level(os.getenv("APP_LOG_LEVEL", "INFO"))
        fmt = os.getenv(
            "APP_LOG_FORMAT",
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        datefmt = os.getenv("APP_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        root = logging.getLogger()
        root.setLevel(level)

        if not root.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)
        else:
            for handler in root.handlers:
                handler.setLevel(level)
                if handler.formatter is None:
                    handler.setFormatter(formatter)

        file_path = str(os.getenv("APP_LOG_FILE", "")).strip()
        if file_path:
            max_bytes = _parse_int(os.getenv("APP_LOG_MAX_BYTES", "10485760"), 10 * 1024 * 1024, 1024)
            backup_count = _parse_int(os.getenv("APP_LOG_BACKUP_COUNT", "5"), 5, 1)
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            has_same_file_handler = False
            for handler in root.handlers:
                if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "") == os.path.abspath(file_path):
                    has_same_file_handler = True
                    break

            if not has_same_file_handler:
                file_handler = RotatingFileHandler(
                    file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)

        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

