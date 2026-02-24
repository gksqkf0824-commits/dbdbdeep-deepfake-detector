"""
URL 기반 미디어(이미지/영상) 다운로드 서비스.

- yt-dlp를 사용해 YouTube/Instagram 등 지원 사이트의 URL에서 미디어를 가져온다.
- 다운로드 결과를 메모리(bytes)로 반환해 기존 추론 파이프라인과 연결한다.
"""

import glob
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import urlparse

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
_VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp", "ts"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except Exception:
        return default
    return max(value, minimum)


URL_MEDIA_MAX_MB = _env_int("URL_MEDIA_MAX_MB", 120, 1)
URL_MEDIA_TIMEOUT_SEC = _env_int("URL_MEDIA_TIMEOUT_SEC", 60, 5)


@dataclass
class DownloadedMedia:
    source_url: str
    media_type: str  # image | video
    filename: str
    content: bytes
    extractor: str
    title: str


def _validate_source_url(source_url: str) -> str:
    s = (source_url or "").strip()
    if not s:
        raise ValueError("source_url이 비어 있습니다.")

    parsed = urlparse(s)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("source_url은 http/https URL이어야 합니다.")
    if not parsed.netloc:
        raise ValueError("source_url 형식이 올바르지 않습니다.")
    return s


def _pick_primary_entry(info: Any) -> Dict[str, Any]:
    if isinstance(info, dict):
        entries = info.get("entries")
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    return item
        return info
    return {}


def _pick_downloaded_file(entry: Dict[str, Any], tmp_dir: str) -> str:
    requested = entry.get("requested_downloads")
    if isinstance(requested, list):
        for item in requested:
            if not isinstance(item, dict):
                continue
            path = item.get("filepath") or item.get("_filename")
            if isinstance(path, str) and os.path.isfile(path):
                return path

    fallback = entry.get("_filename")
    if isinstance(fallback, str) and os.path.isfile(fallback):
        return fallback

    candidates = sorted(glob.glob(os.path.join(tmp_dir, "**", "*"), recursive=True))
    for path in candidates:
        if not os.path.isfile(path):
            continue
        basename = os.path.basename(path).lower()
        if basename.endswith((".part", ".ytdl", ".json", ".description", ".txt", ".vtt", ".srt")):
            continue
        return path

    raise ValueError("다운로드된 미디어 파일을 찾지 못했습니다.")


def _detect_media_type(entry: Dict[str, Any], file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"

    entry_ext = str(entry.get("ext", "")).strip().lower()
    if entry_ext in _IMAGE_EXTS:
        return "image"
    if entry_ext in _VIDEO_EXTS:
        return "video"

    vcodec = str(entry.get("vcodec", "")).strip().lower()
    acodec = str(entry.get("acodec", "")).strip().lower()
    if (vcodec and vcodec != "none") or (acodec and acodec != "none"):
        return "video"

    return "video"


def _load_yt_dlp_cls():
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp가 설치되어 있지 않습니다. `pip install -r backend/requirements.txt` 후 서버를 재시작하세요."
        ) from exc
    return YoutubeDL


def download_media_from_url(source_url: str) -> DownloadedMedia:
    validated_url = _validate_source_url(source_url)
    max_bytes = URL_MEDIA_MAX_MB * 1024 * 1024
    YoutubeDL = _load_yt_dlp_cls()

    with tempfile.TemporaryDirectory(prefix="url-media-") as tmp_dir:
        base_opts = {
            "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "playlist_items": "1",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": URL_MEDIA_TIMEOUT_SEC,
            "max_filesize": max_bytes,
            "restrictfilenames": True,
            "overwrites": True,
            "skip_download": False,
        }

        # YouTube/Instagram 등에서 가변 포맷 구성을 고려해 format selector를 순차 fallback한다.
        # 추론에는 오디오가 필수 아님(프레임 기반)이라 video-only를 우선 시도한다.
        format_candidates = [
            "bestvideo*/bestvideo/best",
            "best/bestvideo*/bestvideo",
        ]

        info = None
        last_error: Exception = None
        for fmt in format_candidates:
            try:
                opts = dict(base_opts)
                opts["format"] = fmt
                with YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(validated_url, download=True)
                break
            except Exception as exc:
                last_error = exc
                msg = str(exc)
                if "Requested format is not available" in msg:
                    continue
                raise ValueError(f"URL에서 미디어를 가져오지 못했습니다: {exc}") from exc

        if info is None:
            raise ValueError(f"URL에서 미디어를 가져오지 못했습니다: {last_error}")

        entry = _pick_primary_entry(info)
        file_path = _pick_downloaded_file(entry, tmp_dir)
        size = os.path.getsize(file_path)
        if size <= 0:
            raise ValueError("다운로드된 미디어 파일이 비어 있습니다.")
        if size > max_bytes:
            raise ValueError(f"다운로드한 미디어가 제한 용량({URL_MEDIA_MAX_MB}MB)을 초과했습니다.")

        with open(file_path, "rb") as fp:
            content = fp.read()

        media_type = _detect_media_type(entry, file_path)
        extractor = str(entry.get("extractor_key") or entry.get("extractor") or "")
        title = str(entry.get("title") or "")
        filename = os.path.basename(file_path)

        return DownloadedMedia(
            source_url=validated_url,
            media_type=media_type,
            filename=filename,
            content=content,
            extractor=extractor,
            title=title,
        )
