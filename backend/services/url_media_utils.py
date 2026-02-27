"""
URL 기반 미디어(이미지/영상) 다운로드 서비스.

지원 분기:
1) Instagram URL 동영상/이미지 추론(Instaloader + OpenGraph + yt-dlp fallback)
2) 기타 웹사이트 URL 동영상/이미지 추론(직접 다운로드 또는 yt-dlp)
"""

import glob
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from log_config import get_logger

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
_VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp", "ts"}
_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

logger = get_logger(__name__)
_YTDLP_HELP_TEXT_CACHE: Optional[str] = None


class _SilentYTDLPLogger:
    def debug(self, msg: str) -> None:
        raw = (os.getenv("URL_MEDIA_DEBUG") or "").strip().lower()
        if raw in {"1", "true", "yes", "on", "y"}:
            logger.debug("[url-media][yt-dlp][debug] %s", msg)

    def warning(self, msg: str) -> None:
        raw = (os.getenv("URL_MEDIA_DEBUG") or "").strip().lower()
        if raw in {"1", "true", "yes", "on", "y"}:
            logger.warning("[url-media][yt-dlp][warn] %s", msg)

    def error(self, msg: str) -> None:
        raw = (os.getenv("URL_MEDIA_DEBUG") or "").strip().lower()
        if raw in {"1", "true", "yes", "on", "y"}:
            logger.error("[url-media][yt-dlp][error] %s", msg)


# =========================
# ENV helpers
# =========================

def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except Exception:
        return default
    return max(value, minimum)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except Exception:
        return default
    return max(value, minimum)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


def _env_csv(name: str, default: str) -> List[str]:
    raw = (os.getenv(name, default) or "").strip()
    tokens: List[str] = []
    for item in raw.split(","):
        token = item.strip()
        if token:
            tokens.append(token)
    return tokens


URL_MEDIA_MAX_MB = _env_int("URL_MEDIA_MAX_MB", 120, 1)
URL_MEDIA_TIMEOUT_SEC = _env_int("URL_MEDIA_TIMEOUT_SEC", 60, 5)
URL_MEDIA_DEBUG = _env_bool("URL_MEDIA_DEBUG", False)
YTDLP_PROCESS_TIMEOUT_SEC = _env_int("YTDLP_PROCESS_TIMEOUT_SEC", 90, 10)
YTDLP_TOTAL_TIMEOUT_SEC = _env_int("YTDLP_TOTAL_TIMEOUT_SEC", 240, 30)
# Backward compatibility: legacy single cookie env is still accepted.
YTDLP_COOKIEFILE_LEGACY = (os.getenv("YTDLP_COOKIEFILE") or "").strip()
YTDLP_YOUTUBE_COOKIEFILE = (os.getenv("YTDLP_YOUTUBE_COOKIEFILE") or "").strip()
YTDLP_INSTAGRAM_COOKIEFILE = (os.getenv("YTDLP_INSTAGRAM_COOKIEFILE") or "").strip()
YTDLP_YOUTUBE_FORMAT = (
    os.getenv("YTDLP_YOUTUBE_FORMAT")
    or "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
).strip()
YTDLP_YOUTUBE_MERGE_OUTPUT_FORMAT = (os.getenv("YTDLP_YOUTUBE_MERGE_OUTPUT_FORMAT") or "mp4").strip()
YTDLP_YOUTUBE_REMOTE_COMPONENTS = (os.getenv("YTDLP_YOUTUBE_REMOTE_COMPONENTS") or "ejs:github").strip()

INSTAGRAM_SESSION_ID = (os.getenv("INSTAGRAM_SESSION_ID") or "").strip()
INSTAGRAM_USER_AGENT = (os.getenv("INSTAGRAM_USER_AGENT") or _HTTP_USER_AGENT).strip() or _HTTP_USER_AGENT
INSTAGRAM_MAX_RETRIES = _env_int("INSTAGRAM_MAX_RETRIES", 3, 1)
INSTAGRAM_RETRY_MIN_DELAY_SEC = _env_float("INSTAGRAM_RETRY_MIN_DELAY_SEC", 2.0, 0.0)
INSTAGRAM_RETRY_MAX_DELAY_SEC = _env_float("INSTAGRAM_RETRY_MAX_DELAY_SEC", 6.0, 0.0)
if INSTAGRAM_RETRY_MIN_DELAY_SEC > INSTAGRAM_RETRY_MAX_DELAY_SEC:
    INSTAGRAM_RETRY_MIN_DELAY_SEC, INSTAGRAM_RETRY_MAX_DELAY_SEC = (
        INSTAGRAM_RETRY_MAX_DELAY_SEC,
        INSTAGRAM_RETRY_MIN_DELAY_SEC,
    )


# =========================
# Model
# =========================

@dataclass
class DownloadedMedia:
    source_url: str
    media_type: str  # image | video
    filename: str
    content: bytes
    extractor: str
    title: str


# =========================
# Common utils
# =========================

def _debug_log(message: str) -> None:
    if not URL_MEDIA_DEBUG:
        return
    logger.info("[url-media] %s", message)


def _summarize_err_text(message: str, max_len: int = 240) -> str:
    text = " ".join(str(message or "").split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _elapsed_sec(start_ts: float) -> float:
    return max(0.0, float(time.monotonic() - float(start_ts)))


def _ensure_time_budget(start_ts: float, stage: str) -> None:
    elapsed = _elapsed_sec(start_ts)
    if elapsed > float(YTDLP_TOTAL_TIMEOUT_SEC):
        raise ValueError(
            f"{stage} 처리 시간이 제한({YTDLP_TOTAL_TIMEOUT_SEC}s)을 초과했습니다. "
            f"(elapsed={elapsed:.1f}s)"
        )


def _run_yt_dlp_command(cmd: List[str], stage: str, start_ts: float) -> subprocess.CompletedProcess:
    _ensure_time_budget(start_ts, stage=stage)
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=YTDLP_PROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = _elapsed_sec(start_ts)
        raise ValueError(
            f"{stage} 실행 타임아웃({YTDLP_PROCESS_TIMEOUT_SEC}s) "
            f"(elapsed={elapsed:.1f}s)"
        ) from exc

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


def _host_from_url(source_url: str) -> str:
    return (urlparse(source_url).netloc or "").lower()


def _host_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def _is_youtube_url(source_url: str) -> bool:
    host = _host_from_url(source_url)
    return (
        _host_matches(host, "youtube.com")
        or _host_matches(host, "youtu.be")
        or _host_matches(host, "youtube-nocookie.com")
    )


_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _sanitize_youtube_video_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    value = value.split("?")[0].split("&")[0].strip("/")
    return value if _YOUTUBE_VIDEO_ID_RE.match(value) else ""


def _extract_youtube_video_id(source_url: str) -> str:
    parsed = urlparse(source_url)
    host = _host_from_url(source_url)
    path_parts = [p for p in (parsed.path or "").split("/") if p]

    if _host_matches(host, "youtu.be"):
        if path_parts:
            return _sanitize_youtube_video_id(path_parts[0])
        return ""

    if not (_host_matches(host, "youtube.com") or _host_matches(host, "youtube-nocookie.com")):
        return ""

    if path_parts:
        head = path_parts[0].lower()
        if head == "shorts" and len(path_parts) >= 2:
            return _sanitize_youtube_video_id(path_parts[1])
        if head == "watch":
            query_v = parse_qs(parsed.query or "").get("v", [])
            if query_v:
                return _sanitize_youtube_video_id(query_v[0])
        if head in {"embed", "live", "v"} and len(path_parts) >= 2:
            return _sanitize_youtube_video_id(path_parts[1])

    query_v = parse_qs(parsed.query or "").get("v", [])
    if query_v:
        return _sanitize_youtube_video_id(query_v[0])
    return ""


def _normalize_youtube_watch_url(source_url: str) -> str:
    if not _is_youtube_url(source_url):
        return source_url
    video_id = _extract_youtube_video_id(source_url)
    if not video_id:
        return source_url
    return f"https://www.youtube.com/watch?v={video_id}"


def _is_instagram_url(source_url: str) -> bool:
    host = _host_from_url(source_url)
    return _host_matches(host, "instagram.com")


def _path_ext_from_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    path = unquote(parsed.path or "")
    _, ext = os.path.splitext(path)
    return ext.lower().lstrip(".")


def _looks_like_direct_media_url(source_url: str) -> bool:
    ext = _path_ext_from_url(source_url)
    return ext in _IMAGE_EXTS or ext in _VIDEO_EXTS


def _media_type_from_content_type(content_type: str) -> str:
    low = str(content_type or "").lower()
    if low.startswith("image/"):
        return "image"
    if low.startswith("video/"):
        return "video"
    return "video"


def _content_ext_from_type(content_type: str) -> str:
    low = str(content_type or "").lower()
    if "jpeg" in low or "jpg" in low:
        return "jpg"
    if "png" in low:
        return "png"
    if "webp" in low:
        return "webp"
    if "gif" in low:
        return "gif"
    if "mp4" in low:
        return "mp4"
    if "webm" in low:
        return "webm"
    if "quicktime" in low:
        return "mov"
    return "bin"


def _filename_from_media_url(media_url: str, default_ext: str, fallback_name: str = "url_media") -> str:
    parsed = urlparse(media_url)
    basename = os.path.basename(unquote(parsed.path or "")).strip()
    if basename:
        root, ext = os.path.splitext(basename)
        if ext:
            return basename
        root = root or fallback_name
        return f"{root}.{default_ext}"
    return f"{fallback_name}.{default_ext}"


class _MetaTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        if str(tag or "").lower() != "meta":
            return

        attr_map: Dict[str, str] = {}
        for key, value in attrs:
            k = str(key or "").strip().lower()
            v = str(value or "").strip()
            if k:
                attr_map[k] = v

        key = str(attr_map.get("property") or attr_map.get("name") or "").strip().lower()
        content = str(attr_map.get("content") or "").strip()
        if key and content and key not in self.meta:
            self.meta[key] = content


def _pick_opengraph_media_url(meta: Dict[str, str], source_url: str) -> Tuple[str, str]:
    low = source_url.lower()
    prefer_video = "/reel/" in low or "/reels/" in low

    candidates: List[Tuple[str, str]] = []
    if prefer_video:
        candidates.extend(
            [
                ("video", meta.get("og:video:secure_url", "")),
                ("video", meta.get("og:video:url", "")),
                ("video", meta.get("og:video", "")),
            ]
        )
    candidates.extend(
        [
            ("image", meta.get("og:image:secure_url", "")),
            ("image", meta.get("og:image:url", "")),
            ("image", meta.get("og:image", "")),
            ("image", meta.get("twitter:image:src", "")),
            ("image", meta.get("twitter:image", "")),
        ]
    )
    if not prefer_video:
        candidates.extend(
            [
                ("video", meta.get("og:video:secure_url", "")),
                ("video", meta.get("og:video:url", "")),
                ("video", meta.get("og:video", "")),
            ]
        )

    seen: set[str] = set()
    for media_type, raw_url in candidates:
        url = str(raw_url or "").strip()
        if not url:
            continue
        absolute_url = urljoin(source_url, url)
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        return media_type, absolute_url

    raise ValueError("OpenGraph 메타에서 미디어 URL을 찾지 못했습니다.")


def _download_media_from_opengraph(source_url: str, max_bytes: int) -> DownloadedMedia:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("OpenGraph 파싱을 위해 requests 라이브러리가 필요합니다.") from exc

    headers = {"User-Agent": _HTTP_USER_AGENT}
    with requests.get(
        source_url,
        timeout=max(URL_MEDIA_TIMEOUT_SEC, 10),
        headers=headers,
        allow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        parser = _MetaTagParser()
        parser.feed(resp.text or "")
        meta = dict(parser.meta)

    media_type_hint, media_url = _pick_opengraph_media_url(meta, source_url=source_url)
    content, content_type = _stream_download_bytes(
        url=media_url,
        max_bytes=max_bytes,
        timeout_sec=URL_MEDIA_TIMEOUT_SEC,
        session=None,
        headers=headers,
    )

    detected_type = media_type_hint
    if not media_type_hint:
        detected_type = _media_type_from_content_type(content_type)
    elif media_type_hint == "image" and _media_type_from_content_type(content_type) == "video":
        detected_type = "video"

    default_ext = _path_ext_from_url(media_url) or _content_ext_from_type(content_type) or (
        "mp4" if detected_type == "video" else "jpg"
    )
    filename = _filename_from_media_url(media_url, default_ext=default_ext, fallback_name="og_media")

    title = str(meta.get("og:title") or meta.get("twitter:title") or "").strip()

    return DownloadedMedia(
        source_url=source_url,
        media_type="video" if detected_type == "video" else "image",
        filename=filename,
        content=content,
        extractor="opengraph_meta",
        title=title,
    )


def _stream_download_bytes(
    url: str,
    max_bytes: int,
    timeout_sec: int,
    session=None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[bytes, str]:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("URL 다운로드를 위해 requests 라이브러리가 필요합니다.") from exc

    request_headers = dict(headers or {})
    if "User-Agent" not in request_headers:
        request_headers["User-Agent"] = _HTTP_USER_AGENT

    requester = session.get if session is not None else requests.get

    with requester(url, stream=True, timeout=max(timeout_sec, 10), headers=request_headers) as resp:
        resp.raise_for_status()
        content_type = str(resp.headers.get("Content-Type", "") or "")

        declared_length = int(resp.headers.get("Content-Length", "0") or 0)
        if declared_length > max_bytes:
            raise ValueError(f"다운로드한 미디어가 제한 용량({URL_MEDIA_MAX_MB}MB)을 초과했습니다.")

        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"다운로드한 미디어가 제한 용량({URL_MEDIA_MAX_MB}MB)을 초과했습니다.")
            chunks.append(chunk)

        content = b"".join(chunks)

    if not content:
        raise ValueError("다운로드된 미디어 파일이 비어 있습니다.")

    return content, content_type


def _download_direct_media(source_url: str, max_bytes: int) -> DownloadedMedia:
    content, content_type = _stream_download_bytes(
        url=source_url,
        max_bytes=max_bytes,
        timeout_sec=URL_MEDIA_TIMEOUT_SEC,
        session=None,
        headers={"User-Agent": _HTTP_USER_AGENT},
    )

    ext = _path_ext_from_url(source_url)
    media_type = (
        "image"
        if ext in _IMAGE_EXTS
        else "video"
        if ext in _VIDEO_EXTS
        else _media_type_from_content_type(content_type)
    )

    default_ext = ext or _content_ext_from_type(content_type)
    filename = _filename_from_media_url(source_url, default_ext=default_ext, fallback_name="url_media")

    return DownloadedMedia(
        source_url=source_url,
        media_type=media_type,
        filename=filename,
        content=content,
        extractor="direct_http",
        title="",
    )


# =========================
# yt-dlp helpers (Generic / Instagram)
# =========================

def _resolve_cookiefile_for_source(source_group: str = "generic") -> str:
    group = str(source_group or "generic").strip().lower()
    candidates: List[str] = []

    if group == "youtube":
        candidates.extend(
            [
                YTDLP_YOUTUBE_COOKIEFILE,
                YTDLP_COOKIEFILE_LEGACY,
                "/run/secrets/www.youtube.com_cookies.txt",
                "/run/secrets/youtube_cookies.txt",
                "/run/secrets/social_cookies.txt",
            ]
        )
    elif group == "instagram":
        candidates.extend(
            [
                YTDLP_INSTAGRAM_COOKIEFILE,
                YTDLP_COOKIEFILE_LEGACY,
                "/run/secrets/www.instagram.com_cookies.txt",
                "/run/secrets/instagram_cookies.txt",
                "/run/secrets/social_cookies.txt",
            ]
        )
    else:
        candidates.extend([YTDLP_COOKIEFILE_LEGACY])

    seen: set[str] = set()
    for raw in candidates:
        path = str(raw or "").strip()
        if not path:
            continue
        low = path.lower()
        if low in seen:
            continue
        seen.add(low)
        if os.path.isfile(path):
            return path
    return ""


def _cookiefile_env_hint(source_group: str = "generic") -> str:
    group = str(source_group or "generic").strip().lower()
    if group == "youtube":
        return "YTDLP_YOUTUBE_COOKIEFILE"
    if group == "instagram":
        return "YTDLP_INSTAGRAM_COOKIEFILE"
    return "cookiefile"


def _ensure_parent_dir(path_obj: Path) -> None:
    parent = path_obj.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _prepare_writable_cookiefile(tmp_dir: str, source_group: str) -> str:
    cookie_path = _resolve_cookiefile_for_source(source_group)
    if not cookie_path:
        return ""
    if not os.path.isfile(cookie_path):
        return ""
    dst = os.path.join(tmp_dir, f"yt_{source_group}_cookies.txt")
    shutil.copyfile(cookie_path, dst)
    return dst


def _detect_node_runtime_path() -> str:
    # 쉘에서의 `$(command -v node)`와 동일한 방식으로 우선 탐지한다.
    try:
        proc = subprocess.run(
            ["sh", "-lc", "command -v node || command -v nodejs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        detected = str(proc.stdout or "").strip()
        if detected:
            return detected
    except Exception:
        pass

    # 쉘 탐지 실패 시 python 경로 탐지로 한 번 더 보완한다.
    node_path = shutil.which("node")
    if node_path:
        return node_path
    nodejs_path = shutil.which("nodejs")
    if nodejs_path:
        return nodejs_path
    return ""


def _build_cli_js_runtimes_value() -> str:
    runtime_cfg = YTDLP_JS_RUNTIMES if isinstance(YTDLP_JS_RUNTIMES, dict) else {}
    node_cfg = runtime_cfg.get("node") if isinstance(runtime_cfg, dict) else None
    node_path = ""
    if isinstance(node_cfg, dict):
        node_path = str(node_cfg.get("path") or "").strip()
    if not node_path:
        node_path = _detect_node_runtime_path()
    return f"node:{node_path}" if node_path else "node"


def _clear_download_candidates(base_path: Path) -> None:
    pattern = base_path.stem + "*"
    for candidate in base_path.parent.glob(pattern):
        if not candidate.is_file():
            continue
        try:
            candidate.unlink()
        except Exception:
            continue


def _pick_latest_video_download(base_path: Path) -> Optional[Path]:
    candidates = sorted(
        base_path.parent.glob(base_path.stem + "*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    best: Optional[Path] = None
    best_size = -1
    for candidate in candidates:
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        if name.endswith((".part", ".ytdl", ".json", ".description", ".txt", ".vtt", ".srt")):
            continue
        ext = candidate.suffix.lower().lstrip(".")
        if ext not in _VIDEO_EXTS:
            continue
        try:
            size = candidate.stat().st_size
        except Exception:
            size = 0
        if size > best_size:
            best = candidate
            best_size = size
    return best

def _parse_js_runtimes(raw: str) -> Dict[str, Dict[str, str]]:
    """
    yt-dlp 신버전은 js_runtimes를 {runtime: {config}} 형태(dict)로 기대한다.

    지원 입력 형식:
    - "node"
    - "node,deno"
    - "node:/usr/bin/node,deno:/usr/bin/deno"
    - '{"node":{"path":"/usr/bin/node"}}'
    """
    s = (raw or "").strip()
    if not s:
        return {}

    if s.startswith("{"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                out: Dict[str, Dict[str, str]] = {}
                for runtime, cfg in parsed.items():
                    name = str(runtime).strip()
                    if not name:
                        continue
                    if isinstance(cfg, dict):
                        path = str(cfg.get("path", "")).strip()
                        out[name] = {"path": path} if path else {}
                    else:
                        out[name] = {}
                if out:
                    return out
        except Exception:
            pass

    out: Dict[str, Dict[str, str]] = {}
    for token in s.split(","):
        item = token.strip()
        if not item:
            continue
        if item.startswith("/"):
            # YTDLP_JS_RUNTIMES=/usr/bin/node 같은 값도 허용한다.
            out["node"] = {"path": item}
            continue
        if ":" in item:
            name, path = item.split(":", 1)
            name = name.strip()
            path = path.strip()
            if not name:
                continue
            out[name] = {"path": path} if path else {}
        else:
            low = item.lower()
            if low in {"nodejs"}:
                out["node"] = {}
            else:
                out[item] = {}
    return out


YTDLP_JS_RUNTIMES = _parse_js_runtimes(os.getenv("YTDLP_JS_RUNTIMES", "node") or "")


def _load_yt_dlp_cls():
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp가 설치되어 있지 않습니다. `pip install -r backend/requirements.txt` 후 서버를 재시작하세요."
        ) from exc
    return YoutubeDL


def _apply_cookiefile_option(opts: Dict[str, Any], tmp_dir: str, source_group: str = "generic") -> None:
    cookie_path = _resolve_cookiefile_for_source(source_group)
    if not cookie_path:
        return

    if not os.path.isfile(cookie_path):
        _debug_log(f"{_cookiefile_env_hint(source_group)} missing: {cookie_path}")
        return

    req_cookie_path = os.path.join(tmp_dir, "yt_cookies.txt")
    try:
        shutil.copyfile(cookie_path, req_cookie_path)
    except Exception as exc:
        _debug_log(f"{_cookiefile_env_hint(source_group)} copy failed: {exc}")
        return
    opts["cookiefile"] = req_cookie_path


def _apply_js_runtime_option(opts: Dict[str, Any]) -> None:
    if YTDLP_JS_RUNTIMES and isinstance(YTDLP_JS_RUNTIMES, dict):
        opts["js_runtimes"] = YTDLP_JS_RUNTIMES


def _yt_dlp_cli_help_text() -> str:
    global _YTDLP_HELP_TEXT_CACHE
    if _YTDLP_HELP_TEXT_CACHE is not None:
        return _YTDLP_HELP_TEXT_CACHE
    try:
        proc = subprocess.run(
            ["yt-dlp", "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        text = ""
    _YTDLP_HELP_TEXT_CACHE = text.lower()
    return _YTDLP_HELP_TEXT_CACHE


def _yt_dlp_cli_supports_option(option_name: str) -> bool:
    opt = str(option_name or "").strip().lower()
    if not opt:
        return False
    return opt in _yt_dlp_cli_help_text()


def _download_youtube_with_ytdlp_cli(source_url: str, max_bytes: int) -> DownloadedMedia:
    started_at = time.monotonic()
    normalized_source_url = _normalize_youtube_watch_url(source_url)

    with tempfile.TemporaryDirectory(prefix="yt-cli-") as tmp_dir:
        out_base = Path(tmp_dir) / "youtube_media"
        _ensure_parent_dir(out_base)
        _clear_download_candidates(out_base)

        cmd: List[str] = [
            "yt-dlp",
            "--no-playlist",
            "--no-part",
            "--check-formats",
            "--socket-timeout",
            str(URL_MEDIA_TIMEOUT_SEC),
            "--max-filesize",
            str(max_bytes),
            "--retries",
            "2",
            "--fragment-retries",
            "2",
            "--restrict-filenames",
        ]
        runtime_value = _build_cli_js_runtimes_value()
        if _yt_dlp_cli_supports_option("--js-runtimes"):
            cmd += ["--js-runtimes", runtime_value]
        else:
            logger.warning(
                "yt-dlp CLI가 --js-runtimes 옵션을 지원하지 않습니다. "
                "구버전일 수 있으니 최신 버전으로 업데이트를 권장합니다."
            )

        if YTDLP_YOUTUBE_REMOTE_COMPONENTS:
            if _yt_dlp_cli_supports_option("--remote-components"):
                cmd += ["--remote-components", YTDLP_YOUTUBE_REMOTE_COMPONENTS]
            else:
                logger.warning(
                    "yt-dlp CLI가 --remote-components 옵션을 지원하지 않습니다. "
                    "구버전일 수 있으니 최신 버전으로 업데이트를 권장합니다."
                )

        cookiefile = ""
        try:
            cookiefile = _prepare_writable_cookiefile(tmp_dir, source_group="youtube")
        except Exception:
            cookiefile = ""
        if cookiefile:
            cmd += ["--cookies", cookiefile]

        if YTDLP_YOUTUBE_FORMAT:
            cmd += ["-f", YTDLP_YOUTUBE_FORMAT]
        if YTDLP_YOUTUBE_MERGE_OUTPUT_FORMAT:
            cmd += ["--merge-output-format", YTDLP_YOUTUBE_MERGE_OUTPUT_FORMAT]

        cmd += [
            "-o",
            str(out_base) + ".%(ext)s",
            normalized_source_url,
        ]

        _debug_log(
            "youtube-cli:run "
            f"url={normalized_source_url} cookie={bool(cookiefile)} "
            f"runtime={runtime_value} format={YTDLP_YOUTUBE_FORMAT}"
        )
        proc = _run_yt_dlp_command(cmd, stage="youtube-cli", start_ts=started_at)
        if proc.returncode != 0:
            err_text = (proc.stderr or proc.stdout or "").strip()
            low = err_text.lower()
            if "no such option: --js-runtimes" in low or "no such option: --remote-components" in low:
                raise ValueError(
                    "현재 서버의 yt-dlp CLI가 필요한 옵션(--js-runtimes/--remote-components)을 지원하지 않습니다. "
                    "컨테이너의 yt-dlp를 최신 버전으로 업데이트해 주세요."
                )
            raise ValueError(f"yt-dlp CLI 실패: {err_text or 'unknown error'}")

        file_path = _pick_latest_video_download(out_base)
        if file_path is None or not file_path.is_file():
            raise ValueError("YouTube 다운로드가 완료되었지만 영상 파일을 찾지 못했습니다.")

        size = file_path.stat().st_size
        if size <= 0:
            raise ValueError("다운로드된 미디어 파일이 비어 있습니다.")
        if size > max_bytes:
            raise ValueError(f"다운로드한 미디어가 제한 용량({URL_MEDIA_MAX_MB}MB)을 초과했습니다.")

        with open(file_path, "rb") as fp:
            content = fp.read()

        return DownloadedMedia(
            source_url=source_url,
            media_type="video",
            filename=file_path.name,
            content=content,
            extractor="youtube_ytdlp_cli",
            title="",
        )


def _build_format_candidates(source_group: str, source_url: str) -> List[Optional[str]]:
    low = source_url.lower()
    if source_group == "youtube":
        return [
            YTDLP_YOUTUBE_FORMAT or "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "best[ext=mp4]/best",
            "best",
            None,
        ]
    if source_group == "instagram":
        if "/reel/" in low or "/reels/" in low:
            return [
                "best[ext=mp4][vcodec!=none]/best[ext=webm][vcodec!=none]/best[vcodec!=none]/best",
                "best",
                None,
            ]
        return [
            "best",
            "best[ext=jpg]/best[ext=jpeg]/best[ext=png]/best[ext=webp]",
            "best[ext=mp4]/best[ext=webm]/best",
            None,
        ]

    return [
        "best",
        "best[ext=mp4]/best[ext=webm]/best",
        None,
    ]


def _is_login_or_rate_limit_error(message: str) -> bool:
    low = str(message or "").lower()
    keywords = [
        "login required",
        "cookies",
        "rate-limit reached",
        "requested content is not available",
        "sign in to confirm",
        "provided youtube account cookies are no longer valid",
        "confirm you’re not a bot",
        "confirm you're not a bot",
        "please sign in",
        "not a bot",
        "detected as a bot",
        "n challenge solving failed",
        "only images are available for download",
        "no supported javascript runtime could be found",
        "po_token",
        "proof of origin token",
        "private account",
        "private video",
        "login to view",
    ]
    return any(keyword in low for keyword in keywords)


def _is_format_unavailable_error(message: str) -> bool:
    low = str(message or "").lower()
    keywords = [
        "requested format is not available",
        "requested format not available",
        "no video formats found",
        "no formats found",
        "only images are available for download",
    ]
    return any(keyword in low for keyword in keywords)


def _is_cookie_invalid_error(message: str) -> bool:
    low = str(message or "").lower()
    keywords = [
        "provided youtube account cookies are no longer valid",
        "cookies are no longer valid",
        "cookie file",
    ]
    return any(keyword in low for keyword in keywords)


def _is_ffmpeg_merge_error(message: str) -> bool:
    low = str(message or "").lower()
    return "ffmpeg is not installed" in low or "requested merging of multiple formats" in low


def _error_signature(message: str) -> str:
    low = " ".join(str(message or "").lower().split())
    if not low:
        return ""
    head = low.split("\n", 1)[0]
    return head[:220]


def _login_or_rate_limit_detail(source_url: str = "") -> str:
    if _is_youtube_url(source_url):
        return (
            "유튜브 접근이 제한되었습니다(봇 확인/로그인 필요). "
            "유효한 YouTube 쿠키를 YTDLP_YOUTUBE_COOKIEFILE로 설정하고 "
            "node 런타임/yt-dlp 설정을 확인해 주세요."
        )
    if _is_instagram_url(source_url):
        return (
            "인스타그램 접근이 제한되었습니다(로그인/레이트리밋). "
            "INSTAGRAM_SESSION_ID 또는 YTDLP_INSTAGRAM_COOKIEFILE 설정을 확인해 주세요."
        )
    return (
        "URL 접근이 제한되었습니다(로그인 또는 레이트리밋). "
        "서버 환경변수/쿠키 설정을 확인해 주세요."
    )


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


def _format_sort_key(fmt: Dict[str, Any]) -> Tuple[int, int, int, float]:
    vcodec = str(fmt.get("vcodec", "")).strip().lower()
    acodec = str(fmt.get("acodec", "")).strip().lower()
    progressive = 1 if (vcodec not in {"", "none"} and acodec not in {"", "none"}) else 0

    ext = str(fmt.get("ext", "")).strip().lower()
    ext_rank = 2 if ext == "mp4" else 1 if ext in {"webm", "m4v", "mov"} else 0

    try:
        height = int(float(fmt.get("height") or 0))
    except Exception:
        height = 0
    try:
        tbr = float(fmt.get("tbr") or 0.0)
    except Exception:
        tbr = 0.0
    quality = float(height) + (tbr / 1000.0)
    return progressive, ext_rank, int(quality), quality


def _entry_format_candidates(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    direct_url = str(entry.get("url") or "").strip()
    direct_ext = str(entry.get("ext") or "").strip().lower()
    if direct_url:
        candidates.append(
            {
                "url": direct_url,
                "ext": direct_ext,
                "format_id": str(entry.get("format_id") or "direct"),
                "vcodec": str(entry.get("vcodec") or ""),
                "acodec": str(entry.get("acodec") or ""),
            }
        )

    formats = entry.get("formats")
    if isinstance(formats, list):
        for item in formats:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            protocol = str(item.get("protocol") or "").lower()
            if "dash" in protocol or protocol.startswith("m3u8"):
                continue
            candidates.append(item)

    if not candidates:
        return []

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(candidates, key=_format_sort_key, reverse=True):
        key = str(item.get("url") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _download_by_entry_format_urls(
    *,
    source_url: str,
    entry: Dict[str, Any],
    max_bytes: int,
    source_group: str,
) -> DownloadedMedia:
    candidates = _entry_format_candidates(entry)
    if not candidates:
        raise ValueError("추출한 포맷 후보가 없어 직접 스트림 다운로드를 시도할 수 없습니다.")

    last_error: Optional[Exception] = None
    for fmt in candidates:
        media_url = str(fmt.get("url") or "").strip()
        if not media_url:
            continue

        headers = {"User-Agent": _HTTP_USER_AGENT}
        if _is_instagram_url(source_url):
            headers["Referer"] = "https://www.instagram.com/"

        try:
            content, content_type = _stream_download_bytes(
                url=media_url,
                max_bytes=max_bytes,
                timeout_sec=URL_MEDIA_TIMEOUT_SEC,
                session=None,
                headers=headers,
            )
        except Exception as exc:
            last_error = exc
            continue

        ext = str(fmt.get("ext") or "").strip().lower() or _content_ext_from_type(content_type)
        format_id = str(fmt.get("format_id") or "fmt").strip() or "fmt"
        entry_id = str(entry.get("id") or "url_media").strip() or "url_media"
        filename = f"{entry_id}_{format_id}.{ext}"

        media_type = (
            "image"
            if ext in _IMAGE_EXTS
            else "video"
            if ext in _VIDEO_EXTS
            else _media_type_from_content_type(content_type)
        )

        extractor = str(entry.get("extractor_key") or entry.get("extractor") or "")
        if not extractor:
            if source_group == "instagram":
                extractor = "instagram_ytdlp_format_url"
            else:
                extractor = "generic_ytdlp_format_url"

        title = str(entry.get("title") or "")
        return DownloadedMedia(
            source_url=source_url,
            media_type=media_type,
            filename=filename,
            content=content,
            extractor=extractor,
            title=title,
        )

    raise ValueError(f"포맷 URL 직접 다운로드 실패: {last_error}")


def _download_with_ytdlp(source_url: str, max_bytes: int, source_group: str) -> DownloadedMedia:
    YoutubeDL = _load_yt_dlp_cls()
    started_at = time.monotonic()
    _debug_log(
        f"ytdlp-api:start source_group={source_group} source={source_url}"
    )
    http_headers: Dict[str, str] = {"User-Agent": _HTTP_USER_AGENT}
    if source_group == "youtube":
        http_headers["Referer"] = "https://www.youtube.com/"

    with tempfile.TemporaryDirectory(prefix="url-media-") as tmp_dir:
        base_opts: Dict[str, Any] = {
            "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "playlist_items": "1",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _SilentYTDLPLogger(),
            "socket_timeout": URL_MEDIA_TIMEOUT_SEC,
            "max_filesize": max_bytes,
            "check_formats": True,
            "restrictfilenames": True,
            "overwrites": True,
            "skip_download": False,
            "http_headers": http_headers,
            "retries": 2,
            "fragment_retries": 2,
        }
        _apply_js_runtime_option(base_opts)

        format_candidates = _build_format_candidates(source_group=source_group, source_url=source_url)
        opts = dict(base_opts)
        _apply_cookiefile_option(opts, tmp_dir, source_group=source_group)
        option_variants: List[Dict[str, Any]] = [opts]
        _debug_log(
            "ytdlp-api:config "
            f"variants={len(option_variants)} formats={len(format_candidates)}"
        )

        info = None
        last_error: Optional[Exception] = None
        saw_login_error = False
        for variant_opts in option_variants:
            for fmt in format_candidates:
                _ensure_time_budget(started_at, stage="ytdlp-api")
                _debug_log(f"ytdlp-api:attempt fmt={fmt or 'none'}")
                try:
                    opts = dict(variant_opts)
                    if fmt:
                        opts["format"] = fmt
                    else:
                        opts.pop("format", None)
                    with YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(source_url, download=True)
                    _debug_log("ytdlp-api:attempt-success download=True")
                    break
                except Exception as exc:
                    last_error = exc
                    msg = str(exc)
                    _debug_log(f"ytdlp-api:attempt-fail err={_summarize_err_text(msg)}")
                    if _is_format_unavailable_error(msg):
                        continue
                    if _is_ffmpeg_merge_error(msg):
                        continue
                    if _is_login_or_rate_limit_error(msg):
                        saw_login_error = True
                        break
                    raise ValueError(f"URL에서 미디어를 가져오지 못했습니다: {exc}") from exc
            if saw_login_error:
                break
            if info is not None:
                break

        if info is None:
            if saw_login_error or _is_login_or_rate_limit_error(str(last_error or "")):
                _debug_log("ytdlp-api:stop login_or_rate_limit_detected")
                logger.warning(
                    "ytdlp-api stop login_or_rate_limit elapsed=%.1fs",
                    _elapsed_sec(started_at),
                )
                raise ValueError(_login_or_rate_limit_detail(source_url))
            _debug_log(f"ytdlp-api:stop failed err={_summarize_err_text(last_error)}")
            logger.warning(
                "ytdlp-api failed elapsed=%.1fs err=%s",
                _elapsed_sec(started_at),
                _summarize_err_text(last_error),
            )
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
        if not extractor:
            if source_group == "instagram":
                extractor = "instagram_ytdlp"
            else:
                extractor = "generic_ytdlp"
        title = str(entry.get("title") or "")
        filename = os.path.basename(file_path)

        return DownloadedMedia(
            source_url=source_url,
            media_type=media_type,
            filename=filename,
            content=content,
            extractor=extractor,
            title=title,
        )


# =========================
# Instagram helpers (Instaloader)
# =========================

_INSTAGRAM_SHORTCODE_RE = re.compile(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)")


def _load_instaloader_symbols():
    try:
        from instaloader import Instaloader, Post, ConnectionException
    except Exception as exc:
        raise RuntimeError(
            "instaloader가 설치되어 있지 않습니다. `pip install -r backend/requirements.txt` 후 서버를 재시작하세요."
        ) from exc
    return Instaloader, Post, ConnectionException


def _extract_instagram_shortcode(source_url: str) -> str:
    match = _INSTAGRAM_SHORTCODE_RE.search(source_url)
    if not match:
        raise ValueError("올바른 인스타그램 게시물 URL 형식이 아닙니다. (/p/, /reel/, /reels/)를 확인하세요.")
    return match.group(1)


def _build_instagram_sessions():
    Instaloader, Post, ConnectionException = _load_instaloader_symbols()

    loader = Instaloader(
        download_pictures=False,
        download_videos=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        user_agent=INSTAGRAM_USER_AGENT,
    )
    if INSTAGRAM_SESSION_ID:
        loader.context._session.cookies.set("sessionid", INSTAGRAM_SESSION_ID, domain=".instagram.com")

    try:
        import requests
    except Exception as exc:
        raise RuntimeError("Instagram 미디어 다운로드를 위해 requests 라이브러리가 필요합니다.") from exc

    req_session = requests.Session()
    if INSTAGRAM_SESSION_ID:
        req_session.cookies.set("sessionid", INSTAGRAM_SESSION_ID, domain=".instagram.com")
    req_session.headers.update({"User-Agent": INSTAGRAM_USER_AGENT})

    return loader, req_session, Post, ConnectionException


def _collect_instagram_media_items(post) -> List[Dict[str, str]]:
    media_items: List[Dict[str, str]] = []

    if str(getattr(post, "typename", "")) == "GraphSidecar":
        nodes = list(post.get_sidecar_nodes())
        for node in nodes:
            is_video = bool(getattr(node, "is_video", False))
            media_url = getattr(node, "video_url", None) if is_video else getattr(node, "display_url", None)
            if not media_url:
                continue
            media_items.append({
                "type": "video" if is_video else "image",
                "url": str(media_url),
            })
    else:
        is_video = bool(getattr(post, "is_video", False))
        media_url = getattr(post, "video_url", None) if is_video else getattr(post, "url", None)
        if media_url:
            media_items.append({
                "type": "video" if is_video else "image",
                "url": str(media_url),
            })

    if not media_items:
        raise ValueError("인스타그램 게시물에서 분석 가능한 미디어를 찾지 못했습니다.")
    return media_items


def _select_instagram_media_item(items: List[Dict[str, str]], source_url: str) -> Dict[str, str]:
    low = source_url.lower()
    if "/reel/" in low or "/reels/" in low:
        for item in items:
            if item.get("type") == "video":
                return item
    return items[0]


def _download_instagram_media(source_url: str, max_bytes: int) -> DownloadedMedia:
    loader, req_session, Post, ConnectionException = _build_instagram_sessions()
    shortcode = _extract_instagram_shortcode(source_url)

    last_error: Optional[Exception] = None
    for attempt in range(INSTAGRAM_MAX_RETRIES):
        if attempt > 0 and INSTAGRAM_RETRY_MAX_DELAY_SEC > 0:
            time.sleep(random.uniform(INSTAGRAM_RETRY_MIN_DELAY_SEC, INSTAGRAM_RETRY_MAX_DELAY_SEC))

        try:
            post = Post.from_shortcode(loader.context, shortcode)
            media_items = _collect_instagram_media_items(post)
            selected = _select_instagram_media_item(media_items, source_url=source_url)

            media_url = selected.get("url") or ""
            media_type = selected.get("type") or "video"

            content, content_type = _stream_download_bytes(
                url=media_url,
                max_bytes=max_bytes,
                timeout_sec=URL_MEDIA_TIMEOUT_SEC,
                session=req_session,
                headers={"User-Agent": INSTAGRAM_USER_AGENT},
            )

            default_ext = "mp4" if media_type == "video" else "jpg"
            media_ext = _path_ext_from_url(media_url) or _content_ext_from_type(content_type) or default_ext
            filename = _filename_from_media_url(
                media_url,
                default_ext=media_ext,
                fallback_name=f"instagram_{shortcode}",
            )

            owner = str(getattr(post, "owner_username", "") or "").strip()
            title = f"@{owner}" if owner else ""

            return DownloadedMedia(
                source_url=source_url,
                media_type="video" if media_type == "video" else "image",
                filename=filename,
                content=content,
                extractor="instagram_instaloader",
                title=title,
            )

        except ConnectionException as exc:
            last_error = exc
            if attempt < INSTAGRAM_MAX_RETRIES - 1:
                continue
            raise ValueError(
                "인스타그램 접속 오류가 반복되어 미디어를 가져오지 못했습니다. "
                "세션 상태/접근 제한을 확인해 주세요."
            ) from exc
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if _is_login_or_rate_limit_error(msg):
                raise ValueError(_login_or_rate_limit_detail(source_url)) from exc
            if attempt < INSTAGRAM_MAX_RETRIES - 1:
                continue
            raise ValueError(f"인스타그램 URL에서 미디어를 가져오지 못했습니다: {exc}") from exc

    raise ValueError(f"인스타그램 URL에서 미디어를 가져오지 못했습니다: {last_error}")


# =========================
# Public API
# =========================

def download_media_from_url(source_url: str) -> DownloadedMedia:
    """
    URL 미디어 다운로드 진입점.
    - YouTube URL: yt-dlp CLI 우선 + yt-dlp API fallback
    - Instagram URL: Instaloader 우선 + OpenGraph + yt-dlp fallback 경로
    - 기타 URL: direct-http 우선, 실패 시 yt-dlp generic 경로
    """
    validated_url = _validate_source_url(source_url)
    max_bytes = URL_MEDIA_MAX_MB * 1024 * 1024
    _debug_log(
        f"entry source={validated_url} youtube={_is_youtube_url(validated_url)} "
        f"instagram={_is_instagram_url(validated_url)} max_bytes={max_bytes}"
    )

    if _is_instagram_url(validated_url):
        fallback_errors: List[str] = []
        _debug_log("branch instagram")

        try:
            return _download_instagram_media(validated_url, max_bytes=max_bytes)
        except Exception as insta_exc:
            fallback_errors.append(f"instaloader: {insta_exc}")

        try:
            return _download_media_from_opengraph(validated_url, max_bytes=max_bytes)
        except Exception as og_exc:
            fallback_errors.append(f"open_graph: {og_exc}")

        for source_group in ("instagram", "generic"):
            try:
                return _download_with_ytdlp(validated_url, max_bytes=max_bytes, source_group=source_group)
            except Exception as ytdlp_exc:
                fallback_errors.append(f"yt-dlp/{source_group}: {ytdlp_exc}")

        joined_errors = " | ".join(fallback_errors) if fallback_errors else "알 수 없는 오류"
        raise ValueError(
            "Instagram URL 처리 실패(Instaloader + OpenGraph + yt-dlp fallback). "
            f"{joined_errors}"
        )

    if _is_youtube_url(validated_url):
        youtube_errors: List[str] = []
        _debug_log("branch youtube")

        try:
            return _download_youtube_with_ytdlp_cli(validated_url, max_bytes=max_bytes)
        except Exception as cli_exc:
            youtube_errors.append(f"yt-dlp-cli: {cli_exc}")
            _debug_log(f"branch youtube cli_fail err={_summarize_err_text(cli_exc)}")
            if _is_login_or_rate_limit_error(str(cli_exc)):
                raise ValueError(_login_or_rate_limit_detail(validated_url)) from cli_exc

        try:
            return _download_with_ytdlp(validated_url, max_bytes=max_bytes, source_group="youtube")
        except Exception as ytdlp_exc:
            youtube_errors.append(f"yt-dlp/youtube: {ytdlp_exc}")
            _debug_log(f"branch youtube api_fail err={_summarize_err_text(ytdlp_exc)}")
            if _is_login_or_rate_limit_error(str(ytdlp_exc)):
                raise ValueError(_login_or_rate_limit_detail(validated_url)) from ytdlp_exc
            raise ValueError(
                "YouTube URL 처리 실패(yt-dlp CLI + yt-dlp API fallback). "
                + " | ".join(youtube_errors)
            ) from ytdlp_exc

    # 기타 웹사이트: 직링크면 direct, 아니면 yt-dlp generic
    if _looks_like_direct_media_url(validated_url):
        try:
            return _download_direct_media(validated_url, max_bytes=max_bytes)
        except Exception:
            pass

    return _download_with_ytdlp(validated_url, max_bytes=max_bytes, source_group="generic")
