"""
URL 기반 미디어(이미지/영상) 다운로드 서비스.

지원 분기:
1) YouTube Shorts URL 동영상 추론(pytubefix)
2) YouTube 일반 URL 동영상/이미지 추론(yt-dlp)
3) Instagram URL 동영상/이미지 추론(Instaloader + OpenGraph + yt-dlp fallback)
4) 기타 웹사이트 URL 동영상/이미지 추론(직접 다운로드 또는 yt-dlp)
"""

import glob
import json
import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
_VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp", "ts"}
_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


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
YTDLP_COOKIEFILE = (os.getenv("YTDLP_COOKIEFILE") or "").strip()
YTDLP_YOUTUBE_CLIENTS = _env_csv("YTDLP_YOUTUBE_CLIENTS", "android,web,tv_embedded")
YTDLP_YOUTUBE_ALT_CLIENTS = _env_csv("YTDLP_YOUTUBE_ALT_CLIENTS", "ios,mweb,web,web_safari")
PYTUBEFIX_YOUTUBE_CLIENTS = _env_csv("PYTUBEFIX_YOUTUBE_CLIENTS", "ANDROID,WEB,IOS,MWEB")

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


def _is_youtube_shorts_url(source_url: str) -> bool:
    if not _is_youtube_url(source_url):
        return False
    path = (urlparse(source_url).path or "").lower()
    return path.startswith("/shorts/")


def _is_instagram_url(source_url: str) -> bool:
    host = _host_from_url(source_url)
    return _host_matches(host, "instagram.com")


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


def _load_pytubefix_youtube_cls():
    try:
        from pytubefix import YouTube
    except Exception as exc:
        raise RuntimeError(
            "pytubefix가 설치되어 있지 않습니다. `pip install -r backend/requirements.txt` 후 서버를 재시작하세요."
        ) from exc
    return YouTube


def _pytubefix_client_candidates() -> List[str]:
    clients: List[str] = []
    seen: set[str] = set()
    for raw in PYTUBEFIX_YOUTUBE_CLIENTS:
        token = str(raw or "").strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        clients.append(token)
    if not clients:
        clients = ["ANDROID", "WEB", "IOS", "MWEB"]
    return clients


def _build_pytubefix_instance(YouTube, watch_url: str, client: str):
    # pytubefix 버전별 ctor 시그니처 차이를 흡수한다.
    kwargs: Dict[str, Any] = {}
    try:
        import inspect

        sig = inspect.signature(YouTube)
        if "client" in sig.parameters:
            kwargs["client"] = client
        if "use_oauth" in sig.parameters:
            kwargs["use_oauth"] = False
        if "allow_oauth_cache" in sig.parameters:
            kwargs["allow_oauth_cache"] = False
    except Exception:
        kwargs["client"] = client

    return YouTube(watch_url, **kwargs)


def _first_non_empty_stream(candidates: List[Any]) -> Any:
    for item in candidates:
        if item is not None:
            return item
    return None


def _select_pytubefix_video_stream(yt) -> Any:
    query = yt.streams
    candidates: List[Any] = []

    selectors = [
        lambda q: q.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first(),
        lambda q: q.filter(progressive=True).order_by("resolution").desc().first(),
        lambda q: q.filter(adaptive=True, only_video=True, file_extension="mp4").order_by("resolution").desc().first(),
        lambda q: q.filter(adaptive=True, only_video=True).order_by("resolution").desc().first(),
        lambda q: q.filter(file_extension="mp4").order_by("resolution").desc().first(),
        lambda q: q.order_by("resolution").desc().first(),
    ]

    for pick in selectors:
        try:
            candidates.append(pick(query))
        except Exception:
            candidates.append(None)

    stream = _first_non_empty_stream(candidates)
    if stream is not None:
        return stream

    try:
        all_streams = list(query)
    except Exception:
        all_streams = []

    best = None
    best_key: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    for s in all_streams:
        has_video = bool(getattr(s, "includes_video_track", False))
        if not has_video:
            continue
        subtype = str(getattr(s, "subtype", "") or "").lower()
        ext_rank = 2 if subtype == "mp4" else 1 if subtype == "webm" else 0
        progressive = 1 if bool(getattr(s, "is_progressive", False)) else 0
        has_audio = 1 if bool(getattr(s, "includes_audio_track", False)) else 0
        res_text = str(getattr(s, "resolution", "") or "")
        try:
            res_num = int(re.sub(r"[^0-9]", "", res_text) or 0)
        except Exception:
            res_num = 0
        key = (progressive, has_audio, ext_rank, res_num)
        if key > best_key:
            best_key = key
            best = s
    return best


def _download_youtube_thumbnail_via_pytubefix(yt, source_url: str, video_id: str, max_bytes: int) -> DownloadedMedia:
    thumb_url = str(getattr(yt, "thumbnail_url", "") or "").strip()
    if not thumb_url:
        raise ValueError("pytubefix 썸네일 URL을 찾지 못했습니다.")

    content, content_type = _stream_download_bytes(
        url=thumb_url,
        max_bytes=max_bytes,
        timeout_sec=URL_MEDIA_TIMEOUT_SEC,
        session=None,
        headers={"User-Agent": _HTTP_USER_AGENT, "Referer": "https://www.youtube.com/"},
    )
    ext = _path_ext_from_url(thumb_url) or _content_ext_from_type(content_type) or "jpg"
    title = str(getattr(yt, "title", "") or "").strip()
    filename = f"youtube_{video_id}_thumbnail.{ext}"
    return DownloadedMedia(
        source_url=source_url,
        media_type="image",
        filename=filename,
        content=content,
        extractor="youtube_shorts_pytubefix_thumbnail_fallback",
        title=title,
    )


def _download_youtube_shorts_via_pytubefix(source_url: str, max_bytes: int) -> DownloadedMedia:
    video_id = _extract_youtube_video_id(source_url)
    if not video_id:
        raise ValueError("YouTube Shorts URL에서 video id를 추출하지 못했습니다.")

    YouTube = _load_pytubefix_youtube_cls()
    watch_url = f"https://www.youtube.com/watch?v={video_id}"

    last_error: Optional[Exception] = None
    last_yt = None
    for client in _pytubefix_client_candidates():
        with tempfile.TemporaryDirectory(prefix=f"yt-shorts-{client.lower()}-") as tmp_dir:
            try:
                yt = _build_pytubefix_instance(YouTube, watch_url=watch_url, client=client)
                last_yt = yt
            except Exception as exc:
                last_error = exc
                continue

            try:
                stream = _select_pytubefix_video_stream(yt)
            except Exception as exc:
                last_error = exc
                stream = None

            if stream is None:
                last_error = ValueError(f"client={client}: 다운로드 가능한 스트림을 찾지 못했습니다.")
                continue

            subtype = str(getattr(stream, "subtype", "") or "").strip().lower() or "mp4"
            filename_hint = f"{video_id}.{subtype}"
            try:
                downloaded_path = stream.download(output_path=tmp_dir, filename=filename_hint)
            except Exception as exc:
                last_error = exc
                continue

            file_path = str(downloaded_path or "").strip()
            if not file_path or not os.path.isfile(file_path):
                file_candidates = sorted(glob.glob(os.path.join(tmp_dir, "*")))
                file_path = next((p for p in file_candidates if os.path.isfile(p)), "")
            if not file_path or not os.path.isfile(file_path):
                last_error = ValueError("pytubefix 다운로드 파일을 찾지 못했습니다.")
                continue

            size = os.path.getsize(file_path)
            if size <= 0:
                last_error = ValueError("다운로드된 미디어 파일이 비어 있습니다.")
                continue
            if size > max_bytes:
                raise ValueError(f"다운로드한 미디어가 제한 용량({URL_MEDIA_MAX_MB}MB)을 초과했습니다.")

            with open(file_path, "rb") as fp:
                content = fp.read()

            title = str(getattr(yt, "title", "") or "").strip()
            filename = os.path.basename(file_path)
            return DownloadedMedia(
                source_url=source_url,
                media_type="video",
                filename=filename,
                content=content,
                extractor=f"youtube_shorts_pytubefix_{client.lower()}",
                title=title,
            )

    if last_yt is not None:
        try:
            return _download_youtube_thumbnail_via_pytubefix(
                yt=last_yt,
                source_url=source_url,
                video_id=video_id,
                max_bytes=max_bytes,
            )
        except Exception as thumb_exc:
            last_error = thumb_exc

    raise ValueError(f"YouTube Shorts에서 다운로드 가능한 스트림을 찾지 못했습니다. last_error={last_error}")


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
# yt-dlp helpers (YouTube / Generic)
# =========================

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
        if ":" in item:
            name, path = item.split(":", 1)
            name = name.strip()
            path = path.strip()
            if not name:
                continue
            out[name] = {"path": path} if path else {}
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


def _apply_cookiefile_option(opts: Dict[str, Any], tmp_dir: str) -> None:
    if not YTDLP_COOKIEFILE:
        return

    if not os.path.isfile(YTDLP_COOKIEFILE):
        raise ValueError(f"YTDLP_COOKIEFILE 파일을 찾을 수 없습니다: {YTDLP_COOKIEFILE}")

    req_cookie_path = os.path.join(tmp_dir, "yt_cookies.txt")
    try:
        shutil.copyfile(YTDLP_COOKIEFILE, req_cookie_path)
    except Exception as exc:
        raise ValueError(f"YTDLP_COOKIEFILE 복사 실패: {exc}") from exc
    opts["cookiefile"] = req_cookie_path


def _apply_js_runtime_option(opts: Dict[str, Any]) -> None:
    if YTDLP_JS_RUNTIMES and isinstance(YTDLP_JS_RUNTIMES, dict):
        opts["js_runtimes"] = YTDLP_JS_RUNTIMES


def _apply_youtube_extractor_option(opts: Dict[str, Any], player_clients: Optional[List[str]] = None) -> None:
    clients = [str(c or "").strip() for c in (player_clients or YTDLP_YOUTUBE_CLIENTS) if str(c or "").strip()]
    if not clients:
        return

    extractor_args = dict(opts.get("extractor_args") or {})
    yt_args = dict(extractor_args.get("youtube") or {})
    yt_args["player_client"] = clients
    extractor_args["youtube"] = yt_args
    opts["extractor_args"] = extractor_args


def _youtube_client_profiles() -> List[List[str]]:
    profiles: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()

    for raw in (YTDLP_YOUTUBE_CLIENTS, YTDLP_YOUTUBE_ALT_CLIENTS):
        profile = [str(c or "").strip() for c in raw if str(c or "").strip()]
        if not profile:
            continue
        key = tuple(profile)
        if key in seen:
            continue
        seen.add(key)
        profiles.append(profile)

    if not profiles:
        profiles.append(["android", "web", "tv_embedded"])
    return profiles


def _build_youtube_option_variants(base_opts: Dict[str, Any], tmp_dir: str) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    profiles = _youtube_client_profiles()

    if YTDLP_COOKIEFILE:
        for clients in profiles:
            opts = dict(base_opts)
            try:
                _apply_cookiefile_option(opts, tmp_dir)
            except Exception:
                continue
            _apply_youtube_extractor_option(opts, player_clients=clients)
            variants.append(opts)

    for clients in profiles:
        opts = dict(base_opts)
        _apply_youtube_extractor_option(opts, player_clients=clients)
        variants.append(opts)

    # extractor_args를 완전히 제거한 raw 변형도 마지막에 시도
    raw_opts = dict(base_opts)
    if YTDLP_COOKIEFILE:
        try:
            _apply_cookiefile_option(raw_opts, tmp_dir)
        except Exception:
            pass
    variants.append(raw_opts)

    return variants


def _build_format_candidates(source_group: str, source_url: str) -> List[Optional[str]]:
    low = source_url.lower()
    if source_group == "youtube":
        return [
            None,
            "best",
            "best[ext=mp4]/best[ext=webm]/best",
            "bv*+ba/b",
            "bestvideo*+bestaudio/bestvideo+bestaudio/best",
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
    ]
    return any(keyword in low for keyword in keywords)


def _is_ffmpeg_merge_error(message: str) -> bool:
    low = str(message or "").lower()
    return "ffmpeg is not installed" in low or "requested merging of multiple formats" in low


def _login_or_rate_limit_detail(source_url: str = "") -> str:
    if _is_youtube_url(source_url):
        return (
            "유튜브 접근이 제한되었습니다(봇 확인/로그인 필요). "
            "공개 접근 프로필로 재시도했지만 실패했습니다. "
            "유효한 YouTube 쿠키를 재발급해 YTDLP_COOKIEFILE로 설정해 주세요."
        )
    if _is_instagram_url(source_url):
        return (
            "인스타그램 접근이 제한되었습니다(로그인/레이트리밋). "
            "INSTAGRAM_SESSION_ID 또는 YTDLP_COOKIEFILE 설정을 확인해 주세요."
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
        if _is_youtube_url(source_url):
            headers["Referer"] = "https://www.youtube.com/"
        elif _is_instagram_url(source_url):
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
            if source_group == "youtube":
                extractor = "youtube_ytdlp_format_url"
            elif source_group == "instagram":
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
    treat_as_youtube = _is_youtube_url(source_url)

    with tempfile.TemporaryDirectory(prefix="url-media-") as tmp_dir:
        base_opts: Dict[str, Any] = {
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
            "http_headers": {"User-Agent": _HTTP_USER_AGENT},
            "retries": 2,
            "fragment_retries": 2,
        }
        _apply_js_runtime_option(base_opts)

        format_candidates = _build_format_candidates(source_group=source_group, source_url=source_url)

        option_variants: List[Dict[str, Any]]
        if source_group == "youtube" or treat_as_youtube:
            option_variants = _build_youtube_option_variants(base_opts=base_opts, tmp_dir=tmp_dir)
        else:
            opts = dict(base_opts)
            _apply_cookiefile_option(opts, tmp_dir)
            option_variants = [opts]

        info = None
        last_error: Optional[Exception] = None
        saw_login_error = False
        for variant_opts in option_variants:
            for fmt in format_candidates:
                try:
                    opts = dict(variant_opts)
                    if fmt:
                        opts["format"] = fmt
                    else:
                        opts.pop("format", None)
                    with YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(source_url, download=True)
                    break
                except Exception as exc:
                    last_error = exc
                    msg = str(exc)
                    if _is_format_unavailable_error(msg):
                        continue
                    if _is_ffmpeg_merge_error(msg):
                        continue
                    if _is_login_or_rate_limit_error(msg):
                        saw_login_error = True
                        break
                    raise ValueError(f"URL에서 미디어를 가져오지 못했습니다: {exc}") from exc
            if info is not None:
                break

        if info is None and treat_as_youtube:
            # format expression이 전부 실패한 경우 메타데이터의 실제 스트림 URL로 한 번 더 시도
            for variant_opts in option_variants:
                try:
                    meta_opts = dict(variant_opts)
                    meta_opts["skip_download"] = True
                    meta_opts["simulate"] = True
                    meta_opts.pop("format", None)
                    with YoutubeDL(meta_opts) as ydl:
                        meta_info = ydl.extract_info(source_url, download=False)
                    meta_entry = _pick_primary_entry(meta_info)
                    if meta_entry:
                        return _download_by_entry_format_urls(
                            source_url=source_url,
                            entry=meta_entry,
                            max_bytes=max_bytes,
                            source_group=source_group,
                        )
                except Exception as exc:
                    last_error = exc
                    continue

        if info is None:
            if saw_login_error or _is_login_or_rate_limit_error(str(last_error or "")):
                raise ValueError(_login_or_rate_limit_detail(source_url))
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
            if source_group == "youtube":
                extractor = "youtube_ytdlp"
            elif source_group == "instagram":
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
    - YouTube Shorts URL: pytubefix 전용 경로
    - YouTube 일반 URL: yt-dlp 전용 경로
    - Instagram URL: Instaloader 우선 + OpenGraph + yt-dlp fallback 경로
    - 기타 URL: direct-http 우선, 실패 시 yt-dlp generic 경로
    """
    validated_url = _validate_source_url(source_url)
    max_bytes = URL_MEDIA_MAX_MB * 1024 * 1024

    if _is_instagram_url(validated_url):
        fallback_errors: List[str] = []

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
        if _is_youtube_shorts_url(validated_url):
            try:
                return _download_youtube_shorts_via_pytubefix(validated_url, max_bytes=max_bytes)
            except Exception as shorts_exc:
                if _is_login_or_rate_limit_error(str(shorts_exc)):
                    raise ValueError(_login_or_rate_limit_detail(validated_url)) from shorts_exc
                raise ValueError(f"YouTube Shorts 처리 실패(pytubefix): {shorts_exc}") from shorts_exc

        try:
            return _download_with_ytdlp(validated_url, max_bytes=max_bytes, source_group="youtube")
        except Exception as ytdlp_exc:
            try:
                return _download_with_ytdlp(validated_url, max_bytes=max_bytes, source_group="generic")
            except Exception as generic_exc:
                if _is_login_or_rate_limit_error(str(ytdlp_exc)) or _is_login_or_rate_limit_error(str(generic_exc)):
                    raise ValueError(_login_or_rate_limit_detail(validated_url)) from generic_exc
                raise ValueError(
                    "YouTube URL 처리 실패(yt-dlp youtube + generic fallback). "
                    f"youtube: {ytdlp_exc}; generic: {generic_exc}"
                ) from generic_exc

    # 기타 웹사이트: 직링크면 direct, 아니면 yt-dlp generic
    if _looks_like_direct_media_url(validated_url):
        try:
            return _download_direct_media(validated_url, max_bytes=max_bytes)
        except Exception:
            pass

    return _download_with_ytdlp(validated_url, max_bytes=max_bytes, source_group="generic")
