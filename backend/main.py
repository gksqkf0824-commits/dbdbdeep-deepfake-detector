from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import secrets
import base64
import os
import mimetypes
import tempfile
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
try:
    import yt_dlp
except Exception:
    yt_dlp = None

from model import detector
from redis_client import redis_db

from utils import (
    sha256_bytes,
    redis_get_json,
    redis_set_json,
    video_to_uniform_sampled_frames,
    aggregate_scores,
    trimmed_mean_confidence,
    build_analysis_result,
    detect_faces_with_aligned_crops,
    get_cam_target_layer,
    GradCAM,
    build_evidence_for_face,
    explain_from_evidence,
    generate_video_ai_comment,
)

app = FastAPI()

# --- Redis 연결 체크 ---
@app.on_event("startup")
def check_redis_connection():
    try:
        redis_db.ping()
        print("✅ Redis 연결 성공! (준비 완료)")
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        print("   👉 Docker가 켜져 있는지, 'docker run -p 6379:6379 -d redis'를 했는지 확인하세요!")

# --- CORS ---
# NOTE: allow_credentials=True + allow_origins=["*"] 조합은 브라우저/환경에 따라 문제가 될 수 있음.
# (일단 기존 유지. 운영에서는 도메인을 명시하는 게 안전)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
REAL_MEAN = 15.0
REAL_STD = 8.0

# Video sampling
VIDEO_MAX_SIDE = 640
VIDEO_MIN_FRAMES = 12
VIDEO_MAX_FRAMES_CAP = 48
VIDEO_FRAMES_PER_MINUTE = 24

# Aggregation
AGG_MODE_VIDEO = "mean"
TOPK = 5
VIDEO_TRIM_RATIO = 0.10

# Redis TTL
RESULT_TTL_SEC = 3600
CACHE_TTL_SEC = 24 * 3600
REMOTE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
REMOTE_VIDEO_MAX_BYTES = 200 * 1024 * 1024
REMOTE_IMAGE_TIMEOUT_SEC = 10
REMOTE_VIDEO_TIMEOUT_SEC = 45
YTDLP_COOKIEFILE = (os.getenv("YTDLP_COOKIEFILE") or "").strip()
REMOTE_COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def store_result_and_make_response(analysis_result: dict, stored_result: dict = None) -> dict:
    """
    결과를 Redis(res:{token})에 저장하고 프론트가 쓰는 형태로 반환.
    """
    token = secrets.token_urlsafe(16)
    payload_for_store = stored_result if stored_result is not None else analysis_result
    redis_set_json(redis_db, f"res:{token}", payload_for_store, ex=RESULT_TTL_SEC)
    return {
        "result_url": f"http://127.0.0.1:8000/get-result/{token}",
        "data": analysis_result,
    }


def has_frame_series(payload: dict, key: str) -> bool:
    values = payload.get(key)
    return isinstance(values, list) and len(values) >= 2


def delete_keys_by_patterns(patterns, batch_size: int = 500) -> int:
    deleted_total = 0

    for pattern in patterns:
        batch = []
        for key in redis_db.scan_iter(match=pattern, count=1000):
            batch.append(key)
            if len(batch) >= batch_size:
                deleted_total += int(redis_db.delete(*batch))
                batch.clear()

        if batch:
            deleted_total += int(redis_db.delete(*batch))

    return deleted_total


def _validate_evidence_level(level: str) -> str:
    lv = (level or "mvp").strip().lower()
    if lv not in {"off", "mvp", "full"}:
        raise HTTPException(status_code=400, detail="evidence_level은 off/mvp/full 중 하나여야 합니다.")
    return lv


def _safe_score_agg(values):
    return float(max(values)) if values else 0.0


def _bytes_to_data_url(payload: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _frame_to_preview_data_url(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("프리뷰 프레임 인코딩 실패")
    return _bytes_to_data_url(buf.tobytes(), "image/jpeg")


def _read_response_content_limited(resp: requests.Response, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=1024 * 512):
        if not chunk:
            continue
        total += len(chunk)
        if total > int(max_bytes):
            raise HTTPException(status_code=413, detail=f"다운로드 파일이 너무 큽니다. (최대 {max_bytes // (1024 * 1024)}MB)")
        chunks.append(chunk)
    return b"".join(chunks)


def _filename_from_url(url: str, fallback: str = "media.bin") -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path or "").strip()
    return name or fallback


def _is_likely_social_video_url(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(domain in host for domain in ("youtube.com", "youtu.be", "instagram.com"))


def _is_likely_ext(path: str, ext_set: set) -> bool:
    path = (path or "").lower()
    return any(path.endswith(ext) for ext in ext_set)


def _pick_ytdlp_primary_info(info: dict) -> dict:
    if isinstance(info, dict) and str(info.get("_type", "")).lower() == "playlist":
        entries = info.get("entries") or []
        for entry in entries:
            if isinstance(entry, dict):
                return entry
    return info if isinstance(info, dict) else {}


def _resolve_ytdlp_downloaded_path(ydl: Any, info: dict, workdir: str) -> Optional[str]:
    requested = info.get("requested_downloads")
    if isinstance(requested, list):
        for item in requested:
            if not isinstance(item, dict):
                continue
            cand = item.get("filepath") or item.get("_filename")
            if isinstance(cand, str) and os.path.exists(cand):
                return cand

    try:
        cand = ydl.prepare_filename(info)
        if isinstance(cand, str) and os.path.exists(cand):
            return cand
    except Exception:
        pass

    candidates = []
    for root, _, files in os.walk(workdir):
        for name in files:
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            candidates.append((size, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _humanize_ytdlp_error(raw_msg: str) -> str:
    msg = (raw_msg or "").strip()
    low = msg.lower()

    if "unsupported url" in low:
        return "지원하지 않는 URL입니다. 원본 이미지/영상 링크 또는 공개된 Shorts/Reels 링크를 입력해 주세요."
    if "sign in to confirm you're not a bot" in low:
        return (
            "YouTube가 자동화 접근을 차단했습니다. "
            "서버에 yt-dlp cookie 설정(예: cookiefile)을 추가하거나 다른 공개 URL로 시도해 주세요."
        )
    if "instagram" in low and "unable to extract video url" in low:
        return (
            "Instagram URL에서 영상 주소를 추출하지 못했습니다. "
            "공개 게시물인지 확인하고, 필요 시 서버의 yt-dlp/cookie 설정을 점검해 주세요."
        )
    return msg or "알 수 없는 yt-dlp 오류"


def _pick_stream_url_from_info(info: dict) -> Optional[str]:
    if not isinstance(info, dict):
        return None

    direct = info.get("url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    formats = info.get("formats")
    if isinstance(formats, list):
        # progressive(mp4) 형태를 우선 선택, 없으면 가장 큰 비트레이트 후보 사용.
        scored = []
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            cand = fmt.get("url")
            if not isinstance(cand, str) or not cand.strip():
                continue
            ext = str(fmt.get("ext") or "").lower()
            vcodec = str(fmt.get("vcodec") or "")
            acodec = str(fmt.get("acodec") or "")
            tbr = float(fmt.get("tbr") or 0.0)
            prefer_progressive = 1 if vcodec != "none" and acodec != "none" else 0
            prefer_mp4 = 1 if ext == "mp4" else 0
            scored.append(((prefer_progressive, prefer_mp4, tbr), cand.strip()))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
    return None


def _download_remote_bytes(url: str, timeout_sec: int, max_bytes: int) -> tuple[bytes, str]:
    with requests.get(
        url,
        stream=True,
        timeout=timeout_sec,
        allow_redirects=True,
        headers=REMOTE_COMMON_HEADERS,
    ) as resp:
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        data = _read_response_content_limited(resp, max_bytes)
    if not data:
        raise HTTPException(status_code=400, detail="다운로드한 미디어 데이터가 비어 있습니다.")
    return data, content_type


def _download_media_with_ytdlp(url: str) -> Dict[str, Any]:
    if yt_dlp is None:
        raise HTTPException(
            status_code=500,
            detail="yt-dlp가 설치되지 않았습니다. backend/requirements.txt 설치 후 다시 시도해 주세요.",
        )

    with tempfile.TemporaryDirectory(prefix="url_media_") as tmp_dir:
        outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        base_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": REMOTE_VIDEO_TIMEOUT_SEC,
            "outtmpl": outtmpl,
            "restrictfilenames": True,
            "merge_output_format": "mp4",
            "geo_bypass": True,
            "retries": 2,
            "extractor_retries": 2,
            "fragment_retries": 2,
            "http_headers": REMOTE_COMMON_HEADERS,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        }
        if YTDLP_COOKIEFILE and os.path.exists(YTDLP_COOKIEFILE):
            base_opts["cookiefile"] = YTDLP_COOKIEFILE
        # 플랫폼/코덱 조합별로 포맷 가용성이 다를 수 있어 순차 fallback.
        format_candidates = [
            "bestvideo*+bestaudio/best",
            "best/bestvideo+bestaudio",
            None,
        ]

        info = {}
        downloaded_path = None
        cert_error_seen = False
        last_exc: Optional[Exception] = None

        for fmt in format_candidates:
            attempt_opts = dict(base_opts)
            if fmt:
                attempt_opts["format"] = fmt
            if cert_error_seen:
                # 런타임 CA 체인 이슈가 있는 환경 fallback.
                attempt_opts["nocheckcertificate"] = True

            try:
                with yt_dlp.YoutubeDL(attempt_opts) as ydl:
                    raw_info = ydl.extract_info(url, download=True)
                    info = _pick_ytdlp_primary_info(raw_info)
                    downloaded_path = _resolve_ytdlp_downloaded_path(ydl, info, tmp_dir)
                if downloaded_path and os.path.exists(downloaded_path):
                    break
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if "CERTIFICATE_VERIFY_FAILED" in msg or "certificate verify failed" in msg.lower():
                    cert_error_seen = True
                continue

        if not downloaded_path or not os.path.exists(downloaded_path):
            # 일부 환경에서는 download=True만 실패하고 metadata 추출(download=False)은 가능한 경우가 있어 fallback 시도.
            try:
                fallback_opts = dict(base_opts)
                fallback_opts["skip_download"] = True
                fallback_opts["format"] = "best[ext=mp4]/best"
                with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                    raw_info = ydl.extract_info(url, download=False)
                    info = _pick_ytdlp_primary_info(raw_info)
                stream_url = _pick_stream_url_from_info(info)
                if stream_url:
                    ext = str(info.get("ext") or "").lower()
                    vcodec = str(info.get("vcodec") or "").lower()
                    is_video = (vcodec not in {"", "none"}) or (f".{ext}" in VIDEO_EXTS)
                    max_bytes = REMOTE_VIDEO_MAX_BYTES if is_video else REMOTE_IMAGE_MAX_BYTES
                    timeout = REMOTE_VIDEO_TIMEOUT_SEC if is_video else REMOTE_IMAGE_TIMEOUT_SEC
                    media_bytes, content_type = _download_remote_bytes(stream_url, timeout, max_bytes)
                    mime_type = content_type or (f"video/{ext}" if is_video and ext else f"image/{ext}" if ext else "")
                    if is_video and not mime_type.startswith("video/"):
                        mime_type = "video/mp4"
                    if (not is_video) and not mime_type.startswith("image/"):
                        mime_type = "image/jpeg"
                    thumbnail_url = info.get("thumbnail") if isinstance(info, dict) else None
                    title = str(info.get("title") or "").strip() if isinstance(info, dict) else ""
                    return {
                        "media_type": "video" if is_video else "image",
                        "mime_type": mime_type,
                        "bytes": media_bytes,
                        "filename": _filename_from_url(url, fallback=f"downloaded_media.{ext or 'bin'}"),
                        "preview": {
                            "kind": "video" if is_video else "image",
                            "url": stream_url,
                            "thumbnail_url": str(thumbnail_url or "").strip() or None,
                            "page_url": url,
                            "title": title or None,
                        },
                    }
            except Exception:
                pass

            if last_exc is not None:
                user_msg = _humanize_ytdlp_error(str(last_exc))
                raise HTTPException(status_code=400, detail=f"URL 미디어 다운로드 실패(yt-dlp): {user_msg}") from last_exc
            raise HTTPException(status_code=400, detail="yt-dlp 다운로드 결과 파일을 찾지 못했습니다.")

        # cert 오류를 겪은 경우, 마지막 fallback로 인증서 검증 비활성화 1회 재시도
        # (상단 포맷 루프에서 cert_error_seen 시 이미 nocheckcertificate=True 로 재시도됨)

        mime_type, _ = mimetypes.guess_type(downloaded_path)
        mime_type = str(mime_type or "").lower()
        ext = os.path.splitext(downloaded_path)[1].lower()
        is_video = mime_type.startswith("video/") or ext in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
        media_type = "video" if is_video else "image"

        file_size = os.path.getsize(downloaded_path)
        size_limit = REMOTE_VIDEO_MAX_BYTES if is_video else REMOTE_IMAGE_MAX_BYTES
        if file_size > size_limit:
            raise HTTPException(
                status_code=413,
                detail=f"다운로드 파일이 너무 큽니다. ({size_limit // (1024 * 1024)}MB 이하)",
            )

        with open(downloaded_path, "rb") as f:
            media_bytes = f.read()

        stream_url = info.get("url") if isinstance(info, dict) else None
        thumbnail_url = info.get("thumbnail") if isinstance(info, dict) else None
        title = str(info.get("title") or "").strip() if isinstance(info, dict) else ""

        return {
            "media_type": media_type,
            "mime_type": mime_type,
            "bytes": media_bytes,
            "filename": os.path.basename(downloaded_path),
            "preview": {
                "kind": media_type,
                "url": str(stream_url or "").strip() or None,
                "thumbnail_url": str(thumbnail_url or "").strip() or None,
                "page_url": url,
                "title": title or None,
            },
        }


def _analyze_evidence_bytes(
    image_bytes: bytes,
    explain: bool = True,
    evidence_level: str = "mvp",
    fusion_w: float = 0.5,
    source_preview: Optional[Dict[str, Any]] = None,
    input_media_type: str = "image",
) -> dict:
    request_id = str(uuid.uuid4())
    lv = _validate_evidence_level(evidence_level)

    img_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    rgb_model = detector.pixel_model
    freq_model = detector.freq_model
    if rgb_model is None or freq_model is None:
        raise HTTPException(
            status_code=500,
            detail="RGB/Frequency 모델 로드 실패: backend/models/*.pth 경로를 확인하세요.",
        )

    try:
        faces = detect_faces_with_aligned_crops(
            image_bgr=bgr,
            margin=0.15,
            target_size=224,
            max_faces=1,
            prioritize_frontal=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"얼굴 분석 실패: {exc}") from exc

    if not faces:
        no_face_result = {
            "request_id": request_id,
            "status": "ok",
            "score": {"p_rgb": 0.0, "p_freq": 0.0, "p_final": 0.0},
            "faces": [],
            "ai_comment": "얼굴이 선명하게 보이지 않아 판독을 진행하지 못했습니다. 얼굴이 크게 보이는 이미지로 다시 시도해 주세요.",
            "ai_comment_source": "fallback:no_face",
        }
        no_face_result["input_media_type"] = str(input_media_type or "image")
        if source_preview:
            no_face_result["source_preview"] = source_preview
        return no_face_result

    cam_target_layer = get_cam_target_layer(rgb_model)
    cam = GradCAM(rgb_model, cam_target_layer)

    faces_out = []
    p_rgb_list, p_freq_list, p_final_list = [], [], []

    try:
        for i, face in enumerate(faces):
            out = build_evidence_for_face(
                face_rgb_uint8=face["crop_rgb"],
                landmarks=face["landmarks"],
                rgb_model=rgb_model,
                freq_model=freq_model,
                cam=cam,
                fusion_w=float(fusion_w),
                evidence_level=lv,
            )

            score = out["score"]
            evidence = out["evidence"]
            assets = out["assets"]

            p_rgb_list.append(float(score["p_rgb"]))
            p_freq_list.append(float(score["p_freq"]))
            p_final_list.append(float(score["p_final"]))

            item = {
                "face_id": i,
                "assets": assets,
                "evidence": evidence,
            }
            if explain:
                item["explanation"] = explain_from_evidence(
                    evidence=evidence,
                    score=score,
                    media_mode_hint="image",
                    use_openai=(i == 0),
                )

            faces_out.append(item)
    finally:
        cam.close()

    ai_comment = ""
    ai_comment_source = "rule_based"
    if explain and faces_out:
        first_explanation = faces_out[0].get("explanation", {})
        if isinstance(first_explanation, dict):
            ai_comment = str(first_explanation.get("summary", "")).strip()
            ai_comment_source = str(first_explanation.get("summary_source", "rule_based")).strip() or "rule_based"

    result = {
        "request_id": request_id,
        "status": "ok",
        "score": {
            "p_rgb": _safe_score_agg(p_rgb_list),
            "p_freq": _safe_score_agg(p_freq_list),
            "p_final": _safe_score_agg(p_final_list),
        },
        "faces": faces_out,
        "ai_comment": ai_comment,
        "ai_comment_source": ai_comment_source,
    }
    result["input_media_type"] = str(input_media_type or "image")
    if source_preview:
        result["source_preview"] = source_preview
    return result


@app.get("/test")
async def test():
    return {"message": "서버가 정상적으로 작동 중입니다."}


@app.post("/clear-cache")
async def clear_cache():
    """
    Redis 캐시 키 삭제.
    현재는 결과/비디오 캐시를 모두 정리한다.
    """
    try:
        patterns = ["cache:*", "res:*"]
        deleted_count = delete_keys_by_patterns(patterns)
        return {
            "message": "Redis cache cleared",
            "deleted_keys": deleted_count,
            "patterns": patterns,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {e}")


@app.post("/api/analyze")
@app.post("/api/analyze-evidence")
@app.post("/analyze-evidence")
async def analyze_with_evidence(
    file: UploadFile = File(...),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
    fusion_w: float = Form(0.5),
):
    data = await file.read()
    return _analyze_evidence_bytes(
        image_bytes=data,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=fusion_w,
    )


@app.post("/api/analyze-url")
@app.post("/analyze-url")
async def analyze_url_with_evidence(
    image_url: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
    fusion_w: float = Form(0.5),
):
    raw_url = (image_url or url or "").strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="분석할 URL을 입력해 주세요.")
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="유효한 http/https URL을 입력하세요.")

    target_url = parsed.geturl()

    # 1) 일반 이미지/비디오 직링크는 requests로 우선 처리
    if not _is_likely_social_video_url(target_url):
        try:
            with requests.get(
                target_url,
                stream=True,
                timeout=REMOTE_IMAGE_TIMEOUT_SEC,
                allow_redirects=True,
                headers=REMOTE_COMMON_HEADERS,
            ) as resp:
                resp.raise_for_status()
                content_type = (resp.headers.get("content-type") or "").lower()
                final_path = urlparse(str(resp.url or target_url)).path.lower()

                if content_type.startswith("image/") or _is_likely_ext(final_path, IMAGE_EXTS):
                    data = _read_response_content_limited(resp, REMOTE_IMAGE_MAX_BYTES)
                    if not data:
                        raise HTTPException(status_code=400, detail="다운로드한 이미지 데이터가 비어 있습니다.")
                    mime = content_type.split(";")[0].strip() or "image/jpeg"
                    source_preview = {
                        "kind": "image",
                        "url": target_url,
                        "data_url": _bytes_to_data_url(data, mime),
                        "page_url": target_url,
                    }
                    return _analyze_evidence_bytes(
                        image_bytes=data,
                        explain=explain,
                        evidence_level=evidence_level,
                        fusion_w=fusion_w,
                        source_preview=source_preview,
                        input_media_type="image",
                    )

                if content_type.startswith("video/") or _is_likely_ext(final_path, VIDEO_EXTS):
                    data = _read_response_content_limited(resp, REMOTE_VIDEO_MAX_BYTES)
                    if not data:
                        raise HTTPException(status_code=400, detail="다운로드한 영상 데이터가 비어 있습니다.")
                    source_preview = {
                        "kind": "video",
                        "url": target_url,
                        "page_url": target_url,
                    }
                    return _analyze_video_bytes(
                        content=data,
                        filename=_filename_from_url(target_url, fallback="remote_video.mp4"),
                        source_preview=source_preview,
                    )
        except HTTPException:
            raise
        except requests.RequestException:
            pass

    # 2) 인스타 릴스 / 유튜브 쇼츠 등은 yt-dlp로 처리
    media = _download_media_with_ytdlp(target_url)
    media_type = str(media.get("media_type") or "").lower()
    payload = media.get("bytes") or b""
    filename = str(media.get("filename") or _filename_from_url(target_url, "downloaded_media.bin"))
    source_preview = dict(media.get("preview") or {})

    if media_type == "image":
        mime = str(media.get("mime_type") or "image/jpeg")
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        source_preview["kind"] = "image"
        source_preview["page_url"] = source_preview.get("page_url") or target_url
        source_preview["data_url"] = _bytes_to_data_url(payload, mime)
        if not source_preview.get("url"):
            source_preview["url"] = target_url
        return _analyze_evidence_bytes(
            image_bytes=payload,
            explain=explain,
            evidence_level=evidence_level,
            fusion_w=fusion_w,
            source_preview=source_preview,
            input_media_type="image",
        )

    source_preview["kind"] = "video"
    source_preview["page_url"] = source_preview.get("page_url") or target_url
    if not source_preview.get("url"):
        source_preview["url"] = target_url
    return _analyze_video_bytes(
        content=payload,
        filename=filename,
        source_preview=source_preview,
    )


# =========================
# Image inference
# =========================
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        score, pixel_score, freq_score, preprocessed = detector.predict(
            image_bytes,
            include_preprocess=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Error: {e}")

    analysis_result = build_analysis_result(
        score, pixel_score, freq_score,
        real_mean=REAL_MEAN, real_std=REAL_STD
    )
    if preprocessed is not None:
        analysis_result["preprocessed"] = preprocessed

    stored_result = dict(analysis_result)
    stored_result.pop("preprocessed", None)
    return store_result_and_make_response(analysis_result, stored_result=stored_result)


# =========================
# Video inference
# =========================
def _analyze_video_bytes(
    content: bytes,
    filename: str,
    source_preview: Optional[Dict[str, Any]] = None,
) -> dict:
    video_hash = sha256_bytes(content)
    video_cache_key = f"cache:video:{video_hash}"

    cached = redis_get_json(redis_db, video_cache_key)
    if cached is not None and has_frame_series(cached, "video_frame_pixel_scores") and has_frame_series(cached, "video_frame_freq_scores"):
        cached_response = dict(cached)
        cached_response["input_media_type"] = "video"
        if source_preview:
            cached_response["source_preview"] = source_preview
        return store_result_and_make_response(cached_response)

    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        frames, meta = video_to_uniform_sampled_frames(
            tmp_path,
            max_side=VIDEO_MAX_SIDE,
            min_frames=VIDEO_MIN_FRAMES,
            max_frames=VIDEO_MAX_FRAMES_CAP,
            frames_per_minute=VIDEO_FRAMES_PER_MINUTE,
        )
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="비디오에서 프레임을 추출하지 못했습니다.")

        if source_preview is not None and not source_preview.get("thumbnail_data_url"):
            try:
                source_preview["thumbnail_data_url"] = _frame_to_preview_data_url(frames[0])
            except Exception:
                pass

        scores, pixel_scores, freq_scores = [], [], []
        successful_frames = []
        successful_frame_indices = []
        failed = 0

        for frame_idx, fr in enumerate(frames):
            try:
                score, p_score, f_score, _ = detector.predict_from_bgr(
                    fr,
                    include_preprocess=False,
                )
                scores.append(score)
                pixel_scores.append(p_score)
                freq_scores.append(f_score)
                successful_frames.append(fr)
                successful_frame_indices.append(frame_idx)
            except Exception:
                failed += 1
                continue

        if len(scores) == 0:
            raise HTTPException(
                status_code=500,
                detail=f"모든 프레임 추론 실패 (sampled={len(frames)}, failed={failed})."
            )

        video_score, trimmed_meta = trimmed_mean_confidence(
            scores,
            trim_ratio=VIDEO_TRIM_RATIO,
        )
        video_pixel = aggregate_scores(pixel_scores, mode=AGG_MODE_VIDEO, topk=TOPK)
        video_freq = aggregate_scores(freq_scores, mode=AGG_MODE_VIDEO, topk=TOPK)

        if video_score is None:
            raise HTTPException(status_code=500, detail="영상 점수 집계 실패")

        analysis_result = build_analysis_result(
            video_score, video_pixel, video_freq,
            real_mean=REAL_MEAN, real_std=REAL_STD
        )
        analysis_result["video_representative_confidence"] = round(float(video_score), 2)
        analysis_result["video_frame_confidences"] = [round(float(s), 2) for s in scores]
        analysis_result["video_frame_pixel_scores"] = [round(float(s), 2) for s in pixel_scores]
        analysis_result["video_frame_freq_scores"] = [round(float(s), 2) for s in freq_scores]

        analysis_result["video_meta"] = {
            "used_frames": len(scores),
            "failed_frames": failed,
            "agg_mode": "Trimmed Mean 10 Percent",
            "pixel_freq_agg_mode": AGG_MODE_VIDEO,
            "topk": TOPK,
        }
        analysis_result["video_meta"].update(trimmed_meta)
        analysis_result["video_meta"].update(meta)

        representative_payload = None
        try:
            if successful_frames and detector.pixel_model is not None and detector.freq_model is not None:
                score_arr = np.asarray(scores, dtype=np.float32)
                rep_pos = int(np.argmin(np.abs(score_arr - float(video_score))))
                rep_frame_bgr = successful_frames[rep_pos]
                rep_sample_index = int(successful_frame_indices[rep_pos])
                rep_score = float(scores[rep_pos])

                rep_faces = detect_faces_with_aligned_crops(
                    image_bgr=rep_frame_bgr,
                    margin=0.15,
                    target_size=224,
                    max_faces=1,
                    prioritize_frontal=False,
                )

                if rep_faces:
                    rep_cam = GradCAM(detector.pixel_model, get_cam_target_layer(detector.pixel_model))
                    try:
                        rep_out = build_evidence_for_face(
                            face_rgb_uint8=rep_faces[0]["crop_rgb"],
                            landmarks=rep_faces[0]["landmarks"],
                            rgb_model=detector.pixel_model,
                            freq_model=detector.freq_model,
                            cam=rep_cam,
                            fusion_w=0.5,
                            evidence_level="mvp",
                        )
                    finally:
                        rep_cam.close()

                    representative_payload = {
                        "sample_index": rep_sample_index,
                        "frame_score": round(rep_score, 2),
                        "target_score": round(float(video_score), 2),
                        "abs_diff": round(abs(rep_score - float(video_score)), 2),
                        "assets": rep_out.get("assets", {}),
                        "evidence": rep_out.get("evidence", {}),
                        "explanation": explain_from_evidence(
                            evidence=rep_out.get("evidence", {}),
                            score=rep_out.get("score", {}),
                            media_mode_hint="video",
                            use_openai=True,
                        ),
                    }
        except Exception:
            representative_payload = None

        if representative_payload is not None:
            analysis_result["representative_analysis"] = representative_payload

        ai_comment = generate_video_ai_comment(
            final_scores=[float(s) for s in scores],
            pixel_scores=[float(s) for s in pixel_scores],
            freq_scores=[float(s) for s in freq_scores],
            is_fake=bool(analysis_result.get("is_fake")) if isinstance(analysis_result.get("is_fake"), bool) else None,
        )
        ai_comment_source = "openai" if ai_comment else "rule_based"
        if not ai_comment:
            if bool(analysis_result.get("is_fake")):
                ai_comment = "영상 전체 흐름을 보면 조작 가능성이 조금 더 높게 보입니다. 아래 근거를 함께 확인해 주세요."
            else:
                ai_comment = "영상 전체 흐름을 보면 원본일 가능성이 조금 더 높게 보입니다. 아래 근거를 함께 확인해 주세요."
        analysis_result["ai_comment"] = ai_comment
        analysis_result["ai_comment_source"] = ai_comment_source
        analysis_result["input_media_type"] = "video"
        if source_preview:
            analysis_result["source_preview"] = source_preview

        cache_payload = dict(analysis_result)
        cache_payload.pop("source_preview", None)
        redis_set_json(redis_db, video_cache_key, cache_payload, ex=CACHE_TTL_SEC)

        return store_result_and_make_response(analysis_result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/api/analyze-video")
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    content = await file.read()
    return _analyze_video_bytes(content=content, filename=file.filename or "upload.mp4")


@app.get("/get-result/{token}")
async def get_analysis_result(token: str):
    data = redis_get_json(redis_db, f"res:{token}")
    if data is None:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    return data
