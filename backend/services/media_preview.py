"""Media preview builders for URL/image/video analysis results."""

import base64
import os
import tempfile

import cv2
import numpy as np


def build_source_preview(source_url: str, media_type: str) -> dict:
    kind = "video" if str(media_type).lower() == "video" else "image"
    return {
        "kind": kind,
        "url": source_url,
        "page_url": source_url,
        "pageUrl": source_url,
    }


def _to_data_url_jpeg(image_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("미리보기 JPEG 인코딩 실패")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_keep_ratio(image_bgr: np.ndarray, max_side: int = 720) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_bgr
    scale = float(max_side) / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _image_preview_from_bytes(content: bytes) -> str | None:
    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = _resize_keep_ratio(img, max_side=720)
        return _to_data_url_jpeg(img, quality=88)
    except Exception:
        return None


def _video_thumbnail_from_bytes(content: bytes, filename: str) -> str | None:
    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    tmp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None

        frame = _resize_keep_ratio(frame, max_side=720)
        return _to_data_url_jpeg(frame, quality=86)
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def build_source_preview_from_downloaded(source_url: str, media_type: str, filename: str, content: bytes, title: str) -> dict:
    preview = build_source_preview(source_url=source_url, media_type=media_type)
    if title:
        preview["title"] = title

    if str(media_type).lower() == "image":
        data_url = _image_preview_from_bytes(content)
        if data_url:
            preview["data_url"] = data_url
    elif str(media_type).lower() == "video":
        thumb = _video_thumbnail_from_bytes(content=content, filename=filename)
        if thumb:
            preview["thumbnail_data_url"] = thumb

    return preview

