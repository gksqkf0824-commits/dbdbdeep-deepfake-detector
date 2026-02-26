"""Core inference and face/CAM preprocessing utilities."""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from pytorch_grad_cam import GradCAM as PytorchGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception:
    PytorchGradCAM = None
    ClassifierOutputTarget = None

from model import build_4ch_srm_y, detector, make_square_bbox_with_margin, resize_with_padding


def infer_prob_binary(model: torch.nn.Module, x: torch.Tensor) -> float:
    if model is None:
        raise RuntimeError("추론 실패: 모델 인스턴스가 없습니다.")

    model.eval()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]

    if y.ndim == 2 and y.shape[1] == 2:
        prob_fake = F.softmax(y, dim=1)[:, 1]
        return float(prob_fake.item())

    prob_fake = torch.sigmoid(y.view(-1))
    return float(prob_fake.item())


def fuse_probs(p_rgb: float, p_freq: float, w: float = 0.5) -> float:
    w = float(max(0.0, min(1.0, w)))
    p = (w * float(p_rgb)) + ((1.0 - w) * float(p_freq))
    return float(max(0.0, min(1.0, p)))


def _ensure_224_rgb(img_rgb_uint8: np.ndarray) -> np.ndarray:
    if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
        raise ValueError("RGB uint8 이미지(3채널) 입력이 필요합니다.")
    if img_rgb_uint8.dtype != np.uint8:
        img_rgb_uint8 = np.clip(img_rgb_uint8, 0, 255).astype(np.uint8)
    if img_rgb_uint8.shape[0] == 224 and img_rgb_uint8.shape[1] == 224:
        return img_rgb_uint8
    return cv2.resize(img_rgb_uint8, (224, 224), interpolation=cv2.INTER_LINEAR)


def rgb_preprocess_tensor(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    from PIL import Image

    img = _ensure_224_rgb(img_rgb_uint8)
    pil = Image.fromarray(img)
    return detector.pixel_transform(pil).unsqueeze(0).to(detector.device, non_blocking=True)


def freq_preprocess_tensor(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    img = _ensure_224_rgb(img_rgb_uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    combined_4ch = build_4ch_srm_y(bgr, detector.srm_filters)
    return (
        torch.from_numpy(combined_4ch.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(detector.device, non_blocking=True)
    )


def get_cam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "model") and hasattr(model.model, "features"):
        return model.model.features[-1]
    raise RuntimeError("CAM target layer를 자동 선택할 수 없습니다. 모델 구조를 확인하세요.")


def _extract_landmarks_5pt(face_obj) -> Optional[np.ndarray]:
    kps = getattr(face_obj, "kps", None)
    if kps is not None:
        arr = np.asarray(kps, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] >= 2:
            return arr[:5, :2]

    kps106 = getattr(face_obj, "landmark_2d_106", None)
    if kps106 is not None:
        arr = np.asarray(kps106, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] >= 2:
            return arr[:5, :2]

    return None


def _default_landmarks_5pt(target_size: int = 224) -> np.ndarray:
    return np.asarray(
        [
            [0.31 * target_size, 0.40 * target_size],
            [0.69 * target_size, 0.40 * target_size],
            [0.50 * target_size, 0.56 * target_size],
            [0.39 * target_size, 0.73 * target_size],
            [0.61 * target_size, 0.73 * target_size],
        ],
        dtype=np.float32,
    )


def _map_landmarks_to_crop(
    landmarks_xy: np.ndarray,
    square_bbox: List[int],
    crop_h: int,
    crop_w: int,
    target_size: int,
) -> np.ndarray:
    x1, y1, _, _ = square_bbox

    scale = float(target_size) / float(max(crop_h, crop_w))
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    pad_t = (target_size - new_h) // 2
    pad_l = (target_size - new_w) // 2

    local = landmarks_xy.astype(np.float32).copy()
    local[:, 0] = (local[:, 0] - float(x1)) * scale + float(pad_l)
    local[:, 1] = (local[:, 1] - float(y1)) * scale + float(pad_t)

    local[:, 0] = np.clip(local[:, 0], 0, target_size - 1)
    local[:, 1] = np.clip(local[:, 1], 0, target_size - 1)
    return local.astype(np.float32)


def _face_area_ratio(face_obj, img_w: int, img_h: int) -> float:
    bbox = np.asarray(getattr(face_obj, "bbox", [0, 0, 0, 0]), dtype=np.float32)
    bw = max(1.0, float(bbox[2] - bbox[0]))
    bh = max(1.0, float(bbox[3] - bbox[1]))
    img_area = max(1.0, float(img_w * img_h))
    return float((bw * bh) / img_area)


def _pose_frontal_score(face_obj) -> Optional[float]:
    pose = getattr(face_obj, "pose", None)
    if pose is None:
        return None

    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None

    yaw = abs(float(arr[0]))
    pitch = abs(float(arr[1]))
    roll = abs(float(arr[2])) if arr.size >= 3 else 0.0

    yaw_score = max(0.0, 1.0 - (yaw / 45.0))
    pitch_score = max(0.0, 1.0 - (pitch / 30.0))
    roll_score = max(0.0, 1.0 - (roll / 40.0))

    return float((0.6 * yaw_score) + (0.3 * pitch_score) + (0.1 * roll_score))


def _landmark_frontal_score(face_obj) -> float:
    lm = _extract_landmarks_5pt(face_obj)
    if lm is None:
        return 0.5

    le, re, nose, ml, mr = lm[0], lm[1], lm[2], lm[3], lm[4]
    eye_dist = float(max(np.linalg.norm(le - re), 1e-6))

    eye_center_x = float((le[0] + re[0]) * 0.5)
    mouth_center_x = float((ml[0] + mr[0]) * 0.5)
    center_x = (eye_center_x + mouth_center_x) * 0.5
    nose_center_dev = abs(float(nose[0]) - float(center_x))
    center_score = max(0.0, 1.0 - (nose_center_dev / (0.35 * eye_dist)))

    nose_to_ml = abs(float(nose[0]) - float(ml[0]))
    mr_to_nose = abs(float(mr[0]) - float(nose[0]))
    den = max(nose_to_ml, mr_to_nose, 1e-6)
    lr_balance = abs(nose_to_ml - mr_to_nose) / den
    symmetry_score = max(0.0, 1.0 - lr_balance)

    eye_center_y = float((le[1] + re[1]) * 0.5)
    mouth_center_y = float((ml[1] + mr[1]) * 0.5)
    y_order_score = 1.0 if (eye_center_y < float(nose[1]) < mouth_center_y) else 0.0

    return float((0.5 * center_score) + (0.4 * symmetry_score) + (0.1 * y_order_score))


def _face_frontal_score(face_obj) -> float:
    pose_score = _pose_frontal_score(face_obj)
    if pose_score is not None:
        return float(max(0.0, min(1.0, pose_score)))
    return float(max(0.0, min(1.0, _landmark_frontal_score(face_obj))))


def _rank_faces_by_primary_priority(
    faces: List[Any],
    img_w: int,
    img_h: int,
) -> List[Any]:
    if not faces:
        return []

    scored = []
    area_ratios = []
    for face in faces:
        ar = _face_area_ratio(face, img_w=img_w, img_h=img_h)
        area_ratios.append(ar)
        scored.append({"face": face, "area_ratio": ar, "frontal": _face_frontal_score(face)})

    max_area = max(area_ratios) if area_ratios else 1.0
    max_area = max(max_area, 1e-6)

    for item in scored:
        area_norm = float(item["area_ratio"] / max_area)
        frontal = float(item["frontal"])
        # 큰 얼굴을 우선하되, 비슷한 크기라면 정면성 높은 얼굴 선택.
        item["priority"] = float((0.8 * area_norm) + (0.2 * frontal))

    scored.sort(key=lambda x: (x["priority"], x["area_ratio"], x["frontal"]), reverse=True)
    return [x["face"] for x in scored]


def detect_faces_with_aligned_crops(
    image_bgr: np.ndarray,
    margin: float = 0.15,
    target_size: int = 224,
    max_faces: int = 8,
    prioritize_frontal: bool = False,
) -> List[Dict[str, np.ndarray]]:
    face_app = getattr(getattr(detector, "face_cropper", None), "app", None)
    if face_app is None:
        raise RuntimeError("InsightFace 초기화 실패: detector.face_cropper.app를 찾을 수 없습니다.")

    faces = face_app.get(image_bgr)
    if not faces:
        return []

    img_h, img_w = image_bgr.shape[:2]
    if prioritize_frontal:
        faces_ranked = _rank_faces_by_primary_priority(faces, img_w=img_w, img_h=img_h)
    else:
        faces_ranked = sorted(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            reverse=True,
        )
    limit = max(1, int(max_faces))
    out: List[Dict[str, np.ndarray]] = []

    for face in faces_ranked[:limit]:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        square_bbox = make_square_bbox_with_margin(
            bbox.tolist(),
            margin=margin,
            img_width=img_w,
            img_height=img_h,
        )

        x1, y1, x2, y2 = [int(v) for v in square_bbox]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_224_bgr = resize_with_padding(crop, target_size=target_size)
        crop_224_rgb = cv2.cvtColor(crop_224_bgr, cv2.COLOR_BGR2RGB)

        landmarks = _extract_landmarks_5pt(face)
        if landmarks is None:
            lm_crop = _default_landmarks_5pt(target_size=target_size)
        else:
            lm_crop = _map_landmarks_to_crop(
                landmarks_xy=landmarks,
                square_bbox=square_bbox,
                crop_h=crop.shape[0],
                crop_w=crop.shape[1],
                target_size=target_size,
            )

        out.append(
            {
                "crop_rgb": crop_224_rgb,
                "landmarks": lm_crop,
                "bbox": bbox,
            }
        )

    return out


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        if PytorchGradCAM is None or ClassifierOutputTarget is None:
            raise RuntimeError(
                "pytorch-grad-cam 라이브러리가 필요합니다. requirements.txt에 grad-cam 패키지를 설치해 주세요."
            )
        self.model = model.eval()
        self.target_layer = target_layer
        self._cam = PytorchGradCAM(model=self.model, target_layers=[self.target_layer])

    def close(self) -> None:
        try:
            if hasattr(self._cam, "activations_and_grads"):
                self._cam.activations_and_grads.release()
        except Exception:
            pass

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        requested_idx = 1 if class_idx is None else int(class_idx)
        idx = requested_idx
        try:
            with torch.no_grad():
                y = self.model(x)
                if isinstance(y, (tuple, list)) and y:
                    y = y[0]
            if isinstance(y, torch.Tensor) and y.ndim >= 2:
                num_classes = int(y.shape[1])
                if num_classes <= 1:
                    idx = 0
                else:
                    idx = max(0, min(num_classes - 1, requested_idx))
            else:
                idx = 0 if class_idx is None else requested_idx
        except Exception:
            idx = 0 if class_idx is None else requested_idx

        targets = [ClassifierOutputTarget(idx)]
        cam_out = self._cam(input_tensor=x, targets=targets)
        cam_np = np.asarray(cam_out[0] if isinstance(cam_out, (list, tuple)) else cam_out, dtype=np.float32)
        if cam_np.ndim == 3:
            cam_np = cam_np[0]
        cam_np = cv2.resize(cam_np, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_LINEAR)
        cam_np = cam_np - cam_np.min()
        cam_np = cam_np / (cam_np.max() + 1e-6)
        return cam_np.astype(np.float32)


def overlay_cam(rgb_img_uint8: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hmap = (np.clip(heatmap01, 0.0, 1.0) * 255).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    out = (rgb_img_uint8 * (1.0 - alpha) + hmap * alpha).astype(np.uint8)
    return out


__all__ = [
    "infer_prob_binary",
    "fuse_probs",
    "rgb_preprocess_tensor",
    "freq_preprocess_tensor",
    "get_cam_target_layer",
    "detect_faces_with_aligned_crops",
    "GradCAM",
    "overlay_cam",
]
