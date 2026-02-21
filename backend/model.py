import os
import io
import uuid
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# RetinaFace (InsightFace)
from insightface.app import FaceAnalysis


# =========================================================
# 1) Freq 모델 아키텍처 (4채널 입력 EfficientNetV2-S)
# =========================================================
class MultiChannelV2S(nn.Module):
    def __init__(self):
        super(MultiChannelV2S, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)

        old_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            4, old_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )

        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(num_ftrs, 1)  # binary logit
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# 2) SRM + Y 채널 전처리 함수들 (두 번째 코드 통합)
# =========================================================
def get_srm_filters():
    f1 = np.array([
        [0,0,0,0,0],
        [0,-1,2,-1,0],
        [0,2,-4,2,0],
        [0,-1,2,-1,0],
        [0,0,0,0,0]
    ], dtype=np.float32) / 4.0

    f2 = np.array([
        [-1,2,-2,2,-1],
        [2,-6,8,-6,2],
        [-2,8,-12,8,-2],
        [2,-6,8,-6,2],
        [-1,2,-2,2,-1]
    ], dtype=np.float32) / 12.0

    f3 = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,-2,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ], dtype=np.float32) / 2.0

    return [f1, f2, f3]


def make_square_bbox_with_margin(bbox, margin, img_width, img_height):
    """
    bbox: [x1,y1,x2,y2] (float/int 가능)
    margin: 0.15 같은 비율 (양쪽으로 margin만큼 확장)
    """
    x1, y1, x2, y2 = bbox
    w, h = (x2 - x1), (y2 - y1)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    size = max(w, h) * (1 + margin * 2)
    half_size = size / 2.0

    nx1 = max(0, int(center_x - half_size))
    ny1 = max(0, int(center_y - half_size))
    nx2 = min(img_width, int(center_x + half_size))
    ny2 = min(img_height, int(center_y + half_size))

    return [nx1, ny1, nx2, ny2]


def resize_with_padding(img_bgr, target_size=224):
    """
    aspect ratio 유지하면서 target_size로 맞추고, 남는 부분은 검정 padding
    """
    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_t = (target_size - new_h) // 2
    pad_l = (target_size - new_w) // 2
    pad_b = target_size - new_h - pad_t
    pad_r = target_size - new_w - pad_l

    padded = cv2.copyMakeBorder(
        resized, pad_t, pad_b, pad_l, pad_r,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded


def build_4ch_srm_y(crop_224_bgr, srm_filters):
    """
    입력: 224x224 BGR
    출력: 224x224x4 (SRM1, SRM2, SRM3, Y)
    """
    ycc = cv2.cvtColor(crop_224_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycc[:, :, 0]  # 0~255 uint8

    srm_maps = []
    for f in srm_filters:
        m = cv2.filter2D(y_channel.astype(np.float32), -1, f)
        m = np.clip(m, 0, 255).astype(np.uint8)
        srm_maps.append(m)

    combined = np.concatenate(
        [np.stack(srm_maps, axis=-1), y_channel[:, :, np.newaxis]],
        axis=-1
    )  # (224,224,4)
    return combined


# =========================================================
# 3) RetinaFace Cropper (bbox만 뽑고, square+margin은 외부에서)
# =========================================================
class RetinaFaceCropper:
    def __init__(self, device: torch.device, det_size=(320, 320)):
        """
        det_size는 (224,224)처럼 너무 작게 주지 말고 320/640 권장
        """
        ctx_id = 0 if (device.type == "cuda") else -1

        # providers 직접 지정하고 싶으면 아래처럼:
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        # self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=det_size)

    def get_largest_face_bbox(self, img_bgr: np.ndarray):
        """
        img_bgr: OpenCV BGR
        return: bbox [x1,y1,x2,y2] float or None
        """
        faces = self.app.get(img_bgr)
        if not faces:
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        bbox = face.bbox  # np.array([x1,y1,x2,y2], dtype=float)
        return bbox


# =========================================================
# 4) Detector (Pixel + Freq(4ch SRM+Y) 앙상블)
# =========================================================
class DeepfakeDetectorEnsemble:
    def __init__(self, pixel_model_path, freq_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # SRM 필터 미리 준비
        self.srm_filters = get_srm_filters()

        # 얼굴 검출기(1회 로드 후 재사용)
        self.face_cropper = RetinaFaceCropper(device=self.device, det_size=(320, 320))

        # Pixel 전처리 (RGB 3ch + ImageNet norm)
        self.pixel_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # -------------------
        # Pixel Model (EfficientNetV2-S, 3ch)
        # -------------------
        self.pixel_model = models.efficientnet_v2_s(weights=None)
        in_features = self.pixel_model.classifier[1].in_features
        self.pixel_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1)  # binary logit
        )

        if os.path.exists(pixel_model_path):
            ckpt = torch.load(pixel_model_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.pixel_model.load_state_dict(state_dict, strict=True
            )
            self.pixel_model.to(self.device).eval()
            print("✅ Pixel 모델 로드 완료!")
        else:
            self.pixel_model = None
            print(f"⚠️ {pixel_model_path} 파일을 찾을 수 없습니다. Pixel 모델 비활성화.")

        # -------------------
        # Freq Model (MultiChannelV2S, 4ch SRM+Y)
        # -------------------
        self.freq_model = MultiChannelV2S()

        if os.path.exists(freq_model_path):
            ckpt = torch.load(freq_model_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.freq_model.load_state_dict(state_dict, strict=True)
            self.freq_model.to(self.device).eval()
            print("✅ Freq(MultiChannelV2S) 모델 로드 완료!")
        else:
            self.freq_model = None
            print(f"⚠️ {freq_model_path} 파일을 찾을 수 없습니다. Freq 모델 비활성화.")

    def _decode_image_bytes_to_bgr(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Decode failed (cv2.imdecode returned None)")
        return img_bgr
    
    def _extract_face_crop224_bgr(self, img_bgr: np.ndarray, margin: float = 0.15) -> np.ndarray:
        """
        RetinaFace bbox -> square+margin -> crop -> padding resize(224)
        """
        bbox = self.face_cropper.get_largest_face_bbox(img_bgr)
        if bbox is None:
            raise ValueError("Face not detected")

        bbox_int = bbox.astype(int)
        sq = make_square_bbox_with_margin(
            bbox_int.tolist(),
            margin=margin,
            img_width=img_bgr.shape[1],
            img_height=img_bgr.shape[0]
        )

        crop = img_bgr[sq[1]:sq[3], sq[0]:sq[2]]
        if crop.size == 0:
            raise ValueError("Invalid crop region")

        crop_224 = resize_with_padding(crop, target_size=224)
        return crop_224

    def predict(self, image_bytes: bytes):
        """
        return:
          avg_conf(real-confidence), pixel_real_conf, freq_real_conf
        """
        # 1) bytes -> BGR
        img_bgr = self._decode_image_bytes_to_bgr(image_bytes)

        # 2) face crop (224,224) in BGR
        face_224_bgr = self._extract_face_crop224_bgr(img_bgr, margin=0.15)

        # -------------------
        # 3) Pixel 입력 (RGB 3ch + ImageNet norm)
        # -------------------
        face_224_rgb = cv2.cvtColor(face_224_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_224_rgb)
        pixel_tensor = self.pixel_transform(face_pil).unsqueeze(0).to(self.device)

        # -------------------
        # 4) Freq 입력 (SRM*3 + Y) 4ch, 0~1
        # -------------------
        combined_4ch = build_4ch_srm_y(face_224_bgr, self.srm_filters)  # (224,224,4) uint8
        freq_tensor = (
            torch.from_numpy(combined_4ch.astype(np.float32) / 255.0)
            .permute(2, 0, 1)  # (4,224,224)
            .unsqueeze(0)
            .to(self.device)
        )

        # -------------------
        # 5) Inference (둘 다 binary logit)
        # -------------------
        with torch.no_grad():
            # Freq 모델: sigmoid -> prob_fake
            if self.freq_model is not None:
                logit_f = self.freq_model(freq_tensor)  # (1,1)
                prob_fake_f = torch.sigmoid(logit_f).item()
                # "REAL confidence"로 맞추기
                s_f = 100.0 * (1.0 - prob_fake_f)
            else:
                s_f = 50.0

            # Pixel 모델: sigmoid -> prob_fake (기존 코드와 동일한 해석)
            if self.pixel_model is not None:
                logit_p = self.pixel_model(pixel_tensor)  # (1,1)
                prob_fake_p = torch.sigmoid(logit_p).item()
                s_p = 100.0 * (1.0 - prob_fake_p)
            else:
                s_p = 50.0

        # 6) 앙상블 (Pixel 0.7 : Freq 0.3)
        avg_conf = (s_p * 0.5) + (s_f * 0.5)

        return round(avg_conf, 2), round(s_p, 2), round(s_f, 2)


# =========================================================
# 5) model.py 하단: detector 인스턴스 생성
# =========================================================
# ✅ weights를 model.py와 같은 폴더의 "models/"에 둔다면:
#   project/
#     backend/
#       model.py
#       models/
#         best_test1.pth
#         freq.pt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

pixel_path = os.path.join(MODELS_DIR, "best_test1.pth")
freq_path  = os.path.join(MODELS_DIR, "freq.pt")

detector = DeepfakeDetectorEnsemble(
    pixel_model_path=pixel_path,
    freq_model_path=freq_path
)

# freq_model = MultiChannelV2S()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ckpt = torch.load(freq_path, map_location=device)
# state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
# missing, unexpected = freq_model.load_state_dict(state_dict, strict=False)
# print("FREQ missing keys:", missing[:20])
# print("FREQ unexpected keys:", unexpected[:20])