"""
model.py — DBDBDEEP Deepfake Detector
Model architectures + ensemble inference class.

Models
------
- ImageModel  : EfficientNet-V2-S (RGB 3-ch)  → trained as image.pth
- FreqModel   : EfficientNet-V2-S (SRM×3 + Y-ch 4-ch input) → trained as freq.pt
- Ensemble    : Weighted Soft Voting  →  w·p_image + (1-w)·p_freq
                Best w = 0.37  (Image 37% / Freq 63%)
"""

import io
import os
import uuid

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Architectures
# ──────────────────────────────────────────────────────────────────────────────

class ImageModel(nn.Module):
    """
    EfficientNet-V2-S for RGB deepfake detection.
    Input : (B, 3, 224, 224) — standard ImageNet-normalized RGB
    Output: (B, 1)           — logit (apply sigmoid → P(fake))
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)


class FreqModel(nn.Module):
    """
    EfficientNet-V2-S with 4-channel input for frequency-domain deepfake detection.
    Input channels: SRM residual map ×3  +  Y (luminance) channel ×1
    Input : (B, 4, 224, 224)
    Output: (B, 1)            — logit (apply sigmoid → P(fake))

    Weight initialisation for the modified first conv:
      - ch 0-2 : copied from ImageNet pretrained 3-ch weights
      - ch 3   : average of ch 0-2  ×  1.2  (Y-channel booster)
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_v2_s(weights=weights)

        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )

        if pretrained:
            with torch.no_grad():
                base.features[0][0].weight[:, :3] = old_conv.weight
                base.features[0][0].weight[:, 3]  = old_conv.weight.mean(dim=1) * 1.2

        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, 1),
        )
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

_RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_SRM_FILTERS = None

def _get_srm_filters():
    global _SRM_FILTERS
    if _SRM_FILTERS is None:
        f1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],
                       [0,-1,2,-1,0],[0,0,0,0,0]]) / 4.0
        f2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
                       [2,-6,8,-6,2],[-1,2,-2,2,-1]]) / 12.0
        f3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
                       [0,0,0,0,0],[0,0,0,0,0]]) / 2.0
        _SRM_FILTERS = [f1, f2, f3]
    return _SRM_FILTERS


def pil_to_freq_tensor(img_pil: Image.Image) -> torch.Tensor:
    """Convert a PIL image → 4-ch SRM+Y tensor (1, 4, 224, 224)."""
    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    ycc     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y       = ycc[:, :, 0]

    srm_maps = []
    for f in _get_srm_filters():
        res = cv2.filter2D(y.astype(np.float32), -1, f)
        srm_maps.append(np.clip(res, 0, 255).astype(np.uint8))

    combined = np.concatenate([np.stack(srm_maps, axis=-1),
                                y[:, :, np.newaxis]], axis=-1)   # (H,W,4)
    combined = cv2.resize(combined, (224, 224), interpolation=cv2.INTER_CUBIC)
    tensor   = torch.from_numpy(combined.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor.unsqueeze(0)   # (1, 4, 224, 224)


def detect_and_crop_face(img_pil: Image.Image, margin: float = 0.15):
    """
    Detect the largest face and return a square-cropped PIL image.
    Returns None if no face is found.
    Requires: opencv-python (haarcascade) or insightface if available.
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w // 2, y + h // 2
    size   = int(max(w, h) * (1 + margin * 2))
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(img_rgb.shape[1], cx + size // 2)
    y2 = min(img_rgb.shape[0], cy + size // 2)
    return Image.fromarray(img_rgb[y1:y2, x1:x2])


# ──────────────────────────────────────────────────────────────────────────────
# 3. Ensemble Detector (inference only)
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeDetectorEnsemble:
    """
    Weighted Soft Voting ensemble of ImageModel + FreqModel.

    Final P(fake) = W_IMAGE * p_image + W_FREQ * p_freq

    Score convention
    ----------------
    - fake_score (0–100) : higher → more likely fake
    - real_score (0–100) : 100 - fake_score
    """

    W_IMAGE: float = 0.37
    W_FREQ:  float = 0.63

    def __init__(self, image_model_path: str, freq_model_path: str):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_model = self._load(ImageModel(), image_model_path)
        self.freq_model  = self._load(FreqModel(),  freq_model_path)

    def _load(self, model: nn.Module, path: str) -> nn.Module:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        ckpt       = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """
        Parameters
        ----------
        image_bytes : raw bytes of the uploaded image

        Returns
        -------
        dict with keys:
          fake_score  (float, 0-100)
          real_score  (float, 0-100)
          p_image     (float, 0-1)  — image model P(fake)
          p_freq      (float, 0-1)  — freq  model P(fake)
          is_fake     (bool)
          risk_level  ("Safe" | "Caution" | "Danger")
        """
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        face = detect_and_crop_face(img_pil)
        if face is None:
            raise ValueError("No face detected in the image.")

        # Image model input
        img_tensor  = _RGB_TRANSFORM(face).unsqueeze(0).to(self.device)
        # Freq model input
        freq_tensor = pil_to_freq_tensor(face).to(self.device)

        p_image = torch.sigmoid(self.image_model(img_tensor)).item()
        p_freq  = torch.sigmoid(self.freq_model(freq_tensor)).item()

        p_fake     = self.W_IMAGE * p_image + self.W_FREQ * p_freq
        fake_score = round(p_fake * 100, 2)
        real_score = round((1 - p_fake) * 100, 2)

        # 3-level risk (based on eval_ensemble thresholds)
        if p_fake < 0.3:
            risk = "Safe"
        elif p_fake < 0.7:
            risk = "Caution"
        else:
            risk = "Danger"

        return {
            "fake_score": fake_score,
            "real_score": real_score,
            "p_image":    round(p_image, 4),
            "p_freq":     round(p_freq,  4),
            "is_fake":    p_fake >= 0.5,
            "risk_level": risk,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Instantiate for import by main.py
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

detector = DeepfakeDetectorEnsemble(
    image_model_path=os.path.join(BASE_DIR, "image.pth"),
    freq_model_path =os.path.join(BASE_DIR, "freq.pt"),
)
