import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# 1. 모델 구조 정의 (FreqNet & Blocks)
#    (Gradio 코드와 동일한 구조)
# -------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class FreqNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=2):
        super(FreqNet, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(64, 3, 1, 1))
        self.bias1 = nn.Parameter(torch.zeros(64))
        self.realconv1 = conv1x1(64, 64)
        self.imagconv1 = conv1x1(64, 64)
        self.se1 = SEBlock(64)
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128 * block.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def hfreqWH(self, x, scale=4):
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=[-2, -1])
        b, c, h, w = x_fft.shape
        x_fft[:, :, h//2-h//scale:h//2+h//scale, w//2-w//scale:w//2+w//scale] = 0
        x_ifft = torch.fft.ifftshift(x_fft, dim=[-2, -1])
        return torch.fft.ifft2(x_ifft, norm="ortho").real

    def forward(self, x):
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight1, self.bias1)
        x = F.relu(x, inplace=True)
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=[-2, -1])
        x_fft = torch.complex(self.realconv1(x_fft.real), self.imagconv1(x_fft.imag))
        x_fft = torch.fft.ifftshift(x_fft, dim=[-2, -1])
        x = torch.fft.ifft2(x_fft, norm="ortho").real
        x = F.relu(x, inplace=True)
        x = self.se1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -------------------------------
# 2. 백엔드용 Detector 클래스
# -------------------------------
class DeepfakeDetectorEnsemble:
    def __init__(self, pixel_model_path, freq_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # 전처리 정의
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 1) Pixel Model (EfficientNetV2) 로드
        self.pixel_model = models.efficientnet_v2_s(weights=None)
        in_features = self.pixel_model.classifier[1].in_features
        self.pixel_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1)
        )
        if os.path.exists(pixel_model_path):
            ckpt = torch.load(pixel_model_path, map_location=self.device)
            # state_dict 키 처리 (만약 체크포인트가 dict 형태라면)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.pixel_model.load_state_dict(state_dict, strict=False)
            self.pixel_model.to(self.device).eval()
            print("✅ Pixel 모델 로드 완료!")
        else:
            print(f"⚠️ {pixel_model_path} 파일을 찾을 수 없습니다.")

        # 2) FreqNet 로드
        self.freq_model = FreqNet()
        if os.path.exists(freq_model_path):
            ckpt = torch.load(freq_model_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.freq_model.load_state_dict(state_dict, strict=False)
            self.freq_model.to(self.device).eval()
            print("✅ Freq 모델 로드 완료!")
        else:
            print(f"⚠️ {freq_model_path} 파일을 찾을 수 없습니다.")

    def get_cropped_face(self, img_pil):
        """얼굴 검출 및 크롭 (Gradio 코드와 동일)"""
        img_rgb = np.array(img_pil)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # OpenCV 내장 cascade 경로 사용
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0: return None
        
        # 가장 큰 얼굴 선택
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w//2, y + h//2
        size = int(max(w, h) * 1.5)
        nx1, ny1 = max(0, cx - size//2), max(0, cy - size//2)
        nx2, ny2 = min(img_rgb.shape[1], cx + size//2), min(img_rgb.shape[0], cy + size//2)
        return Image.fromarray(img_rgb[ny1:ny2, nx1:nx2])

    def create_result_plot(self, score, title, color):
        """원형 그래프 생성 및 저장"""
        plt.figure(figsize=(4, 4))
        plt.pie([max(0, score), max(0, 100-score)], colors=[color, '#E5E7EB'], 
                startangle=90, counterclock=False, wedgeprops={'width': 0.3})
        plt.text(0, 0, f"{int(score)}%", ha='center', va='center', fontsize=20, fontweight='bold', color=color)
        plt.title(title, fontsize=10, pad=10)
        
        # 파일명 생성
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(self.output_dir, filename)
        
        plt.savefig(path, transparent=True, bbox_inches='tight')
        plt.close() # 메모리 누수 방지
        return filename # 백엔드에서는 전체 경로보다는 파일명만 리턴하는 게 일반적임

    def predict(self, image_bytes):
        # 1. 이미지 로드
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 2. 얼굴 검출 (얼굴 없으면 에러 처리)
        face_pil = self.get_cropped_face(image)
        if face_pil is None:
            # 얼굴을 못 찾았을 경우의 처리는 백엔드 API 설계에 따라 예외를 던지거나 None을 리턴
            raise ValueError("Face not detected")
            
        # 3. 전처리 및 텐서 변환
        input_tensor = self.transform(face_pil.resize((224, 224))).unsqueeze(0).to(self.device)

        # 4. 추론 (Gradio 로직과 동일)
        with torch.no_grad():
            # FreqNet
            if self.freq_model:
                outputs_f = self.freq_model(input_tensor)
                # Gradio: 100 * F.softmax(outputs_f, dim=1)[0][0].item()
                s_f = 100 * F.softmax(outputs_f, dim=1)[0][0].item()
            else:
                s_f = 50.0

            # PixelNet (EfficientNet)
            if self.pixel_model:
                # Gradio: 100 * (1 - torch.sigmoid(model_image(img_tensor)).item())
                s_p = 100 * (1 - torch.sigmoid(self.pixel_model(input_tensor)).item())
            else:
                s_p = 50.0

        # 5. 가중치 평균 (Pixel 0.7 : Freq 0.3)
        avg_conf = (s_p * 0.7) + (s_f * 0.3)

        # 6. 결과 그래프 생성
        # 반환값은 파일명(string)입니다. 프론트엔드에서 이 경로로 이미지를 요청해야 합니다.
        freq_plot_file = self.create_result_plot(s_f, "Frequency Analysis", "#EC4899")
        pixel_plot_file = self.create_result_plot(s_p, "Pixel Analysis", "#8B5CF6")

        return round(avg_conf, 2), round(s_p, 2), round(s_f, 2), pixel_plot_file, freq_plot_file

# 사용 예시
# model.py 맨 아래쪽 수정

# 1. 현재 이 파일(model.py)이 있는 폴더의 절대 경로를 구합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 그 경로를 기준으로 가중치 파일의 위치를 지정합니다.
pixel_path = os.path.join(BASE_DIR, "best_test1.pth")
freq_path = os.path.join(BASE_DIR, "freq.pt")

# 3. 이제 경로가 정확하므로 파일을 잘 찾을 것입니다.
detector = DeepfakeDetectorEnsemble(
    pixel_model_path=pixel_path, 
    freq_model_path=freq_path
)