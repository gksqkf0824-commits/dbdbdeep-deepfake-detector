import { useEffect, useMemo, useRef, useState } from "react";

import Header from "../components/Header";
import UploadCard from "../components/UploadCard";
import ResultPanel from "../components/ResultPanel";
import ExplainPanel from "../components/ExplainPanel";

const DEFAULT_API_BASE =
  typeof window !== "undefined" ? window.location.origin : "";
const API_BASE = (process.env.REACT_APP_API_BASE || DEFAULT_API_BASE).replace(/\/$/, "");

const EMPTY_RESULT = {
  requestId: "",
  confidence: null,
  pixelScore: null,
  freqScore: null,
  isFake: null,
  pValue: null,
  reliability: "",
  videoMeta: null,
  videoRepresentativeConfidence: null,
  videoFrameConfidences: [],
  videoFramePixelScores: [],
  videoFrameFreqScores: [],
  timeline: [],
  preprocessed: null,
  comment: "",
  topRegions: [],
  dominantBand: "",
  dominantEnergyBand: "",
  explanationSummary: "",
  spatialFindings: [],
  frequencyFindings: [],
  bandAblation: [],
  bandEnergy: [],
  nextSteps: [],
  caveats: [],
};

const clampProb = (v) => Math.max(0, Math.min(1, Number(v)));
const toRealConfidence = (fakeProb) =>
  Number.isFinite(Number(fakeProb)) ? (1 - clampProb(fakeProb)) * 100 : null;

const toFiniteArray = (values) =>
  Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : [];

function toDataUrl(base64Payload, mimeType = "image/jpeg") {
  if (typeof base64Payload !== "string" || base64Payload.length === 0) {
    return null;
  }
  return `data:${mimeType};base64,${base64Payload}`;
}

function toRenderableImageUrl(url) {
  if (typeof url !== "string" || url.length === 0) return null;
  if (url.startsWith("data:")) return url;
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/")) return `${API_BASE}${url}`;
  return null;
}

function buildLegacyComment(data) {
  const confidence = Number(data?.confidence);
  const pixel = Number(data?.pixel_score);
  const freq = Number(data?.freq_score);
  const isFake = typeof data?.is_fake === "boolean" ? data.is_fake : null;

  if (isFake === true) {
    return "겉보기는 그럴듯했지만, 미세 결의 박자가 어긋나 조작 가능성이 높게 관측됐습니다.";
  }
  if (isFake === false) {
    return "큰 윤곽과 미세 결이 같은 리듬으로 맞물려, 현재 근거상 원본 가능성이 우세합니다.";
  }
  if (Number.isFinite(confidence) || Number.isFinite(pixel) || Number.isFinite(freq)) {
    return "여러 신호를 합쳐 읽은 결과가 도착했습니다. 아래 근거 항목에서 단서를 함께 확인해보세요.";
  }
  return "분석이 완료되었습니다. 결과 패널의 근거 정보를 함께 확인해보세요.";
}

function parseLegacyResult(response) {
  const data = response?.data || response || {};

  const videoFrameConfidences = toFiniteArray(data.video_frame_confidences);
  const videoFramePixelScores = toFiniteArray(data.video_frame_pixel_scores);
  const videoFrameFreqScores = toFiniteArray(data.video_frame_freq_scores);

  const timelineLength = Math.max(
    videoFrameConfidences.length,
    videoFramePixelScores.length,
    videoFrameFreqScores.length
  );

  const timeline = Array.from({ length: timelineLength }, (_, idx) => ({
    time: idx + 1,
    pixel: idx < videoFramePixelScores.length ? videoFramePixelScores[idx] : null,
    srm: idx < videoFrameFreqScores.length ? videoFrameFreqScores[idx] : null,
    final: idx < videoFrameConfidences.length ? videoFrameConfidences[idx] : null,
  }));

  const preprocessed =
    data.preprocessed && typeof data.preprocessed === "object"
      ? {
          cropImage: toDataUrl(
            data.preprocessed.face_crop_image_b64,
            data.preprocessed.mime_type || "image/jpeg"
          ),
        }
      : null;

  return {
    ...EMPTY_RESULT,
    requestId: "",
    confidence: Number.isFinite(data.confidence) ? data.confidence : null,
    pixelScore: Number.isFinite(data.pixel_score) ? data.pixel_score : null,
    freqScore: Number.isFinite(data.freq_score) ? data.freq_score : null,
    isFake: typeof data.is_fake === "boolean" ? data.is_fake : null,
    pValue: Number.isFinite(data.p_value) ? data.p_value : null,
    reliability: data.reliability || "",
    videoMeta: data.video_meta || null,
    videoRepresentativeConfidence: Number.isFinite(data.video_representative_confidence)
      ? data.video_representative_confidence
      : null,
    videoFrameConfidences,
    videoFramePixelScores,
    videoFrameFreqScores,
    timeline,
    preprocessed,
    comment: buildLegacyComment(data),
    nextSteps: [],
    caveats: [],
  };
}

function parseEvidenceResult(response) {
  const score = response?.score || {};
  const faces = Array.isArray(response?.faces) ? response.faces : [];
  const firstFace = faces[0] || {};

  const explanation = firstFace?.explanation || {};
  const spatialEvidence = firstFace?.evidence?.spatial || {};
  const freqEvidence = firstFace?.evidence?.frequency || {};

  const confidence = toRealConfidence(score.p_final);
  const isFake = Number.isFinite(confidence) ? confidence < 50 : null;

  const cropImage = toRenderableImageUrl(firstFace?.assets?.face_crop_url || "");
  const preprocessed = cropImage ? { cropImage } : null;

  return {
    ...EMPTY_RESULT,
    requestId: response?.request_id || "",
    confidence,
    pixelScore: toRealConfidence(score.p_rgb),
    freqScore: toRealConfidence(score.p_freq),
    isFake,
    pValue: null,
    reliability: "",
    preprocessed,
    comment:
      explanation?.summary ||
      (isFake
        ? "[경고] 비정상 징후가 감지되었습니다. 추가 검증을 권장합니다."
        : "[판독 완료] 비정상 징후가 낮게 관찰되었습니다."),
    topRegions: Array.isArray(spatialEvidence?.regions_topk) ? spatialEvidence.regions_topk : [],
    dominantBand: freqEvidence?.dominant_band || "",
    dominantEnergyBand: freqEvidence?.dominant_energy_band || "",
    explanationSummary: explanation?.summary || "",
    spatialFindings: Array.isArray(explanation?.spatial_findings) ? explanation.spatial_findings : [],
    frequencyFindings: Array.isArray(explanation?.frequency_findings)
      ? explanation.frequency_findings
      : [],
    bandAblation: Array.isArray(freqEvidence?.band_ablation) ? freqEvidence.band_ablation : [],
    bandEnergy: Array.isArray(freqEvidence?.band_energy) ? freqEvidence.band_energy : [],
    nextSteps: Array.isArray(explanation?.next_steps) ? explanation.next_steps : [],
    caveats: Array.isArray(explanation?.caveats) ? explanation.caveats : [],
  };
}

function parseAnalyzeResponse(response, fileType) {
  if (fileType === "video") return parseLegacyResult(response);
  if (response?.score && Array.isArray(response?.faces)) return parseEvidenceResult(response);
  return parseLegacyResult(response);
}

async function analyzeWithFastAPI(file, fileType) {
  const formData = new FormData();
  formData.append("file", file);

  if (fileType !== "video") {
    formData.append("explain", "true");
    formData.append("evidence_level", "mvp");
    formData.append("fusion_w", "0.5");
  }

  const endpoint = fileType === "video" ? "/api/analyze-video" : "/api/analyze-evidence";
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  const bodyText = await response.text();
  let json;

  try {
    json = JSON.parse(bodyText);
  } catch {
    throw new Error(`서버 응답 파싱 실패 (status: ${response.status})`);
  }

  if (!response.ok) {
    throw new Error(json?.detail || `분석 요청 실패 (status: ${response.status})`);
  }

  return json;
}

async function analyzeImageUrlWithFastAPI(imageUrl) {
  const trimmed = (imageUrl || "").trim();
  if (!/^https?:\/\//i.test(trimmed)) {
    throw new Error("URL은 http:// 또는 https:// 형식이어야 합니다.");
  }

  const formData = new FormData();
  formData.append("image_url", trimmed);
  formData.append("explain", "true");
  formData.append("evidence_level", "mvp");
  formData.append("fusion_w", "0.5");

  const response = await fetch(`${API_BASE}/api/analyze-url`, {
    method: "POST",
    body: formData,
  });

  const bodyText = await response.text();
  let json;

  try {
    json = JSON.parse(bodyText);
  } catch {
    throw new Error(`서버 응답 파싱 실패 (status: ${response.status})`);
  }

  if (!response.ok) {
    throw new Error(json?.detail || `분석 요청 실패 (status: ${response.status})`);
  }

  return json;
}

export default function Analyze() {
  const [inputMode, setInputMode] = useState("file");
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [videoDuration, setVideoDuration] = useState(0);

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const progressTimerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const startProgress = (seconds) => {
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);

    setProgress(0);
    const tickMs = 150;
    const safeSeconds = Math.max(seconds, 1);
    const step = 95 / ((safeSeconds * 1000) / tickMs);

    progressTimerRef.current = setInterval(() => {
      setProgress((prev) => (prev >= 95 ? 95 : prev + step));
    }, tickMs);
  };

  const stopProgress = (finalProgress = 100) => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setProgress(finalProgress);
  };

  const onPickFile = (pickedFile) => {
    if (!pickedFile) return;

    const nextType = pickedFile.type.startsWith("video") ? "video" : "image";

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const objectUrl = URL.createObjectURL(pickedFile);

    setInputMode("file");
    setFile(pickedFile);
    setFileType(nextType);
    setPreviewUrl(objectUrl);
    setImageUrl("");
    setResult(null);
    setError("");
    stopProgress(0);

    if (nextType === "video") {
      const videoEl = document.createElement("video");
      videoEl.preload = "metadata";
      videoEl.onloadedmetadata = () => setVideoDuration(videoEl.duration || 0);
      videoEl.onerror = () => setVideoDuration(0);
      videoEl.src = objectUrl;
    } else {
      setVideoDuration(0);
    }
  };

  const onChangeMode = (mode) => {
    setInputMode(mode);
    setResult(null);
    setError("");
    stopProgress(0);
  };

  const analyze = async () => {
    try {
      setError("");
      setResult(null);

      let requestFile = null;
      let requestType = "image";

      if (inputMode === "file") {
        if (!file) throw new Error("분석할 파일을 먼저 선택해 주세요.");
        requestFile = file;
        requestType = fileType === "video" ? "video" : "image";
      } else {
        if (!imageUrl.trim()) throw new Error("분석할 이미지 URL을 입력해 주세요.");
        requestType = "image";
      }

      setLoading(true);
      const estimatedSeconds = requestType === "video" ? Math.max(videoDuration * 2, 8) : 5;
      startProgress(estimatedSeconds);

      const response =
        inputMode === "file"
          ? await analyzeWithFastAPI(requestFile, requestType)
          : await analyzeImageUrlWithFastAPI(imageUrl);
      const parsed = parseAnalyzeResponse(response, requestType);
      setResult(parsed);
      stopProgress(100);
    } catch (e) {
      stopProgress(0);
      setError(e?.message || "분석 실패");
    } finally {
      setLoading(false);
    }
  };

  const aiComment = useMemo(() => {
    if (!result) return "";
    const summary = String(result.comment || result.explanationSummary || "").trim();
    return summary;
  }, [result]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-indigo-50 py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-8">
        <Header />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <UploadCard
            mode={inputMode}
            fileType={fileType}
            previewUrl={previewUrl}
            imageUrl={imageUrl}
            loading={loading}
            hasResult={Boolean(result)}
            aiComment={aiComment}
            onModeChange={onChangeMode}
            onPickFile={onPickFile}
            onUrlChange={setImageUrl}
            onAnalyze={analyze}
          />

          <ResultPanel
            progress={progress}
            result={result}
            error={error}
            faceImageUrl={result?.preprocessed?.cropImage || null}
          />
        </div>

        <ExplainPanel result={result} />
      </div>
    </div>
  );
}
