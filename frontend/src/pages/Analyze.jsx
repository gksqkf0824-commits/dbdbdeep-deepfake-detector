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
  isUndetermined: false,
  confidence: null,
  pixelScore: null,
  freqScore: null,
  isFake: null,
  pValue: null,
  reliability: "",
  aiCommentSource: "",
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
  spatialVisualUrl: null,
  sourcePreview: null,
  interpretationGuide: [],
  nextSteps: [],
  caveats: [],
  evidenceBasis: "",
  representativeSampleIndex: null,
  inputMediaType: "",
};

const INTERPRETATION_GUIDE_FALLBACK = [
  "CAM은 모델이 상대적으로 주목한 위치를 보여주는 참고 지표입니다.",
  "우세 대역과 Δfake(대역 제거 전후 fake 확률 변화)를 함께 보면 영향 방향을 읽기 쉽습니다.",
  "밴드 에너지 비율은 저/중/고주파 중 어느 구간의 신호가 더 강한지 보여줍니다.",
  "저주파는 큰 윤곽, 중주파는 얼굴 경계/피부 결, 고주파는 미세 경계·압축 흔적 해석에 유용합니다.",
  "주파수 단위는 Hz가 아니라 cycles/pixel이므로 해상도·압축 상태에 따라 해석 민감도가 달라질 수 있습니다.",
];

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

function normalizeSourcePreview(raw) {
  if (!raw || typeof raw !== "object") return null;
  const kind = String(raw.kind || "").toLowerCase() === "video" ? "video" : "image";
  const directUrl = toRenderableImageUrl(raw.url || "");
  const dataUrl = toRenderableImageUrl(raw.data_url || "");
  const thumbnailDataUrl = toRenderableImageUrl(raw.thumbnail_data_url || "");
  const thumbnailUrl = toRenderableImageUrl(raw.thumbnail_url || "");
  const pageUrl = String(raw.page_url || "").trim() || null;
  const title = String(raw.title || "").trim() || null;

  return {
    kind,
    url: directUrl,
    dataUrl,
    thumbnailDataUrl,
    thumbnailUrl,
    pageUrl,
    title,
  };
}

function buildCommonComment({ isFake, confidence, pixelScore, freqScore }) {
  if (isFake === true) {
    return "자동 분석 기준으로 조작 가능성이 조금 더 높게 나왔습니다. 아래 근거를 함께 확인해 주세요.";
  }
  if (isFake === false) {
    return "자동 분석 기준으로 원본일 가능성이 조금 더 높게 나왔습니다. 아래 근거를 함께 확인해 주세요.";
  }
  if ([confidence, pixelScore, freqScore].some((v) => Number.isFinite(Number(v)))) {
    return "분석이 완료되었습니다. 아래 근거를 확인해 주세요.";
  }
  return "분석이 완료되었습니다. 결과 패널의 근거를 확인해 주세요.";
}

function summarizeSeries(values) {
  const series = toFiniteArray(values);
  if (series.length === 0) return null;

  const start = series[0];
  const mid = series[Math.floor((series.length - 1) / 2)];
  const end = series[series.length - 1];
  const max = Math.max(...series);
  const min = Math.min(...series);
  const swing = Math.max(0, max - min);
  const drift = end - start;
  const trend = drift > 3 ? "상승" : drift < -3 ? "하강" : "유지";

  return { start, mid, end, max, min, swing, drift, trend };
}

function buildTimelineExplainData(timeline, isFake) {
  const finalStats = summarizeSeries(timeline.map((item) => item?.final));
  const pixelStats = summarizeSeries(timeline.map((item) => item?.pixel));
  const freqStats = summarizeSeries(timeline.map((item) => item?.srm));

  const spatialFindings = [];
  const frequencyFindings = [];
  const caveats = [];
  const nextSteps = [
    "급변한 구간(피크/저점) 주변 프레임을 원본 파일 기준으로 다시 확인하세요.",
    "가능하면 같은 장면의 다른 영상 또는 고해상도 소스로 교차 검증하세요.",
  ];

  if (finalStats) {
    spatialFindings.push({
      claim: "영상 초반·중반·후반의 최종 점수 변화를 비교했습니다.",
      evidence: `시작 ${finalStats.start.toFixed(1)}% · 중간 ${finalStats.mid.toFixed(
        1
      )}% · 종료 ${finalStats.end.toFixed(1)}% (추세 ${finalStats.trend}, 변동폭 ${finalStats.swing.toFixed(
        1
      )}%)`,
    });
  }

  if (pixelStats) {
    spatialFindings.push({
      claim: "화면 질감(픽셀) 점수의 구간별 변화도 함께 확인했습니다.",
      evidence: `시작 ${pixelStats.start.toFixed(1)}% · 종료 ${pixelStats.end.toFixed(
        1
      )}% (변동폭 ${pixelStats.swing.toFixed(1)}%)`,
    });
  }

  if (freqStats) {
    frequencyFindings.push({
      claim: "주파수 패턴 점수의 시간대별 변화를 확인했습니다.",
      evidence: `시작 ${freqStats.start.toFixed(1)}% · 중간 ${freqStats.mid.toFixed(
        1
      )}% · 종료 ${freqStats.end.toFixed(1)}% (추세 ${freqStats.trend})`,
    });
  }

  if (finalStats && freqStats) {
    const driftAligned =
      (finalStats.drift >= 0 && freqStats.drift >= 0) || (finalStats.drift <= 0 && freqStats.drift <= 0);
    const endGap = Math.abs(finalStats.end - freqStats.end);
    frequencyFindings.push({
      claim: "최종 점수와 주파수 점수가 같은 방향으로 움직였는지 점검했습니다.",
      evidence: `${driftAligned ? "동일 방향" : "상반 방향"}, 종료 시점 차이 ${endGap.toFixed(1)}%`,
    });
  }

  if (finalStats) {
    if (finalStats.swing >= 20) {
      caveats.push("최종 신뢰도 변동폭이 큰 편이라 장면 전환/압축 영향 가능성을 함께 고려하세요.");
    } else if (finalStats.swing >= 10) {
      caveats.push("중간 수준의 변동이 있어 급변 구간 프레임을 추가 확인하는 편이 안전합니다.");
    } else {
      caveats.push("전체 변동폭은 크지 않지만, 단일 프레임만으로 결론을 확정하긴 어렵습니다.");
    }
  }
  caveats.push("영상 길이, 해상도, 재인코딩 여부에 따라 타임라인 해석 민감도가 달라질 수 있습니다.");

  let summary = "타임라인 근거가 충분하지 않아 단일 구간 중심으로 해석했습니다.";
  if (finalStats) {
    const verdict =
      isFake === true
        ? "전체 흐름은 조작 가능성 쪽에 가깝습니다."
        : isFake === false
          ? "전체 흐름은 원본 가능성 쪽에 가깝습니다."
          : "최종 방향은 추가 근거와 함께 보는 편이 안전합니다.";
    summary = `타임라인 기준 시작 ${finalStats.start.toFixed(1)}%에서 종료 ${finalStats.end.toFixed(
      1
    )}%로 ${finalStats.trend} 추세가 관측됐고, 변동폭은 ${finalStats.swing.toFixed(1)}%입니다. ${verdict}`;
  }

  return {
    summary,
    spatialFindings: spatialFindings.slice(0, 3),
    frequencyFindings: frequencyFindings.slice(0, 3),
    caveats: caveats.slice(0, 3),
    nextSteps: nextSteps.slice(0, 3),
  };
}

function parseLegacyResult(response) {
  const data = response?.data || response || {};
  const representative = data?.representative_analysis || {};
  const representativeAssets = representative?.assets || {};
  const representativeEvidence = representative?.evidence || {};
  const representativeExplanation = representative?.explanation || {};
  const hasRepresentativeEvidence =
    representative && typeof representative === "object" && Object.keys(representative).length > 0;
  const representativeSampleIndex = Number.isFinite(Number(representative?.sample_index))
    ? Number(representative.sample_index)
    : null;

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

  const inferredIsFake =
    typeof data.is_fake === "boolean"
      ? data.is_fake
      : Number.isFinite(Number(data.confidence))
        ? Number(data.confidence) < 50
        : null;
  const timelineExplain = buildTimelineExplainData(timeline, inferredIsFake);

  const representativeSpatialFindings = Array.isArray(representativeExplanation?.spatial_findings)
    ? representativeExplanation.spatial_findings
    : [];
  const representativeFrequencyFindings = Array.isArray(representativeExplanation?.frequency_findings)
    ? representativeExplanation.frequency_findings
    : [];
  const representativeCaveats = Array.isArray(representativeExplanation?.caveats)
    ? representativeExplanation.caveats
    : [];
  const representativeNextSteps = Array.isArray(representativeExplanation?.next_steps)
    ? representativeExplanation.next_steps
    : [];
  const representativeInterpretationGuide = Array.isArray(representativeExplanation?.interpretation_guide)
    ? representativeExplanation.interpretation_guide
    : [];

  const spatialVisualUrl = toRenderableImageUrl(representativeAssets?.gradcam_overlay_url || "");

  const representativeSpatialEvidence = representativeEvidence?.spatial || {};
  const representativeFreqEvidence = representativeEvidence?.frequency || {};

  const preprocessed =
    data.preprocessed && typeof data.preprocessed === "object"
      ? {
          cropImage: toDataUrl(
            data.preprocessed.face_crop_image_b64,
            data.preprocessed.mime_type || "image/jpeg"
          ),
        }
      : null;
  const sourcePreview = normalizeSourcePreview(data?.source_preview || response?.source_preview || null);

  return {
    ...EMPTY_RESULT,
    requestId: "",
    isUndetermined: false,
    confidence: Number.isFinite(data.confidence) ? data.confidence : null,
    pixelScore: Number.isFinite(data.pixel_score) ? data.pixel_score : null,
    freqScore: Number.isFinite(data.freq_score) ? data.freq_score : null,
    isFake: inferredIsFake,
    pValue: Number.isFinite(data.p_value) ? data.p_value : null,
    reliability: data.reliability || "",
    aiCommentSource: String(data.ai_comment_source || "").trim(),
    videoMeta: data.video_meta || null,
    videoRepresentativeConfidence: Number.isFinite(data.video_representative_confidence)
      ? data.video_representative_confidence
      : null,
    videoFrameConfidences,
    videoFramePixelScores,
    videoFrameFreqScores,
    timeline,
    preprocessed,
    comment:
      String(data.ai_comment || "").trim() ||
      buildCommonComment({
        isFake: inferredIsFake,
        confidence: data.confidence,
        pixelScore: data.pixel_score,
        freqScore: data.freq_score,
      }),
    topRegions: Array.isArray(representativeSpatialEvidence?.regions_topk)
      ? representativeSpatialEvidence.regions_topk
      : [],
    dominantBand: representativeFreqEvidence?.dominant_band || "",
    dominantEnergyBand: representativeFreqEvidence?.dominant_energy_band || "",
    bandAblation: Array.isArray(representativeFreqEvidence?.band_ablation)
      ? representativeFreqEvidence.band_ablation
      : [],
    bandEnergy: Array.isArray(representativeFreqEvidence?.band_energy)
      ? representativeFreqEvidence.band_energy
      : [],
    explanationSummary: timelineExplain.summary,
    spatialFindings:
      representativeSpatialFindings.length > 0
        ? representativeSpatialFindings
        : timelineExplain.spatialFindings,
    frequencyFindings:
      representativeFrequencyFindings.length > 0
        ? representativeFrequencyFindings
        : timelineExplain.frequencyFindings,
    spatialVisualUrl,
    sourcePreview,
    inputMediaType: sourcePreview?.kind || "video",
    interpretationGuide:
      representativeInterpretationGuide.length > 0
        ? representativeInterpretationGuide
        : INTERPRETATION_GUIDE_FALLBACK,
    nextSteps: representativeNextSteps.length > 0 ? representativeNextSteps : timelineExplain.nextSteps,
    caveats: representativeCaveats.length > 0 ? representativeCaveats : timelineExplain.caveats,
    evidenceBasis: hasRepresentativeEvidence ? "video_representative_frame" : "",
    representativeSampleIndex,
  };
}

function parseEvidenceResult(response) {
  const score = response?.score || {};
  const faces = Array.isArray(response?.faces) ? response.faces : [];
  const firstFace = faces[0] || {};
  const topLevelComment = String(response?.ai_comment || "").trim();
  const topLevelSource = String(response?.ai_comment_source || "").trim();

  const explanation = firstFace?.explanation || {};
  const spatialEvidence = firstFace?.evidence?.spatial || {};
  const freqEvidence = firstFace?.evidence?.frequency || {};
  const spatialVisualUrl = toRenderableImageUrl(firstFace?.assets?.gradcam_overlay_url || "");

  const hasDetectedFace = faces.length > 0;
  const confidence = hasDetectedFace ? toRealConfidence(score.p_final) : null;
  const isFake = Number.isFinite(confidence) ? confidence < 50 : null;

  const cropImage = toRenderableImageUrl(firstFace?.assets?.face_crop_url || "");
  const preprocessed = cropImage ? { cropImage } : null;
  const sourcePreview = normalizeSourcePreview(response?.source_preview || null);

  return {
    ...EMPTY_RESULT,
    requestId: response?.request_id || "",
    isUndetermined: !hasDetectedFace,
    confidence,
    pixelScore: hasDetectedFace ? toRealConfidence(score.p_rgb) : null,
    freqScore: hasDetectedFace ? toRealConfidence(score.p_freq) : null,
    isFake,
    pValue: null,
    reliability: "",
    aiCommentSource:
      topLevelSource || String(firstFace?.explanation?.summary_source || "").trim(),
    preprocessed,
    comment:
      topLevelComment ||
      String(explanation?.summary || "").trim() ||
      buildCommonComment({
        isFake,
        confidence,
        pixelScore: hasDetectedFace ? toRealConfidence(score.p_rgb) : null,
        freqScore: hasDetectedFace ? toRealConfidence(score.p_freq) : null,
      }),
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
    spatialVisualUrl,
    sourcePreview,
    inputMediaType: sourcePreview?.kind || "image",
    interpretationGuide: Array.isArray(explanation?.interpretation_guide)
      ? explanation.interpretation_guide
      : INTERPRETATION_GUIDE_FALLBACK,
    nextSteps: Array.isArray(explanation?.next_steps) ? explanation.next_steps : [],
    caveats: Array.isArray(explanation?.caveats) ? explanation.caveats : [],
    evidenceBasis: "image_face",
    representativeSampleIndex: null,
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
  let json = null;

  try {
    json = JSON.parse(bodyText);
  } catch {
    if (!response.ok) {
      throw new Error(bodyText || `분석 요청 실패 (status: ${response.status})`);
    }
    throw new Error(`서버 응답 파싱 실패 (status: ${response.status})`);
  }

  if (!response.ok) {
    throw new Error(json?.detail || bodyText || `분석 요청 실패 (status: ${response.status})`);
  }

  return json;
}

async function analyzeMediaUrlWithFastAPI(imageUrl) {
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
  let json = null;

  try {
    json = JSON.parse(bodyText);
  } catch {
    if (!response.ok) {
      throw new Error(bodyText || `분석 요청 실패 (status: ${response.status})`);
    }
    throw new Error(`서버 응답 파싱 실패 (status: ${response.status})`);
  }

  if (!response.ok) {
    throw new Error(json?.detail || bodyText || `분석 요청 실패 (status: ${response.status})`);
  }

  const mediaHint = String(json?.input_media_type || json?.data?.input_media_type || "").toLowerCase();
  const mediaType = mediaHint === "video" || Boolean(json?.video_meta || json?.data?.video_meta) ? "video" : "image";
  return { json, mediaType };
}

export default function Analyze() {
  const [inputMode, setInputMode] = useState("file");
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [urlPreview, setUrlPreview] = useState(null);
  const [videoDuration, setVideoDuration] = useState(0);

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const progressTimerRef = useRef(null);

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, []);

  useEffect(() => {
    return () => {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
      if (previewUrl) {
        if (String(previewUrl).startsWith("blob:")) URL.revokeObjectURL(previewUrl);
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
    setUrlPreview(null);
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
    if (previewUrl && String(previewUrl).startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setInputMode(mode);
    setFile(null);
    setFileType("");
    setPreviewUrl(null);
    setImageUrl("");
    setUrlPreview(null);
    setVideoDuration(0);
    setResult(null);
    setError("");
    stopProgress(0);
  };

  const resetAnalysis = () => {
    if (previewUrl && String(previewUrl).startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setInputMode("file");
    setFile(null);
    setFileType("");
    setPreviewUrl(null);
    setImageUrl("");
    setUrlPreview(null);
    setVideoDuration(0);
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
        if (!imageUrl.trim()) throw new Error("분석할 이미지/영상 URL을 입력해 주세요.");
      }

      setLoading(true);
      const estimatedSeconds =
        inputMode === "file"
          ? requestType === "video"
            ? Math.max(videoDuration * 2, 8)
            : 5
          : 10;
      startProgress(estimatedSeconds);

      let response;
      if (inputMode === "file") {
        response = await analyzeWithFastAPI(requestFile, requestType);
      } else {
        const urlPayload = await analyzeMediaUrlWithFastAPI(imageUrl);
        requestType = urlPayload.mediaType;
        response = urlPayload.json;
      }
      const parsed = parseAnalyzeResponse(response, requestType);
      if (inputMode === "url") {
        setFile(null);
        setPreviewUrl(null);
        setVideoDuration(0);
        setFileType(requestType);
        const fallbackPreview =
          requestType === "video"
            ? { kind: "video", url: imageUrl.trim(), pageUrl: imageUrl.trim() }
            : { kind: "image", url: imageUrl.trim(), pageUrl: imageUrl.trim() };
        setUrlPreview(parsed?.sourcePreview || fallbackPreview);
      } else {
        setUrlPreview(null);
      }
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

  const aiCommentSource = useMemo(() => {
    if (!result) return "";
    return String(result.aiCommentSource || "").trim();
  }, [result]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-indigo-50 py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-8">
        <Header />

        {/* [수정 부분] grid-cols-2 대신 flex 레이아웃 사용 */}
        <div className="flex flex-col lg:flex-row items-stretch gap-6">
          
          {/* 1. 왼쪽 업로드 카드: 고정 너비 또는 최소 너비 설정 */}
          <div className="w-full lg:w-[420px] flex-shrink-0">
            <UploadCard
              mode={inputMode}
              fileType={fileType}
              previewUrl={previewUrl}
              urlPreview={urlPreview}
              imageUrl={imageUrl}
              loading={loading}
              hasResult={Boolean(result)}
              aiComment={aiComment}
              aiCommentSource={aiCommentSource}
              onReset={resetAnalysis}
              onModeChange={onChangeMode}
              onPickFile={onPickFile}
              onUrlChange={(value) => {
                setImageUrl(value);
                if (urlPreview) setUrlPreview(null);
              }}
              onAnalyze={analyze}
            />
          </div>

          {/* 2. 오른쪽 결과 판넬: flex-grow(또는 flex-1)를 통해 왼쪽으로 팽창 */}
          <div className="flex-1 min-w-0">
            <ResultPanel
              progress={progress}
              result={result}
              error={error}
              fileType={fileType || result?.inputMediaType || ""}
              faceImageUrl={result?.preprocessed?.cropImage || null}
            />
          </div>

        </div>

        <ExplainPanel result={result} />
      </div>
    </div>
  );
}
