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

function buildCommonComment({ isFake, confidence, pixelScore, freqScore }) {
  if (isFake === true) {
    return "분석 결과, 조작 가능성이 상대적으로 높게 관측되었습니다. 아래 세부 근거를 함께 확인해 주세요.";
  }
  if (isFake === false) {
    return "분석 결과, 원본 가능성이 상대적으로 높게 관측되었습니다. 아래 세부 근거를 함께 확인해 주세요.";
  }
  if ([confidence, pixelScore, freqScore].some((v) => Number.isFinite(Number(v)))) {
    return "분석이 완료되었습니다. 아래 세부 근거를 확인해 주세요.";
  }
  return "분석이 완료되었습니다. 결과 패널의 세부 근거를 확인해 주세요.";
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
      claim: "최종 신뢰도 추이를 시간축으로 확인했습니다.",
      evidence: `시작 ${finalStats.start.toFixed(1)}% · 중간 ${finalStats.mid.toFixed(
        1
      )}% · 종료 ${finalStats.end.toFixed(1)}% (추세 ${finalStats.trend}, 변동폭 ${finalStats.swing.toFixed(
        1
      )}%)`,
    });
  }

  if (pixelStats) {
    spatialFindings.push({
      claim: "픽셀 계열 신호의 구간별 변화도 함께 반영했습니다.",
      evidence: `시작 ${pixelStats.start.toFixed(1)}% · 종료 ${pixelStats.end.toFixed(
        1
      )}% (변동폭 ${pixelStats.swing.toFixed(1)}%)`,
    });
  }

  if (freqStats) {
    frequencyFindings.push({
      claim: "주파수(SRM) 신호의 시간대별 변화를 확인했습니다.",
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
      claim: "최종 점수와 주파수 신호의 방향성 일치 여부를 점검했습니다.",
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
    explanationSummary: timelineExplain.summary,
    spatialFindings: timelineExplain.spatialFindings,
    frequencyFindings: timelineExplain.frequencyFindings,
    nextSteps: timelineExplain.nextSteps,
    caveats: timelineExplain.caveats,
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

  const hasDetectedFace = faces.length > 0;
  const confidence = hasDetectedFace ? toRealConfidence(score.p_final) : null;
  const isFake = Number.isFinite(confidence) ? confidence < 50 : null;

  const cropImage = toRenderableImageUrl(firstFace?.assets?.face_crop_url || "");
  const preprocessed = cropImage ? { cropImage } : null;

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

  const resetAnalysis = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setInputMode("file");
    setFile(null);
    setFileType("");
    setPreviewUrl(null);
    setImageUrl("");
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

  const aiCommentSource = useMemo(() => {
    if (!result) return "";
    return String(result.aiCommentSource || "").trim();
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
            aiCommentSource={aiCommentSource}
            onReset={resetAnalysis}
            onModeChange={onChangeMode}
            onPickFile={onPickFile}
            onUrlChange={setImageUrl}
            onAnalyze={analyze}
          />

          <ResultPanel
            progress={progress}
            result={result}
            error={error}
            fileType={fileType}
            faceImageUrl={result?.preprocessed?.cropImage || null}
          />
        </div>

        <ExplainPanel result={result} />
      </div>
    </div>
  );
}
