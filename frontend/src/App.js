import React, { useEffect, useRef, useState } from 'react';
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './index.css';

const API_BASE = (process.env.REACT_APP_API_BASE || '').replace(/\/$/, '');

const REGION_LABEL = {
  eyes: '눈 주변',
  nose: '코 주변',
  mouth: '입 주변',
  forehead: '이마',
  jawline: '턱선',
  cheeks: '볼',
};

const FREQ_BAND_META = {
  low: { label: '저주파', range: '0 ~ 0.125 cycles/pixel' },
  mid: { label: '중주파', range: '0.125 ~ 0.25 cycles/pixel' },
  high: { label: '고주파', range: '0.25 ~ 0.5 cycles/pixel' },
};

const toRegionLabel = (region) => REGION_LABEL[region] || region || '미확정';

const toBandLabel = (band, withRange = false) => {
  if (typeof band === 'string' && band.trim().toLowerCase() === 'unknown') {
    return 'UNKNOWN';
  }
  const meta = FREQ_BAND_META[band];
  if (!meta) return band || '미확정';
  return withRange ? `${meta.label}(${meta.range})` : meta.label;
};

const formatSignedDelta = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return 'N/A';
  return `${n >= 0 ? '+' : ''}${n.toFixed(3)}`;
};

const formatRatioPercent = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return 'N/A';
  return `${(n * 100).toFixed(1)}%`;
};

const EMPTY_RESULT = {
  requestId: '',
  confidence: null,
  pixelScore: null,
  freqScore: null,
  isFake: null,
  pValue: null,
  reliability: '',
  videoMeta: null,
  videoRepresentativeConfidence: null,
  videoFrameConfidences: [],
  videoFramePixelScores: [],
  videoFrameFreqScores: [],
  preprocessed: null,
  comment: '',
  topRegions: [],
  dominantBand: '',
  dominantEnergyBand: '',
  explanationSummary: '',
  spatialFindings: [],
  frequencyFindings: [],
  bandAblation: [],
  bandEnergy: [],
  camOverlayUrl: '',
  spectrumUrl: '',
};

const DonutChart = ({ score, label, color = '#00f2ff' }) => {
  const safeScore = Math.max(0, Math.min(100, Number(score ?? 0)));
  const data = [{ value: safeScore }, { value: 100 - safeScore }];

  return (
    <div className="flex flex-col items-center justify-center w-full h-full">
      <ResponsiveContainer width="100%" height="80%">
        <PieChart>
          <Pie
            data={data}
            innerRadius="70%"
            outerRadius="90%"
            dataKey="value"
            startAngle={90}
            endAngle={-270}
            stroke="none"
          >
            <Cell fill={color} />
            <Cell fill="#1a2634" />
          </Pie>
          <text
            x="50%"
            y="50%"
            textAnchor="middle"
            dominantBaseline="middle"
            fill={color}
            className="text-2xl font-black italic"
          >
            {`${Math.floor(safeScore)}%`}
          </text>
        </PieChart>
      </ResponsiveContainer>
      <p className="text-[10px] mt-2 text-[#00f2ff]/60 tracking-widest uppercase">{label}</p>
    </div>
  );
};

async function analyzeWithFastAPI(file, fileType) {
  const formData = new FormData();
  formData.append('file', file);
  if (fileType !== 'video') {
    formData.append('explain', 'true');
    formData.append('evidence_level', 'mvp');
    formData.append('fusion_w', '0.5');
  }

  const endpoint = fileType === 'video' ? '/analyze-video' : '/api/analyze';
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
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

function toDataUrl(base64Payload, mimeType = 'image/jpeg') {
  if (typeof base64Payload !== 'string' || base64Payload.length === 0) {
    return null;
  }
  return `data:${mimeType};base64,${base64Payload}`;
}

function toRenderableImageUrl(url) {
  if (typeof url !== 'string' || url.length === 0) {
    return null;
  }
  if (url.startsWith('data:')) {
    return url;
  }
  if (/^https?:\/\//i.test(url)) {
    return url;
  }
  if (url.startsWith('/')) {
    return `${API_BASE}${url}`;
  }
  return null;
}

const clampProb = (v) => Math.max(0, Math.min(1, Number(v)));
const toRealConfidence = (fakeProb) =>
  Number.isFinite(Number(fakeProb)) ? (1 - clampProb(fakeProb)) * 100 : null;

const parseLegacyResult = (response) => {
  const data = response?.data || response || {};
  const videoFrameConfidences = Array.isArray(data.video_frame_confidences)
    ? data.video_frame_confidences.map(Number).filter(Number.isFinite)
    : [];
  const videoFramePixelScores = Array.isArray(data.video_frame_pixel_scores)
    ? data.video_frame_pixel_scores.map(Number).filter(Number.isFinite)
    : [];
  const videoFrameFreqScores = Array.isArray(data.video_frame_freq_scores)
    ? data.video_frame_freq_scores.map(Number).filter(Number.isFinite)
    : [];

  const preprocessed =
    data.preprocessed && typeof data.preprocessed === 'object'
      ? {
          cropImage: toDataUrl(
            data.preprocessed.face_crop_image_b64,
            data.preprocessed.mime_type || 'image/jpeg'
          ),
        }
      : null;

  return {
    ...EMPTY_RESULT,
    requestId: '',
    confidence: Number.isFinite(data.confidence) ? data.confidence : null,
    pixelScore: Number.isFinite(data.pixel_score) ? data.pixel_score : null,
    freqScore: Number.isFinite(data.freq_score) ? data.freq_score : null,
    isFake: typeof data.is_fake === 'boolean' ? data.is_fake : null,
    pValue: Number.isFinite(data.p_value) ? data.p_value : null,
    reliability: data.reliability || '',
    videoMeta: data.video_meta || null,
    videoRepresentativeConfidence: Number.isFinite(data.video_representative_confidence)
      ? data.video_representative_confidence
      : null,
    videoFrameConfidences,
    videoFramePixelScores,
    videoFrameFreqScores,
    preprocessed,
    comment:
      data.is_fake === true
        ? '[경고] 조작 가능성이 높습니다. 추가 검증을 권장합니다.'
        : '[판독 완료] 무결성 지표가 정상 범위입니다.',
  };
};

const parseEvidenceResult = (response) => {
  const score = response?.score || {};
  const faces = Array.isArray(response?.faces) ? response.faces : [];
  const firstFace = faces[0] || {};
  const explanation = firstFace?.explanation || {};
  const spatialEvidence = firstFace?.evidence?.spatial || {};
  const freqEvidence = firstFace?.evidence?.frequency || {};

  const confidence = toRealConfidence(score.p_final);
  const isFake = Number.isFinite(confidence) ? confidence < 50 : null;

  const cropImage = toRenderableImageUrl(firstFace?.assets?.face_crop_url || '');
  const preprocessed = cropImage ? { cropImage } : null;

  return {
    ...EMPTY_RESULT,
    requestId: response?.request_id || '',
    confidence,
    pixelScore: toRealConfidence(score.p_rgb),
    freqScore: toRealConfidence(score.p_freq),
    isFake,
    pValue: null,
    reliability: '',
    preprocessed,
    comment:
      explanation?.summary ||
      (isFake
        ? '[경고] 비정상 징후가 감지되었습니다. 추가 검증을 권장합니다.'
        : '[판독 완료] 비정상 징후가 낮게 관찰되었습니다.'),
    topRegions: Array.isArray(spatialEvidence?.regions_topk) ? spatialEvidence.regions_topk : [],
    dominantBand: freqEvidence?.dominant_band || '',
    dominantEnergyBand: freqEvidence?.dominant_energy_band || '',
    explanationSummary: explanation?.summary || '',
    spatialFindings: Array.isArray(explanation?.spatial_findings) ? explanation.spatial_findings : [],
    frequencyFindings: Array.isArray(explanation?.frequency_findings) ? explanation.frequency_findings : [],
    bandAblation: Array.isArray(freqEvidence?.band_ablation) ? freqEvidence.band_ablation : [],
    bandEnergy: Array.isArray(freqEvidence?.band_energy) ? freqEvidence.band_energy : [],
    camOverlayUrl: toRenderableImageUrl(firstFace?.assets?.cam_overlay_url || ''),
    spectrumUrl: toRenderableImageUrl(firstFace?.assets?.spectrum_url || ''),
  };
};

const parseAnalyzeResponse = (response, fileType) => {
  if (fileType === 'video') {
    return parseLegacyResult(response);
  }
  if (response?.score && Array.isArray(response?.faces)) {
    return parseEvidenceResult(response);
  }
  return parseLegacyResult(response);
};

function App() {
  const [selectedFileUrl, setSelectedFileUrl] = useState(null);
  const [rawFile, setRawFile] = useState(null);
  const [fileType, setFileType] = useState('');
  const [videoDuration, setVideoDuration] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(EMPTY_RESULT);

  const progressTimerRef = useRef(null);
  const objectUrlRef = useRef(null);

  useEffect(() => {
    return () => {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  const startProgress = (seconds) => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
    }

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

  const resetResult = () => {
    setAnalysisResult(EMPTY_RESULT);
    stopProgress(0);
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    const objectUrl = URL.createObjectURL(file);
    objectUrlRef.current = objectUrl;

    const detectedType = file.type.startsWith('video') ? 'video' : 'image';

    setRawFile(file);
    setSelectedFileUrl(objectUrl);
    setFileType(detectedType);
    resetResult();

    if (detectedType === 'video') {
      const videoEl = document.createElement('video');
      videoEl.preload = 'metadata';
      videoEl.onloadedmetadata = () => setVideoDuration(videoEl.duration || 0);
      videoEl.src = objectUrl;
    } else {
      setVideoDuration(0);
    }
  };

  const handleAnalyze = async () => {
    if (!rawFile || !fileType) {
      alert('분석할 파일을 먼저 업로드하세요.');
      return;
    }

    setIsAnalyzing(true);
    const estimatedSeconds = fileType === 'video' ? Math.max(videoDuration * 2, 8) : 5;
    startProgress(estimatedSeconds);

    try {
      const response = await analyzeWithFastAPI(rawFile, fileType);
      setAnalysisResult(parseAnalyzeResponse(response, fileType));
      stopProgress(100);
    } catch (error) {
      stopProgress(0);
      alert(error?.message || '분석 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const mainConfidence =
    fileType === 'video'
      ? (analysisResult.videoRepresentativeConfidence ?? analysisResult.confidence)
      : analysisResult.confidence;
  const displayScore = mainConfidence !== null ? Math.floor(mainConfidence) : null;
  const trendLength = Math.max(
    analysisResult.videoFramePixelScores.length,
    analysisResult.videoFrameFreqScores.length
  );
  const confidenceTrendData = Array.from({ length: trendLength }, (_, idx) => ({
    frame: idx + 1,
    pixelConfidence:
      idx < analysisResult.videoFramePixelScores.length
        ? Math.max(0, Math.min(100, Number(analysisResult.videoFramePixelScores[idx])))
        : null,
    freqConfidence:
      idx < analysisResult.videoFrameFreqScores.length
        ? Math.max(0, Math.min(100, Number(analysisResult.videoFrameFreqScores[idx])))
        : null,
  }));
  const hasTrendLine = confidenceTrendData.length >= 2;
  const verdict =
    analysisResult.isFake === null
      ? '대기'
      : analysisResult.isFake
        ? '검거'
        : '통과';

  return (
    <div className="min-h-screen forensic-grid p-6 md:p-12 text-[#00f2ff] bg-[#0a0e14]">
      <header className="max-w-[1600px] mx-auto mb-10 flex justify-between items-center border-b-4 border-[#00f2ff] pb-6">
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 bg-[#00f2ff] flex items-center justify-center rounded-sm shadow-[0_0_15px_#00f2ff]">
            <span className="text-black text-xl font-black">dbdb</span>
          </div>
          <h1 className="text-4xl font-black tracking-tighter uppercase">디비디비딥페이크</h1>
        </div>
        <button
          onClick={() => window.location.reload()}
          className="px-8 py-3 border-2 border-[#00f2ff] hover:bg-[#00f2ff] hover:text-black transition-all font-black italic"
        >
          새로고침
        </button>
      </header>

      <main className="max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10">
        <section className="lg:col-span-5 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-6 shadow-inner">
            <label
              htmlFor="evidence-upload"
              className="relative aspect-video bg-black/70 border-2 border-dashed border-[#00f2ff]/50 flex flex-col items-center justify-center cursor-pointer overflow-hidden"
            >
              {selectedFileUrl ? (
                fileType === 'video' ? (
                  <video src={selectedFileUrl} className="w-full h-full object-contain" controls />
                ) : (
                  <img src={selectedFileUrl} alt="업로드 파일 미리보기" className="w-full h-full object-contain" />
                )
              ) : (
                <p className="text-[#00f2ff]/50 font-bold text-center">증거물(이미지/영상)을 업로드하세요.</p>
              )}
              <input
                id="evidence-upload"
                type="file"
                accept="image/*,video/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className={`w-full py-4 font-black text-xl border-4 transition-all ${
              isAnalyzing
                ? 'bg-gray-800 text-gray-500 border-gray-700'
                : 'border-[#00f2ff] hover:bg-[#00f2ff] hover:text-black'
            }`}
          >
            {isAnalyzing ? '데이터 정밀 분석 중...' : '판별하기'}
          </button>

          {(analysisResult.camOverlayUrl || analysisResult.spectrumUrl) && (
            <div className="p-4 bg-black/80 border-l-4 border-[#ff007f] text-sm space-y-2">
              <h3 className="text-[#ff007f] text-base font-bold">시각 근거 이미지</h3>
              <div className="grid grid-cols-2 gap-3 pt-1">
                {analysisResult.camOverlayUrl && (
                  <div>
                    <p className="text-xs opacity-70 mb-1">CAM Overlay</p>
                    <img
                      src={analysisResult.camOverlayUrl}
                      alt="CAM Overlay"
                      className="w-full border border-cyan-700/50"
                    />
                  </div>
                )}
                {analysisResult.spectrumUrl && (
                  <div>
                    <p className="text-xs opacity-70 mb-1">Wavelet</p>
                    <img
                      src={analysisResult.spectrumUrl}
                      alt="Wavelet Map"
                      className="w-full border border-pink-700/50"
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="p-4 bg-black/80 border-l-4 border-[#00f2ff]">
            <h3 className="text-[#00f2ff] text-lg font-bold mb-1 underline">AI 분석관의 한마디</h3>
            <p className="text-gray-200 text-sm font-mono italic whitespace-pre-line leading-relaxed">
              {analysisResult.comment || '> 가짜는 반드시 흔적을 남깁니다.'}
            </p>
          </div>

          {(analysisResult.topRegions.length > 0 ||
            analysisResult.spatialFindings.length > 0 ||
            analysisResult.dominantBand ||
            analysisResult.dominantEnergyBand ||
            analysisResult.bandAblation.length > 0 ||
            analysisResult.bandEnergy.length > 0 ||
            analysisResult.frequencyFindings.length > 0) && (
            <div className="p-4 bg-black/80 border-l-4 border-[#ff007f] text-sm space-y-2">
              <h3 className="text-[#ff007f] text-base font-bold">상세 분석 (Spatial + Frequency)</h3>
              {analysisResult.topRegions.length > 0 && (
                <p>
                  주요 부위:{' '}
                  {analysisResult.topRegions
                    .slice(0, 2)
                    .map((r) => {
                      const cam = Number(r.importance_cam);
                      const camText = Number.isFinite(cam) ? cam.toFixed(2) : 'N/A';
                      return `${toRegionLabel(r.region)} (CAM ${camText})`;
                    })
                    .join(', ')}
                </p>
              )}
              {analysisResult.dominantBand && (
                <p>우세 주파수 대역: {toBandLabel(analysisResult.dominantBand, false)}</p>
              )}
              {analysisResult.dominantEnergyBand && (
                <p>Wavelet 에너지 우세 대역: {toBandLabel(analysisResult.dominantEnergyBand, false)}</p>
              )}
              {analysisResult.bandAblation.length > 0 && (
                <p>
                  밴드 제거 민감도(Δfake):{' '}
                  {analysisResult.bandAblation
                    .map((b) => `${toBandLabel(b.band, false)} ${formatSignedDelta(b.delta_fake_prob)}`)
                    .join(' | ')}
                </p>
              )}
              {analysisResult.bandEnergy.length > 0 && (
                <p>
                  밴드 에너지 비율:{' '}
                  {analysisResult.bandEnergy
                    .map((b) => `${toBandLabel(b.band, false)} ${formatRatioPercent(b.energy_ratio)}`)
                    .join(' | ')}
                </p>
              )}
              {analysisResult.spatialFindings.length > 0 && (
                <div className="pt-1 space-y-1">
                  <p className="text-xs opacity-80">이미지 세부 해석</p>
                  {analysisResult.spatialFindings.slice(0, 3).map((s, idx) => (
                    <p key={`spatial-finding-${idx}`} className="text-xs leading-relaxed text-cyan-100/90">
                      {idx + 1}. {s.claim} ({s.evidence})
                    </p>
                  ))}
                </div>
              )}
              {analysisResult.frequencyFindings.length > 0 && (
                <div className="pt-1 space-y-1">
                  <p className="text-xs opacity-80">주파수 세부 해석</p>
                  {analysisResult.frequencyFindings.slice(0, 3).map((f, idx) => (
                    <p key={`freq-finding-${idx}`} className="text-xs leading-relaxed text-pink-100/90">
                      {idx + 1}. {f.claim} ({f.evidence})
                    </p>
                  ))}
                </div>
              )}
              <div className="mt-2 pt-3 border-t border-[#ff007f]/30 text-xs text-gray-200/90 space-y-1 leading-relaxed">
                <p className="font-bold text-[#ffd6ea]">해석 가이드</p>
                <p>주요 부위: 모델이 얼굴에서 특히 주목한 위치(CAM 기반)입니다.</p>
                <p>우세 주파수 대역: 밴드를 제거했을 때 예측 변화가 가장 큰 구간입니다.</p>
                <p>밴드 제거 민감도(Δfake): 각 대역 제거 전후의 fake 확률 변화량입니다.</p>
                <p>밴드 에너지 비율: Wavelet 에너지가 각 대역에 분포한 상대 비율입니다.</p>
                <p>
                  저주파({FREQ_BAND_META.low.range}): 얼굴의 큰 윤곽, 완만한 밝기/색 변화 같은 저해상 구조 성분입니다.
                </p>
                <p>
                  중주파({FREQ_BAND_META.mid.range}): 눈/코/입 주변 경계, 피부 결 등 중간 규모 텍스처 성분입니다.
                </p>
                <p>
                  고주파({FREQ_BAND_META.high.range}): 미세 경계, 세부 노이즈, 과도한 샤프닝/압축 잔상에 민감한 성분입니다.
                </p>
              </div>
            </div>
          )}

          {analysisResult.videoMeta && (
            <div className="p-4 bg-black/80 border-l-4 border-[#00f2ff] text-sm">
              <p>샘플링 프레임: {analysisResult.videoMeta.sampled_frames}</p>
              <p>추론 사용 프레임: {analysisResult.videoMeta.used_frames}</p>
              <p>추론 실패 프레임: {analysisResult.videoMeta.failed_frames}</p>
              <p>집계 방식: {analysisResult.videoMeta.agg_mode}</p>
              {Number.isFinite(analysisResult.videoMeta.excluded_low_count) &&
                Number.isFinite(analysisResult.videoMeta.excluded_high_count) && (
                  <p>
                    대표 신뢰도 제외 프레임(하위/상위): {analysisResult.videoMeta.excluded_low_count}/
                    {analysisResult.videoMeta.excluded_high_count}
                  </p>
                )}
            </div>
          )}

          {fileType === 'image' && analysisResult.preprocessed && (
            <div className="p-4 bg-black/80 border-l-4 border-[#00f2ff] text-sm space-y-3">
              <h3 className="text-[#00f2ff] text-lg font-bold underline">얼굴 전처리 결과</h3>
              <div className="bg-[#121b28] border border-[#00f2ff]/30 p-2">
                <p className="text-xs mb-2">얼굴 크롭(224x224)</p>
                {analysisResult.preprocessed.cropImage ? (
                  <img
                    src={analysisResult.preprocessed.cropImage}
                    alt="얼굴 크롭 전처리 결과"
                    className="w-full max-w-[320px] aspect-square object-contain bg-black mx-auto"
                  />
                ) : (
                  <div className="w-full max-w-[320px] aspect-square flex items-center justify-center text-gray-600 bg-black mx-auto">
                    NO IMAGE
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        <section className="lg:col-span-7 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-10 flex flex-col min-h-[600px] shadow-2xl relative">
            <div className="flex justify-between items-start mb-12">
              <div>
                <p className="text-[#00f2ff]/60 uppercase font-bold mb-2 tracking-widest">
                  {fileType === 'video' ? '대표 신뢰도' : '신뢰도'}
                </p>
                <div className="flex items-baseline gap-4">
                  <span className="text-9xl font-black italic text-[#00f2ff]">{displayScore ?? '00'}</span>
                  <span className="text-4xl font-bold">%</span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-[#00f2ff]/60 mb-4 font-bold uppercase">진위여부</p>
                <div
                  className={`px-8 py-4 text-2xl font-black border-4 ${
                    verdict === '통과'
                      ? 'border-green-500 text-green-500'
                      : verdict === '검거'
                        ? 'border-red-600 text-red-600 animate-pulse'
                        : 'border-[#00f2ff]/30 text-[#00f2ff]/40'
                  }`}
                >
                  {verdict}
                </div>
              </div>
            </div>

            <div className="mb-10">
              <div className="flex justify-between text-[10px] mb-1 font-mono">
                <span>ANALYSIS PROGRESS</span>
                <span>{Math.floor(progress)}%</span>
              </div>
              <div className="h-3 bg-black border border-[#00f2ff]/30 relative overflow-hidden">
                <div
                  className="h-full bg-[#00f2ff] shadow-[0_0_15px_#00f2ff] transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-grow">
              <div className="border-2 border-[#00f2ff]/20 p-4 bg-black/50 flex flex-col items-center justify-center">
                <p className="text-sm mb-3 text-[#00f2ff] font-bold border-l-4 border-[#00f2ff] pl-3 self-start">
                  PIXEL SCORE
                </p>
                <div className="w-full h-full min-h-[200px]">
                  {analysisResult.pixelScore !== null ? (
                    <DonutChart score={analysisResult.pixelScore} label="Pixel Integrity" />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-700">WAITING...</div>
                  )}
                </div>
              </div>

              <div className="border-2 border-[#00f2ff]/20 p-4 bg-black/50 flex flex-col items-center justify-center">
                <p className="text-sm mb-3 text-[#00f2ff] font-bold border-l-4 border-[#00f2ff] pl-3 self-start">
                  FREQUENCY SCORE
                </p>
                <div className="w-full h-full min-h-[200px]">
                  {analysisResult.freqScore !== null ? (
                    <DonutChart score={analysisResult.freqScore} label="Frequency Analysis" color="#ff007f" />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-700">WAITING...</div>
                  )}
                </div>
              </div>
            </div>

            {fileType === 'video' && analysisResult.confidence !== null && hasTrendLine && (
              <div className="mt-8 border-2 border-[#00f2ff]/20 p-4 bg-black/50">
                <p className="text-sm mb-3 text-[#00f2ff] font-bold border-l-4 border-[#00f2ff] pl-3">
                  VIDEO PIXEL/FREQUENCY TREND
                </p>
                <div className="h-[220px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={confidenceTrendData} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2f40" />
                      <XAxis
                        dataKey="frame"
                        stroke="#00f2ff"
                        tick={{ fontSize: 10, fill: '#7dd3fc' }}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        stroke="#00f2ff"
                        tick={{ fontSize: 10, fill: '#7dd3fc' }}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0a0e14', border: '1px solid #00f2ff' }}
                        labelStyle={{ color: '#00f2ff' }}
                        formatter={(value, name) => [
                          `${Number(value).toFixed(2)}%`,
                          name === 'pixelConfidence' ? 'Pixel 신뢰도' : 'Frequency 신뢰도',
                        ]}
                        labelFormatter={(label) => `프레임 #${label}`}
                      />
                      <Line
                        type="monotone"
                        dataKey="pixelConfidence"
                        stroke="#00f2ff"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, stroke: '#00f2ff', fill: '#0a0e14' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="freqConfidence"
                        stroke="#ff007f"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, stroke: '#ff007f', fill: '#0a0e14' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
            {fileType === 'video' && analysisResult.confidence !== null && !hasTrendLine && (
              <div className="mt-8 border-2 border-[#00f2ff]/20 p-4 bg-black/50 text-sm text-[#00f2ff]/70">
                라인 그래프를 그리기 위한 유효 프레임이 부족합니다.
              </div>
            )}

            <div className="mt-8 text-xs font-mono opacity-70 border-t border-[#00f2ff]/20 pt-3">
              <span>
                p-value: {analysisResult.pValue ?? '-'} / reliability: {analysisResult.reliability || '-'}
              </span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
