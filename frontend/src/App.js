import React, { useEffect, useRef, useState } from 'react';
import { Cell, Pie, PieChart, ResponsiveContainer } from 'recharts';
import './index.css';

const FALLBACK_API_BASE =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://127.0.0.1:8000';

const API_BASE = (process.env.REACT_APP_API_BASE || FALLBACK_API_BASE).replace(/\/$/, '');

const EMPTY_RESULT = {
  confidence: null,
  pixelScore: null,
  freqScore: null,
  isFake: null,
  pValue: null,
  reliability: '',
  videoMeta: null,
  comment: '',
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

  const endpoint = fileType === 'video' ? '/analyze-video' : '/analyze';
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
      const data = response?.data || {};

      setAnalysisResult({
        confidence: Number.isFinite(data.confidence) ? data.confidence : null,
        pixelScore: Number.isFinite(data.pixel_score) ? data.pixel_score : null,
        freqScore: Number.isFinite(data.freq_score) ? data.freq_score : null,
        isFake: typeof data.is_fake === 'boolean' ? data.is_fake : null,
        pValue: Number.isFinite(data.p_value) ? data.p_value : null,
        reliability: data.reliability || '',
        videoMeta: data.video_meta || null,
        comment:
          data.is_fake === true
            ? '[경고] 조작 가능성이 높습니다. 추가 검증을 권장합니다.'
            : '[판독 완료] 무결성 지표가 정상 범위입니다.',
      });
      stopProgress(100);
    } catch (error) {
      stopProgress(0);
      alert(error?.message || '분석 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const displayScore =
    analysisResult.confidence !== null ? Math.floor(analysisResult.confidence) : null;
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

          <div className="p-4 bg-black/80 border-l-4 border-[#00f2ff]">
            <h3 className="text-[#00f2ff] text-lg font-bold mb-1 underline">AI 분석관의 한마디</h3>
            <p className="text-gray-200 text-sm font-mono italic">
              {analysisResult.comment || '> 가짜는 반드시 흔적을 남깁니다.'}
            </p>
          </div>

          {analysisResult.videoMeta && (
            <div className="p-4 bg-black/80 border-l-4 border-[#00f2ff] text-sm">
              <p>샘플링 프레임: {analysisResult.videoMeta.sampled_frames}</p>
              <p>추론 사용 프레임: {analysisResult.videoMeta.used_frames}</p>
              <p>추론 실패 프레임: {analysisResult.videoMeta.failed_frames}</p>
              <p>집계 방식: {analysisResult.videoMeta.agg_mode}</p>
            </div>
          )}
        </section>

        <section className="lg:col-span-7 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-10 flex flex-col min-h-[600px] shadow-2xl relative">
            <div className="flex justify-between items-start mb-12">
              <div>
                <p className="text-[#00f2ff]/60 uppercase font-bold mb-2 tracking-widest">신뢰도</p>
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
