import React, { useState } from 'react';
// ✅ Gradio client 제거: import { client } from "@gradio/client";
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import './index.css';

// ✅ FastAPI base (Nginx 프록시 사용 권장: /api)
const API_BASE = process.env.REACT_APP_API_BASE || "/api";

// ----------------------
// 1) 도넛 차트 컴포넌트
// ----------------------
const DonutChart = ({ score, label, color = "#00f2ff" }) => {
  const safeScore = Math.max(0, Math.min(100, Number(score ?? 0)));

  const data = [
    { value: safeScore },
    { value: 100 - safeScore },
  ];
  const COLORS = [color, "#1a2634"];

  return (
    <div className="flex flex-col items-center justify-center w-full h-full">
      <ResponsiveContainer width="100%" height="80%">
        <PieChart>
          <Pie
            data={data}
            innerRadius="70%"
            outerRadius="90%"
            paddingAngle={0}
            dataKey="value"
            startAngle={90}
            endAngle={-270}
            stroke="none"
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index]} />
            ))}
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

// ----------------------
// 2) FastAPI 호출 함수
// ----------------------
async function analyzeWithFastAPI(file, fileType) {
  const formData = new FormData();
  formData.append("file", file);

  const endpoint = fileType === "video" ? "/analyze-video" : "/analyze";

  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  // ✅ 413 등에서 HTML이 올 수 있어 안전 파싱
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    // JSON이 아니면 원문을 에러로
    throw new Error(`Server returned non-JSON response (status ${res.status}).`);
  }

  if (!res.ok) {
    throw new Error(json?.detail || `분석 요청 실패 (status ${res.status})`);
  }

  // json: { result_url, data }
  return json;
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [rawFile, setRawFile] = useState(null);
  const [fileType, setFileType] = useState(''); // 'image' | 'video'
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [progress, setProgress] = useState(0);

  const [isUrlMode, setIsUrlMode] = useState(false);
  const [inputUrl, setInputUrl] = useState("");

  // ✅ 이미지 경로 삭제, 점수만 표시
  const [analysisResult, setAnalysisResult] = useState({
    realConfidence: null,
    pixelScore: null,
    freqScore: null,
    comment: ""
  });

  const newsData = [
    { id: 1, src: "/image/news_1.png", label: "EVIDENCE_01" },
    { id: 2, src: "/image/news_2.jpeg", label: "EVIDENCE_02" },
    { id: 3, src: "/image/news_3.jpg", label: "EVIDENCE_03" }
  ];

  const resetResult = () => {
    setAnalysisResult({ realConfidence: null, pixelScore: null, freqScore: null, comment: "" });
    setProgress(0);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // ✅ (선택) 프론트에서 업로드 크기 제한 (Nginx 413 방지)
    // Nginx에서 client_max_body_size도 반드시 올려야 함.
    const MAX_MB = 50;
    if (file.size > MAX_MB * 1024 * 1024) {
      alert(`파일이 너무 큽니다. (${MAX_MB}MB 이하만 업로드 가능)`);
      return;
    }

    setRawFile(file);

    const blobUrl = URL.createObjectURL(file);
    setSelectedFile(blobUrl);

    const isVideo = file.type.startsWith('video');
    setFileType(isVideo ? 'video' : 'image');

    if (isVideo) {
      const v = document.createElement('video');
      v.preload = 'metadata';
      v.onloadedmetadata = () => setVideoDuration(v.duration || 0);
      v.src = blobUrl;
    } else {
      setVideoDuration(0);
    }

    resetResult();
  };

  const handleAnalyze = async () => {
    if (!rawFile) {
      alert("분석할 증거물을 확보하십시오.");
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);

    // 진행바 (대략치)
    const estimatedTime = fileType === 'video'
      ? Math.max(videoDuration * 2, 8)
      : 5;

    const intervalTime = 150;
    const totalSteps = (estimatedTime * 1000) / intervalTime;
    const stepIncrement = 95 / Math.max(totalSteps, 1);

    const timer = setInterval(() => {
      setProgress((prev) => (prev >= 95 ? 95 : prev + stepIncrement));
    }, intervalTime);

    try {
      const { data } = await analyzeWithFastAPI(rawFile, fileType);

      clearInterval(timer);
      setProgress(100);

      // ✅ 백엔드(data) 스키마 기준:
      // image: { confidence, pixel_score, freq_score, is_fake, ... }
      // video: { confidence, ... } (video는 pixel/freq score를 안 주면 null 처리)
      setAnalysisResult({
        realConfidence: data?.confidence ?? null,
        pixelScore: data?.pixel_score ?? null,
        freqScore: data?.freq_score ?? null,
        comment: data?.is_fake
          ? "[경고] 딥러닝 기반 생성 노이즈 패턴 및 프레임 변조 포착."
          : "[판독완료] 데이터 무결성 검증됨. 정상 파일입니다."
      });

    } catch (error) {
      clearInterval(timer);
      setProgress(0);
      alert(error?.message || "분석 오류 발생");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const displayScore = analysisResult.realConfidence !== null ? Math.floor(analysisResult.realConfidence) : null;

  return (
    <div className="min-h-screen forensic-grid p-6 md:p-12 text-[#00f2ff] bg-[#0a0e14]">
      <header className="max-w-[1600px] mx-auto mb-10 flex justify-between items-center border-b-4 border-[#00f2ff] pb-6">
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 bg-[#00f2ff] flex items-center justify-center rounded-sm shadow-[0_0_15px_#00f2ff]">
            <span className="text-black text-xl font-black">NPA</span>
          </div>
          <div>
            <h1 className="text-4xl font-black tracking-tighter">DIGITAL FORENSIC ANALYSIS TERMINAL</h1>
            <p className="text-sm text-[#00f2ff]/70 tracking-[0.3em]">UNIT CODE: 0429-DEEPFAKE-DETECTOR</p>
          </div>
        </div>

        <button
          onClick={() => window.location.reload()}
          className="px-6 py-3 border-2 border-[#00f2ff] hover:bg-[#00f2ff] hover:text-black transition-all text-sm font-black italic"
        >
          REBOOT
        </button>
      </header>

      <main className="max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10">

        {/* LEFT */}
        <section className="lg:col-span-5 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-6 relative">
            {/* URL 모드 토글 (원하면 버튼 UI 다시 붙여도 됨) */}
            {!isUrlMode ? (
              <label htmlFor="file-upload" className="relative aspect-video bg-black/70 border-2 border-dashed border-[#00f2ff]/50 flex flex-col items-center justify-center cursor-pointer overflow-hidden">
                {selectedFile ? (
                  fileType === 'video'
                    ? <video src={selectedFile} className="w-full h-full object-contain" controls />
                    : <img src={selectedFile} alt="Evidence" className="w-full h-full object-contain" />
                ) : (
                  <div className="text-center">
                    <p className="text-2xl font-bold mb-2 opacity-50">SECURE UPLOAD AREA</p>
                    <p className="text-xs text-[#00f2ff]/50">image/*, video/* supported</p>
                  </div>
                )}
                {isAnalyzing && <div className="scan-line"></div>}
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*,video/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>
            ) : (
              <div className="aspect-video bg-black/70 border-2 border-[#00f2ff]/50 p-8 flex flex-col justify-center gap-5">
                <input
                  type="text"
                  value={inputUrl}
                  onChange={(e) => setInputUrl(e.target.value)}
                  placeholder="INPUT TARGET URL..."
                  className="bg-black border-2 border-[#00f2ff]/50 p-4 text-white outline-none"
                />
                <button
                  disabled
                  className="bg-gray-700 text-gray-300 font-black py-4 cursor-not-allowed"
                  title="URL 모드는 서버 다운로드/검증 로직 필요. 운영에서는 비활성 권장"
                >
                  RUN REMOTE ANALYSIS (DISABLED)
                </button>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className={`w-full py-6 font-black text-2xl border-4 ${isAnalyzing ? 'bg-gray-800 text-gray-500 border-gray-700' : 'border-[#00f2ff] hover:bg-[#00f2ff] hover:text-black'}`}
          >
            {isAnalyzing ? "SCANNING DATA..." : "EXECUTE FORENSIC SCAN"}
          </button>

          <div className="p-6 bg-black/80 border-l-8 border-[#00f2ff]">
            <h3 className="text-[#00f2ff] text-xl font-bold mb-3 underline">CHIEF INVESTIGATOR'S LOG</h3>
            <p className="text-gray-200 text-lg font-mono italic">{analysisResult.comment || "> SYSTEM IDLE: AWAITING INPUT..."}</p>
          </div>

          <div className="pt-4 border-t border-[#00f2ff]/20">
            <p className="text-sm font-bold mb-4 tracking-widest text-[#00f2ff]/60 uppercase">Reference Deepfake Cases</p>
            <div className="grid grid-cols-3 gap-4">
              {newsData.map((news) => (
                <div key={news.id} className="aspect-[4/3] bg-gray-900 border-2 border-white/10 hover:border-[#00f2ff] cursor-pointer relative group overflow-hidden transition-all">
                  <img src={news.src} alt={news.label} className="w-full h-full object-cover opacity-40 group-hover:opacity-100 group-hover:scale-110 transition-all duration-500" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent"></div>
                  <div className="absolute bottom-2 left-2 text-[10px] bg-[#00f2ff] text-black px-2 font-bold">{news.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* RIGHT */}
        <section className="lg:col-span-7 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-10 relative h-full flex flex-col">
            <div className="flex justify-between items-start mb-12">
              <div>
                <p className="text-lg text-[#00f2ff]/60 uppercase font-bold tracking-widest mb-2">Integrity Confidence</p>
                <div className="flex items-baseline gap-4">
                  <span className="text-9xl font-black italic text-[#00f2ff] drop-shadow-[0_0_15px_rgba(0,242,255,0.5)]">
                    {displayScore ?? "00"}
                  </span>
                  <span className="text-4xl font-bold">%</span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-[#00f2ff]/60 mb-4 uppercase font-bold tracking-widest">Final Verdict</p>
                {displayScore !== null && (
                  <div className={`px-8 py-4 text-2xl font-black border-4 ${displayScore > 50 ? 'border-green-500 text-green-500' : 'border-red-600 text-red-600 animate-pulse'}`}>
                    {displayScore > 50 ? 'VERIFIED: AUTHENTIC' : 'ALERT: FORGERY DETECTED'}
                  </div>
                )}
              </div>
            </div>

            <div className="mb-12">
              <div className="flex justify-between text-lg font-bold mb-3">
                <span className="tracking-widest">SCANNING FREQUENCY & PIXEL INTEGRITY...</span>
                <span>{Math.floor(progress)}%</span>
              </div>
              <div className="h-4 bg-black border-2 border-[#00f2ff]/30 p-[2px]">
                <div className="h-full bg-[#00f2ff] transition-all duration-300" style={{ width: `${progress}%` }}></div>
              </div>
            </div>

            {/* Donut Charts */}
            <div className="grid grid-cols-2 gap-6 flex-grow">
              <div className="border-2 border-[#00f2ff]/20 p-4 bg-black/50 flex flex-col items-center justify-center">
                <p className="text-sm mb-3 text-[#00f2ff] font-bold border-l-4 border-[#00f2ff] pl-3 self-start">PIXEL INTEGRITY</p>
                <div className="w-full h-full min-h-[200px]">
                  {analysisResult.pixelScore !== null ? (
                    <DonutChart score={analysisResult.pixelScore} label="Pixel Consistency" />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-700">WAITING...</div>
                  )}
                </div>
              </div>

              <div className="border-2 border-[#00f2ff]/20 p-4 bg-black/50 flex flex-col items-center justify-center">
                <p className="text-sm mb-3 text-[#00f2ff] font-bold border-l-4 border-[#00f2ff] pl-3 self-start">FREQUENCY SPECTRUM</p>
                <div className="w-full h-full min-h-[200px]">
                  {analysisResult.freqScore !== null ? (
                    <DonutChart score={analysisResult.freqScore} label="Freq Domain Analysis" color="#ff007f" />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-700">WAITING...</div>
                  )}
                </div>
              </div>
            </div>

            <div className="mt-8 flex justify-between items-center opacity-40 text-xs font-mono border-t border-[#00f2ff]/20 pt-4">
              <span>LOCAL_SECURE_SERVER_PORT: 8080</span>
              <span>ANALYSIS_COMPLETE_DATE: {new Date().toLocaleDateString()}</span>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;