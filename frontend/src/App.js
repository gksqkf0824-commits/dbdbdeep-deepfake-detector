/* eslint-disable jsx-a11y/alt-text */
import React, { useState, useRef } from 'react';
// import { client } from "@gradio/client";
import './index.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [rawFile, setRawFile] = useState(null);
  const [fileType, setFileType] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const progressTimer = useRef(null); // 타이머 관리를 위한 ref

  const [isUrlMode, setIsUrlMode] = useState(false);
  const [inputUrl, setInputUrl] = useState("");
  const [isExtracting, setIsExtracting] = useState(false);

  const [analysisResult, setAnalysisResult] = useState({
    srmImg: null,
    pixelImg: null,
    graphImg: null,
    urlGridImg: null,
    realConfidence: null,
    comment: ""
  });

  const newsData = [
    { id: 1, src: "/image/news_1.png", label: "EVIDENCE_01" },
    { id: 2, src: "/image/news_2.jpeg", label: "EVIDENCE_02" },
    { id: 3, src: "/image/news_3.jpg", label: "EVIDENCE_03" }
  ];

  // 프로그레스 바 로직: 목표 시간(초) 동안 서서히 증가
  const startProgress = (seconds) => {
    clearInterval(progressTimer.current);
    setProgress(0);
    const interval = 100; // 0.1초마다 업데이트
    const step = 100 / (seconds * (1000 / interval));

    progressTimer.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) { // 응답 대기를 위해 95%에서 멈춤
          clearInterval(progressTimer.current);
          return prev;
        }
        return prev + step;
      });
    }, interval);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setRawFile(file);
      setSelectedFile(URL.createObjectURL(file));
      const isVideo = file.type.startsWith('video');
      setFileType(isVideo ? 'video' : 'image');
      setAnalysisResult({ srmImg: null, pixelImg: null, graphImg: null, urlGridImg: null, realConfidence: null, comment: "" });
      setProgress(0);
      clearInterval(progressTimer.current);
    }
  };

  const handleUrlAnalyze = async () => {
    if (!inputUrl) {
      alert("타겟 URL을 입력하십시오.");
      return;
    }
    setIsExtracting(true);
    setAnalysisResult({ srmImg: null, pixelImg: null, graphImg: null, urlGridImg: null, realConfidence: null, comment: "" });
    
    startProgress(10); // URL 모드: 10초 설정

    try {
      const app = await client("euntaejang/deepfake");
      const result = await app.predict("/extract_url", [inputUrl]);
      
      if (result.data) {
        clearInterval(progressTimer.current);
        setProgress(100); // 즉시 완료
        setAnalysisResult({
          realConfidence: result.data[0],
          urlGridImg: result.data[1]?.url,
          comment: `[원격분석완료] ${result.data[2]}`
        });
      }
    } catch (error) {
      clearInterval(progressTimer.current);
      setProgress(0);
      alert("URL 분석 실패");
    } finally {
      setIsExtracting(false);
    }
  };

  const handleAnalyze = async () => {
    if (!rawFile) {
      alert("분석할 증거물을 확보하십시오.");
      return;
    }
    setIsAnalyzing(true);
    
    // 시간 설정: 이미지는 5초, 동영상은 메타데이터 기반(없으면 기본 20초)
    if (fileType === 'image') {
      startProgress(5);
    } else {
      // 동영상 소요 시간 계산을 위해 비디오 객체 생성
      const tempVideo = document.createElement('video');
      tempVideo.src = selectedFile;
      tempVideo.onloadedmetadata = () => {
        const duration = tempVideo.duration || 10;
        startProgress(duration * 2);
      };
    }

    try {
      const app = await client("euntaejang/deepfake");
      const endpoint = fileType === 'video' ? "/predict_video" : "/predict";
      const apiResult = await app.predict(endpoint, [rawFile]);

      clearInterval(progressTimer.current);
      setProgress(100);

      if (fileType === 'video') {
        setAnalysisResult({
          realConfidence: apiResult.data[0],
          graphImg: apiResult.data[1]?.url,
          comment: "[영상 타임라인 분석 완료] 데이터 무결성 검증됨."
        });
      } else {
        setAnalysisResult({
          realConfidence: apiResult.data[0],
          srmImg: apiResult.data[1]?.url,
          pixelImg: apiResult.data[2]?.url,
          comment: apiResult.data[0] > 50 
            ? "[판독완료] 픽셀 및 주파수 무결성 통과." 
            : "[경고] 생성 노이즈 및 주파수 변조 감지."
        });
      }
    } catch (error) {
      clearInterval(progressTimer.current);
      setProgress(0);
      alert("얼굴을 검출할 수 없습니다.");
    } finally {
      setIsAnalyzing(false);
    }
  };

 const displayScore = analysisResult.realConfidence !== null
  ? (isUrlMode
      ? analysisResult.realConfidence
      : Math.floor(analysisResult.realConfidence))
  : null;
  return (
    <div className="min-h-screen forensic-grid p-6 md:p-12 text-[#00f2ff] bg-[#0a0e14]">
      <header className="max-w-[1600px] mx-auto mb-10 flex justify-between items-center border-b-4 border-[#00f2ff] pb-6">
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 bg-[#00f2ff] flex items-center justify-center rounded-sm shadow-[0_0_15px_#00f2ff]">
            <span className="text-black text-xl font-black">dbdb</span>
          </div>
          <h1 className="text-4xl font-black tracking-tighter uppercase">디비디비딥페이크</h1>
        </div>
        <button onClick={() => {
          clearInterval(progressTimer.current);
          window.location.reload();
        }} className="px-8 py-3 border-2 border-[#00f2ff] hover:bg-[#00f2ff] hover:text-black transition-all font-black italic">
          새로고침
        </button>
      </header>

      <main className="max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10">
        <section className="lg:col-span-5 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-6 shadow-inner">
            <div className="flex justify-between mb-6">
              <button onClick={() => { setIsUrlMode(false); setProgress(0); }} className={`flex-1 py-3 font-bold border-b-4 ${!isUrlMode ? 'border-[#00f2ff] bg-[#00f2ff]/10' : 'border-transparent text-gray-500'}`}>사진/동영상</button>
              <button onClick={() => { setIsUrlMode(true); setProgress(0); }} className={`flex-1 py-3 font-bold border-b-4 ${isUrlMode ? 'border-[#00f2ff] bg-[#00f2ff]/10' : 'border-transparent text-gray-500'}`}>URL링크</button>
            </div>

            {!isUrlMode ? (
              <label className="relative aspect-video bg-black/70 border-2 border-dashed border-[#00f2ff]/50 flex flex-col items-center justify-center cursor-pointer overflow-hidden">
                {selectedFile ? (
                  fileType === 'video' ? <video src={selectedFile} className="w-full h-full object-contain" /> : <img src={selectedFile} className="w-full h-full object-contain" />
                ) : (
                  <p className="text-[#00f2ff]/50 font-bold text-center">증거물을 업로드하세요..</p>
                )}
                {isAnalyzing && <div className="scan-line"></div>}
                <input type="file" className="hidden" onChange={handleFileChange} />
              </label>
            ) : (
              <div className="aspect-video bg-black/70 border-2 border-[#00f2ff]/50 p-8 flex flex-col justify-center gap-5">
                <input 
                  type="text" 
                  placeholder="url을 넣으세요..."
                  className="bg-black border-2 border-[#00f2ff]/50 p-4 outline-none text-white font-mono"
                  value={inputUrl}
                  onChange={(e) => setInputUrl(e.target.value)}
                />
                <button onClick={handleUrlAnalyze} disabled={isExtracting} className="bg-[#00f2ff] text-black font-black py-4 hover:bg-white transition-all disabled:bg-gray-600">
                  {isExtracting ? "수사팀 진입 중..." : "해당 url로 수사팀 투입하기 "}
                </button>
              </div>
            )}
          </div>

          <button onClick={handleAnalyze} disabled={isAnalyzing || isUrlMode} className="w-full py-4 font-black text-xl border-4 border-[#00f2ff] hover:bg-[#00f2ff] hover:text-black transition-all">
            {isAnalyzing ? "데이터 정밀 분석 중..." : "판별하기"}
          </button>

          <div className="grid grid-cols-3 gap-4">
            {newsData.map((news) => (
              <div key={news.id} className="border-2 border-[#00f2ff]/30 bg-black shadow-lg">
                <img src={news.src} className="w-full h-32 object-cover" />
                <p className="text-[10px] text-center font-bold py-1 bg-[#00f2ff]/10">{news.label}</p>
              </div>
            ))}
          </div>

          <div className="p-4 bg-black/80 border-l-4 border-[#00f2ff]">
            <h3 className="text-[#00f2ff] text-lg font-bold mb-1 underline">AI 분석관의 한마디</h3>
            <p className="text-gray-200 text-sm font-mono italic">{analysisResult.comment || "> 가짜는 반드시 흔적을 남깁니다."}</p>
          </div>
        </section>

        <section className="lg:col-span-7 space-y-6">
          <div className="bg-[#121b28] border-2 border-[#00f2ff]/40 p-10 flex flex-col min-h-[600px] shadow-2xl relative">
            <div className="flex justify-between items-start mb-12">
              <div>
                <p className="text-[#00f2ff]/60 uppercase font-bold mb-2 tracking-widest">신뢰도</p>
                <div className="flex items-baseline gap-4">
                  <span className="text-9xl font-black italic text-[#00f2ff]">{displayScore ?? "00"}</span>
                  {!isUrlMode && (
                  <span className="text-4xl font-bold">%</span>
                  )}
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-[#00f2ff]/60 mb-4 font-bold uppercase">진위여부</p>
                {/* 일반 모드 */}
                {!isUrlMode && displayScore !== null && (
                  <div className={`px-8 py-4 text-2xl font-black border-4 ${
                     displayScore > 50
                    ? 'border-green-500 text-green-500'
                    : 'border-red-600 text-red-600 animate-pulse'
                    }`}>
                    {displayScore > 50 ? '통과' : '검거'}
                    </div>
                      )}

                  {/* URL 모드 */}
                  {isUrlMode && displayScore && (
                  <div className="px-8 py-4 text-2xl font-black border-4 border-[#00f2ff] text-[#00f2ff]">
                  url 분석 완료
                  </div>
                  )}                
              </div>
            </div>

            {/* [상황판 로딩바] */}
            <div className="mb-10">
              <div className="flex justify-between text-[10px] mb-1 font-mono">
                <span>ANALYSIS PROGRESS</span>
                <span>{Math.floor(progress)}%</span>
              </div>
              <div className="h-3 bg-black border border-[#00f2ff]/30 relative overflow-hidden">
                <div 
                  className="h-full bg-[#00f2ff] shadow-[0_0_15px_#00f2ff] transition-all duration-300 ease-out" 
                  style={{ width: `${progress}%` }}
                ></div>
                {/* 로딩 바 위를 지나가는 광원 효과 추가 */}
                <div className="absolute top-0 left-0 w-full h-full scan-bar-light"></div>
              </div>
            </div>

            <div className="flex-grow">
              <p className="text-sm mb-4 text-[#00f2ff] font-bold border-l-4 border-[#00f2ff] pl-3 uppercase tracking-tighter">
                상황실 메인 스크린
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
                {!isUrlMode && fileType === 'image' && analysisResult.srmImg && (
                  <>
                    <div className="border border-[#00f2ff]/20 bg-black/40 p-4 flex flex-col items-center">
                      <span className="text-[10px] text-[#00f2ff]/50 mb-2 font-mono uppercase tracking-widest">주파수 분석결과</span>
                      <img src={analysisResult.srmImg} className="w-full h-auto object-contain" />
                    </div>
                    <div className="border border-[#00f2ff]/20 bg-black/40 p-4 flex flex-col items-center">
                      <span className="text-[10px] text-[#00f2ff]/50 mb-2 font-mono uppercase tracking-widest">이미지 분석결과</span>
                      <img src={analysisResult.pixelImg} className="w-full h-auto object-contain" />
                    </div>
                  </>
                )}

                {!isUrlMode && fileType === 'video' && analysisResult.graphImg && (
                  <div className="col-span-2 border border-[#00f2ff]/20 bg-black/40 p-4">
                    <img src={analysisResult.graphImg} className="w-full h-auto object-contain" />
                  </div>
                )}

                {isUrlMode && analysisResult.urlGridImg && (
                  <div className="col-span-2 border border-[#00f2ff]/20 bg-black/40 p-4 overflow-auto">
                    <img src={analysisResult.urlGridImg} className="w-full h-auto object-contain" />
                  </div>
                )}

                {!analysisResult.srmImg && !analysisResult.graphImg && !analysisResult.urlGridImg && (
                  <div className="col-span-2 aspect-video bg-gray-900/50 border border-dashed border-[#00f2ff]/10 flex items-center justify-center">
                    <span className="text-sm opacity-20 uppercase tracking-[0.4em]">디지털 판독 대기중...</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;