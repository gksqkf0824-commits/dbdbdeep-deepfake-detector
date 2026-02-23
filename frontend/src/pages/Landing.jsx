import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
// 💡 아이콘 라이브러리 추가
import { UploadCloud, Cpu, FileCheck } from "lucide-react"; 
import Header from "../components/Header";
import ServicePR from "../components/ServicePR";
import FAQSection from "../components/FAQSection";
import Footer from "../components/Footer";

// 스르륵 나타나는 페이드인 애니메이션 컴포넌트
const FadeInSection = ({ children, delay = "duration-1000" }) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setIsVisible(true);
        });
      },
      { threshold: 0.1 }
    );
    if (domRef.current) observer.observe(domRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={domRef}
      className={`transition-all ease-out transform ${delay} ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-12"
      }`}
    >
      {children}
    </div>
  );
};

export default function Landing() {
  const nav = useNavigate();

  // 💡 전문적인 텍스트와 Lucide 아이콘으로 교체된 배열
  const steps = [
    {
      step: "01",
      title: "검증 대상 업로드",
      desc: "진위 여부를 확인할 이미지, 영상 파일을 시스템에 업로드하거나 분석할 대상 URL을 입력합니다.",
      icon: <UploadCloud className="w-7 h-7 text-[#3182f6]" strokeWidth={2} />,
    },
    {
      step: "02",
      title: "AI 코어 엔진 분석",
      desc: "자체 개발한 딥페이크 탐지 엔진이 픽셀 및 주파수 단위의 미세한 조작 흔적을 다각도로 스캔합니다.",
      icon: <Cpu className="w-7 h-7 text-indigo-500" strokeWidth={2} />,
    },
    {
      step: "03",
      title: "분석 리포트 산출",
      desc: "위조 확률(%) 데이터와 종합 점수가 포함된 리포트를 즉각적으로 제공합니다.",
      icon: <FileCheck className="w-7 h-7 text-teal-500" strokeWidth={2} />,
    },
  ];

  return (
    <div className="min-h-screen bg-[#f9fafb] text-slate-900 font-sans break-keep">
      <Header />

      {/* 1. HERO SECTION (기존과 동일하므로 생략하지 않고 그대로 유지) */}
      <section className="pt-40 pb-32 flex flex-col items-center justify-center relative w-full px-6 mt-10">
        <div className="mb-6 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 text-[#3182f6] text-sm font-semibold">
          <span className="w-2 h-2 rounded-full bg-[#3182f6] animate-pulse"></span>
          실시간 딥페이크 탐지 중
        </div>

        <h1 className="text-5xl md:text-[68px] font-extrabold tracking-tight text-center leading-[1.2] mb-6 text-slate-900">
          업로드 한 번,<br />
          딥페이크·신뢰도 확인
        </h1>
        
        <p className="text-lg md:text-xl text-slate-500 text-center max-w-lg mb-10 font-medium">
          의심되는 사진과 영상을 AI로 분석해 조작 가능성과 신뢰도를 제공합니다.
        </p>

        <button
          onClick={() => nav("/analyze")}
          className="px-10 py-4 rounded-2xl bg-[#3182f6] hover:bg-[#1b64da] text-white text-lg font-bold shadow-[0_8px_20px_rgba(49,130,246,0.3)] hover:-translate-y-1 transition-all duration-300"
        >
          분석하기
        </button>

        <div className="mt-28 grid grid-cols-2 md:grid-cols-4 gap-12 text-center w-full max-w-4xl">
          <div className="hover:-translate-y-1 transition-transform">
            <div className="text-3xl md:text-4xl font-extrabold text-[#3182f6] mb-2">Real-Time</div>
            <div className="text-sm text-slate-500 font-medium">대기 없는 즉시 판별</div>
          </div>
          <div className="hover:-translate-y-1 transition-transform">
            <div className="text-3xl md:text-4xl font-extrabold text-slate-800 mb-2">Deep-Scan</div>
            <div className="text-sm text-slate-500 font-medium">주파수 및 픽셀 다각도 분석</div>
          </div>
          <div className="hover:-translate-y-1 transition-transform">
            <div className="text-3xl md:text-4xl font-extrabold text-slate-800 mb-2">All-in-One</div>
            <div className="text-sm text-slate-500 font-medium">사진·영상·URL 통합 지원</div>
          </div>
          <div className="hover:-translate-y-1 transition-transform">
            <div className="text-3xl md:text-4xl font-extrabold text-slate-800 mb-2">Zero-Log</div>
            <div className="text-sm text-slate-500 font-medium">분석 후 즉시 영구 파기</div>
          </div>
        </div>
      </section>

      {/* 2. 본문 컨텐츠 영역 */}
      <section className="bg-white py-32 rounded-t-[3rem] shadow-[0_-10px_40px_rgba(0,0,0,0.02)]">
        <div className="max-w-5xl mx-auto px-6 space-y-40">

          <div id="pr-section">
            <FadeInSection>
              <ServicePR />
            </FadeInSection>
          </div>

          {/* 💡 Step-by-Step 디자인 변경됨 */}
          <div id="how-to-section">
            <FadeInSection delay="duration-[1000ms]">
              <div className="text-center mb-16">
                <span className="text-[#3182f6] font-bold tracking-wider uppercase text-sm mb-3 block">
                  Workflow
                </span>
                <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4">
                  단 3단계로 끝나는 검증 프로세스
                </h2>
                <p className="text-slate-500 font-medium">
                  복잡한 절차 없이, 고도화된 AI 모델이 위조 여부를 신속하게 판독합니다.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
                {/* 데스크톱 환경에서 스텝 사이를 잇는 점선 배경 */}
                <div className="hidden md:block absolute top-12 left-[15%] right-[15%] h-0.5 border-t-2 border-dashed border-gray-200 z-0"></div>

                {steps.map((item, idx) => (
                  <div key={idx} className="relative z-10 flex flex-col items-center text-center bg-white p-6 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                    
                    {/* 숫자와 아이콘이 겹쳐진 고급스러운 배지 스타일 */}
                    <div className="w-20 h-20 rounded-2xl bg-white border border-gray-100 shadow-sm flex items-center justify-center text-3xl mb-8 font-black text-slate-200 relative">
                      {/* Lucide 아이콘 컨테이너 */}
                      <div className="absolute -top-3 -right-3 bg-white p-2.5 rounded-xl shadow-md border border-gray-50 flex items-center justify-center">
                        {item.icon}
                      </div>
                      {item.step}
                    </div>

                    <h3 className="text-xl font-bold text-slate-900 mb-3">{item.title}</h3>
                    <p className="text-slate-500 font-medium leading-relaxed text-sm">
                      {item.desc}
                    </p>
                  </div>
                ))}
              </div>
            </FadeInSection>
          </div>

          <div id="faq-section">
            <FadeInSection delay="duration-[1000ms]">
              <FAQSection />
            </FadeInSection>
          </div>

        </div>
      </section>

      <Footer />
      
    </div>
  );
}