import { useNavigate, useLocation } from "react-router-dom"; // 💡 useLocation 추가
import { useEffect, useRef, useState } from "react";
import { UploadCloud, Cpu, FileCheck } from "lucide-react"; 
import Header from "../components/Header";
import ServicePR from "../components/ServicePR";
import FAQSection from "../components/FAQSection";
import Footer from "../components/Footer";

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
  const location = useLocation(); // 💡 현재 경로 및 해시를 추적하기 위해 추가

  // 💡 [추가] 외부 페이지(추론페이지 등)에서 해시를 들고 왔을 때 해당 위치로 스크롤
  useEffect(() => {
    if (location.hash) {
      // #pr-section 에서 #을 제거하고 id만 추출
      const id = location.hash.replace("#", "");
      const element = document.getElementById(id);
      
      if (element) {
        // 페이지 렌더링 후 약간의 시간차를 두어 정확한 위치를 잡습니다.
        setTimeout(() => {
          const headerOffset = 64;
          const elementPosition = element.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.scrollY - headerOffset;

          window.scrollTo({
            top: offsetPosition,
            behavior: "smooth"
          });
        }, 100);
      }
    }
  }, [location]); // location 정보가 바뀔 때마다 실행

  const featureCards = [
    { title: "Real-Time", desc: "대기 없는 즉시 판별" },
    { title: "Deep-Scan", desc: "주파수 및 픽셀 다각도 분석" },
    { title: "All-in-One", desc: "사진·영상·URL 통합 지원" },
    { title: "Zero-Log", desc: "분석 후 즉시 영구 파기" },
  ];
  
  const [activeFeatureIdx, setActiveFeatureIdx] = useState(0);
  const [hoveredFeatureIdx, setHoveredFeatureIdx] = useState(null);
  const featureCycleRef = useRef(0);

  useEffect(() => {
    if (hoveredFeatureIdx !== null) return;
    const timer = setInterval(() => {
      featureCycleRef.current = (featureCycleRef.current + 1) % featureCards.length;
      setActiveFeatureIdx(featureCycleRef.current);
    }, 1800);
    return () => clearInterval(timer);
  }, [hoveredFeatureIdx, featureCards.length]);

  const onFeatureEnter = (idx) => {
    featureCycleRef.current = idx;
    setHoveredFeatureIdx(idx);
    setActiveFeatureIdx(idx);
  };

  const onFeatureLeave = () => {
    const nextIdx = (featureCycleRef.current + 1) % featureCards.length;
    featureCycleRef.current = nextIdx;
    setHoveredFeatureIdx(null);
    setActiveFeatureIdx(nextIdx);
  };

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
          {featureCards.map((item, idx) => {
            const isActive = activeFeatureIdx === idx;
            return (
              <div
                key={item.title}
                onMouseEnter={() => onFeatureEnter(idx)}
                onMouseLeave={onFeatureLeave}
                className={`cursor-default transition-all duration-500 ${
                  isActive ? "-translate-y-1 scale-[1.03]" : ""
                }`}
              >
                <div
                  className={`text-3xl md:text-4xl font-extrabold mb-2 whitespace-nowrap transition-colors duration-500 ${
                    isActive ? "text-[#3182f6]" : "text-slate-800"
                  }`}
                >
                  {item.title}
                </div>
                <div
                  className={`text-sm font-medium transition-colors duration-500 ${
                    isActive ? "text-[#3182f6]" : "text-slate-500"
                  }`}
                >
                  {item.desc}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="bg-white py-32 rounded-t-[3rem] shadow-[0_-10px_40px_rgba(0,0,0,0.02)]">
        <div className="max-w-5xl mx-auto px-6 space-y-40">

          <div id="pr-section">
            <FadeInSection>
              <ServicePR />
            </FadeInSection>
          </div>

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
                <div className="hidden md:block absolute top-12 left-[15%] right-[15%] h-0.5 border-t-2 border-dashed border-gray-200 z-0"></div>

                {steps.map((item, idx) => (
                  <div key={idx} className="relative z-10 flex flex-col items-center text-center bg-white p-6 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                    <div className="w-20 h-20 rounded-2xl bg-white border border-gray-100 shadow-sm flex items-center justify-center text-3xl mb-8 font-black text-slate-200 relative">
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