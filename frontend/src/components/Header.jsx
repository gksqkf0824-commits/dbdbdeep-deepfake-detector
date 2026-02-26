import React from 'react';
import { useNavigate, useLocation } from "react-router-dom";

export default function Header() {
  const nav = useNavigate();
  const location = useLocation();

  const handleLogoClick = () => {
    if (location.pathname !== "/") {
      nav("/");
    } else {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const scrollToSection = (sectionId) => {
    if (location.pathname !== "/") {
      nav(`/#${sectionId}`);
    } else {
      const element = document.getElementById(sectionId);
      if (element) {
        const headerOffset = 64; 
        const elementPosition = element.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.scrollY - headerOffset;
        window.scrollTo({ top: offsetPosition, behavior: "smooth" });
      }
    }
  };

  const handleSolutionClick = () => {
    if (location.pathname === "/analyze") {
      window.location.reload();
      return;
    }
    nav("/analyze");
  };

  // 🎮 게임 버튼 클릭 핸들러 추가
  const handleGameClick = () => {
    if (location.pathname === "/game") {
      // 이미 게임 페이지라면 페이지를 새로고침하여 새로운 랜덤 문제 생성
      window.location.reload();
    } else {
      // 아니라면 게임 페이지로 이동
      nav("/game");
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div 
          className="text-xl font-extrabold text-slate-900 cursor-pointer tracking-tighter hover:text-[#3182f6] transition-colors duration-300"
          onClick={handleLogoClick} 
        >
          DBDBDEEP
        </div>

        <nav className="hidden md:flex gap-8 text-slate-600 font-medium text-sm">
          <button onClick={() => scrollToSection("pr-section")} className="hover:text-[#3182f6] transition">기술 및 특징</button>
          <button onClick={() => scrollToSection("how-to-section")} className="hover:text-[#3182f6] transition">검증 프로세스</button>
          <button onClick={() => scrollToSection("faq-section")} className="hover:text-[#3182f6] transition">지원 센터</button>
        </nav>

        <div className="flex items-center gap-3">
          <button
            onClick={handleGameClick}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-bold py-2 px-5 rounded-lg transition border border-slate-200"
          >
            🎮 AI와 대결해보기
          </button>

          <button
            onClick={handleSolutionClick}
            className="bg-[#3182f6] hover:bg-[#1b64da] text-white text-sm font-semibold py-2 px-5 rounded-lg transition shadow-md shadow-blue-100"
          >
            솔루션 체험하기
          </button>
        </div>
      </div>
    </header>
  );
}