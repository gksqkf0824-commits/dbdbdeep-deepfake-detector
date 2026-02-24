import { useNavigate, useLocation } from "react-router-dom";

export default function Header() {
  const nav = useNavigate();
  const location = useLocation();

  // 💡 로고 클릭 시 실행될 함수 추가
  const handleLogoClick = () => {
    if (location.pathname !== "/") {
      // 메인 페이지가 아니라면 메인으로 이동 (해시 없이)
      nav("/");
      // 페이지 이동 후 자동으로 최상단에 위치하게 됩니다.
    } else {
      // 메인 페이지라면 부드럽게 최상단으로 스크롤
      window.scrollTo({
        top: 0,
        behavior: "smooth"
      });
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

        window.scrollTo({
          top: offsetPosition,
          behavior: "smooth"
        });
      }
    }
  };

  const handleSolutionClick = () => {
    const scrollToTopNow = () => {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    };

    if (location.pathname === "/analyze") {
      scrollToTopNow();
      return;
    }

    nav("/analyze");
    setTimeout(scrollToTopNow, 0);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* 로고 영역 - handleLogoClick 연결 */}
        <div 
          className="text-xl font-extrabold text-slate-900 cursor-pointer tracking-tighter hover:text-[#3182f6] transition-colors duration-300"
          onClick={handleLogoClick} 
        >
          DBDBDEEP
        </div>

        {/* 네비게이션 메뉴 */}
        <nav className="hidden md:flex gap-8 text-slate-600 font-medium text-sm">
          <button onClick={() => scrollToSection("pr-section")} className="hover:text-[#3182f6] transition">
            기술 및 특징
          </button>
          <button onClick={() => scrollToSection("how-to-section")} className="hover:text-[#3182f6] transition">
            검증 프로세스
          </button>
          <button onClick={() => scrollToSection("faq-section")} className="hover:text-[#3182f6] transition">
            지원 센터
          </button>
        </nav>

        {/* 솔루션 버튼 */}
        <button
          onClick={handleSolutionClick}
          className="bg-[#3182f6] hover:bg-[#1b64da] text-white text-sm font-semibold py-2 px-5 rounded-lg transition"
        >
          솔루션 체험하기
        </button>
      </div>
    </header>
  );
}
