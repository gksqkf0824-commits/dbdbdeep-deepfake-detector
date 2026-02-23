import { useNavigate } from "react-router-dom";

export default function Header() {
  const nav = useNavigate();

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      // 헤더의 높이(h-16 = 64px) + 여백(옵션)만큼 빼줍니다.
      const headerOffset = 64; 
      // 현재 요소의 위치를 계산합니다.
      const elementPosition = element.getBoundingClientRect().top;
      // 현재 스크롤 위치에 요소의 위치를 더하고 헤더 높이만큼 뺍니다.
      const offsetPosition = elementPosition + window.scrollY - headerOffset;

      // 계산된 위치로 부드럽게 스크롤합니다.
      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* 로고 영역 */}
        <div 
          className="text-xl font-extrabold text-slate-900 cursor-pointer tracking-tighter hover:text-[#3182f6] transition-colors duration-300"
          onClick={() => nav("/")} 
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
          onClick={() => nav("/analyze")}
          className="bg-[#3182f6] hover:bg-[#1b64da] text-white text-sm font-semibold py-2 px-5 rounded-lg transition"
        >
          솔루션 체험하기
        </button>
      </div>
    </header>
  );
}