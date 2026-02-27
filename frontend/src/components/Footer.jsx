export default function Footer() {
  return (
    <footer className="bg-white border-t border-gray-100 pt-16 pb-8">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
          
          {/* 브랜드 영역 */}
          <div className="md:col-span-2">
            <div className="text-2xl font-extrabold text-slate-900 tracking-tighter mb-4">
              DBDBDEEP
            </div>
            {/* 거창한 표현을 빼고 직관적이고 담백한 설명으로 수정 */}
            <p className="text-slate-500 text-sm leading-relaxed max-w-sm font-medium">
              딥러닝 기반 이미지 및 영상 진위 판별 솔루션.<br />
              육안으로 확인하기 어려운 조작된 미디어를<br />
              데이터를 통해 객관적이고 정확하게 검증합니다.
            </p>
          </div>

          {/* 링크 영역 1: Quick Links (헤더처럼 메인 페이지 내 섹션으로 이동) */}
          <div>
            <h4 className="font-bold text-slate-900 mb-4">Menu</h4>
            <ul className="space-y-3 text-sm text-slate-500 font-medium">
              {/* 단순 페이지 이동 대신, 존재하는 섹션으로 앵커 링크 연결 */}
              <li><a href="#pr-section" className="hover:text-[#3182f6] transition">기술 및 특징</a></li>
              <li><a href="#how-to-section" className="hover:text-[#3182f6] transition">검증 프로세스</a></li>
              <li><a href="#faq-section" className="hover:text-[#3182f6] transition">지원 센터</a></li>
            </ul>
          </div>

          {/* 링크 영역 2: GitHub 연동 (프로젝트 저장소 링크) */}
          <div>
            <h4 className="font-bold text-slate-900 mb-4">GitHub</h4>
            <ul className="space-y-3 text-sm text-slate-500 font-medium">
              {/* 실제 깃허브 주소로 변경해서 사용하세요. 새 창에서 열리도록 target="_blank" 적용 */}
              <li><a href="https://github.com/gksqkf0824-commits/multimodal-face-auth-security/blob/main/model.py" target="_blank" rel="noopener noreferrer" className="hover:text-[#3182f6] transition">AI Model</a></li>
              <li><a href="https://github.com/gksqkf0824-commits/multimodal-face-auth-security/blob/main/main.py" target="_blank" rel="noopener noreferrer" className="hover:text-[#3182f6] transition">Backend</a></li>
              <li><a href="https://github.com/gksqkf0824-commits/dbdbdeep-deepfake-detector/tree/main/frontend" target="_blank" rel="noopener noreferrer" className="hover:text-[#3182f6] transition">Frontend</a></li>
            </ul>
          </div>
        </div>

        {/* 저작권 및 부가 정보 */}
        <div className="border-t border-gray-100 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-slate-400 font-medium">
          <div>
            © 2026 DBDBDEEP Team. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
}