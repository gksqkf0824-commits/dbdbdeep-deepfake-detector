import { ScanFace, BarChart3, FileSearch } from "lucide-react";

export default function ServicePR() {
  return (
    <div className="w-full">
      
      {/* 상단: 타이틀 및 소개 */}
      <div className="text-center mb-16">
        <span className="text-[#3182f6] font-bold tracking-wider uppercase text-sm mb-3 block">
          Core Values
        </span>
        <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-5 leading-tight">
          DBDBDEEP이 제공하는 핵심 가치
        </h2>
        <p className="text-slate-500 text-base max-w-2xl mx-auto leading-relaxed font-medium">
          육안으로 식별하기 어려운 정교한 위조 미디어, 이제 직감이 아닌 데이터로 검증하세요. 
          독자적인 AI 모델이 투명하고 객관적인 판별 기준을 제시합니다.
        </p>
      </div>

      {/* 하단: 특징 카드 그리드 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Card 1: 정밀 탐지 */}
        <div className="bg-white border border-gray-200 rounded-3xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 cursor-pointer group">
          <div className="w-14 h-14 rounded-2xl bg-blue-50/50 flex items-center justify-center mb-6 group-hover:bg-blue-50 transition-colors">
            <ScanFace className="w-7 h-7 text-[#3182f6]" strokeWidth={2} />
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-3">정교한 위조 미디어 판별</h3>
          <p className="text-slate-500 leading-relaxed text-sm font-medium">
            인물의 얼굴, 표정, 경계선 등 픽셀 단위의 미세한 왜곡을 분석하여 육안으로 놓치기 쉬운 조작 요소까지 정확하게 식별합니다.
          </p>
        </div>

        {/* Card 2: 수치화된 데이터 */}
        <div className="bg-white border border-gray-200 rounded-3xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 cursor-pointer group">
          <div className="w-14 h-14 rounded-2xl bg-indigo-50/50 flex items-center justify-center mb-6 group-hover:bg-indigo-50 transition-colors">
            <BarChart3 className="w-7 h-7 text-indigo-500" strokeWidth={2} />
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-3">객관적인 수치 지표</h3>
          <p className="text-slate-500 leading-relaxed text-sm font-medium">
            단순한 진위 여부를 넘어, 위조 확률을 명확한 % 수치로 제공합니다. 주관적인 판단 대신 데이터 기반의 의사결정을 지원합니다.
          </p>
        </div>

        {/* Card 3: 분석 근거/투명성 */}
        <div className="bg-white border border-gray-200 rounded-3xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 cursor-pointer group">
          <div className="w-14 h-14 rounded-2xl bg-teal-50/50 flex items-center justify-center mb-6 group-hover:bg-teal-50 transition-colors">
            <FileSearch className="w-7 h-7 text-teal-500" strokeWidth={2} />
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-3">직관적인 결과 시각화</h3>
          <p className="text-slate-500 leading-relaxed text-sm font-medium">
            주파수 및 이미지 검사 결과를 깔끔한 도넛 차트 형태로 제공합니다. AI의 판별 결과를 누구나 한눈에 쉽게 파악할 수 있습니다.
          </p>
        </div>

      </div>
    </div>
  );
}