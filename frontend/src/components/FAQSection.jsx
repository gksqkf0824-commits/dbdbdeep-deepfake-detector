export default function FAQSection() {
  // 💡 업데이트된 상세한 질문과 답변 리스트
  const faqs = [
    { 
      q: "분석 가능한 파일 형식은 무엇인가요?", 
      a: "일반적인 이미지 파일(JPG, PNG)과 영상 파일(MP4, AVI, MOV 등) 분석을 지원합니다." 
    },
    { 
      q: "URL 분석 기능은 어떻게 사용하나요?", 
      a: "의심되는 영상이나 사진이 포함된 웹페이지 주소를 입력하면, 시스템이 해당 데이터를 추출하여 원격 분석합니다." 
    },
    { 
      q: "판독 결과의 '인물 신뢰 지수'는 무엇을 의미하나요?", 
      a: "신뢰 지수가 100%에 가까울수록 원본(Authentic)일 확률이 높으며, 낮을수록 조작된 딥페이크일 가능성이 큽니다." 
    },
    { 
      q: "업로드한 이미지는 서버에 저장되나요?", 
      a: "분석 데이터는 판독 즉시 휘발성으로 처리되며, 별도로 서버에 저장되지 않습니다." 
    },
    { 
      q: "여러 명의 얼굴이 나올 경우 누구를 분석하나요?", 
      a: "화면 내에서 가장 비중이 크거나 정면에 위치한 타겟을 자동으로 탐지하여 정밀 분석을 수행합니다." 
    },
    { 
      q: "분석 시간이 얼마나 걸리나요?", 
      a: "이미지는 약 5초 내외, 영상은 재생 시간 이내로 분석이 완료됩니다." 
    },
    { 
      q: "판독 결과가 50% 근처라면 어떻게 해석해야 하나요?", 
      a: "화질 저하나 필터 영향으로 판독이 모호한 상태입니다. 더 선명한 고화질 증거물을 확보하여 재분석하시길 권장합니다." 
    }
  ];

  return (
    <div className="w-full h-full max-w-4xl mx-auto">
      {/* 제목 */}
      <div className="font-extrabold text-3xl text-slate-900 mb-8 border-b border-gray-100 pb-5">
        자주 묻는 질문
      </div>

      {/* 스크롤 영역 (질문이 7개로 늘어났으므로 스크롤 기능이 아주 유용하게 작동합니다) */}
      <div className="max-h-[500px] overflow-y-auto pr-2 space-y-4">
        {faqs.map((item, i) => (
          <details
            key={i}
            className="group border border-gray-200 rounded-2xl bg-white open:bg-gray-50 hover:bg-gray-50 transition-all duration-300 shadow-sm"
          >
            <summary className="flex items-center justify-between cursor-pointer p-6 list-none">
              <span className="font-bold text-lg text-slate-800">
                <span className="text-[#3182f6] mr-2">Q{i + 1}.</span> 
                {item.q}
              </span>
              
              {/* 회전하는 화살표 아이콘 */}
              <span className="text-slate-400 transition-transform duration-300 group-open:rotate-180 flex-shrink-0 ml-4">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </span>
            </summary>

            <div className="px-6 pb-6 text-slate-500 text-base leading-relaxed font-medium border-t border-gray-100 pt-4 mt-2">
              {item.a}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}