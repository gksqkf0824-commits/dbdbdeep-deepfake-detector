import { useRef } from "react";

export default function UploadCard({
  mode = "file",
  fileType = "",
  previewUrl,
  loading,
  imageUrl,
  onModeChange,
  onPickFile,
  onAnalyze,
  onUrlChange,
}) {
  const inputRef = useRef(null);

  const openPicker = () => {
    if (mode === "file") inputRef.current?.click();
  };

  const onChangeFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    onPickFile(f);
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 h-full flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="font-semibold text-slate-900 text-lg">
          분석 대상 업로드
        </div>
        <span className="text-xs font-semibold text-slate-500 border border-gray-200 px-3 py-1 rounded-full bg-gray-50 uppercase tracking-wide">
          Evidence
        </span>
      </div>

      {/* 슬라이드 토글 스위치 */}
      <div className="flex bg-slate-100 p-1 rounded-lg mb-5 flex-shrink-0">
        <button
          onClick={() => onModeChange?.("file")}
          className={`flex-1 py-2 rounded-md font-semibold text-sm transition-all duration-200 ${
            mode === "file" 
              ? "bg-white text-blue-600 shadow-sm" 
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          파일 업로드
        </button>
        <button
          onClick={() => onModeChange?.("url")}
          className={`flex-1 py-2 rounded-md font-semibold text-sm transition-all duration-200 ${
            mode === "url" 
              ? "bg-white text-blue-600 shadow-sm" 
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          URL 주소
        </button>
      </div>

      {/* 입력 영역 */}
      <div className="h-[340px] flex flex-col flex-shrink-0">
        {mode === "file" ? (
          // 1. 파일 업로드 모드
          <div
            onClick={openPicker}
            // 💡 핵심 수정 포인트: relative와 overflow-hidden을 추가하여 내부 이미지가 밖으로 나가는 것을 차단합니다.
            className="relative flex-1 border-2 border-dashed border-blue-200 bg-slate-50 rounded-lg flex flex-col items-center justify-center text-center cursor-pointer transition hover:border-blue-400 overflow-hidden"
          >
            {previewUrl ? (
              fileType === "video" ? (
                <video
                  src={previewUrl}
                  controls
                  className="absolute inset-0 w-full h-full object-contain p-2 bg-black"
                />
              ) : (
                <img
                  src={previewUrl}
                  alt="preview"
                  // 💡 핵심 수정 포인트: 이미지를 박스 안에 절대 좌표로 띄우고(absolute inset-0) object-contain으로 비율을 맞춥니다.
                  className="absolute inset-0 w-full h-full object-contain p-2"
                />
              )
            ) : (
              <div>
                <div className="text-3xl mb-3">📁</div>
                <div className="font-medium text-slate-700">
                  상대의 사진/영상을 올려주세요
                </div>
                <div className="text-sm text-slate-500 mt-2">
                  클릭해서 파일 선택
                </div>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*,video/*"
              hidden
              onChange={onChangeFile}
            />
          </div>
        ) : (
          // 2. URL 입력 모드
          <div className="flex-1 border-2 border-gray-100 bg-slate-50 rounded-lg flex flex-col items-center justify-center p-8 transition-all duration-300">
            <div className="text-3xl mb-3">🔗</div>
            <div className="font-semibold text-slate-700 mb-6">
              이미지 URL을 입력하세요
            </div>
            <input 
              type="text" 
              placeholder="https://example.com/image.jpg"
              value={imageUrl || ""}
              onChange={(e) => onUrlChange?.(e.target.value)}
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-sm font-medium bg-white"
            />
            <div className="text-xs text-slate-500 mt-4">
              웹상에 공개된 이미지 주소만으로도 즉시 분석이 가능합니다.
            </div>
          </div>
        )}
      </div>

      {/* Analyze Button */}
      <button
        onClick={onAnalyze}
        disabled={loading}
        className={`w-full mt-5 py-3 rounded-lg font-semibold text-white transition flex-shrink-0 ${
          loading
            ? "bg-blue-300 cursor-not-allowed"
            : "bg-gradient-to-r from-blue-600 to-indigo-500 hover:opacity-90"
        }`}
      >
        {loading ? "분석 중..." : "판독 시작"}
      </button>

      {/* Comment Box */}
      <div className="mt-5 bg-slate-50 border border-gray-200 rounded-lg p-4 flex-shrink-0">
        <div className="font-semibold text-slate-800 mb-2">
          AI 코멘트
        </div>
        <div className="text-sm text-slate-500">
          {mode === "file" 
            ? (
                previewUrl
                  ? fileType === "video"
                    ? "영상이 준비됐어요. 판독을 시작해보세요."
                    : "파일이 준비됐어요. 판독을 시작해보세요."
                  : "분석 전입니다. 파일을 업로드하세요."
              )
            : (imageUrl ? "URL이 입력되었습니다. 판독을 시작해보세요." : "분석 전입니다. 이미지 주소를 입력해주세요.")
          }
        </div>
      </div>
      
    </div>
  );
}
