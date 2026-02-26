import { useRef, useState } from "react";

export default function UploadCard({
  mode = "file",
  fileType = "",
  previewUrl,
  urlPreview = null,
  loading,
  hasResult = false,
  aiComment = "",
  aiCommentSource = "",
  imageUrl,
  onReset,
  onModeChange,
  onPickFile,
  onAnalyze,
  onUrlChange,
}) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false); // 드래그 상태 관리

  const openPicker = () => {
    if (mode === "file") inputRef.current?.click();
  };

  const onChangeFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    onPickFile(f);
  };

  // --- 드래그 앤 드롭 핸들러 추가 ---
  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const f = e.dataTransfer.files?.[0];
    if (f && mode === "file") {
      onPickFile(f);
    }
  };
  // ------------------------------

  const defaultComment =
    mode === "file"
      ? previewUrl
        ? fileType === "video"
          ? "영상이 준비됐어요. 판독을 시작해보세요."
          : "파일이 준비됐어요. 판독을 시작해보세요."
        : "분석 전입니다. 파일을 업로드하세요."
      : imageUrl
        ? "URL이 입력되었습니다. 판독을 시작해보세요."
        : "분석 전입니다. 이미지/영상 주소를 입력해주세요.";

  const commentText = (() => {
    const cleaned = String(aiComment || "").trim();
    if (cleaned.length > 0) return cleaned;
    if (hasResult) return "분석이 완료되었습니다. 결과 패널의 상세 근거를 함께 확인해보세요.";
    return defaultComment;
  })();

  const formattedCommentText = (() => {
    const raw = String(commentText || "").trim();
    if (!raw) return "";
    return raw.replace(/\.\s+/g, ".\n");
  })();

  const sourceLabel = (() => {
    const source = String(aiCommentSource || "").trim().toLowerCase();
    if (!source) return "";
    if (source.startsWith("openai")) return "생성: OpenAI LLM";
    if (source.startsWith("fallback")) return "생성: Fallback";
    return "생성: Rule Based";
  })();

  const onClickPrimary = () => {
    if (loading) return;
    if (hasResult) {
      if (inputRef.current) inputRef.current.value = "";
      onReset?.();
      return;
    }
    onAnalyze?.();
  };

  const urlPreviewKind = String(urlPreview?.kind || "").toLowerCase() === "video" ? "video" : "image";
  const urlPreviewImage =
    urlPreview?.dataUrl || urlPreview?.thumbnailDataUrl || urlPreview?.thumbnailUrl || null;
  const urlPreviewVideo = urlPreview?.url || null;

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5 flex flex-col h-full w-full max-w-[400px] ml-0 mr-auto">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-5 flex-shrink-0">
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
      <div className="flex-grow flex flex-col mb-6 min-h-[360px]">
        {mode === "file" ? (
          <div
            onClick={openPicker}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            className={`relative flex-1 border-2 border-dashed rounded-lg flex flex-col items-center justify-center text-center cursor-pointer transition-all overflow-hidden ${
              isDragging 
                ? "border-blue-500 bg-blue-50 scale-[0.99]" 
                : "border-blue-200 bg-slate-50 hover:border-blue-400"
            }`}
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
                  className="absolute inset-0 w-full h-full object-contain p-2"
                />
              )
            ) : (
              <div className="px-4 pointer-events-none">
                <div className="text-4xl mb-4">{isDragging ? "📥" : "📁"}</div>
                <div className="font-bold text-slate-700 text-lg">
                  {isDragging ? "여기에 놓아주세요" : "상대의 사진/영상을 올려주세요"}
                </div>
                <div className="text-sm text-slate-500 mt-2 font-medium">
                  {isDragging ? "파일을 감지했습니다" : "드래그하거나 클릭해서 파일 선택"}
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
          <div className="flex-1 border-2 border-gray-100 bg-slate-50 rounded-lg flex flex-col p-6 transition-all duration-300">
            {urlPreview ? (
              <div className="mb-5">
                <div className="w-full aspect-video rounded-lg overflow-hidden bg-black/90 border border-slate-200">
                  {urlPreviewKind === "video" && urlPreviewVideo && !urlPreviewImage ? (
                    <video
                      src={urlPreviewVideo}
                      controls
                      className="w-full h-full object-contain"
                    />
                  ) : urlPreviewImage ? (
                    <img
                      src={urlPreviewImage}
                      alt="URL preview"
                      className="w-full h-full object-contain bg-white"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-300 text-sm font-medium">
                      URL 미리보기를 표시할 수 없습니다.
                    </div>
                  )}
                </div>
                <div className="mt-2 text-xs text-slate-500 font-medium">
                  {urlPreviewKind === "video" ? "URL 영상 미리보기" : "URL 이미지 미리보기"}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-center py-6">
                <div className="text-4xl mb-4">🔗</div>
                <div className="font-bold text-slate-700 text-lg mb-2">
                  이미지/영상 URL을 입력하세요
                </div>
                <div className="text-sm text-slate-500 font-medium">
                  인스타 릴스, 유튜브 쇼츠 URL도 지원합니다.
                </div>
              </div>
            )}
            <input 
              type="text" 
              placeholder="https://example.com/image.jpg 또는 https://youtube.com/shorts/..."
              value={imageUrl || ""}
              onChange={(e) => onUrlChange?.(e.target.value)}
              className="w-full px-5 py-4 rounded-xl border border-gray-300 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 outline-none transition-all text-sm font-medium bg-white"
            />
          </div>
        )}
      </div>

      {/* 하단 버튼 및 코멘트 */}
      <div className="mt-auto space-y-5">
        <button
          onClick={onClickPrimary}
          disabled={loading}
          className={`w-full py-4 rounded-xl font-bold text-white shadow-lg transition-all flex-shrink-0 ${
            loading
              ? "bg-blue-300 cursor-not-allowed"
              : hasResult
                ? "bg-slate-800 hover:bg-slate-900"
                : "bg-gradient-to-r from-blue-600 to-indigo-500 hover:opacity-90 active:scale-95"
          }`}
        >
          {loading ? "데이터 분석 중..." : hasResult ? "다른 파일 분석하기" : "판독 시작"}
        </button>

        <div className="bg-blue-50/50 border border-blue-100 rounded-xl p-5">
          <div className="flex items-center justify-between mb-2.5">
            <div className="font-bold text-blue-900 text-sm flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
              AI 코멘트
            </div>
            {hasResult && sourceLabel && (
              <div className="text-[10px] text-blue-400 font-bold uppercase tracking-widest">{sourceLabel}</div>
            )}
          </div>
          <div className="text-[13px] text-blue-800/80 leading-relaxed font-medium whitespace-pre-line break-words">
            {formattedCommentText}
          </div>
        </div>
      </div>
      
    </div>
  );
}
