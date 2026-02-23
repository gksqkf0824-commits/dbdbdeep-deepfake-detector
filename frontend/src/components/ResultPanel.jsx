import React from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from 'recharts';

const toFiniteNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const formatAggModeLabel = (mode) => {
  const raw = String(mode || "").trim();
  if (!raw) return "-";
  if (raw === "trimmed_mean_10pct") return "Trimmed Mean 10 Percent";
  return raw
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const VideoTimelinePlaceholder = () => {
  const xTicks = Array.from({ length: 10 }, (_, idx) => idx + 1);

  return (
    <div className="w-full h-full rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="relative h-[150px]">
        {[15, 40, 65, 90].map((top) => (
          <div
            key={`grid-${top}`}
            className="absolute left-0 right-0 border-t border-dashed border-slate-200"
            style={{ top: `${top}%` }}
          />
        ))}
        <div className="absolute bottom-0 left-0 right-0 border-t border-slate-300" />
        <div className="absolute -bottom-6 left-0 right-0 flex justify-between text-[10px] text-slate-400">
          {xTicks.map((value) => (
            <span key={`tick-${value}`}>{value}</span>
          ))}
        </div>
      </div>
      <div className="mt-8 flex items-center justify-center gap-4 text-xs text-slate-400 font-medium">
        <span className="inline-flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-300" />
          주파수(SRM)
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-300" />
          최종(Final)
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-300" />
          픽셀(Pixel)
        </span>
      </div>
    </div>
  );
};

// 1. 도넛 차트 컴포넌트 
const ScoreDonutChart = ({ score, color, cross = false }) => {
  const safeScore = cross ? 100 : Math.max(0, Math.min(100, Number(score)));
  const data = [
    { name: "Score", value: safeScore },
    { name: "Rest", value: cross ? 0 : 100 - safeScore },
  ];

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            innerRadius={45}
            outerRadius={60}
            paddingAngle={0}
            startAngle={90}
            endAngle={450}
            dataKey="value"
            stroke="none"
          >
            <Cell fill={color} />
            <Cell fill="#f1f5f9" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {cross ? (
          <>
            <span className="text-3xl font-extrabold text-red-600 leading-none">X</span>
            <span className="text-[9px] text-red-300 font-semibold tracking-wider">N/A</span>
          </>
        ) : (
          <>
            <span className="text-xl font-bold text-slate-700">{safeScore.toFixed(1)}</span>
            <span className="text-[9px] text-slate-400 font-semibold tracking-wider">SCORE</span>
          </>
        )}
      </div>
    </div>
  );
};

// 2. 동영상/이미지 타임라인 라인 차트 컴포넌트
const VideoTimelineChart = ({ data }) => {
  return (
    <div className="w-full h-full p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis 
            dataKey="time" 
            tick={{fontSize: 11, fill: "#64748b"}} 
            tickLine={false}
            axisLine={{ stroke: "#cbd5e1" }}
          />
          <YAxis 
            domain={[0, 100]} 
            tick={{fontSize: 11, fill: "#64748b"}} 
            tickLine={false}
            axisLine={false}
          />
          <Tooltip 
            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
          />
          <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} />
          <Line name="픽셀(Pixel)" type="monotone" dataKey="pixel" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          <Line name="주파수(SRM)" type="monotone" dataKey="srm" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
          <Line name="최종(Final)" type="monotone" dataKey="final" stroke="#10b981" strokeWidth={3} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// 💡 props에 faceImageUrl 추가
export default function ResultPanel({ progress, result, error, faceImageUrl, fileType = "" }) {
  const pixelScore = toFiniteNumber(result?.pixelScore ?? result?.pixel_score);
  const freqScore = toFiniteNumber(result?.freqScore ?? result?.freq_score);

  const timelineRaw = Array.isArray(result?.timeline) ? result.timeline : [];
  const timeline = timelineRaw
    .map((item, idx) => ({
      time: toFiniteNumber(item?.time) ?? idx + 1,
      pixel: toFiniteNumber(item?.pixel),
      srm: toFiniteNumber(item?.srm),
      final: toFiniteNumber(item?.final),
    }))
    .filter((item) => item.pixel !== null || item.srm !== null || item.final !== null);

  const latestTimeline = timeline.length > 0 ? timeline[timeline.length - 1] : null;
  const hasTimelineData = timeline.length > 1;
  const timelineFinal = toFiniteNumber(latestTimeline?.final);
  const isVideo = fileType === "video" || Boolean(result?.videoMeta);
  const isUndetermined = Boolean(result?.isUndetermined);

  const trust = (() => {
    if (isUndetermined) return null;
    const representative = toFiniteNumber(result?.videoRepresentativeConfidence);
    const rawConfidence = toFiniteNumber(result?.confidence);

    const explicit = [representative, rawConfidence, timelineFinal].filter((v) => v !== null);
    const positiveExplicit = explicit.find((v) => v > 0);
    if (positiveExplicit !== undefined) return positiveExplicit;

    const scoreCandidates = [pixelScore, freqScore].filter((v) => v !== null);
    if (scoreCandidates.length > 0) {
      const avg = scoreCandidates.reduce((acc, cur) => acc + cur, 0) / scoreCandidates.length;
      return Math.max(0, Math.min(100, avg));
    }

    if (explicit.length > 0) return explicit[0];
    return null;
  })();

  const isFake = (() => {
    if (typeof result?.isFake === "boolean") return result.isFake;
    if (typeof result?.is_fake === "boolean") return result.is_fake;
    return trust !== null ? trust < 50 : null;
  })();

  const badge = (() => {
    if (!result) return { text: "대기", color: "text-slate-400", bg: "bg-slate-100" };
    if (isUndetermined) return { text: "판별 불가", color: "text-slate-500", bg: "bg-slate-100" };
    if (isFake === true) return { text: "주의 요망", color: "text-red-600", bg: "bg-red-50" };
    if (isFake === false) return { text: "매우 안전", color: "text-emerald-600", bg: "bg-emerald-50" };
    return { text: "판독 완료", color: "text-blue-600", bg: "bg-blue-50" };
  })();

  const pValue = toFiniteNumber(result?.pValue ?? result?.p_value);

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 w-full">

      {/* Header 영역 변경됨: 얼굴 이미지 + 텍스트 */}
      <div className="flex justify-between items-start">
        
        <div className="flex items-center gap-5">
          {/* 💡 전처리된 얼굴 이미지 띄워주는 공간 (새로 추가됨) */}
          <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-2xl bg-slate-50 border border-gray-200 flex-shrink-0 overflow-hidden flex items-center justify-center shadow-sm">
            {isVideo ? (
              <div className="text-center text-slate-400">
                <div className="text-2xl mb-1 mt-1">▶</div>
                <span className="text-[10px] font-bold uppercase tracking-wider block">Video</span>
                <span className="text-[9px] text-slate-300">Timeline</span>
              </div>
            ) : faceImageUrl ? (
              <img src={faceImageUrl} alt="Detected Face" className="w-full h-full object-cover" />
            ) : (
              <div className="text-center text-slate-400">
                <div className="text-3xl mb-1 mt-1">👤</div>
                <span className="text-[10px] font-bold uppercase tracking-wider">Face</span>
              </div>
            )}
          </div>

          {/* 기존 텍스트 및 점수 영역 */}
          <div>
            <div className="font-semibold text-slate-900 mb-1">
              AI 판별 결과
            </div>
            <div
              className={`text-4xl sm:text-5xl font-bold tracking-tight ${
                isUndetermined ? "text-red-600" : "text-blue-600"
              }`}
            >
              {isUndetermined ? "판별 불가" : trust !== null ? `${trust.toFixed(2)}%` : "--%"}
            </div>
            <div className="text-sm text-slate-500 mt-2 font-medium">
              {result
                ? isUndetermined
                  ? "얼굴이 감지되지 않아 판별할 수 없습니다."
                  : result.reliability
                    ? `신뢰도: ${result.reliability}`
                    : "분석 완료"
                : "분석 결과가 여기에 표시됩니다."}
            </div>
          </div>
        </div>

        {/* 우측 배지 영역 */}
        <div className="text-right pt-1">
          <span className={`px-3 py-1.5 rounded-full text-xs font-bold ${badge.color} ${badge.bg}`}>
            {badge.text}
          </span>
          {result && pValue !== null && (
            <div className="text-xs text-slate-500 mt-2 font-medium">
              p-value: {pValue}
            </div>
          )}
        </div>
      </div>

      {/* Progress */}
      <div className="mt-8">
        <div className="flex justify-between text-xs text-slate-500 font-medium mb-2">
          <span>분석 진행률</span>
          <span>{Math.floor(progress)}%</span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-5 border border-red-200 bg-red-50 rounded-lg p-4">
          <div className="font-semibold text-red-600 mb-1">분석 실패</div>
          <div className="text-sm text-red-500">{error}</div>
        </div>
      )}

      {/* Charts & Images 통합 영역 */}
      <div className="mt-6">
        {isVideo ? (
          <div className="border border-gray-200 rounded-lg p-5 bg-white">
            <div className="font-semibold text-slate-800 mb-4 flex justify-between items-center">
              <span>타임라인 분석 (신뢰도 추이)</span>
              <span className="text-xs bg-slate-100 text-slate-500 px-2 py-1 rounded">Video</span>
            </div>
            <div className="h-[240px] w-full">
              {hasTimelineData ? <VideoTimelineChart data={timeline} /> : <VideoTimelinePlaceholder />}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            
            {/* Frequency (주파수) */}
            <div className="border border-gray-200 rounded-lg p-4 bg-white flex flex-col items-center">
              <div className="font-semibold text-slate-800 w-full mb-3">주파수 분석</div>
              <div className="h-[120px] w-full mb-4">
                {isUndetermined ? (
                  <ScoreDonutChart score={100} color="#ef4444" cross />
                ) : toFiniteNumber(latestTimeline?.srm) !== null ? (
                  <ScoreDonutChart score={latestTimeline.srm} color="#6366f1" />
                ) : freqScore !== null ? (
                  <ScoreDonutChart score={freqScore} color="#6366f1" />
                ) : (
                  <div className="text-xs text-slate-500 text-center mt-10">
                    --
                  </div>
                )}
              </div>
            </div>

            {/* Pixel (픽셀) */}
            <div className="border border-gray-200 rounded-lg p-4 bg-white flex flex-col items-center">
              <div className="font-semibold text-slate-800 w-full mb-3">픽셀 분석</div>
              <div className="h-[120px] w-full mb-4">
                {isUndetermined ? (
                  <ScoreDonutChart score={100} color="#ef4444" cross />
                ) : toFiniteNumber(latestTimeline?.pixel) !== null ? (
                  <ScoreDonutChart score={latestTimeline.pixel} color="#3b82f6" />
                ) : pixelScore !== null ? (
                  <ScoreDonutChart score={pixelScore} color="#3b82f6" />
                ) : (
                  <div className="text-xs text-slate-500 text-center mt-10">
                    --
                  </div>
                )}
              </div>
            </div>

          </div>
        )}
      </div>

      {result?.videoMeta && (
        <div className="mt-5 border border-slate-200 rounded-lg p-3 bg-slate-50 text-xs text-slate-600 grid grid-cols-2 gap-2">
          <div>전체 샘플링 프레임: {result.videoMeta.sampled_frames ?? "-"}</div>
          <div>추론 사용 프레임: {result.videoMeta.used_frames ?? "-"}</div>
          <div>얼굴 검출 실패 프레임: {result.videoMeta.failed_frames ?? "-"}</div>
          <div>집계 방식: {formatAggModeLabel(result.videoMeta.agg_mode)}</div>
        </div>
      )}

    </div>
  );
}
