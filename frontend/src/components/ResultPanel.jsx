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
            innerRadius={50}
            outerRadius={70}
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
            <span className="text-4xl font-extrabold text-red-600 leading-none">X</span>
            <span className="text-[10px] text-red-300 font-semibold tracking-wider">N/A</span>
          </>
        ) : (
          <>
            <span className="text-2xl font-bold text-slate-700">{safeScore.toFixed(1)}</span>
            <span className="text-[10px] text-slate-400 font-semibold tracking-wider">SCORE</span>
          </>
        )}
      </div>
    </div>
  );
};

const VideoTimelineChart = ({ data }) => {
  return (
    <div className="w-full h-full p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="time" tick={{fontSize: 12, fill: "#64748b"}} tickLine={false} axisLine={{ stroke: "#cbd5e1" }} />
          <YAxis domain={[0, 100]} tick={{fontSize: 12, fill: "#64748b"}} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }} />
          <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px', fontSize: '13px' }} />
          <Line name="픽셀(Pixel)" type="monotone" dataKey="pixel" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line name="주파수(SRM)" type="monotone" dataKey="srm" stroke="#6366f1" strokeWidth={3} dot={{ r: 4 }} />
          <Line name="최종(Final)" type="monotone" dataKey="final" stroke="#10b981" strokeWidth={4} dot={{ r: 5 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

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

  const hasTimelineData = timeline.length > 1;
  const isVideo = fileType === "video" || Boolean(result?.videoMeta);
  const isUndetermined = Boolean(result?.isUndetermined);

  const trust = (() => {
    if (isUndetermined) return null;
    const representative = toFiniteNumber(result?.videoRepresentativeConfidence);
    const rawConfidence = toFiniteNumber(result?.confidence);
    const explicit = [representative, rawConfidence].filter((v) => v !== null);
    const positiveExplicit = explicit.find((v) => v > 0);
    if (positiveExplicit !== undefined) return positiveExplicit;
    const scoreCandidates = [pixelScore, freqScore].filter((v) => v !== null);
    if (scoreCandidates.length > 0) {
      const avg = scoreCandidates.reduce((acc, cur) => acc + cur, 0) / scoreCandidates.length;
      return Math.max(0, Math.min(100, avg));
    }
    return null;
  })();

  const badge = (() => {
    if (!result) return { text: "대기", color: "text-slate-400", bg: "bg-slate-100" };
    if (isUndetermined) return { text: "추론 실패", color: "text-red-600", bg: "bg-red-50" };
    const pReal = trust !== null ? trust / 100 : null;
    if (pReal !== null && pReal < 0.335) return { text: "가짜", color: "text-red-600", bg: "bg-red-50" };
    if (pReal !== null && pReal < 0.52) return { text: "주의", color: "text-amber-600", bg: "bg-amber-50" };
    if (pReal !== null) return { text: "진짜", color: "text-emerald-600", bg: "bg-emerald-50" };
    return { text: "판독 완료", color: "text-blue-600", bg: "bg-blue-50" };
  })();

  return (
    /**
     * 핵심 수정: flex-grow를 적용하여 왼쪽의 빈 공간을 모두 차지하도록 팽창시켰습니다.
     * 상위 레이아웃에서 UploadCard 옆에 바로 붙게 됩니다.
     */
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 flex-grow flex flex-col h-full">
      {/* Top Section */}
      <div className="flex justify-between items-start flex-shrink-0">
        <div className="flex items-center gap-6">
          <div className="w-24 h-24 sm:w-28 sm:h-28 rounded-2xl bg-slate-50 border border-gray-200 flex-shrink-0 overflow-hidden flex items-center justify-center shadow-sm">
            {isVideo ? (
              <div className="text-center text-slate-400">
                <div className="text-3xl mb-1">▶</div>
                <span className="text-xs font-bold uppercase tracking-wider block">Video</span>
              </div>
            ) : faceImageUrl ? (
              <img src={faceImageUrl} alt="Detected Face" className="w-full h-full object-cover" />
            ) : (
              <div className="text-center text-slate-400">
                <div className="text-4xl mb-1">👤</div>
                <span className="text-xs font-bold uppercase tracking-wider font-sans">Face</span>
              </div>
            )}
          </div>
          <div>
            <div className="font-semibold text-slate-900 mb-1 text-lg">AI 판별 결과</div>
            <div className={`text-5xl sm:text-6xl font-bold tracking-tight ${isUndetermined ? "text-red-600" : "text-blue-600"}`}>
              {isUndetermined ? "추론 실패" : trust !== null ? `${trust.toFixed(2)}%` : "--%"}
            </div>
            <div className="text-base text-slate-500 mt-2 font-medium">
              {result ? (isUndetermined ? "얼굴 미탐지" : "분석 완료") : "분석 결과 대기"}
            </div>
          </div>
        </div>
        <div className="text-right pt-2">
          <span className={`px-4 py-2 rounded-full text-sm font-bold ${badge.color} ${badge.bg}`}>
            {badge.text}
          </span>
        </div>
      </div>

      {/* Progress */}
      <div className="mt-10 flex-shrink-0">
        <div className="flex justify-between text-sm text-slate-500 font-medium mb-3">
          <span>분석 진행률</span>
          <span>{Math.floor(progress)}%</span>
        </div>
        <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-500" style={{ width: `${progress}%` }} />
        </div>
      </div>

      {/* Analysis Charts */}
      <div className="mt-8 flex-grow flex flex-col justify-end">
          {result && isUndetermined ? (
          <div className="border border-red-200 rounded-lg px-6 py-14 bg-red-50/40 shadow-sm flex items-center justify-center text-center">
            <div className="text-red-600 font-semibold text-lg">얼굴 미탐지로 인해 추론이 실패했습니다</div>
          </div>
        ) : isVideo ? (
          <div className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm">
            <div className="font-semibold text-slate-800 mb-5 text-base">타임라인 정밀 분석</div>
            <div className="h-[280px] w-full">
              {result && hasTimelineData ? <VideoTimelineChart data={timeline} /> : <VideoTimelinePlaceholder />}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-6 h-full">
            {/* 주파수 분석 */}
            <div className="border border-gray-200 rounded-lg p-6 bg-white flex flex-col items-center shadow-sm">
              <div className="font-semibold text-slate-800 w-full mb-5 text-base font-sans">주파수 분석 (Frequency)</div>
              <div className="w-full flex-grow flex items-center justify-center min-h-[220px]">
                {result ? (
                  isUndetermined ? <ScoreDonutChart score={100} color="#ef4444" cross /> : <ScoreDonutChart score={freqScore} color="#6366f1" />
                ) : (
                  <div className="w-32 h-32 rounded-full border-8 border-slate-50 bg-slate-50/20 flex items-center justify-center">
                    <span className="text-slate-300 text-xs font-bold uppercase tracking-widest">Waiting</span>
                  </div>
                )}
              </div>
            </div>
            {/* 픽셀 분석 */}
            <div className="border border-gray-200 rounded-lg p-6 bg-white flex flex-col items-center shadow-sm">
              <div className="font-semibold text-slate-800 w-full mb-5 text-base font-sans">픽셀 분석 (Pixel-level)</div>
              <div className="w-full flex-grow flex items-center justify-center min-h-[220px]">
                {result ? (
                  isUndetermined ? <ScoreDonutChart score={100} color="#ef4444" cross /> : <ScoreDonutChart score={pixelScore} color="#3b82f6" />
                ) : (
                  <div className="w-32 h-32 rounded-full border-8 border-slate-50 bg-slate-50/20 flex items-center justify-center">
                    <span className="text-slate-300 text-xs font-bold uppercase tracking-widest">Waiting</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {result?.videoMeta && (
        <div className="mt-6 border border-slate-200 rounded-lg p-4 bg-slate-50 text-xs text-slate-600 grid grid-cols-2 gap-4 flex-shrink-0">
          <div className="flex gap-2"><span className="font-bold text-slate-400">샘플:</span> {result.videoMeta.sampled_frames ?? "-"} frames</div>
          <div className="flex gap-2"><span className="font-bold text-slate-400">방식:</span> {formatAggModeLabel(result.videoMeta.agg_mode)}</div>
        </div>
      )}
    </div>
  );
}
