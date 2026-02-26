import React from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from 'recharts';

const toFiniteNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const toNonNegativeInt = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return null;
  return Math.round(n);
};

const toPercentFromRatio = (value) => {
  const n = toFiniteNumber(value);
  if (n === null) return null;
  return n <= 1 ? n * 100 : n;
};

const parseTrimPercentsFromMode = (mode) => {
  const raw = String(mode || "");
  const lowMatch = raw.match(/low\s*([0-9]+(?:\.[0-9]+)?)\s*percent/i);
  const highMatch = raw.match(/high\s*([0-9]+(?:\.[0-9]+)?)\s*percent/i);
  return {
    lowPct: lowMatch ? Number(lowMatch[1]) : null,
    highPct: highMatch ? Number(highMatch[1]) : null,
  };
};

const formatAggModeLabel = (mode) => {
  const raw = String(mode || "").trim();
  if (!raw) return "-";
  if (raw === "trimmed_mean_10pct") return "Trimmed Mean";
  if (/trimmed/i.test(raw)) return "Trimmed Mean";
  if (raw === "mean") return "Mean";
  if (raw === "median") return "Median";
  if (raw === "topk_mean") return "Top-K Mean";
  return raw
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const formatFrameCount = (value) => {
  const n = toNonNegativeInt(value);
  return n === null ? "-" : `${n} 프레임`;
};

const formatPercentText = (value) => {
  if (value === null) return null;
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}%` : `${rounded.toFixed(1)}%`;
};

const buildAggModeGuide = (videoMeta) => {
  const rawMode = String(videoMeta?.agg_mode || "").trim();
  const label = formatAggModeLabel(rawMode);
  if (!rawMode || label === "-") {
    return {
      label: "-",
      lines: ["집계 방식 정보가 없어 기본 평균 집계로 처리됩니다."],
    };
  }

  if (/trimmed/i.test(rawMode)) {
    const parsed = parseTrimPercentsFromMode(rawMode);
    const lowPct = toPercentFromRatio(videoMeta?.trim_low_ratio) ?? parsed.lowPct;
    const highPct = toPercentFromRatio(videoMeta?.trim_high_ratio) ?? parsed.highPct;
    const rawCount = toNonNegativeInt(videoMeta?.raw_count);
    const usedCount = toNonNegativeInt(videoMeta?.used_count);
    const excludedLowCount = toNonNegativeInt(videoMeta?.excluded_low_count);
    const excludedHighCount = toNonNegativeInt(videoMeta?.excluded_high_count);

    const trimParts = [];
    if (lowPct !== null) trimParts.push(`하위 ${formatPercentText(lowPct)} 제외`);
    if (highPct !== null) trimParts.push(`상위 ${formatPercentText(highPct)} 제외`);
    const trimText = trimParts.length > 0 ? trimParts.join(" · ") : "극단값 일부 제외";

    const lines = [
      `현재 방식은 ${label}입니다.`,
      `${trimText} 후 남은 프레임 점수를 평균해 대표 점수를 계산합니다.`,
    ];

    const detailParts = [];
    if (rawCount !== null) detailParts.push(`집계 대상 ${rawCount}개`);
    if (excludedLowCount !== null) detailParts.push(`하위 제외 ${excludedLowCount}개`);
    if (excludedHighCount !== null) detailParts.push(`상위 제외 ${excludedHighCount}개`);
    if (usedCount !== null) detailParts.push(`최종 반영 ${usedCount}개`);
    if (detailParts.length > 0) lines.push(detailParts.join(" · "));

    return { label, lines };
  }

  return {
    label,
    lines: [
      `현재 방식은 ${label}입니다.`,
      "분석 프레임 점수를 같은 비중으로 합산해 대표 점수를 계산합니다.",
    ],
  };
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
            innerRadius={60}
            outerRadius={85}
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
            <span className="text-3xl font-bold text-slate-700">{safeScore.toFixed(1)}</span>
            <span className="text-[11px] text-slate-400 font-semibold tracking-wider">SCORE</span>
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
  const videoMeta = result?.videoMeta && typeof result.videoMeta === "object" ? result.videoMeta : null;

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
  const isVideo = fileType === "video" || Boolean(videoMeta);
  const isUndetermined = Boolean(result?.isUndetermined);
  const totalSampledFrames = formatFrameCount(videoMeta?.sampled_frames);
  const analyzedSampledFrames = formatFrameCount(videoMeta?.used_frames);
  const failedSampledFrames = formatFrameCount(videoMeta?.failed_frames);
  const aggGuide = buildAggModeGuide(videoMeta);
  const videoMetaRows = [
    { key: "sampled", label: "전체 샘플링 프레임", value: totalSampledFrames },
    { key: "used", label: "분석 샘플링 프레임", value: analyzedSampledFrames },
    { key: "failed", label: "분석 실패 프레임", value: failedSampledFrames },
    { key: "agg", label: "방식", value: aggGuide.label || "-" },
  ];

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
    if (!result) return { text: "대기", color: "text-slate-400", bg: "bg-slate-100", padding: "px-10" };
    if (isUndetermined) return { text: "추론 실패", color: "text-red-600", bg: "bg-red-50", padding: "px-10" };
    const pReal = trust !== null ? trust / 100 : null;
    if (pReal !== null && pReal < 0.335) return { text: "FAKE", color: "text-red-600", bg: "bg-red-50", padding: "px-10" };
    // WARNING일 때만 글자수가 많으므로 px-6으로 패딩을 줄여 전체 박스 크기를 맞춤
    if (pReal !== null && pReal < 0.52) return { text: "WARNING", color: "text-amber-600", bg: "bg-amber-50", padding: "px-6" };
    if (pReal !== null) return { text: "REAL", color: "text-emerald-600", bg: "bg-emerald-50", padding: "px-10" };
    return { text: "판독 완료", color: "text-blue-600", bg: "bg-blue-50", padding: "px-10" };
  })();

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-8 flex-grow flex flex-col h-full">
      {/* Top Section - 추론 전 배치(이미지 좌측)로 통일 */}
      <div className="flex justify-between items-start flex-shrink-0 mb-12">
        <div className="flex items-center gap-8">
          <div className="w-32 h-32 sm:w-40 sm:h-40 rounded-3xl bg-slate-50 border border-gray-200 flex-shrink-0 overflow-hidden flex items-center justify-center shadow-md">
            {isVideo ? (
              <div className="text-center text-slate-400">
                <div className="text-4xl mb-1">▶</div>
                <span className="text-sm font-bold uppercase tracking-wider block">Video</span>
              </div>
            ) : faceImageUrl ? (
              <img src={faceImageUrl} alt="Detected Face" className="w-full h-full object-cover" />
            ) : (
              <div className="text-center text-slate-400">
                <div className="text-5xl mb-2">👤</div>
                <span className="text-xs font-bold uppercase tracking-wider font-sans">Face</span>
              </div>
            )}
          </div>
          <div>
            <div className="font-semibold text-slate-900 mb-2 text-xl">AI 판별 결과</div>
            <div className={`text-6xl sm:text-7xl font-bold tracking-tight ${isUndetermined ? "text-red-600" : "text-blue-600"}`}>
              {isUndetermined ? "추론 실패" : trust !== null ? `${trust.toFixed(2)}%` : "--%"}
            </div>
            <div className="text-lg text-slate-500 mt-3 font-medium">
              {result ? (isUndetermined ? "얼굴 미탐지" : "분석 완료") : "분석 결과 대기"}
            </div>
          </div>
        </div>
        
        {/* 수정 포인트: 배지 종류에 따라 dynamic padding 적용 */}
        <div className="text-right">
          <span className={`inline-block py-5 rounded-2xl text-2xl font-black shadow-sm ${badge.padding} ${badge.color} ${badge.bg}`}>
            {badge.text}
          </span>
        </div>
      </div>

      {/* Progress & Analysis Charts */}
      <div className="mt-auto flex flex-col">
        <div className="mb-10">
          <div className="flex justify-between text-base text-slate-500 font-medium mb-3">
            <span>분석 진행률</span>
            <span>{Math.floor(progress)}%</span>
          </div>
          <div className="h-3 bg-slate-100 rounded-full overflow-hidden shadow-inner">
            <div className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-500" style={{ width: `${progress}%` }} />
          </div>
        </div>

        <div className="flex flex-col flex-1">
          {result && isUndetermined ? (    
            <div className="flex flex-1 items-center">
              <div className="w-full border border-red-200 rounded-lg px-6 py-14 bg-red-50/40 shadow-sm flex items-center justify-center text-center">
                <div className="text-red-600 font-semibold text-lg">
                  얼굴 미탐지로 인해 추론이 실패했습니다
                </div>
              </div>
            </div>
          ) : isVideo ? (
            <div className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm">
              <div className="font-semibold text-slate-800 mb-5 text-base">타임라인 정밀 분석</div>
              <div className="h-[280px] w-full">
                {result && hasTimelineData ? <VideoTimelineChart data={timeline} /> : <VideoTimelinePlaceholder />}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-8">
              <div className="border border-gray-200 rounded-xl p-8 bg-white flex flex-col items-center shadow-sm">
                <div className="font-semibold text-slate-800 w-full mb-6 text-lg font-sans">주파수 분석 (Frequency)</div>
                <div className="w-full flex-grow flex items-center justify-center min-h-[250px]">
                  {result ? (
                    isUndetermined ? <ScoreDonutChart score={100} color="#ef4444" cross /> : <ScoreDonutChart score={freqScore} color="#6366f1" />
                  ) : (
                    <div className="w-40 h-40 rounded-full border-8 border-slate-50 bg-slate-50/20 flex items-center justify-center shadow-inner">
                      <span className="text-slate-300 text-sm font-bold uppercase tracking-widest">Waiting</span>
                    </div>
                  )}
                </div>
              </div>
              <div className="border border-gray-200 rounded-xl p-8 bg-white flex flex-col items-center shadow-sm">
                <div className="font-semibold text-slate-800 w-full mb-6 text-lg font-sans">픽셀 분석 (Pixel-level)</div>
                <div className="w-full flex-grow flex items-center justify-center min-h-[250px]">
                  {result ? (
                    isUndetermined ? <ScoreDonutChart score={100} color="#ef4444" cross /> : <ScoreDonutChart score={pixelScore} color="#3b82f6" />
                  ) : (
                    <div className="w-40 h-40 rounded-full border-8 border-slate-50 bg-slate-50/20 flex items-center justify-center shadow-inner">
                      <span className="text-slate-300 text-sm font-bold uppercase tracking-widest">Waiting</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {videoMeta && (
        <div className="mt-8 border border-slate-200 rounded-lg p-4 bg-slate-50 flex-shrink-0">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
            {videoMetaRows.map((item) => (
              <div key={item.key} className="rounded-md border border-slate-200 bg-white px-3 py-2.5">
                <div className="text-[11px] font-semibold text-slate-500">{item.label}</div>
                <div className="mt-1 text-sm font-semibold text-slate-800 break-words">{item.value}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex justify-end">
            <div className="relative group">
              <button
                type="button"
                className="text-xs font-medium text-slate-400 hover:text-slate-500 transition-colors"
              >
                가이드
              </button>
              <div
                className="absolute right-0 bottom-6 z-20 w-[320px] rounded-lg border border-slate-200 bg-white p-4 shadow-xl opacity-0 invisible transition-all duration-150 group-hover:opacity-100 group-hover:visible group-focus-within:opacity-100 group-focus-within:visible"
              >
                <div className="text-sm font-semibold text-slate-900 mb-2">방식 설명</div>
                <div className="space-y-2">
                  {aggGuide.lines.map((line, idx) => (
                    <div key={`agg-guide-${idx}`} className="text-xs text-slate-600 leading-relaxed">
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
