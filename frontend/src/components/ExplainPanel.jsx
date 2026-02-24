const REGION_LABEL = {
  eyes: "눈 주변",
  nose: "코 주변",
  mouth: "입 주변",
  forehead: "이마",
  jawline: "턱선",
  cheeks: "볼",
};

const BAND_LABEL = {
  low: "저주파",
  mid: "중주파",
  high: "고주파",
  unknown: "미확정",
};

const toRegionLabel = (region) => REGION_LABEL[region] || region || "미확정";
const toBandLabel = (band) => BAND_LABEL[band] || band || "미확정";

const toFiniteNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const toEnergyPercent = (value) => {
  const n = toFiniteNumber(value);
  if (n === null) return null;
  return n <= 1 ? n * 100 : n;
};

const summarizeSeries = (values) => {
  const series = Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : [];
  if (series.length === 0) return null;
  const start = series[0];
  const mid = series[Math.floor((series.length - 1) / 2)];
  const end = series[series.length - 1];
  const max = Math.max(...series);
  const min = Math.min(...series);
  const swing = Math.max(0, max - min);
  const drift = end - start;
  const trend = drift > 3 ? "상승" : drift < -3 ? "하강" : "유지";
  return { start, mid, end, swing, trend };
};

const compactFinding = (item) => {
  const claim = String(item?.claim || "").trim();
  const evidence = String(item?.evidence || "").trim();
  if (claim && evidence) return `${claim} (${evidence})`;
  return claim || evidence;
};

const ListBox = ({ title, items, emptyText, visualTitle = "", visualUrl = null }) => (
  <div className="rounded-md border border-slate-200 bg-white p-4">
    <div className="text-sm font-semibold text-slate-900">{title}</div>
    {visualUrl ? (
      <div className="mt-3">
        {visualTitle ? <div className="text-xs font-semibold text-slate-500 mb-2">{visualTitle}</div> : null}
        <div className="w-[200px] h-[200px] rounded-md border border-slate-200 bg-slate-100 overflow-hidden flex items-center justify-center">
          <img src={visualUrl} alt={visualTitle || title} className="w-full h-full object-contain" />
        </div>
      </div>
    ) : null}
    {items.length > 0 ? (
      <div className="mt-3 space-y-2">
        {items.map((item, idx) => (
          <div key={`${title}-${idx}`} className="text-sm text-slate-600 leading-relaxed">
            {item}
          </div>
        ))}
      </div>
    ) : (
      <div className="mt-3 text-sm text-slate-400">{emptyText}</div>
    )}
  </div>
);

export default function ExplainPanel({ result }) {
  const topRegions = Array.isArray(result?.topRegions) ? result.topRegions.slice(0, 2) : [];
  const spatialFindings = Array.isArray(result?.spatialFindings)
    ? result.spatialFindings.map(compactFinding).filter(Boolean).slice(0, 3)
    : [];
  const frequencyFindings = Array.isArray(result?.frequencyFindings)
    ? result.frequencyFindings.map(compactFinding).filter(Boolean).slice(0, 3)
    : [];
  const caveats = Array.isArray(result?.caveats)
    ? result.caveats.map((v) => String(v || "").trim()).filter(Boolean).slice(0, 3)
    : [];
  const nextSteps = Array.isArray(result?.nextSteps)
    ? result.nextSteps.map((v) => String(v || "").trim()).filter(Boolean).slice(0, 3)
    : [];
  const interpretationGuide = Array.isArray(result?.interpretationGuide)
    ? result.interpretationGuide.map((v) => String(v || "").trim()).filter(Boolean).slice(0, 6)
    : [];
  const bandEnergy = Array.isArray(result?.bandEnergy) ? result.bandEnergy : [];

  const dominantBand = result?.dominantBand ? toBandLabel(result.dominantBand) : "";
  const dominantEnergyBand = result?.dominantEnergyBand ? toBandLabel(result.dominantEnergyBand) : "";

  const timelineFinalStats = summarizeSeries(
    (Array.isArray(result?.timeline) ? result.timeline : []).map((item) => item?.final)
  );

  const hasCoreEvidence =
    topRegions.length > 0 ||
    dominantBand ||
    dominantEnergyBand ||
    bandEnergy.length > 0 ||
    Boolean(timelineFinalStats);

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6">
      <div className="font-semibold text-slate-900 mb-4">판독 결과에 대한 세부 근거</div>

      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="text-sm font-semibold text-slate-900">핵심 근거 요약</div>
        {!hasCoreEvidence ? (
          <div className="mt-3 text-sm text-slate-400">
            분석이 완료되면 주요 근거 부위와 타임라인 단서를 요약해 표시합니다.
          </div>
        ) : (
          <div className="mt-3 space-y-3">
            {topRegions.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-2">주요 근거 부위</div>
                <div className="flex flex-wrap gap-2">
                  {topRegions.map((item, idx) => {
                    const cam = toFiniteNumber(item?.importance_cam);
                    return (
                      <span
                        key={`region-${idx}`}
                        className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700"
                      >
                        {toRegionLabel(item?.region)}
                        {cam !== null ? ` · CAM ${cam.toFixed(2)}` : ""}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {(dominantBand || dominantEnergyBand) && (
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-2">주파수 단서</div>
                <div className="flex flex-wrap gap-2">
                  {dominantBand && (
                    <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                      우세 대역: {dominantBand}
                    </span>
                  )}
                  {dominantEnergyBand && (
                    <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                      에너지 우세: {dominantEnergyBand}
                    </span>
                  )}
                </div>
              </div>
            )}

            {bandEnergy.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-2">밴드 에너지</div>
                <div className="flex flex-wrap gap-2">
                  {bandEnergy.map((item, idx) => {
                    const ratioPct = toEnergyPercent(item?.energy_ratio);
                    return (
                      <span
                        key={`energy-${idx}`}
                        className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700"
                      >
                        {toBandLabel(item?.band)} {ratioPct !== null ? `${ratioPct.toFixed(1)}%` : "-"}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {timelineFinalStats && (
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-2">타임라인 요약</div>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                    시작 {timelineFinalStats.start.toFixed(1)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                    중간 {timelineFinalStats.mid.toFixed(1)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                    종료 {timelineFinalStats.end.toFixed(1)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                    변동폭 {timelineFinalStats.swing.toFixed(1)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-xs bg-white border border-slate-200 text-slate-700">
                    추세 {timelineFinalStats.trend}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="text-sm font-semibold text-slate-900 mb-3">분석 근거</div>
        <div className="flex flex-col gap-3">
          <ListBox
            title="공간 분석"
            items={spatialFindings}
            emptyText="공간 분석 근거가 아직 없습니다."
            visualTitle="Grad-CAM 분석 결과"
            visualUrl={result?.spatialVisualUrl || null}
          />
          <ListBox
            title="주파수 분석"
            items={frequencyFindings}
            emptyText="주파수 분석 근거가 아직 없습니다."
            visualTitle="Wavelet 분석 결과"
            visualUrl={result?.frequencyVisualUrl || null}
          />
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="text-sm font-semibold text-slate-900 mb-3">참고 사항</div>
        <div className="flex flex-col gap-3">
          <ListBox title="⚠️ 주의" items={caveats} emptyText="추가 주의사항이 없습니다." />
          <ListBox title="✅ 권장" items={nextSteps} emptyText="추가 권장사항이 없습니다." />
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="text-sm font-semibold text-slate-900 mb-3">📌 해석 가이드</div>
        <div className="rounded-md border border-slate-200 bg-white p-4">
          {interpretationGuide.length > 0 ? (
            <div className="space-y-2">
              {interpretationGuide.map((line, idx) => (
                <div key={`guide-${idx}`} className="text-sm text-slate-600 leading-relaxed">
                  {line}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-slate-400">해석 가이드가 아직 없습니다.</div>
          )}
        </div>
      </div>
    </div>
  );
}
