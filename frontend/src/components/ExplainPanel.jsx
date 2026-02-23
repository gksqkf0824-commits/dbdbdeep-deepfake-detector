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
  unknown: "UNKNOWN",
};

const toRegionLabel = (region) => REGION_LABEL[region] || region || "미확정";
const toBandLabel = (band) => BAND_LABEL[band] || band || "미확정";
const toFiniteNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};
const formatPercent = (value, digits = 1) => {
  const n = toFiniteNumber(value);
  return n === null ? "-" : `${n.toFixed(digits)}%`;
};

export default function ExplainPanel({ result }) {
  const isFake = (() => {
    if (typeof result?.isFake === "boolean") return result.isFake;
    if (typeof result?.is_fake === "boolean") return result.is_fake;
    return null;
  })();

  const summary = (() => {
    if (!result) {
      return "분석이 완료되면 AI가 어떤 근거로 판단했는지 간단하게 설명해드립니다.";
    }
    if (result.comment) return result.comment;
    if (isFake === true) {
      return "픽셀 변조 점수가 높고 주파수 패턴이 비정상적으로 감지되었습니다. 딥페이크 가능성이 있으므로 추가 검증을 권장합니다.";
    }
    if (isFake === false) {
      return "픽셀·주파수 분석에서 변조 흔적이 낮게 관측되었습니다. 비교적 신뢰도가 높지만, 100% 보장은 아니므로 참고용으로 활용하세요.";
    }
    return "분석이 완료되었습니다. 아래 근거 정보를 참고해 결과를 해석해보세요.";
  })();

  const topRegions = Array.isArray(result?.topRegions) ? result.topRegions.slice(0, 2) : [];
  const spatialFindings = Array.isArray(result?.spatialFindings) ? result.spatialFindings.slice(0, 3) : [];
  const frequencyFindings = Array.isArray(result?.frequencyFindings)
    ? result.frequencyFindings.slice(0, 4)
    : [];
  const bandEnergy = Array.isArray(result?.bandEnergy) ? result.bandEnergy : [];
  const caveats = Array.isArray(result?.caveats) ? result.caveats.slice(0, 2) : [];
  const nextSteps = Array.isArray(result?.nextSteps) ? result.nextSteps.slice(0, 2) : [];

  const confidence = toFiniteNumber(result?.videoRepresentativeConfidence ?? result?.confidence);
  const pixelScore = toFiniteNumber(result?.pixelScore ?? result?.pixel_score);
  const freqScore = toFiniteNumber(result?.freqScore ?? result?.freq_score);
  const pValue = toFiniteNumber(result?.pValue ?? result?.p_value);

  const energyText =
    bandEnergy.length > 0
      ? bandEnergy
          .map((item) => `${toBandLabel(item?.band)} ${formatPercent((Number(item?.energy_ratio) || 0) * 100)}`)
          .join(" / ")
      : "";

  const scoreSupportText = (() => {
    if (confidence === null && pixelScore === null && freqScore === null) return "";
    return `최종 ${formatPercent(confidence, 2)} · 픽셀 ${formatPercent(pixelScore, 2)} · 주파수 ${formatPercent(freqScore, 2)}`;
  })();

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6">
      <div className="font-semibold text-slate-900 mb-2">
        판독 결과에 대한 간단 설명
      </div>

      <div className="text-sm text-slate-500 leading-relaxed">
        {summary}
      </div>

      {scoreSupportText && (
        <div className="mt-4 text-sm text-slate-600 leading-relaxed">
          신뢰도 근거: {scoreSupportText}
          {pValue !== null && ` · p-value ${pValue.toFixed(4)}`}
        </div>
      )}

      {topRegions.length > 0 && (
        <div className="mt-4 text-sm text-slate-600">
          주요 근거 부위:{" "}
          {topRegions
            .map((item) => {
              const cam = Number(item?.importance_cam);
              const camText = Number.isFinite(cam) ? cam.toFixed(2) : "N/A";
              return `${toRegionLabel(item?.region)} (CAM ${camText})`;
            })
            .join(", ")}
        </div>
      )}

      {(result?.dominantBand || result?.dominantEnergyBand) && (
        <div className="mt-2 text-sm text-slate-600">
          주파수 근거:{" "}
          {result?.dominantBand && `우세 대역 ${toBandLabel(result.dominantBand)}`}
          {result?.dominantBand && result?.dominantEnergyBand && " / "}
          {result?.dominantEnergyBand && `에너지 우세 ${toBandLabel(result.dominantEnergyBand)}`}
        </div>
      )}

      {energyText && (
        <div className="mt-2 text-sm text-slate-600">
          밴드 에너지 분포: {energyText}
        </div>
      )}

      {(spatialFindings.length > 0 || frequencyFindings.length > 0) && (
        <div className="mt-4 space-y-2">
          {spatialFindings.map((item, idx) => (
            <div key={`spatial-${idx}`} className="text-xs text-slate-500 leading-relaxed">
              [공간] {item?.claim} ({item?.evidence})
            </div>
          ))}
          {frequencyFindings.map((item, idx) => (
            <div key={`freq-${idx}`} className="text-xs text-slate-500 leading-relaxed">
              [주파수] {item?.claim} ({item?.evidence})
            </div>
          ))}
        </div>
      )}

      {caveats.length > 0 && (
        <div className="mt-4 space-y-1">
          {caveats.map((item, idx) => (
            <div key={`caveat-${idx}`} className="text-xs text-slate-500 leading-relaxed">
              [주의] {item}
            </div>
          ))}
        </div>
      )}

      {nextSteps.length > 0 && (
        <div className="mt-3 space-y-1">
          {nextSteps.map((item, idx) => (
            <div key={`next-${idx}`} className="text-xs text-slate-500 leading-relaxed">
              [권장] {item}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
