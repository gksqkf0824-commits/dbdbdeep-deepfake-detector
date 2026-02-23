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
  const spatialFindings = Array.isArray(result?.spatialFindings) ? result.spatialFindings.slice(0, 2) : [];
  const frequencyFindings = Array.isArray(result?.frequencyFindings)
    ? result.frequencyFindings.slice(0, 2)
    : [];

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6">
      <div className="font-semibold text-slate-900 mb-2">
        판독 결과에 대한 간단 설명
      </div>

      <div className="text-sm text-slate-500 leading-relaxed">
        {summary}
      </div>

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
    </div>
  );
}
