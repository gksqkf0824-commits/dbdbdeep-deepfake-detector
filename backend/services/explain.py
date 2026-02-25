"""Explanation and LLM text generation utilities."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import requests


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _extract_responses_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return _normalize_text(output_text)
    if isinstance(output_text, list):
        parts = [str(x).strip() for x in output_text if isinstance(x, str)]
        if parts:
            return _normalize_text(" ".join(parts))

    parts: List[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return _normalize_text(" ".join(parts))


def _extract_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return _normalize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return _normalize_text(" ".join(parts))
    return ""


def _call_openai_comment(system_prompt: str, user_prompt: str, max_output_tokens: int = 200) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1").rstrip("/")
    timeout_sec = _env_float("OPENAI_TIMEOUT_SEC", 20.0)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{base_url}/responses",
            headers=headers,
            json={
                "model": model,
                "temperature": 0.3,
                "max_output_tokens": int(max_output_tokens),
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout_sec,
        )
        if resp.ok:
            text = _extract_responses_text(resp.json())
            if text:
                return text
    except Exception:
        pass

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "temperature": 0.3,
                "max_tokens": int(max_output_tokens),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout_sec,
        )
        if resp.ok:
            text = _extract_chat_text(resp.json())
            if text:
                return text
    except Exception:
        pass

    return None


def generate_interpretation_guide(
    *,
    media_mode_hint: str,
    fake_prob: float,
    real_prob: float,
    top_regions_kor: List[str],
    dominant_band: str,
    dominant_energy_band: str,
    band_ablation: List[Dict[str, Any]],
    band_energy: List[Dict[str, Any]],
    freq_notes: List[str],
    use_openai: bool = True,
) -> List[str]:
    del media_mode_hint, fake_prob, real_prob, top_regions_kor
    del dominant_band, dominant_energy_band, band_ablation, band_energy, freq_notes, use_openai
    return [
        "주요 부위: 모델이 얼굴에서 특히 주목한 위치(CAM 기반)입니다.",
        "우세 주파수 대역: 밴드를 제거했을 때 예측 변화가 가장 큰 구간입니다.",
        "밴드 제거 민감도(Δfake): 각 대역 제거 전후의 fake 확률 변화량입니다.",
        "밴드 에너지 비율: Wavelet 에너지가 각 대역에 분포한 상대 비율입니다.",
        "저주파(0 ~ 0.125 cycles/pixel): 얼굴의 큰 윤곽, 완만한 밝기/색 변화 같은 저해상 구조 성분입니다.",
        "중주파(0.125 ~ 0.25 cycles/pixel): 눈/코/입 주변 경계, 피부 결 등 중간 규모 텍스처 성분입니다.",
        "고주파(0.25 ~ 0.5 cycles/pixel): 미세 경계, 세부 노이즈, 과도한 샤프닝/압축 잔상에 민감한 성분입니다.",
        "기준: Nyquist 한계는 0.5 cycles/pixel이며, 각주파수로는 2πf(rad/pixel) 관계를 사용합니다.",
        "주의: 현재 Wavelet 분석의 주파수 단위는 시간 주파수(Hz)가 아니라 공간 주파수(cycles/pixel)입니다.",
        "실세계 단위(cycles/mm)로 환산하려면 이미지의 mm/pixel 스케일 정보가 추가로 필요합니다.",
    ]


def generate_image_ai_comment(
    fake_prob: float,
    real_prob: float,
    top_regions: List[str],
    dominant_band_label: str,
    energy_low_pct: float,
    energy_mid_pct: float,
    energy_high_pct: float,
) -> Optional[str]:
    system_prompt = (
        "너는 비전문가 사용자에게 딥페이크 판독 결과를 설명하는 한국어 안내자다. "
        "중학생도 이해할 수 있는 쉬운 단어를 사용하고, 전문용어는 꼭 필요할 때만 짧게 풀이해서 쓴다. "
        "출력은 1~2문장으로 작성하고, 확정적 단정 표현은 금지한다. "
        "확률 수치(예: xx%)를 그대로 반복하지 말고 가능성 중심으로 말한다. "
        "자연스러운 유머/위트를 한 번만 가볍게 넣되, 과하거나 조롱처럼 들리지 않게 한다."
    )

    region_text = ", ".join(top_regions) if top_regions else "얼굴 핵심 부위"
    user_prompt = (
        f"최종 fake 확률 {fake_prob*100:.1f}%, real 확률 {real_prob*100:.1f}%.\n"
        f"주요 부위: {region_text}\n"
        f"우세 대역: {dominant_band_label}\n"
        f"밴드 에너지: low {energy_low_pct:.1f}%, mid {energy_mid_pct:.1f}%, high {energy_high_pct:.1f}%\n"
        "사용자용 AI 코멘트를 작성해줘.\n"
        "- 누구나 이해할 수 있는 쉬운 한국어\n"
        "- 판단 근거 1~2개를 짧게 포함\n"
        "- '확실하다' 같은 단정 대신 '가능성' 표현 사용\n"
        "- 퍼센트 숫자 반복 금지\n"
        "- 유머/위트를 한 스푼만 자연스럽게 포함"
    )
    return _call_openai_comment(system_prompt=system_prompt, user_prompt=user_prompt, max_output_tokens=180)


def _series_stats(values: List[float]) -> Optional[Dict[str, float]]:
    arr = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not arr:
        return None

    start = arr[0]
    mid = arr[(len(arr) - 1) // 2]
    end = arr[-1]
    swing = max(arr) - min(arr)
    drift = end - start
    trend = "상승" if drift > 3 else ("하강" if drift < -3 else "유지")

    return {
        "start": float(start),
        "mid": float(mid),
        "end": float(end),
        "swing": float(swing),
        "drift": float(drift),
        "trend": trend,
    }


def generate_video_ai_comment(
    final_scores: List[float],
    pixel_scores: List[float],
    freq_scores: List[float],
    is_fake: Optional[bool],
) -> Optional[str]:
    final_stats = _series_stats(final_scores)
    pixel_stats = _series_stats(pixel_scores)
    freq_stats = _series_stats(freq_scores)
    if final_stats is None:
        return None

    system_prompt = (
        "너는 비전문가 사용자에게 영상 판독 결과를 설명하는 한국어 안내자다. "
        "어려운 기술 용어를 피하고, 시간 흐름(처음/중간/끝)을 중심으로 쉽게 설명한다. "
        "출력은 1~2문장으로 작성하고, 확정적 단정 표현은 금지한다. "
        "확률 수치(예: xx%)를 그대로 반복하지 말고 가능성 중심으로 말한다. "
        "자연스러운 유머/위트를 한 번만 가볍게 넣되, 과하거나 조롱처럼 들리지 않게 한다."
    )

    verdict = (
        "조작 가능성 쪽으로 기울었습니다."
        if is_fake is True
        else "원본 가능성 쪽으로 기울었습니다."
        if is_fake is False
        else "추가 검증이 필요합니다."
    )

    def _fmt(stats_obj: Optional[Dict[str, float]], label: str) -> str:
        if not stats_obj:
            return f"{label}: 데이터 부족"
        return (
            f"{label}: 시작 {stats_obj['start']:.1f}%, 중간 {stats_obj['mid']:.1f}%, "
            f"종료 {stats_obj['end']:.1f}%, 추세 {stats_obj['trend']}, 변동폭 {stats_obj['swing']:.1f}%"
        )

    user_prompt = (
        f"{_fmt(final_stats, '최종')}\n"
        f"{_fmt(pixel_stats, '픽셀')}\n"
        f"{_fmt(freq_stats, '주파수')}\n"
        f"판정 방향: {verdict}\n"
        "사용자에게 보여줄 AI 코멘트를 작성해줘.\n"
        "- 처음/중간/끝 변화가 어떻게 보였는지 짧게 설명\n"
        "- 비전문가도 이해 가능한 쉬운 표현\n"
        "- 확정 단정 대신 가능성 표현 사용\n"
        "- 퍼센트 숫자 반복 금지\n"
        "- 유머/위트를 한 스푼만 자연스럽게 포함"
    )
    return _call_openai_comment(system_prompt=system_prompt, user_prompt=user_prompt, max_output_tokens=180)


def explain_from_evidence(
    evidence: Dict[str, Any],
    score: Dict[str, float],
    media_mode_hint: str = "image",
    use_openai: bool = True,
) -> Dict[str, Any]:
    spatial = evidence.get("spatial", {})
    freq = evidence.get("frequency", {})

    region_label = {
        "eyes": "눈 주변",
        "nose": "코 주변",
        "mouth": "입 주변",
        "forehead": "이마",
        "jawline": "턱선",
        "cheeks": "볼",
    }
    band_label = {"low": "저주파", "mid": "중주파", "high": "고주파", "unknown": "미확정"}

    def _region(region_name: str) -> str:
        rr = str(region_name or "미확정")
        return region_label.get(rr, rr)

    def _band(band_name: str) -> str:
        bb = str(band_name or "unknown")
        return band_label.get(bb, bb)

    top = spatial.get("regions_topk", [])[:2]
    dom = str(freq.get("dominant_band", "unknown"))
    band_map = {x["band"]: x["delta_fake_prob"] for x in freq.get("band_ablation", []) if "band" in x}
    energy_map = {x["band"]: x["energy_ratio"] for x in freq.get("band_energy", []) if "band" in x}
    energy_dom = str(freq.get("dominant_energy_band", "unknown"))

    band_semantics = {
        "low": "얼굴의 큰 윤곽과 완만한 밝기/색 변화 같은 저해상 구조",
        "mid": "눈·코·입 경계와 피부 결 같은 중간 규모 텍스처",
        "high": "미세 경계, 세부 노이즈, 샤프닝/압축 잔상에 민감한 성분",
    }

    if dom == "unknown" and band_map:
        try:
            dom = max(band_map.keys(), key=lambda b: abs(float(band_map[b])))
        except Exception:
            dom = "unknown"

    fake_prob = float(score.get("p_final", 0.0))
    fake_prob = max(0.0, min(1.0, fake_prob))
    real_prob = 1.0 - fake_prob
    is_fake_mode = fake_prob >= 0.5
    low = float(energy_map.get("low", 0.0)) * 100.0
    mid = float(energy_map.get("mid", 0.0)) * 100.0
    high = float(energy_map.get("high", 0.0)) * 100.0

    top_regions_kor = [_region(item.get("region", "")) for item in top if item.get("region")]
    region_hint = "얼굴 핵심 부위"
    if top_regions_kor:
        region_hint = ", ".join(top_regions_kor)

    band_hint = _band(dom if dom != "unknown" else energy_dom)
    if is_fake_mode:
        summary = (
            f"{region_hint}에서 모델 반응이 크게 나타났고 {band_hint} 대역에서도 변화가 보여, "
            "이번 샘플은 조작 가능성이 상대적으로 높아 보입니다."
        )
    else:
        summary = (
            f"{region_hint}과 {band_hint} 대역 신호가 전반적으로 안정적으로 보여, "
            "이번 샘플은 원본일 가능성이 상대적으로 높아 보입니다."
        )

    summary_source = "rule_based"
    if use_openai:
        llm_summary = generate_image_ai_comment(
            fake_prob=fake_prob,
            real_prob=real_prob,
            top_regions=top_regions_kor,
            dominant_band_label=band_hint,
            energy_low_pct=low,
            energy_mid_pct=mid,
            energy_high_pct=high,
        )
        if llm_summary:
            summary = llm_summary
            summary_source = "openai"

    spatial_findings = []
    if top:
        region_tokens = []
        for item in top:
            region = _region(item.get("region", "face"))
            importance = float(item.get("importance_cam", 0.0))
            region_tokens.append(f"{region} CAM {importance:.2f}")
        spatial_findings.append(
            {
                "claim": "주요 부위(CAM) 기준으로 모델이 특히 주목한 위치를 확인했습니다.",
                "evidence": " · ".join(region_tokens),
            }
        )

        occlusion_tokens = []
        for item in top:
            delta = item.get("delta_occlusion")
            if delta is None:
                continue
            region = _region(item.get("region", "face"))
            delta_f = float(delta) * 100.0
            direction = "증가" if delta_f > 0 else ("감소" if delta_f < 0 else "변화 거의 없음")
            occlusion_tokens.append(f"{region} {abs(delta_f):.1f}% {direction}")
        if occlusion_tokens:
            spatial_findings.append(
                {
                    "claim": "해당 부위를 가렸을 때 점수 변화를 함께 점검했습니다.",
                    "evidence": " · ".join(occlusion_tokens),
                }
            )

    outside_face_ratio = spatial.get("outside_face_ratio", None)
    localization_conf = str(spatial.get("localization_confidence", "unknown"))
    if outside_face_ratio is not None:
        try:
            outside_pct = float(outside_face_ratio) * 100.0
            if outside_pct <= 25.0:
                claim = "근거가 얼굴 중심에 비교적 잘 모여 있습니다."
            else:
                claim = "근거가 얼굴 외곽에도 일부 분산되어 해석 시 주의가 필요합니다."
            evidence_txt = f"얼굴 바깥 비율 {outside_pct:.1f}%, 집중도 {localization_conf}"
            spatial_findings.append({"claim": claim, "evidence": evidence_txt})
        except Exception:
            pass

    if not spatial_findings:
        spatial_findings.append(
            {
                "claim": "얼굴 전반 패턴을 기반으로 판별했습니다.",
                "evidence": "부위별 상위 근거가 제한되어 전체 정보를 함께 활용했습니다.",
            }
        )

    frequency_findings = []
    if dom in band_map:
        delta_f = float(band_map[dom]) * 100.0
        frequency_findings.append(
            {
                "claim": f"우세 주파수 대역은 {_band(dom)}로 확인됐습니다.",
                "evidence": f"{_band(dom)} 제거 기준 Δfake {delta_f:+.1f}%",
            }
        )
    else:
        frequency_findings.append(
            {
                "claim": "우세 주파수 대역은 미확정입니다.",
                "evidence": "밴드 제거 전후 점수 변화가 작거나 계산되지 않아 대역 우세를 단정하기 어렵습니다.",
            }
        )

    if band_map:
        band_delta_tokens = []
        for band in ("low", "mid", "high"):
            if band not in band_map:
                continue
            delta_f = float(band_map[band]) * 100.0
            band_delta_tokens.append(f"{_band(band)} {delta_f:+.1f}%")
        if band_delta_tokens:
            frequency_findings.append(
                {
                    "claim": "밴드 제거 민감도(Δfake)를 대역별로 비교했습니다.",
                    "evidence": " · ".join(band_delta_tokens) + " (+: fake↑, -: fake↓)",
                }
            )

    dominant_energy_label = _band(energy_dom)
    semantic_hint = band_semantics.get(energy_dom)
    energy_claim = f"밴드 에너지 비율 기준 우세 대역은 {dominant_energy_label}입니다."
    if semantic_hint:
        energy_claim = f"{energy_claim} {semantic_hint} 신호가 상대적으로 두드러졌습니다."
    frequency_findings.append(
        {
            "claim": energy_claim,
            "evidence": f"저주파 {low:.1f}% · 중주파 {mid:.1f}% · 고주파 {high:.1f}%",
        }
    )

    if dom != "unknown" and energy_dom != "unknown":
        consistency = "일관" if dom == energy_dom else "부분 불일치"
        frequency_findings.append(
            {
                "claim": "주파수 민감도와 에너지 우세 대역의 합치도를 확인했습니다.",
                "evidence": f"민감도 우세 {_band(dom)} / 에너지 우세 {_band(energy_dom)} ({consistency})",
            }
        )

    frequency_findings.append(
        {
            "claim": "최종 확률 축에서도 같은 방향의 결론이 확인됩니다.",
            "evidence": f"fake {fake_prob*100.0:.1f}%, real {real_prob*100.0:.1f}%",
        }
    )

    freq_notes = freq.get("notes", [])
    spatial_notes = spatial.get("notes", [])
    caveats = [
        "강한 압축이나 저해상도는 주파수 패턴을 왜곡해 오탐/미탐을 늘릴 수 있습니다.",
        "자동 판별은 보조 근거입니다. 중요한 의사결정은 추가 검증과 함께 진행하세요.",
    ]
    if any("skipped" in str(note) for note in (freq_notes or [])) or any(
        "skipped" in str(note) for note in (spatial_notes or [])
    ):
        caveats.insert(0, "일부 근거 실험이 생략되어, 이번 결과는 보수적으로 해석하는 편이 안전합니다.")

    interpretation_guide = generate_interpretation_guide(
        media_mode_hint=media_mode_hint,
        fake_prob=fake_prob,
        real_prob=real_prob,
        top_regions_kor=top_regions_kor,
        dominant_band=dom,
        dominant_energy_band=energy_dom,
        band_ablation=freq.get("band_ablation", []) if isinstance(freq.get("band_ablation", []), list) else [],
        band_energy=freq.get("band_energy", []) if isinstance(freq.get("band_energy", []), list) else [],
        freq_notes=[str(x) for x in (freq_notes or [])],
        use_openai=use_openai,
    )

    return {
        "summary": summary,
        "summary_source": summary_source,
        "spatial_findings": spatial_findings[:4],
        "frequency_findings": frequency_findings[:4],
        "interpretation_guide": interpretation_guide[:10],
        "next_steps": [
            "원본에 가까운 고해상도 파일(재인코딩 전)로 한 번 더 교차 검증하세요.",
            "가능하면 다른 각도/조명 샘플을 추가해 같은 결론이 반복되는지 확인하세요.",
        ],
        "caveats": caveats[:3],
    }


__all__ = [
    "explain_from_evidence",
    "generate_interpretation_guide",
    "generate_image_ai_comment",
    "generate_video_ai_comment",
]
