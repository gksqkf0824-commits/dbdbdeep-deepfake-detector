"""Explanation and LLM text generation utilities."""

import os
import re
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


_NUMERIC_MENTION_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?\s*(?:%|％|퍼센트)?")


def _strip_numeric_mentions(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _NUMERIC_MENTION_RE.sub("", text)
    cleaned = cleaned.replace("%", "").replace("％", "")
    cleaned = _normalize_text(cleaned)
    cleaned = cleaned.strip(" .,!?:;")
    return cleaned


def sanitize_ai_comment(text: str) -> str:
    """Normalize AI comments to avoid numeric/probability-style phrasing."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"(?:진짜|가짜|원본|조작|거짓)?\s*확률(?:은|는|이|가)?\s*[0-9]+(?:[.,][0-9]+)?\s*(?:%|％|퍼센트)?(?:\s*(?:로|으로))?",
        "",
        cleaned,
    )
    cleaned = _strip_numeric_mentions(cleaned)
    cleaned = re.sub(r"\s+(은|는|이|가|을|를|로|으로)\s*([,.!?]|$)", r"\2", cleaned)
    cleaned = re.sub(r"가능성\s+입니다", "가능성이 있습니다", cleaned)
    cleaned = re.sub(r"가능성이\s+입니다", "가능성이 있습니다", cleaned)
    cleaned = _normalize_text(cleaned)
    cleaned = re.sub(r"\.\s*,\s*", ". ", cleaned)
    cleaned = re.sub(r"^[,.;:]+\s*", "", cleaned)
    cleaned = cleaned.strip(" .,!?:;")
    if cleaned.startswith("입니다"):
        return ""
    if len(cleaned) < 6:
        return ""
    return cleaned


def _qualitative_confidence(fake_prob: float) -> str:
    margin = abs(float(fake_prob) - 0.5)
    if margin >= 0.30:
        return "판단 신호가 비교적 또렷한 편"
    if margin >= 0.15:
        return "판단 신호가 어느 정도 보이는 편"
    return "판단 신호가 팽팽해 추가 확인이 필요한 편"


def _qualitative_image_verdict(fake_prob: float) -> str:
    p = float(fake_prob)
    margin = abs(p - 0.5)
    if p >= 0.5:
        if margin >= 0.25:
            return "조작 가능성 쪽으로 꽤 기울어 보입니다."
        return "조작 가능성 쪽으로 살짝 기울어 보입니다."
    if margin >= 0.25:
        return "원본 가능성 쪽으로 꽤 기울어 보입니다."
    return "원본 가능성 쪽으로 살짝 기울어 보입니다."


def _qualitative_band_energy(low_pct: float, mid_pct: float, high_pct: float) -> str:
    values = {"저주파": float(low_pct), "중주파": float(mid_pct), "고주파": float(high_pct)}
    dominant = max(values.keys(), key=lambda k: values[k])
    if dominant == "저주파":
        return "큰 윤곽 중심 신호가 비교적 도드라집니다."
    if dominant == "중주파":
        return "경계/텍스처 중심 신호가 비교적 도드라집니다."
    return "미세 경계 중심 신호가 비교적 도드라집니다."


def _qualitative_video_flow(stats_obj: Optional[Dict[str, float]], label: str) -> str:
    if not stats_obj:
        return f"{label} 흐름은 데이터가 부족합니다."

    trend = str(stats_obj.get("trend") or "유지")
    swing = float(stats_obj.get("swing") or 0.0)
    if swing >= 20.0:
        variance = "변화 폭이 큰 편"
    elif swing >= 8.0:
        variance = "변화 폭이 중간 정도"
    else:
        variance = "변화 폭이 크지 않은 편"

    if trend == "상승":
        trend_text = "뒤로 갈수록 올라가는 흐름"
    elif trend == "하강":
        trend_text = "뒤로 갈수록 내려가는 흐름"
    else:
        trend_text = "처음부터 끝까지 비교적 유지되는 흐름"

    return f"{label}은 {trend_text}이고 {variance}입니다."


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
    verdict = _qualitative_image_verdict(fake_prob)
    confidence_hint = _qualitative_confidence(fake_prob)
    band_energy_hint = _qualitative_band_energy(energy_low_pct, energy_mid_pct, energy_high_pct)

    system_prompt = (
        "너는 비전문가 사용자에게 딥페이크 판독 결과를 설명하는 한국어 안내자다. "
        "중학생도 이해할 수 있는 쉬운 단어를 사용하고, 전문용어는 꼭 필요할 때만 짧게 풀이해서 쓴다. "
        "아라비아 숫자(0-9), 퍼센트 기호(%), 소수점 표기를 절대 사용하지 않는다. "
        "출력은 한두 문장으로 작성하고, 단정 대신 가능성 중심으로 설명한다. "
        "유머, 위트, 센스, 농담 표현은 사용하지 않는다. "
        "문장은 자연스럽게 이어져야 하며 해석 오해가 없도록 간결하게 작성한다."
    )

    region_text = ", ".join(top_regions) if top_regions else "얼굴 핵심 부위"
    user_prompt = (
        f"판정 방향: {verdict}\n"
        f"판정 강도 힌트: {confidence_hint}\n"
        f"주요 부위: {region_text}\n"
        f"우세 대역: {dominant_band_label}\n"
        f"밴드 에너지 요약: {band_energy_hint}\n"
        "사용자용 AI 코멘트를 작성해줘.\n"
        "- 일반인이 이해하기 쉽게 설명\n"
        "- 수치(숫자, 퍼센트) 표기 금지\n"
        "- 유머/위트/센스/농담 표현 금지\n"
        "- 모든 문장이 자연스럽게 이어지도록 작성\n"
        "- 판단 근거를 한두 개 짧게 포함\n"
        "- 숫자 근거를 그대로 읊지 말고 쉬운 말로 바꿔 설명\n"
        "- '확실하다' 같은 단정 대신 '가능성' 표현 사용"
    )
    raw = _call_openai_comment(system_prompt=system_prompt, user_prompt=user_prompt, max_output_tokens=180)
    if not raw:
        return None
    cleaned = sanitize_ai_comment(raw)
    return cleaned or None


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

    final_flow = _qualitative_video_flow(final_stats, "최종 흐름")
    pixel_flow = _qualitative_video_flow(pixel_stats, "픽셀 흐름")
    freq_flow = _qualitative_video_flow(freq_stats, "주파수 흐름")
    swing = float(final_stats.get("swing") or 0.0)
    if swing >= 20.0:
        confidence_hint = "흐름 신호가 비교적 또렷하게 보입니다."
    elif swing >= 8.0:
        confidence_hint = "흐름 신호가 어느 정도 보이지만 추가 맥락 확인이 좋습니다."
    else:
        confidence_hint = "흐름 신호가 미세해 추가 확인이 특히 중요합니다."

    system_prompt = (
        "너는 비전문가 사용자에게 영상 판독 결과를 설명하는 한국어 안내자다. "
        "어려운 기술 용어를 피하고, 시간 흐름(처음/중간/끝)을 중심으로 쉽게 설명한다. "
        "아라비아 숫자(0-9), 퍼센트 기호(%), 소수점 표기를 절대 사용하지 않는다. "
        "출력은 한두 문장으로 작성하고, 단정 대신 가능성 중심으로 설명한다. "
        "유머, 위트, 센스, 농담 표현은 사용하지 않는다. "
        "문장은 자연스럽게 이어져야 하며 해석 오해가 없도록 간결하게 작성한다."
    )

    verdict = (
        "조작되었을 가능성이 높습니다."
        if is_fake is True
        else "조작이 없을 가능성이 높습니다."
        if is_fake is False
        else "추가 검증이 필요합니다."
    )

    user_prompt = (
        f"{final_flow}\n"
        f"{pixel_flow}\n"
        f"{freq_flow}\n"
        f"판정 방향: {verdict}\n"
        f"판정 강도 힌트: {confidence_hint}\n"
        "사용자에게 보여줄 AI 코멘트를 작성해줘.\n"
        "- 일반인이 이해하기 쉽게 설명\n"
        "- 수치(숫자, 퍼센트) 표기 금지\n"
        "- 유머/위트/센스/농담 표현 금지\n"
        "- 모든 문장이 자연스럽게 이어지도록 작성\n"
        "- 처음/중간/끝 변화 흐름을 짧게 요약\n"
        "- 확정 단정 대신 가능성 표현 사용\n"
        "- 판단 근거를 한두 개 포함"
    )
    raw = _call_openai_comment(system_prompt=system_prompt, user_prompt=user_prompt, max_output_tokens=180)
    if not raw:
        return None
    cleaned = sanitize_ai_comment(raw)
    return cleaned or None


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
            "해당 샘플은 조작 가능성이 상대적으로 높아 보입니다."
        )
    else:
        summary = (
            f"{region_hint}과 {band_hint} 대역 신호가 전반적으로 안정적으로 보여, "
            "해당 샘플은 원본일 가능성이 상대적으로 높아 보입니다."
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
    "sanitize_ai_comment",
]
