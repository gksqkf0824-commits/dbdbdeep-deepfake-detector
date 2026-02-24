"""Statistical and score aggregation utilities."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


def calculate_p_value(score: float, real_mean: float, real_std: float) -> float:
    """score가 real 분포에서 얼마나 드문지(우측 꼬리) p-value로 계산."""
    z_score = (score - real_mean) / real_std
    p_value = 1 - stats.norm.cdf(z_score)
    return round(max(float(p_value), 0.0001), 4)


def make_reliability_label(p_val: float) -> str:
    if p_val < 0.01:
        return "매우 높음"
    if p_val < 0.05:
        return "높음"
    return "보통"


def aggregate_scores(values: List[float], mode: str = "mean", topk: int = 5) -> Optional[float]:
    if not values:
        return None

    arr = np.array(values, dtype=np.float32)

    if mode == "median":
        return float(np.median(arr))

    if mode == "topk_mean":
        k = min(int(topk), len(arr))
        topk_vals = np.sort(arr)[-k:]
        return float(np.mean(topk_vals))

    return float(np.mean(arr))


def trimmed_mean_confidence(
    values: List[float],
    trim_ratio: float = 0.10,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    상/하위 trim_ratio 비율을 제외한 값들의 평균을 계산.
    예: trim_ratio=0.10 이면 하위 10%, 상위 10%를 제외.
    """
    if not values:
        return None, {
            "trim_ratio": float(trim_ratio),
            "raw_count": 0,
            "used_count": 0,
            "excluded_low_count": 0,
            "excluded_high_count": 0,
        }

    arr = np.sort(np.array(values, dtype=np.float32))
    n = len(arr)

    ratio = float(trim_ratio)
    if ratio < 0:
        ratio = 0.0
    if ratio > 0.49:
        ratio = 0.49

    trim_count = int(np.floor(n * ratio))
    max_trim = (n - 1) // 2
    trim_count = min(trim_count, max_trim)

    if trim_count > 0:
        core = arr[trim_count : n - trim_count]
    else:
        core = arr

    if core.size == 0:
        core = arr
        trim_count = 0

    return float(np.mean(core)), {
        "trim_ratio": ratio,
        "raw_count": n,
        "used_count": int(core.size),
        "excluded_low_count": int(trim_count),
        "excluded_high_count": int(trim_count),
    }


def build_analysis_result(
    score: float,
    pixel: float,
    freq: float,
    real_mean: float,
    real_std: float,
) -> Dict[str, Any]:
    p_val = calculate_p_value(score, real_mean=real_mean, real_std=real_std)
    return {
        "confidence": float(score),
        "pixel_score": float(pixel) if pixel is not None else None,
        "freq_score": float(freq) if freq is not None else None,
        "is_fake": float(score) < 50,
        "p_value": p_val,
        "reliability": make_reliability_label(p_val),
    }


__all__ = [
    "calculate_p_value",
    "make_reliability_label",
    "aggregate_scores",
    "trimmed_mean_confidence",
    "build_analysis_result",
]
