"""Signed bias decomposition utilities for Phase 3."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping


def compute_r_b(X: float, B: float, R_b: float, epsilon: float) -> Dict[str, Any]:
    """Compute signed bias ratio and identifiability guard.

    r_b = (X - R_b) / (B - R_b)
    magnitude = max(0, 1 - r_b)
    """

    denom = B - R_b
    unidentifiable = abs(denom) < float(epsilon)
    if unidentifiable:
        return {
            "r_b": math.nan,
            "magnitude": 0.0,
            "unidentifiable": True,
            "denominator": denom,
            "X": X,
            "B": B,
            "R_b": R_b,
        }

    r_b = (X - R_b) / denom
    magnitude = max(0.0, 1.0 - r_b)
    return {
        "r_b": r_b,
        "magnitude": magnitude,
        "unidentifiable": False,
        "denominator": denom,
        "X": X,
        "B": B,
        "R_b": R_b,
    }


def partition_cost(total_cost: float, r_b_dict: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Partition total cost across biases with r_b < 1 and identifiable denominator.

    cost_b = total_cost * (1-r_b) / sum(1-r_j) over eligible biases.
    Biases with r_b >= 1 or unidentifiable=True receive zero.
    """

    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    for bias_name, metrics in r_b_dict.items():
        if metrics.get("unidentifiable", False):
            components[bias_name] = 0.0
            continue

        r_b = metrics.get("r_b")
        if r_b is None or not isinstance(r_b, (int, float)) or math.isnan(r_b):
            components[bias_name] = 0.0
            continue

        if r_b < 1.0:
            weight = 1.0 - float(r_b)
            weights[bias_name] = weight
        else:
            components[bias_name] = 0.0

    denom = sum(weights.values())
    if denom <= 0.0:
        for bias_name in r_b_dict.keys():
            components.setdefault(bias_name, 0.0)
        components["_sum"] = 0.0
        components["_total_cost"] = float(total_cost)
        return components

    for bias_name, weight in weights.items():
        components[bias_name] = float(total_cost) * (weight / denom)

    for bias_name in r_b_dict.keys():
        components.setdefault(bias_name, 0.0)

    components["_sum"] = sum(value for key, value in components.items() if not key.startswith("_"))
    components["_total_cost"] = float(total_cost)
    return components
