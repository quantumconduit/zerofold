"""
ZeroFold Collapse Detector — Governing Dynamics Engine
=======================================================
Based on: Governing_Dynamics.pdf — Universal Growth-Collapse Theorem (Theorem 15.1)

Core theorem:
    S_req(t) = k * [P(t)]^n,  n > 1
    Collapse at t_c = (S_cap / k*a^n)^(1/n)

Where:
    S_req = structural support required by propagation
    P(t)  = propagation intensity at time t
    S_cap = system structural capacity
    k, n  = scaling constants (system-dependent)
    a     = growth rate of P(t) (P(t) = a*t for linear growth)

4-Field structure: F(t) = (T, X, G, E)
    T = Time (structural channel)
    X = Space (propagational channel)
    G = Gravity (structural channel)
    E = Energy (propagational channel)

Phase transition b_crit ≈ 0.20 (empirically validated in critical threshold.png)

Applications:
    - AI training OOM: P = parameter count, S_cap = memory/compute budget
    - Financial leverage: P = leverage ratio, S_cap = collateral capacity
    - Grid stability: P = load growth, S_cap = generation+storage capacity
    - Software buffer overflow: P = input growth rate, S_cap = buffer size
    - Stellar collapse: P = fusion rate, S_cap = gravitational binding energy
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Collapse result
# ---------------------------------------------------------------------------

class CollapsePhase(str, Enum):
    STABLE    = "stable"       # Far from collapse
    WARNING   = "warning"      # Within 2× of t_c
    CRITICAL  = "critical"     # Within 1.1× of t_c
    COLLAPSED = "collapsed"    # Past t_c


@dataclass
class CollapseResult:
    """Result of a collapse prediction."""
    t_c:          float        # Predicted collapse time (same units as input)
    t_current:    float        # Current time
    margin:       float        # (t_c - t_current) / t_c — 0 means imminent
    phase:        CollapsePhase
    s_req_now:    float        # Structural requirement at t_current
    s_cap:        float        # System structural capacity
    doubling_time: float       # Time for S_req to double (danger metric)
    domain:       str          # Application domain label

    @property
    def time_to_collapse(self) -> float:
        return max(0.0, self.t_c - self.t_current)

    @property
    def overload_factor(self) -> float:
        """How much S_req exceeds S_cap. >1 means already overloaded."""
        return self.s_req_now / max(self.s_cap, 1e-30)

    def summary(self) -> str:
        lines = [
            f"Domain: {self.domain}",
            f"Phase:  {self.phase.value.upper()}",
            f"t_c   = {self.t_c:.4g}  (collapse predicted at this time)",
            f"t_now = {self.t_current:.4g}",
            f"Time to collapse: {self.time_to_collapse:.4g}",
            f"Margin: {self.margin*100:.1f}%",
            f"S_req now: {self.s_req_now:.4g}  vs  S_cap: {self.s_cap:.4g}",
            f"Overload factor: {self.overload_factor:.3f}×",
            f"Doubling time: {self.doubling_time:.4g}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core theorem implementation
# ---------------------------------------------------------------------------

class CollapseDetector:
    """
    Universal Growth-Collapse Detector based on Governing Dynamics Theorem 15.1.

    Detects when any system undergoing superlinear structural demand growth
    will exceed its structural capacity — regardless of domain.

    Usage:
        detector = CollapseDetector(domain="AI Training")
        result = detector.predict(
            t_current=1.0,      # current time (any unit)
            p_now=1e9,          # current propagation intensity (e.g., params)
            growth_rate=2.5,    # a: P grows as a*t^gamma
            growth_exp=1.0,     # gamma: linear growth in params
            s_cap=80e9,         # structural capacity (e.g., GPU memory in bytes)
            k=0.1,              # proportionality constant
            n=1.5,              # superlinearity exponent (>1 required)
        )
        print(result.summary())
    """

    def __init__(self, domain: str = "generic"):
        self.domain = domain

    def predict(self,
                t_current:   float,
                p_now:       float,
                growth_rate: float,
                growth_exp:  float = 1.0,
                s_cap:       float = 1.0,
                k:           float = 1.0,
                n:           float = 2.0) -> CollapseResult:
        """
        Predict collapse time given current system state.

        Args:
            t_current:   Current time (seconds, steps, quarters — any unit)
            p_now:       Current propagation intensity P(t_current)
            growth_rate: a — coefficient of P growth: P(t) = a * t^gamma
            growth_exp:  gamma — exponent of P growth (default 1.0 = linear)
            s_cap:       Structural capacity ceiling
            k:           Proportionality constant for S_req = k * P^n
            n:           Superlinearity exponent (must be > 1 for collapse)

        Returns:
            CollapseResult with t_c, phase, margin, and actionable metrics
        """
        if n <= 1.0:
            raise ValueError(f"n must be > 1 for collapse to occur (got n={n}). "
                             "Sublinear/linear systems are stable.")

        # S_req(t) = k * [P(t)]^n  where P(t) = growth_rate * t^growth_exp
        # Collapse when S_req(t_c) = S_cap
        # k * (growth_rate * t_c^growth_exp)^n = S_cap
        # t_c = (S_cap / (k * growth_rate^n)) ^ (1 / (n * growth_exp))
        try:
            inner  = s_cap / max(k * (growth_rate ** n), 1e-300)
            t_c    = inner ** (1.0 / (n * growth_exp))
        except (OverflowError, ZeroDivisionError, ValueError):
            t_c    = float("inf")

        # S_req at current time
        p_t         = growth_rate * (t_current ** growth_exp)
        s_req_now   = k * (p_t ** n)

        # Doubling time: how long until S_req doubles?
        # k * (growth_rate * t_d^gamma)^n = 2 * s_req_now
        # t_d = t_current * 2^(1 / (n * growth_exp))
        doubling_time = t_current * (2.0 ** (1.0 / (n * growth_exp))) - t_current

        # Phase classification
        margin = (t_c - t_current) / max(t_c, 1e-12)
        if t_current >= t_c:
            phase = CollapsePhase.COLLAPSED
        elif margin < 0.1:
            phase = CollapsePhase.CRITICAL
        elif margin < 0.5:
            phase = CollapsePhase.WARNING
        else:
            phase = CollapsePhase.STABLE

        return CollapseResult(
            t_c=t_c,
            t_current=t_current,
            margin=margin,
            phase=phase,
            s_req_now=s_req_now,
            s_cap=s_cap,
            doubling_time=doubling_time,
            domain=self.domain,
        )

    def scan_trajectory(self,
                        t_values: list[float],
                        growth_rate: float,
                        growth_exp:  float = 1.0,
                        s_cap:       float = 1.0,
                        k:           float = 1.0,
                        n:           float = 2.0) -> list[CollapseResult]:
        """Run prediction across a time series — useful for dashboards."""
        return [
            self.predict(t, growth_rate * t**growth_exp,
                         growth_rate, growth_exp, s_cap, k, n)
            for t in t_values
        ]


# ---------------------------------------------------------------------------
# Pre-built domain configurations
# ---------------------------------------------------------------------------

class DomainPresets:
    """
    Pre-configured CollapseDetector instances for common domains.
    All parameters calibrated from Governing_Dynamics.pdf examples.
    """

    @staticmethod
    def ai_training_oom(
        current_params: float = 7e9,    # current model size
        target_params:  float = None,   # optional: projected params at t_target
        gpu_memory_gb:  float = 80.0,   # H100 = 80GB
        t_current:      float = 1.0,    # training step or epoch
        growth_rate:    float = 3.16,   # ~10× per year in param count → sqrt per step
    ) -> CollapseResult:
        """
        AI training out-of-memory collapse.
        P = parameter count growth, S_cap = GPU memory capacity.
        Typical n=1.3 (memory scales superlinearly with params due to activations+gradients).
        """
        s_cap = gpu_memory_gb * 1e9  # bytes
        k     = 4.0                  # 4 bytes per param (fp32), but activations multiply
        n     = 1.3                  # superlinear memory demand
        det   = CollapseDetector("AI Training OOM")
        return det.predict(t_current, current_params, growth_rate,
                           growth_exp=1.0, s_cap=s_cap, k=k, n=n)

    @staticmethod
    def financial_leverage(
        current_leverage: float = 10.0,  # current leverage ratio (e.g., 10×)
        collateral_cap:   float = 1.0,   # normalized collateral capacity
        t_current:        float = 1.0,   # time (quarters, months)
        growth_rate:      float = 1.2,   # leverage growth per period
    ) -> CollapseResult:
        """
        Financial leverage collapse (margin call cascade).
        P = leverage ratio, S_cap = collateral capacity.
        n=2.0 — losses accelerate quadratically with leverage.
        """
        det = CollapseDetector("Financial Leverage")
        return det.predict(t_current, current_leverage, growth_rate,
                           growth_exp=1.0, s_cap=collateral_cap, k=0.01, n=2.0)

    @staticmethod
    def grid_stability(
        current_load_gw:  float = 100.0,  # current grid load (GW)
        capacity_gw:      float = 120.0,  # total generation + storage (GW)
        t_current:        float = 1.0,    # time (hours)
        growth_rate:      float = 5.0,    # load growth (GW/hour)
    ) -> CollapseResult:
        """
        Grid stability collapse (frequency instability).
        P = load, S_cap = generation+storage capacity.
        n=1.5 — grid instability grows superlinearly near capacity.
        """
        det = CollapseDetector("Grid Stability")
        return det.predict(t_current, current_load_gw, growth_rate,
                           growth_exp=1.0, s_cap=capacity_gw, k=1.0, n=1.5)

    @staticmethod
    def buffer_overflow(
        current_rate_mbps: float = 100.0,  # current input rate (Mbps)
        buffer_mb:         float = 512.0,  # buffer size (MB)
        t_current:         float = 1.0,    # seconds
        growth_rate:       float = 50.0,   # rate growth (Mbps/s)
    ) -> CollapseResult:
        """
        Software buffer overflow / queue saturation.
        P = input rate, S_cap = buffer size.
        n=1.2 — queue depth grows faster than rate due to head-of-line blocking.
        """
        det = CollapseDetector("Buffer Overflow")
        return det.predict(t_current, current_rate_mbps, growth_rate,
                           growth_exp=1.0, s_cap=buffer_mb, k=0.001, n=1.2)


# ---------------------------------------------------------------------------
# Phase transition detection (b_crit ≈ 0.20 from oscillator theory)
# ---------------------------------------------------------------------------

def detect_phase_transition(b_values: np.ndarray,
                             recall_values: np.ndarray,
                             threshold: float = 0.85) -> dict:
    """
    Detect the critical b value where recall crosses the phase transition.
    Validated: b_crit ≈ 0.20 (from Oscillator_Theory_and_The_RH.pdf + critical_threshold.png)

    Args:
        b_values:     Array of absorption coefficient values
        recall_values: Array of detection recall values
        threshold:    Recall value defining the phase boundary

    Returns:
        dict with b_crit, transition_sharpness, below_mean, above_mean
    """
    above = recall_values >= threshold
    transitions = np.where(np.diff(above.astype(int)) > 0)[0]

    if len(transitions) == 0:
        b_crit = float("nan")
    else:
        idx    = transitions[0]
        # Linear interpolation between the two points
        b0, b1 = b_values[idx], b_values[idx + 1]
        r0, r1 = recall_values[idx], recall_values[idx + 1]
        b_crit = b0 + (threshold - r0) * (b1 - b0) / max(r1 - r0, 1e-12)

    below_mask = b_values < (b_crit if not math.isnan(b_crit) else b_values[len(b_values)//2])
    sharpness  = (recall_values[~below_mask].mean() - recall_values[below_mask].mean()) \
                  if any(below_mask) and any(~below_mask) else float("nan")

    return {
        "b_crit":               round(float(b_crit), 4) if not math.isnan(b_crit) else None,
        "transition_sharpness": round(float(sharpness), 4) if not math.isnan(sharpness) else None,
        "below_b_crit_recall":  round(float(recall_values[below_mask].mean()), 4) if any(below_mask) else None,
        "above_b_crit_recall":  round(float(recall_values[~below_mask].mean()), 4) if any(~below_mask) else None,
        "expected_b_crit":      0.20,  # From validated experimental results
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ZeroFold Collapse Detector — Governing Dynamics Engine")
    print("=" * 56)
    print()

    # Demo: AI training OOM
    r = DomainPresets.ai_training_oom(current_params=7e9, gpu_memory_gb=80)
    print(r.summary())
    print()

    # Demo: Financial leverage
    r2 = DomainPresets.financial_leverage(current_leverage=15.0)
    print(r2.summary())
    print()

    # Demo: Grid stability
    r3 = DomainPresets.grid_stability(current_load_gw=110.0, capacity_gw=120.0)
    print(r3.summary())
