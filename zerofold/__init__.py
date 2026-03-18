"""
ZeroFold — Role-Aware Compute Substrate
========================================
Core modules:
  pca     — Role-aware PCA and SVD acceleration (drop-in for numpy/sklearn)
  router  — Phase-aware routing: classify matrices by role, route to minimum-energy compute path
  collapse — Governing Dynamics: predict system collapse from growth curves
  zsse    — Zero Substrate Signature Engine: spectral signatures, semiprime factorization
"""
from .pca import pca, svd, PCAResult, SVDResult, substrate_stats, clear_substrate, ZeroSubstrate
from .router import ZeroFoldRouter, classify_matrix, bench
from .collapse import CollapseDetector, CollapseResult
from .zsse import SubstrateSignatureEngine

__version__ = "0.1.0"
__all__ = [
    "pca", "svd", "PCAResult", "SVDResult", "substrate_stats", "clear_substrate", "ZeroSubstrate",
    "ZeroFoldRouter", "classify_matrix", "bench",
    "CollapseDetector", "CollapseResult",
    "SubstrateSignatureEngine",
]
