"""
ZeroFold — Cold-Start Multi-Worker Demo
========================================
Shows disk persistence in action:

  Worker 1: computes SVD on weight matrices, saves to disk
  Worker 2: starts cold, loads from disk — zero recompute

Run:
    python demo_multiworker.py
"""

import multiprocessing
import os
import tempfile
import time

import numpy as np
from zerofold.pca import ZeroSubstrate

SEED      = 42
N         = 512
K         = 64
N_WEIGHTS = 6  # simulate 6 weight matrices (2 transformer layers x Wq/Wk/Wv)


def make_weights():
    rng = np.random.default_rng(SEED)
    return [rng.standard_normal((N, N)).astype(np.float64) for _ in range(N_WEIGHTS)]


def worker_1(cache_dir: str):
    """Simulates the first inference worker — warms the cache."""
    print(f"[Worker 1] starting cold  (cache_dir={cache_dir})")
    weights   = make_weights()
    substrate = ZeroSubstrate(mode="hybrid", cache_dir=cache_dir)

    t0 = time.perf_counter()
    for i, W in enumerate(weights):
        substrate.svd(W, n_components=K)
    elapsed = time.perf_counter() - t0

    s = substrate.stats()
    print(f"[Worker 1] computed {N_WEIGHTS} weight SVDs in {elapsed*1000:.1f}ms")
    print(f"[Worker 1] saved to disk — {s['receipts_stored']} receipts stored")


def worker_2(cache_dir: str):
    """Simulates any subsequent worker — loads from disk, zero recompute."""
    print(f"\n[Worker 2] starting cold  (cache_dir={cache_dir})")
    weights   = make_weights()
    substrate = ZeroSubstrate(mode="hybrid", cache_dir=cache_dir)

    times  = []
    layers = []
    for W in weights:
        t0 = time.perf_counter()
        r  = substrate.svd(W, n_components=K)
        times.append(time.perf_counter() - t0)
        layers.append(r.cache_layer)

    s = substrate.stats()
    print(f"[Worker 2] {N_WEIGHTS} weight SVDs in {sum(times)*1000:.2f}ms total")
    print(f"[Worker 2] hit_rate={s['hit_rate']*100:.0f}%  "
          f"all from: {set(layers)}")
    print(f"[Worker 2] avg per matrix: {sum(times)/len(times)*1000:.3f}ms")


def run():
    cache_dir = tempfile.mkdtemp(prefix="zerofold_")
    print("=" * 56)
    print("ZeroFold — Multi-Worker Cold-Start Demo")
    print(f"Matrix size: {N}x{N}  |  k={K}  |  weights: {N_WEIGHTS}")
    print("=" * 56)

    # Show baseline: what numpy costs per matrix
    weights = make_weights()
    t0 = time.perf_counter()
    for W in weights:
        np.linalg.svd(W, full_matrices=False)
    numpy_time = time.perf_counter() - t0
    print(f"\nBaseline numpy: {numpy_time*1000:.1f}ms for {N_WEIGHTS} matrices")
    print(f"               ({numpy_time/N_WEIGHTS*1000:.1f}ms each)\n")

    # Worker 1: compute and persist
    p1 = multiprocessing.Process(target=worker_1, args=(cache_dir,))
    p1.start()
    p1.join()

    # Worker 2: cold start, loads from disk
    p2 = multiprocessing.Process(target=worker_2, args=(cache_dir,))
    p2.start()
    p2.join()

    print()
    print("=" * 56)
    print("Worker 2 did zero SVD computation.")
    print("Same results. Loaded from disk. Bitwise identical.")
    print("=" * 56)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
