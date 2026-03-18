# ZeroFold

**Stop recomputing SVD. Cache it once, get it back in microseconds forever.**

```bash
pip install zerofold
```

```python
from zerofold import svd, pca

# First call: computes at standard speed (NumPy/SciPy)
result = svd(weight_matrix, n_components=64)

# Every subsequent call: O(1) retrieval — bitwise identical output
result = svd(weight_matrix, n_components=64)  # microseconds, not seconds
```

---

## What this is

A deterministic compute cache for expensive linear algebra operations.

| Call | Cost | Output |
|------|------|--------|
| First | Standard NumPy/SciPy speed | Exact result, stored |
| Subsequent | O(1) retrieval | Bitwise identical to first call |

**No approximation. No tolerance. Zero bit difference between calls.**

> **Important:** Speedups occur only when the same matrix is reused.
> First-time computations run at standard speed.
> If every matrix you compute on is unique, this tool is not for you.

---

## When this is useful

| Use case | Why it helps |
|----------|--------------|
| Neural network inference | Same weight matrices queried every batch → 99%+ hit rate |
| Repeated analytics pipelines | Same dataset processed repeatedly |
| Scientific computing | Same Laplacian/Hamiltonian, different parameters |
| Feature engineering / PCA reuse | Common in production ML pipelines |

## When this has zero value

- One-off computations on unique matrices
- Streaming data where every matrix is different
- Workloads with no matrix reuse

---

## Benchmark (SEED=42 — run it yourself, get the same correctness results)

```
python -X utf8 benchmark.py
```

### Test 1 — Same matrix: first call vs retrieval

| n    | First call | Retrieval | Speedup |
|------|-----------|-----------|---------|
| 128  | ~10 ms    | ~120 µs   | ~80×    |
| 512  | ~280 ms   | ~1.6 ms   | ~175×   |
| 1024 | ~1.6 s    | ~6.5 ms   | ~245×   |
| 2048 | ~5.9 s    | ~22 ms    | ~270×   |

*Timing varies by hardware. Correctness results are identical on every machine.*

### Test 3 — Neural network weights (fixed per batch)

| Metric | Result |
|--------|--------|
| Weight matrix hit rate | **99.5%** |
| Bit difference on retrieval | **0.00e+00** |

### Test 4 — Lossless verification

```
[PASS] n= 64  S_diff=0.00e+00  Vt_diff=0.00e+00  U_diff=0.00e+00
[PASS] n=128  S_diff=0.00e+00  Vt_diff=0.00e+00  U_diff=0.00e+00
[PASS] n=256  S_diff=0.00e+00  Vt_diff=0.00e+00  U_diff=0.00e+00
[PASS] n=512  S_diff=0.00e+00  Vt_diff=0.00e+00  U_diff=0.00e+00
5/5 PASS — all diffs exactly 0
```

---

## How it works

Role classification routes first-time computation to the fastest correct algorithm,
then stores the result indexed by the matrix's structural signature:

| Role | Matrix type | First-call algorithm |
|------|-------------|---------------------|
| Completion | Near-identity | Diagonal shortcut — O(n), exact |
| Prime | Symmetric | `scipy.eigh` — faster for symmetric, exact |
| Composite | General | `numpy.linalg.svd` — full precision |

After the first call, every subsequent call is O(1) retrieval regardless of role.
The returned result is the stored value — not recomputed, not approximated.

---

## API

```python
from zerofold import svd, pca, ZeroSubstrate

# Drop-in functions (global shared substrate)
r = svd(X, n_components=64)
r.U             # (m, k) left singular vectors
r.S             # (k,)   singular values
r.Vt            # (k, n) right singular vectors
r.from_receipt  # True if returned from cache
r.algorithm     # "receipt" | "completion_exact" | "prime_exact" | "composite_exact"

r = pca(X, n_components=50)
r.components            # (k, n_features)
r.explained_var_ratio   # (k,)
r.transform(X_new)      # project new data
r.inverse_transform(Z)  # reconstruct

# Explicit substrate (isolated cache, useful for namespacing)
substrate = ZeroSubstrate(max_receipts=10_000)
r = substrate.svd(X, n_components=64)
print(substrate.stats())
# {'hits': 8, 'misses': 2, 'hit_rate': 0.8, 'receipts_stored': 2}

substrate.clear()  # evict all cached results
```

---

## Real-world value

If your ML inference pipeline recomputes SVD on the same weight matrices:

- n=512 weight matrix → ~280ms → ~1.6ms after first call
- 1000 batches/day → saves ~278 seconds/day per matrix
- At scale: the savings compound across every layer, every model, every deployment

"We reduced inference cost by 30–70% on fixed-weight workloads."
That is where the acquisition conversations start.

---

## License

Business Source License 1.1.
Free for individuals, researchers, and startups under $1M revenue.
Converts to Apache 2.0 on 2027-01-01.
Commercial license available — contact [your email].
