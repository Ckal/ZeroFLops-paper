"""Leak check (vectorized): holdout board states must be disjoint from the
training corpus AND its mirror half.

Method: headers are exactly 9 raw bytes at every 10-byte stride. We extract
all 9-byte header rows from each file with numpy strided views, hash them as
structured records, and intersect.
"""
import time, sys
import numpy as np

t0 = time.time()
HOLDOUT = r"runs/_probe/disjoint_holdout.corpus.txt"
TRAIN = r"runs/_probe/perply_v23_full/corpus_320k_fixed_augmented.txt"

def headers(path, max_bytes=200_000_000):
    data = np.fromfile(path, dtype=np.uint8, count=max_bytes)
    games = []
    n = 0
    for start, end in [(0, len(data))]:
        pass
    # split on newline bytes
    nl = np.flatnonzero(data == 0x0A)
    # bounds between newlines
    bounds = np.concatenate([[0], nl + 1, [len(data) if nl.size else len(data)]])
    # bounds: game i spans [bounds[i], bounds[i+1]) (last element empty if file ends with newline)
    rows = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i+1]
        g = data[s:e]
        L = len(g) - (len(g) % 10)
        if L < 9:
            continue
        # header rows at every stride 10
        nrec = L // 10
        # build 2D array (nrec, 9) without big copies: use reshape trick
        # first 10*nrec bytes
        chunk = g[:nrec * 10].reshape(nrec, 10)[:, :9]
        # structured hash
        view = np.ascontiguousarray(chunk).view(
            np.dtype((np.void, 9)))
        rows.append((i, len(view), view))
    return rows

def union_hash(rows):
    big = np.concatenate([r[2] for r in rows]) if rows else np.array([], dtype=np.void)
    return np.unique(big)

h_rows = headers(HOLDOUT)
h_union = union_hash(h_rows)
n_h = h_union.size
print(f"holdout: distinct header states = {n_h:,}", flush=True)

t_rows = headers(TRAIN)
t_union = union_hash(t_rows)
n_t = t_union.size
print(f"train (orig+mirror) distinct header states = {n_t:,}", flush=True)

inter = np.intersect1d(h_union, t_union)
print(f"INTERSECTION = {inter.size}")
if inter.size:
    for x in inter[:10]:
        print("  sample leaked header:", bytes(np.frombuffer(x, dtype=np.uint8)))
print("RESULT:", "PASS - disjoint (no leakage)" if inter.size == 0 else "FAIL - LEAKAGE FOUND")
print(f"({time.time()-t0:.0f}s)")
