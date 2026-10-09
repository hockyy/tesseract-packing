"""Seeds from the 2D(5) x 2D(5) product packing in 4D: choose n of the 25 cubes, jitter, write for polish.
python prodseeds.py n count out.json jit seed"""
import sys, json, itertools, numpy as np
n, cnt, out, jit, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
s0 = float(sys.argv[6]) if len(sys.argv) > 6 else None
rng = np.random.default_rng(seed)
s = 2 + 1 / np.sqrt(2)
two = [((0.5, 0.5), 0), ((s - .5, .5), 0), ((.5, s - .5), 0), ((s - .5, s - .5), 0), ((s / 2, s / 2), np.pi / 4)]
def r2(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
full = []
for (pa, ta), (pb, tb) in itertools.product(two, two):
    R = np.zeros((4, 4)); R[:2, :2] = r2(ta); R[2:, 2:] = r2(tb)
    full.append((np.array(pa + pb), R))
res = []
for _ in range(cnt):
    idx = rng.choice(25, min(n, 25), replace=False)
    C = []; Rs = []
    for i in idx:
        c, R = full[i]
        A = rng.normal(size=(4, 4)) * jit; A = A - A.T
        w, V = np.linalg.eigh(A @ A.T)  # cheap: use QR of (I + A) as a small rotation
        Q = np.linalg.qr(np.eye(4) + A)[0]; Q = Q * np.sign(np.diag(np.linalg.qr(np.eye(4) + A)[1]))
        C.append(c + rng.normal(size=4) * jit); Rs.append(R @ Q)
    S = s0 or s
    C = [(c - s / 2) * (S / s) + S / 2 for c in C]
    for _ in range(n - len(C)):   # extra cubes: random spot, random orientation
        C.append(rng.uniform(0.5, S - 0.5, 4)); Rs.append(np.linalg.qr(rng.normal(size=(4, 4)))[0])
    C = np.array(C)
    U = [(C[j] - C[i]) / np.linalg.norm(C[j] - C[i]) for i, j in itertools.combinations(range(n), 2)]
    res.append(dict(s=S, centers=C.ravel().tolist(), R=np.array(Rs).ravel().tolist(), u=np.array(U).ravel().tolist()))
json.dump(dict(d=4, n=n, results=res), open(out, 'w'), separators=(',', ':'))
