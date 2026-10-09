"""Structured 4D seeds: 16 corner cubes + 8 'same-group' 2-middle cubes (MM** and **MM, rotated 45deg in their
middle plane) + centre MMMM (45deg in (0,1) and (2,3)) = 25 (the 2Dx2D product), then n-25 extra cubes in
'cross' 2-middle cells (middle coords {a,b}, a in {0,1}, b in {2,3}), rotated ~45deg in their middle plane.
python tseeds.py n count out.json s0 jit seed [drop]   (drop: remove this many random product cubes first)"""
import sys, json, itertools, numpy as np
n, cnt, out, s0, jit, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
drop = int(sys.argv[7]) if len(sys.argv) > 7 else 0
planes = [tuple(int(ch) for ch in p) for p in sys.argv[8].split(',')] if len(sys.argv) > 8 else None
rng = np.random.default_rng(seed)
d = 4
def rot(a, b, th):
    G = np.eye(d); G[a, a] = G[b, b] = np.cos(th); G[a, b] = -np.sin(th); G[b, a] = np.sin(th); return G
def pos(cell, s):
    return np.array([0.5 if x == 'L' else (s - 0.5 if x == 'H' else s / 2) for x in cell])
corners = [''.join(p) for p in itertools.product('LH', repeat=4)]
same = [m + ''.join(p) for m in ['MM'] for p in itertools.product('LH', repeat=2)] + [''.join(p) + 'MM' for p in itertools.product('LH', repeat=2)]
cross = []
for a in (0, 1):
    for b in (2, 3):
        if planes and (a, b) not in planes and (b, a) not in planes: continue
        rest = [k for k in range(4) if k not in (a, b)]
        for p in itertools.product('LH', repeat=2):
            c = ['?'] * 4; c[a] = 'M'; c[b] = 'M'; c[rest[0]] = p[0]; c[rest[1]] = p[1]; cross.append(''.join(c))
res = []
for _ in range(cnt):
    s = s0
    base = corners + same + ['MMMM']
    if drop: base = [base[i] for i in sorted(rng.choice(len(base), len(base) - drop, replace=False))]
    k = n - len(base)
    extra = [cross[i] for i in rng.choice(len(cross), k, replace=False)] if k <= len(cross) else cross + [None] * (k - len(cross))
    C, Rs = [], []
    for cell in base + extra:
        if cell is None:
            C.append(rng.uniform(0.5, s - 0.5, 4)); Rs.append(np.linalg.qr(rng.normal(size=(4, 4)))[0]); continue
        p = pos(cell, s); mid = [k2 for k2 in range(4) if cell[k2] == 'M']
        Rm = np.eye(4)
        if len(mid) == 2: Rm = rot(mid[0], mid[1], np.pi / 4 * rng.choice([-1, 1]) + rng.normal() * 0.05)
        if len(mid) == 4: Rm = rot(0, 1, np.pi / 4) @ rot(2, 3, np.pi / 4)
        A = rng.normal(size=(4, 4)) * jit; Q = np.linalg.qr(np.eye(4) + A - A.T)[0]
        Q = Q * np.sign(np.diag(np.linalg.qr(np.eye(4) + A - A.T)[1]))
        C.append(p + rng.normal(size=4) * jit); Rs.append(Rm @ Q)
    C = np.array(C)
    U = [(C[j] - C[i]) / (np.linalg.norm(C[j] - C[i]) + 1e-12) for i, j in itertools.combinations(range(n), 2)]
    res.append(dict(s=s, centers=C.ravel().tolist(), R=np.array(Rs).ravel().tolist(), u=np.array(U).ravel().tolist()))
json.dump(dict(d=4, n=n, results=res), open(out, 'w'), separators=(',', ':'))
