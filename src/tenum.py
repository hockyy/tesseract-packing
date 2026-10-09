"""Enumerate symmetry-distinct choices of k = n-25 'cross' 2-middle cells on top of the 25-cube 2Dx2D product frame
(16 corners + MM** + **MM + MMMM), one seed per orbit (x reps jitters).
Symmetry group (128): L<->H flips per coordinate, swap(0,1), swap(2,3), swap of the two factors.
python tenum.py n out.json s0 jit reps seed [maxorbits]"""
import sys, json, itertools, numpy as np
n, out, s0, jit, reps, seed = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
maxorb = int(sys.argv[7]) if len(sys.argv) > 7 else 10**9
rng = np.random.default_rng(seed); d = 4; k = n - 25
cross = []
for a in (0, 1):
    for b in (2, 3):
        rest = [q for q in range(4) if q not in (a, b)]
        for p in itertools.product('LH', repeat=2):
            c = ['?'] * 4; c[a] = c[b] = 'M'; c[rest[0]], c[rest[1]] = p; cross.append(''.join(c))
perms = []
for sw01 in (0, 1):
    for sw23 in (0, 1):
        for swf in (0, 1):
            p = [0, 1, 2, 3]
            if sw01: p[0], p[1] = p[1], p[0]
            if sw23: p[2], p[3] = p[3], p[2]
            if swf: p = [p[2], p[3], p[0], p[1]]
            perms.append(p)
G = [(p, f) for p in perms for f in itertools.product((0, 1), repeat=4)]
def act(cell, g):
    p, f = g; out = [None] * 4
    for q in range(4):
        ch = cell[p[q]]
        if f[q] and ch != 'M': ch = 'H' if ch == 'L' else 'L'
        out[q] = ch
    return ''.join(out)
seen = set(); orbits = []
for combo in itertools.combinations(cross, k):
    key = min(tuple(sorted(act(c, g) for c in combo)) for g in G)
    if key not in seen: seen.add(key); orbits.append(key)
print(len(orbits), 'orbits for k =', k)
if len(orbits) > maxorb: orbits = [orbits[i] for i in rng.choice(len(orbits), maxorb, replace=False)]
def rot(a, b, th):
    M = np.eye(d); M[a, a] = M[b, b] = np.cos(th); M[a, b] = -np.sin(th); M[b, a] = np.sin(th); return M
corners = [''.join(p) for p in itertools.product('LH', repeat=4)]
same = ['MM' + ''.join(p) for p in itertools.product('LH', repeat=2)] + [''.join(p) + 'MM' for p in itertools.product('LH', repeat=2)]
res = []; meta = []
for orb in orbits:
    for _ in range(reps):
        s = s0; C, Rs = [], []
        for cell in corners + same + ['MMMM'] + list(orb):
            p = np.array([0.5 if x == 'L' else (s - 0.5 if x == 'H' else s / 2) for x in cell])
            mid = [q for q in range(4) if cell[q] == 'M']
            if len(mid) == 2: Rm = rot(mid[0], mid[1], np.pi / 4 + rng.normal() * 0.05)
            elif len(mid) == 4: Rm = rot(0, 1, np.pi / 4) @ rot(2, 3, np.pi / 4)
            else: Rm = np.eye(4)
            A = rng.normal(size=(4, 4)) * jit; Q, Rq = np.linalg.qr(np.eye(4) + A - A.T); Q = Q * np.sign(np.diag(Rq))
            C.append(p + rng.normal(size=4) * jit); Rs.append(Rm @ Q)
        C = np.array(C)
        U = [(C[j] - C[i]) / (np.linalg.norm(C[j] - C[i]) + 1e-12) for i, j in itertools.combinations(range(n), 2)]
        res.append(dict(s=s, centers=C.ravel().tolist(), R=np.array(Rs).ravel().tolist(), u=np.array(U).ravel().tolist()))
        meta.append(orb)
json.dump(dict(d=4, n=n, results=res), open(out, 'w'), separators=(',', ':'))
json.dump(meta, open(out + '.meta', 'w'))
