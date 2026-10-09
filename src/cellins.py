"""Structured insertion: add one cube into each unoccupied 2-middle cell of a 4D packing (rotated 45deg in its middle plane).
python cellins.py in.json out.json reps jit seed [inflate]"""
import sys, json, itertools, numpy as np
D = json.load(open(sys.argv[1])); d, n = D['d'], D['n']; r = D['results'][0]
reps, jit, seed = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]); infl = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
rng = np.random.default_rng(seed)
c = np.array(r['centers']).reshape(n, d); R = np.array(r['R']).reshape(n, d, d); s = r['s']
cell = lambda x: ''.join('L' if v < 1.0 else ('H' if v > s - 1.0 else 'M') for v in x)
occ = set(cell(x) for x in c)
cells = [''.join(p) for p in itertools.product('LMH', repeat=4) if ''.join(p).count('M') == 2 and ''.join(p) not in occ]
def rot(a, b, th):
    G = np.eye(d); G[a, a] = G[b, b] = np.cos(th); G[a, b] = -np.sin(th); G[b, a] = np.sin(th); return G
res = []
S = s * infl; c2 = (c - s / 2) * infl + S / 2
for cl in cells:
    for _ in range(reps):
        mid = [k for k in range(4) if cl[k] == 'M']
        p = np.array([0.5 if x == 'L' else (S - 0.5 if x == 'H' else S / 2 + rng.choice([-1, 1]) * rng.uniform(0, 0.24)) for x in cl])
        Rm = rot(mid[0], mid[1], np.pi / 4 + rng.normal() * 0.05)
        C = np.vstack([c2 + rng.normal(size=c2.shape) * jit, p]); Rs = np.concatenate([R, Rm[None]], 0)
        m = n + 1
        U = [(C[j] - C[i]) / (np.linalg.norm(C[j] - C[i]) + 1e-12) for i, j in itertools.combinations(range(m), 2)]
        res.append(dict(s=S, centers=C.ravel().tolist(), R=Rs.ravel().tolist(), u=np.array(U).ravel().tolist()))
print(len(cells), 'empty 2-middle cells:', cells)
json.dump(dict(d=4, n=n + 1, results=res), open(sys.argv[2], 'w'), separators=(',', ':'))
