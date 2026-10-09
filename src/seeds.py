"""Template seeds: 2^d axis-aligned corner cubes + tilted cubes in chosen cells, written for polish.exe.
python seeds.py d n count out.json s0 tmin tmax [ones]"""
import sys, json, itertools, numpy as np
d, n, cnt, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
s0, tmin, tmax = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
want = int(sys.argv[8]) if len(sys.argv) > 8 else d - 1
rng = np.random.default_rng(int(sys.argv[9]) if len(sys.argv) > 9 else 0)
cells = list(itertools.product(range(3), repeat=d))
corners = [c for c in cells if 1 not in c]
chan = [c for c in cells if c.count(1) == want]
res = []
for _ in range(cnt):
    pick = [chan[i] for i in rng.choice(len(chan), n - len(corners), replace=False)]
    C, R = [], []
    g = (s0 - 2) / 2
    for cell in corners + pick:
        pos = np.array([0.5 if x == 0 else (s0 - 0.5 if x == 2 else s0 / 2) for x in cell])
        Rm = np.eye(d)
        mid = [k for k in range(d) if cell[k] == 1]
        if mid:
            a, b = rng.choice(mid, 2, replace=False) if len(mid) >= 2 else rng.choice(d, 2, replace=False)
            th = np.radians(rng.uniform(tmin, tmax)) * rng.choice([-1, 1])
            G = np.eye(d); G[a, a] = G[b, b] = np.cos(th); G[a, b] = -np.sin(th); G[b, a] = np.sin(th)
            Rm = G @ Rm
            # push inward along non-middle axes slightly (towards centre)
            for k in range(d):
                if cell[k] != 1: pos[k] += (0.15 if cell[k] == 0 else -0.15) * rng.uniform(0, 1)
        Rm = Rm + 0.01 * rng.normal(size=(d, d)); Rm = np.linalg.qr(Rm)[0]
        C.append(pos + 0.01 * rng.normal(size=d)); R.append(Rm)
    C = np.array(C); U = []
    for i, j in itertools.combinations(range(n), 2):
        w = C[j] - C[i]; U.append(w / np.linalg.norm(w))
    res.append(dict(s=s0, centers=C.ravel().tolist(), R=np.array(R).ravel().tolist(), u=np.array(U).ravel().tolist()))
json.dump(dict(d=d, n=n, results=res), open(out, 'w'), separators=(',', ':'))
