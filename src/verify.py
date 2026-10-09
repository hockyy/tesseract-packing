"""Independent check (numpy only): full separating-axis test over all candidate axes, any d.
python verify.py polished.json"""
import json, sys, itertools
import numpy as np


def check(d, n, c, R, s):
    c = np.asarray(c, float).reshape(n, d)
    R = np.asarray(R, float).reshape(n, d, d)
    orth = max(np.abs(R[i].T @ R[i] - np.eye(d)).max() for i in range(n))
    combos = list(itertools.combinations(range(2 * d), d - 1))
    worst = np.inf
    for i, j in itertools.combinations(range(n), 2):
        E = np.concatenate([R[i].T, R[j].T], 0)  # rows = edge directions
        best = -np.inf
        for cb in combos:
            V = E[list(cb)]
            sv = np.linalg.svd(V, compute_uv=False)
            if sv.min() < 1e-9:
                continue
            u = np.linalg.svd(V)[2][-1]
            h = 0.5 * np.abs(R[i].T @ u).sum() + 0.5 * np.abs(R[j].T @ u).sum()
            best = max(best, abs(u @ (c[j] - c[i])) - h)
        worst = min(worst, best)
    hw = 0.5 * np.abs(R).sum(2)
    wall = min((c - hw).min(), (s - c - hw).min())
    return worst, wall, orth


if __name__ == '__main__':
    data = json.load(open(sys.argv[1]))
    d, n = data['d'], data['n']
    for r in data['results']:
        clr, wall, orth = check(d, n, r['centers'], r['R'], r['s'])
        ok = clr >= 0 and wall >= -1e-12 and orth < 1e-12
        print(f"s={r['s']:.12f}  min pair gap={clr:.2e}  wall slack={wall:.2e}  orthogonality err={orth:.1e}  {'OK' if ok else 'FAIL'}")
