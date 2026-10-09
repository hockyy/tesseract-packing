# Tesseracts in a Box

Packings of n unit 4-cubes (tesseracts) inside the smallest possible 4-cube, with rotations allowed.

Code and data for the findings below. Live page with an in-browser search you can rerun: https://packing.mikira.id

## Results

| n | side found | previous construction | file |
|---|---|---|---|
| 26 | 2.942809042 (≈ 2 + 2√2/3) | 2.956: 3D 13-cube record (Friedman 1998) stacked in 2 layers | `results/n26_side_2.942809042.json` |
| 27 | 2.942809214 (≈ 2 + 2√2/3) | 2.98995: 3D 14-cube record (Friedman 1998) stacked in 2 layers | `results/n27_side_2.942809214.json` |

These are **candidates**. They pass a double-precision check: full separating-axis test, all 56 candidate axes per pair in 4D. They do not yet have an exact rational-arithmetic certificate. The "previous construction" values are our own products of published 2D/3D results, because we found no published 4D table. The closed form 2 + 2√2/3 is a conjecture based on numerical agreement.

## Files

- `src/pack.cpp` runs a multithreaded soft-to-rigid random-start search in any dimension, with per-pair free separating axes and Adam:
  `g++ -O3 -march=native -pthread src/pack.cpp -o pack`, then `pack d n starts soft_steps seed out.json [threads]`.
- `src/polish.cpp` is an augmented-Lagrangian + L-BFGS polish (minimise the box side subject to pair, wall and orthogonality constraints).
  Run `polish in.json out.json [topk]`. Set env `BH=<seconds>` for multithreaded basin hopping.
- `src/prodseeds.py` writes 4D seeds from the 5×5 product construction, spread to side s0, plus n − 25 random cubes.
- `src/verify.py` is an independent NumPy checker. Run `python src/verify.py results/n27_side_2.942809214.json`.
- `src/gpu.cu` is an experimental CUDA port of the search. It is no faster than the CPU version.
- `src/core.js` is a JavaScript port of the polish and basin hopping. It powers the interactive search on the live page.

## JSON format

`{"d":4,"n":..,"results":[{"s":side,"centers":[n*4],"R":[n*16 row-major; columns are edge directions],"u":[pair separating axes]}]}`.
Cube i has vertices `c_i + R_i · (±½, ±½, ±½, ±½)`, and the box is `[0, s]^4`.

## Sources for the baselines

- Erich Friedman, Cubes in Cubes: https://erich-friedman.github.io/packing/cubincub/
- Erich Friedman, Packing Unit Squares in Squares (EJC survey DS7): https://www.combinatorics.org/ojs/index.php/eljc/article/view/DS7
- Yohei Nakajima, soft-to-rigid packing: https://github.com/yoheinakajima/soft-to-rigid-packing
- Tencent Hyra results: https://github.com/Tencent-Hunyuan/hyra-results
- Berthold et al., Out-of-the-Box Global Optimization for Packing Problems: https://arxiv.org/abs/2605.04850
