/* Search core: product-construction seeds, augmented-Lagrangian + L-BFGS polish, basin-hopping moves.
   Mirrors polish.cpp. Works in the browser (window.Core) and in Node (module.exports). */
(function (root) {
const d = 4;
let n, NP, PI, PJ, nd, ndd, NV, NW, NO;
function setup(nn) {
  n = nn; NP = n * (n - 1) / 2; PI = []; PJ = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { PI.push(i); PJ.push(j); }
  nd = n * d; ndd = n * d * d; NV = nd + ndd + NP * d + 1; NW = n * d * 2; NO = n * d * (d + 1) / 2;
}
/* seeded RNG (mulberry32) */
let rs = 1;
function setSeed(k) { rs = (k >>> 0) ^ 0x6d2b79f5; for (let i = 0; i < 4; i++) rnd(); }
function rnd() { rs = (rs + 0x6d2b79f5) >>> 0; let t = rs; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
function gauss() { const u = rnd() || 1e-12, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function gs(R, off) {
  for (let m = 0; m < d; m++) {
    for (let p = 0; p < m; p++) { let dot = 0; for (let k = 0; k < d; k++) dot += R[off + k*d + m] * R[off + k*d + p]; for (let k = 0; k < d; k++) R[off + k*d + m] -= dot * R[off + k*d + p]; }
    let nr = 0; for (let k = 0; k < d; k++) nr += R[off + k*d + m] ** 2; nr = Math.sqrt(nr); for (let k = 0; k < d; k++) R[off + k*d + m] /= nr;
  }
}
const sgn = x => x > 0 ? 1 : x < 0 ? -1 : 0;
function clone(P) { return { n: P.n, c: P.c.slice(), R: P.R.slice(), u: P.u.slice(), s: P.s }; }
function axesFromCentres(P, only) {
  for (let p = 0; p < NP; p++) {
    if (only && !only.has(PI[p]) && !only.has(PJ[p])) continue;
    let nr = 0; for (let k = 0; k < d; k++) { const w = P.c[PJ[p]*d+k] - P.c[PI[p]*d+k]; P.u[p*d+k] = w; nr += w*w; }
    nr = Math.sqrt(nr) + 1e-12; for (let k = 0; k < d; k++) P.u[p*d+k] /= nr;
  }
}
function fromData(D) {
  setup(D.n);
  const P = { n: D.n, c: Float64Array.from(D.centers), R: Float64Array.from(D.R), u: new Float64Array(NP*d), s: D.s };
  axesFromCentres(P); return P;
}
/* the 5x5 product construction (25 cubes) spread to side s0, plus n-25 random cubes */
function productSeed(nn, s0, jit) {
  setup(nn);
  const s = 2 + 1 / Math.SQRT2;
  const two = [[.5,.5,0],[s-.5,.5,0],[.5,s-.5,0],[s-.5,s-.5,0],[s/2,s/2,Math.PI/4]];
  const P = { n, c: new Float64Array(n*d), R: new Float64Array(n*d*d), u: new Float64Array(NP*d), s: s0 };
  let i = 0;
  for (const a of two) for (const b of two) {
    if (i >= n) break;
    const pos = [a[0], a[1], b[0], b[1]];
    for (let k = 0; k < d; k++) P.c[i*d+k] = (pos[k] - s/2) * (s0/s) + s0/2 + jit * gauss();
    const off = i*16, ca = Math.cos(a[2]), sa = Math.sin(a[2]), cb = Math.cos(b[2]), sb = Math.sin(b[2]);
    const M = [ca,-sa,0,0, sa,ca,0,0, 0,0,cb,-sb, 0,0,sb,cb];
    for (let q = 0; q < 16; q++) P.R[off+q] = M[q] + jit * gauss();
    gs(P.R, off); i++;
  }
  for (; i < n; i++) {
    for (let k = 0; k < d; k++) P.c[i*d+k] = 0.5 + (s0 - 1) * rnd();
    for (let q = 0; q < 16; q++) P.R[i*16+q] = gauss();
    gs(P.R, i*16);
  }
  axesFromCentres(P); return P;
}

const MARGIN = 1e-12;
function makeProb(P) {
  const sg = new Float64Array(NP);
  for (let p = 0; p < NP; p++) { let uw = 0; for (let k = 0; k < d; k++) uw += P.u[p*d+k] * (P.c[PJ[p]*d+k] - P.c[PI[p]*d+k]); sg[p] = uw >= 0 ? 1 : -1; }
  const x = new Float64Array(NV);
  x.set(P.c, 0); x.set(P.R, nd); x.set(P.u, nd + ndd); x[NV-1] = P.s;
  return { sg, x, lp: new Float64Array(NP), lw: new Float64Array(NW), lo: new Float64Array(NO), mu: 10 };
}
function evalL(pb, x, g, cons) {
  const { sg, lp, lw, lo, mu } = pb;
  let L = x[NV-1];
  if (g) { g.fill(0); g[NV-1] = 1; }
  const ai = [0,0,0,0], aj = [0,0,0,0], uh = [0,0,0,0], gh = [0,0,0,0];
  for (let p = 0; p < NP; p++) {
    const i = PI[p], j = PJ[p], uo = nd + ndd + p*d, Ri = nd + i*16, Rj = nd + j*16;
    let nr = 0; for (let k = 0; k < d; k++) nr += x[uo+k] ** 2; nr = Math.sqrt(nr);
    for (let k = 0; k < d; k++) uh[k] = x[uo+k] / nr;
    let hs = 0, uw = 0;
    for (let m = 0; m < d; m++) { let a = 0, b = 0; for (let k = 0; k < d; k++) { a += x[Ri+k*d+m] * uh[k]; b += x[Rj+k*d+m] * uh[k]; } ai[m] = a; aj[m] = b; hs += 0.5 * (Math.abs(a) + Math.abs(b)); }
    for (let k = 0; k < d; k++) uw += uh[k] * (x[j*d+k] - x[i*d+k]);
    const gv = sg[p] * uw - hs - MARGIN;
    if (cons) cons.gp[p] = gv;
    const w = -Math.max(0, lp[p] - mu * gv);
    L += (w*w - lp[p]*lp[p]) / (2*mu);
    if (g && w !== 0) {
      for (let k = 0; k < d; k++) { let v = sg[p] * (x[j*d+k] - x[i*d+k]); for (let m = 0; m < d; m++) v -= 0.5 * (sgn(ai[m]) * x[Ri+k*d+m] + sgn(aj[m]) * x[Rj+k*d+m]); gh[k] = v; }
      let dot = 0; for (let k = 0; k < d; k++) dot += gh[k] * uh[k];
      for (let k = 0; k < d; k++) {
        g[uo+k] += w * (gh[k] - dot * uh[k]) / nr;
        g[j*d+k] += w * sg[p] * uh[k]; g[i*d+k] -= w * sg[p] * uh[k];
        for (let m = 0; m < d; m++) { g[Ri+k*d+m] += w * (-0.5 * sgn(ai[m]) * uh[k]); g[Rj+k*d+m] += w * (-0.5 * sgn(aj[m]) * uh[k]); }
      }
    }
  }
  const s = x[NV-1];
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) {
    const Ro = nd + i*16 + k*d; let l1 = 0; for (let m = 0; m < d; m++) l1 += Math.abs(x[Ro+m]);
    const hw = 0.5 * l1, xx = x[i*d+k];
    for (let side = 0; side < 2; side++) {
      const id = (i*d+k)*2 + side, gv = (side === 0 ? xx - hw : s - xx - hw) - MARGIN;
      if (cons) cons.gw[id] = gv;
      const w = -Math.max(0, lw[id] - mu * gv); L += (w*w - lw[id]*lw[id]) / (2*mu);
      if (g && w !== 0) { g[i*d+k] += w * (side === 0 ? 1 : -1); if (side === 1) g[NV-1] += w; for (let m = 0; m < d; m++) g[Ro+m] += w * (-0.5 * sgn(x[Ro+m])); }
    }
  }
  let oid = 0;
  for (let i = 0; i < n; i++) { const Ro = nd + i*16;
    for (let m = 0; m < d; m++) for (let m2 = m; m2 < d; m2++, oid++) {
      let e = m === m2 ? -1 : 0; for (let k = 0; k < d; k++) e += x[Ro+k*d+m] * x[Ro+k*d+m2];
      if (cons) cons.eo[oid] = e;
      const w = lo[oid] + mu * e; L += lo[oid] * e + 0.5 * mu * e * e;
      if (g) for (let k = 0; k < d; k++) {
        if (m === m2) g[Ro+k*d+m] += 2 * w * x[Ro+k*d+m];
        else { g[Ro+k*d+m] += w * x[Ro+k*d+m2]; g[Ro+k*d+m2] += w * x[Ro+k*d+m]; }
      }
    } }
  return L;
}
const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function makeLBFGS(pb) {
  const st = { g: new Float64Array(NV), gn: new Float64Array(NV), xn: new Float64Array(NV), dir: new Float64Array(NV), S: [], Y: [], rho: [], f: 0, done: false };
  st.f = evalL(pb, pb.x, st.g); return st;
}
function lbfgsStep(pb, st) {
  const x = pb.x, g = st.g, N = NV;
  let gmax = 0; for (let i = 0; i < N; i++) gmax = Math.max(gmax, Math.abs(g[i]));
  if (gmax < 1e-11) { st.done = true; return; }
  const dir = st.dir; dir.set(g); const k = st.S.length, al = new Float64Array(k);
  for (let q = k - 1; q >= 0; q--) { al[q] = st.rho[q] * dotv(st.S[q], dir); const Yq = st.Y[q]; for (let i = 0; i < N; i++) dir[i] -= al[q] * Yq[i]; }
  if (k) { const gam = dotv(st.S[k-1], st.Y[k-1]) / dotv(st.Y[k-1], st.Y[k-1]); for (let i = 0; i < N; i++) dir[i] *= gam; }
  else { const nr = Math.sqrt(dotv(g, g)); for (let i = 0; i < N; i++) dir[i] *= 1e-3 / Math.max(nr, 1e-12); }
  for (let q = 0; q < k; q++) { const be = st.rho[q] * dotv(st.Y[q], dir), Sq = st.S[q]; for (let i = 0; i < N; i++) dir[i] += Sq[i] * (al[q] - be); }
  for (let i = 0; i < N; i++) dir[i] = -dir[i];
  let gd = dotv(g, dir);
  if (gd >= 0) { st.S = []; st.Y = []; st.rho = []; const nr = Math.sqrt(dotv(g, g)); for (let i = 0; i < N; i++) dir[i] = -g[i] * 1e-3 / Math.max(nr, 1e-12); gd = dotv(g, dir); }
  let step = 1, ok = false, fn = st.f;
  for (let ls = 0; ls < 40; ls++) {
    for (let i = 0; i < N; i++) st.xn[i] = x[i] + step * dir[i];
    fn = evalL(pb, st.xn, st.gn);
    if (fn <= st.f + 1e-4 * step * gd) { ok = true; break; }
    step *= 0.5;
  }
  if (!ok) { if (!st.S.length) st.done = true; st.S = []; st.Y = []; st.rho = []; return; }
  const sv = new Float64Array(N), yv = new Float64Array(N);
  for (let i = 0; i < N; i++) { sv[i] = st.xn[i] - x[i]; yv[i] = st.gn[i] - g[i]; }
  const sy = dotv(sv, yv);
  if (sy > 1e-14) { st.S.push(sv); st.Y.push(yv); st.rho.push(1 / sy); if (st.S.length > 25) { st.S.shift(); st.Y.shift(); st.rho.shift(); } }
  x.set(st.xn); g.set(st.gn); st.f = fn;
}
function makePolish(P, innerCap) {
  setup(P.n);
  return { n: P.n, pb: makeProb(P), outer: 0, inner: null, innerIts: 0, innerCap: innerCap || 3000, prev: 1e18, viol: 1e9, done: false, total: 0 };
}
function polishWork(job, budgetMs) {
  setup(job.n);
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const t0 = now();
  const cons = { gp: new Float64Array(NP), gw: new Float64Array(NW), eo: new Float64Array(NO) };
  while (!job.done && now() - t0 < budgetMs) {
    if (!job.inner) { job.inner = makeLBFGS(job.pb); job.innerIts = 0; }
    lbfgsStep(job.pb, job.inner); job.innerIts++; job.total++;
    if (job.inner.done || job.innerIts >= job.innerCap) {
      const pb = job.pb; evalL(pb, pb.x, null, cons);
      let viol = 0;
      for (let p = 0; p < NP; p++) { viol = Math.max(viol, -cons.gp[p]); pb.lp[p] = Math.max(0, pb.lp[p] - pb.mu * cons.gp[p]); }
      for (let a = 0; a < NW; a++) { viol = Math.max(viol, -cons.gw[a]); pb.lw[a] = Math.max(0, pb.lw[a] - pb.mu * cons.gw[a]); }
      for (let a = 0; a < NO; a++) { viol = Math.max(viol, Math.abs(cons.eo[a])); pb.lo[a] += pb.mu * cons.eo[a]; }
      job.viol = viol; job.outer++; job.inner = null;
      if (viol < 1e-12 || job.outer >= 40) job.done = true;
      else { if (viol > 0.25 * job.prev) pb.mu = Math.min(pb.mu * 5, 1e9); job.prev = viol; }
    }
  }
}
function jobToPacking(job) {
  setup(job.n); const x = job.pb.x;
  return { n: job.n, c: x.slice(0, nd), R: x.slice(nd, nd + ndd), u: x.slice(nd + ndd, nd + ndd + NP*d), s: x[NV-1] };
}
/* min gap along each pair's stored axis (a valid non-overlap certificate when >= 0) */
function clearances(P) {
  setup(P.n); let mn = 1e18;
  for (let p = 0; p < NP; p++) {
    const i = PI[p], j = PJ[p]; let nr = 0; for (let k = 0; k < d; k++) nr += P.u[p*d+k] ** 2; nr = Math.sqrt(nr) || 1;
    let hs = 0, uw = 0;
    for (let m = 0; m < d; m++) { let a = 0, b = 0; for (let k = 0; k < d; k++) { a += P.R[i*16+k*d+m] * P.u[p*d+k]; b += P.R[j*16+k*d+m] * P.u[p*d+k]; } hs += 0.5 * (Math.abs(a) + Math.abs(b)) / nr; }
    for (let k = 0; k < d; k++) uw += P.u[p*d+k] / nr * (P.c[j*d+k] - P.c[i*d+k]);
    mn = Math.min(mn, Math.abs(uw) - hs);
  }
  return mn;
}
function overlapping(P, tol) {
  setup(P.n); const hot = new Set();
  for (let p = 0; p < NP; p++) {
    const i = PI[p], j = PJ[p]; let nr = 0; for (let k = 0; k < d; k++) nr += P.u[p*d+k] ** 2; nr = Math.sqrt(nr) || 1;
    let hs = 0, uw = 0;
    for (let m = 0; m < d; m++) { let a = 0, b = 0; for (let k = 0; k < d; k++) { a += P.R[i*16+k*d+m] * P.u[p*d+k]; b += P.R[j*16+k*d+m] * P.u[p*d+k]; } hs += 0.5 * (Math.abs(a) + Math.abs(b)) / nr; }
    for (let k = 0; k < d; k++) uw += P.u[p*d+k] / nr * (P.c[j*d+k] - P.c[i*d+k]);
    if (Math.abs(uw) - hs < -tol) { hot.add(i); hot.add(j); }
  }
  return hot;
}
function finalize(P) {
  setup(P.n);
  for (let i = 0; i < n; i++) gs(P.R, i*16);
  const mn = clearances(P);
  if (mn < 0) { const lam = 1 + 2 * (-mn) + 1e-13; for (let q = 0; q < nd; q++) P.c[q] *= lam; }
  let sn = 0;
  for (let k = 0; k < d; k++) {
    let lo = 1e18, hi = -1e18;
    for (let i = 0; i < n; i++) { let l1 = 0; for (let m = 0; m < d; m++) l1 += Math.abs(P.R[i*16+k*d+m]); lo = Math.min(lo, P.c[i*d+k] - 0.5*l1); hi = Math.max(hi, P.c[i*d+k] + 0.5*l1); }
    for (let i = 0; i < n; i++) P.c[i*d+k] -= lo; sn = Math.max(sn, hi - lo);
  }
  P.s = sn; return P;
}
function perturb(P0, maxDeg) {
  setup(P0.n);
  const P = clone(P0), moved = new Set(), kind = Math.floor(rnd() * 3);
  if (kind < 2) {
    for (let q = 0; q < (kind === 0 ? 1 : 2); q++) {
      const i = Math.floor(rnd() * n); moved.add(i);
      const a = Math.floor(rnd() * d); let b = Math.floor(rnd() * (d - 1)); if (b >= a) b++;
      const th = (5 + (maxDeg - 5) * rnd()) * Math.PI / 180 * (rnd() < .5 ? -1 : 1), off = i*16;
      for (let m = 0; m < d; m++) { const ra = P.R[off+a*d+m], rb = P.R[off+b*d+m]; P.R[off+a*d+m] = Math.cos(th)*ra - Math.sin(th)*rb; P.R[off+b*d+m] = Math.sin(th)*ra + Math.cos(th)*rb; }
      for (let k = 0; k < d; k++) P.c[i*d+k] += 0.1 * gauss();
    }
  } else {
    const i = Math.floor(rnd() * n); let j = Math.floor(rnd() * n); if (i === j) j = (j + 1) % n; moved.add(i); moved.add(j);
    for (let k = 0; k < d; k++) { const t = P.c[i*d+k]; P.c[i*d+k] = P.c[j*d+k]; P.c[j*d+k] = t; }
  }
  for (let q = 0; q < nd; q++) P.c[q] += 0.01 * gauss();
  axesFromCentres(P, moved);
  return P;
}
/* run one full polish synchronously (for Node testing) */
function polishSync(P, innerCap) { const job = makePolish(P, innerCap); while (!job.done) polishWork(job, 1e9); return { P: finalize(jobToPacking(job)), job }; }

const Core = { setSeed, rnd, productSeed, fromData, makePolish, polishWork, jobToPacking, finalize, clearances, overlapping, perturb, clone, polishSync };
if (typeof module !== 'undefined' && module.exports) module.exports = Core; else root.Core = Core;
})(typeof window !== 'undefined' ? window : globalThis);
