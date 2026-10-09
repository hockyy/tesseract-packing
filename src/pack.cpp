// Soft-to-rigid packing of n unit d-cubes in [0,s]^d, multithreaded, hand gradients.
// Each pair owns a free unit direction u_ij (a candidate separating axis); a pair is
// non-overlapping iff |u.(cj-ci)| >= h_i(u)+h_j(u), h = support function of the particle.
// Particle shape = (1-t)*Ball(1/2) + t*UnitCube; t: 0 (ball) -> 1 (cube).
// build: g++ -O3 -march=native -pthread pack.cpp -o pack.exe
// run:   pack.exe d n starts soft_steps seed out.json [threads]
#include <bits/stdc++.h>
using namespace std;
typedef double D;
static int d, n, P;
static vector<int> PI, PJ;
static D SIGMA = 0.0;
static D JIT = 0.03;

struct State { vector<D> c, R, u; };
struct Grad { vector<D> c, R, u; };

static void orthonormalize(D* R) { // columns R[k*d+m] (k row, m column)
    for (int m = 0; m < d; m++) {
        for (int p = 0; p < m; p++) {
            D dot = 0; for (int k = 0; k < d; k++) dot += R[k*d+m]*R[k*d+p];
            for (int k = 0; k < d; k++) R[k*d+m] -= dot*R[k*d+p];
        }
        D nr = 0; for (int k = 0; k < d; k++) nr += R[k*d+m]*R[k*d+m];
        nr = sqrt(nr); for (int k = 0; k < d; k++) R[k*d+m] /= nr;
    }
}
static inline D sgn(D x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

// returns energy; fills g if non-null
static D energy(const State& S, Grad* g, D s, D t) {
    D E = 0;
    if (g) { fill(g->c.begin(), g->c.end(), 0); fill(g->R.begin(), g->R.end(), 0); fill(g->u.begin(), g->u.end(), 0); }
    D ai[8], aj[8];
    for (int p = 0; p < P; p++) {
        int i = PI[p], j = PJ[p];
        const D* u = &S.u[p*d]; const D* Ri = &S.R[i*d*d]; const D* Rj = &S.R[j*d*d];
        const D* ci = &S.c[i*d]; const D* cj = &S.c[j*d];
        D si = 0, sj = 0, uw = 0;
        for (int m = 0; m < d; m++) {
            D a = 0, b = 0; for (int k = 0; k < d; k++) { a += Ri[k*d+m]*u[k]; b += Rj[k*d+m]*u[k]; }
            ai[m] = a; aj[m] = b; si += fabs(a); sj += fabs(b);
        }
        for (int k = 0; k < d; k++) uw += u[k]*(cj[k]-ci[k]);
        D sg = uw >= 0 ? 1.0 : -1.0;
        D viol = 0.5*((1-t)*2 + t*(si+sj)) - sg*uw;   // two particles: each (1-t)+t*sum -> halves
        if (viol > 0) {
            E += viol*viol;
            if (g) {
                D f = 2*viol;
                for (int k = 0; k < d; k++) {
                    D gu = -sg*(cj[k]-ci[k]);
                    for (int m = 0; m < d; m++) {
                        gu += 0.5*t*(sgn(ai[m])*Ri[k*d+m] + sgn(aj[m])*Rj[k*d+m]);
                        g->R[i*d*d+k*d+m] += f*0.5*t*sgn(ai[m])*u[k];
                        g->R[j*d*d+k*d+m] += f*0.5*t*sgn(aj[m])*u[k];
                    }
                    g->u[p*d+k] += f*gu;
                    g->c[j*d+k] += f*(-sg*u[k]);
                    g->c[i*d+k] += f*(sg*u[k]);
                }
            }
        }
    }
    for (int i = 0; i < n; i++) {
        const D* R = &S.R[i*d*d];
        for (int k = 0; k < d; k++) {
            D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(R[k*d+m]);
            D hw = 0.5*((1-t) + t*l1);
            D x = S.c[i*d+k];
            D v1 = hw - x, v2 = x + hw - s;
            if (v1 > 0) { E += v1*v1; if (g) { g->c[i*d+k] += -2*v1; for (int m = 0; m < d; m++) g->R[i*d*d+k*d+m] += 2*v1*0.5*t*sgn(R[k*d+m]); } }
            if (v2 > 0) { E += v2*v2; if (g) { g->c[i*d+k] += 2*v2; for (int m = 0; m < d; m++) g->R[i*d*d+k*d+m] += 2*v2*0.5*t*sgn(R[k*d+m]); } }
        }
    }
    return E;
}

struct Adam {
    vector<D> m, v; long k = 0;
    void reset(size_t N) { m.assign(N, 0); v.assign(N, 0); k = 0; }
    void step(vector<D>& x, const vector<D>& g, size_t off, D lr) { (void)off;
        k++; D b1 = 0.9, b2 = 0.999; D c1 = 1 - pow(b1, k), c2 = 1 - pow(b2, k);
        for (size_t q = 0; q < x.size(); q++) {
            m[q] = b1*m[q] + (1-b1)*g[q]; v[q] = b2*v[q] + (1-b2)*g[q]*g[q];
            x[q] -= lr*(m[q]/c1)/(sqrt(v[q]/c2) + 1e-12);
        }
    }
};
struct Opt {
    Adam ac, aR, au;
    void reset(const State& S) { ac.reset(S.c.size()); aR.reset(S.R.size()); au.reset(S.u.size()); }
    void step(State& S, const Grad& g, D lr) {
        ac.step(S.c, g.c, 0, lr); aR.step(S.R, g.R, 0, lr); au.step(S.u, g.u, 0, lr);
        for (int i = 0; i < n; i++) orthonormalize(&S.R[i*d*d]);
        for (int p = 0; p < P; p++) { D nr = 0; for (int k = 0; k < d; k++) nr += S.u[p*d+k]*S.u[p*d+k]; nr = sqrt(nr); if (nr < 1e-12) { S.u[p*d] = 1; nr = 1; } for (int k = 0; k < d; k++) S.u[p*d+k] /= nr; }
    }
};

struct Result { D s = 1e18; State S; D clr = 0, wall = 0; };

static void measure(const State& S, D s, D& clr, D& wall) {
    clr = 1e18; wall = 1e18;
    for (int p = 0; p < P; p++) {
        int i = PI[p], j = PJ[p]; D hs = 0, uw = 0;
        for (int m = 0; m < d; m++) { D a = 0, b = 0; for (int k = 0; k < d; k++) { a += S.R[i*d*d+k*d+m]*S.u[p*d+k]; b += S.R[j*d*d+k*d+m]*S.u[p*d+k]; } hs += 0.5*fabs(a) + 0.5*fabs(b); }
        for (int k = 0; k < d; k++) uw += S.u[p*d+k]*(S.c[j*d+k]-S.c[i*d+k]);
        clr = min(clr, fabs(uw) - hs);
    }
    for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(S.R[i*d*d+k*d+m]); D hw = 0.5*l1; wall = min(wall, min(S.c[i*d+k]-hw, s - S.c[i*d+k] - hw)); }
}

static Result run(unsigned long seed, int steps) {
    mt19937_64 rng(seed); normal_distribution<D> N01(0, 1); uniform_real_distribution<D> U01(0, 1);
    D vol = pow((D)n, 1.0/d);
    int triv = (int)ceil(vol - 1e-9);
    D s_start = 1.35*vol + 0.3, s_end = max(vol*1.02, triv*(getenv("SEF")?atof(getenv("SEF")):1.0));
    State S; S.c.resize(n*d); S.R.resize(n*d*d); S.u.resize(P*d);
    for (auto& x : S.c) x = U01(rng)*s_start;
    for (auto& x : S.R) x = N01(rng);
    for (int i = 0; i < n; i++) orthonormalize(&S.R[i*d*d]);
    for (int p = 0; p < P; p++) { D nr = 0; for (int k = 0; k < d; k++) { D w = S.c[PJ[p]*d+k]-S.c[PI[p]*d+k]; S.u[p*d+k] = w; nr += w*w; } nr = sqrt(nr)+1e-12; for (int k = 0; k < d; k++) S.u[p*d+k] /= nr; }
    bool templ = getenv("TEMPLATE") != nullptr;
    if (templ) {
        // corner-frame template: 2^d axis-aligned corner cubes on a 3^d grid, the rest tilted in channel cells
        int ncell = 1; for (int k = 0; k < d; k++) ncell *= 3;
        vector<int> corners, chan, other;
        for (int cell = 0; cell < ncell; cell++) { int x = cell, ones = 0; for (int k = 0; k < d; k++) { if (x % 3 == 1) ones++; x /= 3; }
            int want = getenv("ONES") ? atoi(getenv("ONES")) : d-1; (ones == 0 ? corners : ones == want ? chan : other).push_back(cell); }
        shuffle(chan.begin(), chan.end(), rng); shuffle(other.begin(), other.end(), rng);
        D pchan = getenv("PCHAN") ? atof(getenv("PCHAN")) : 0.85;
        vector<int> cells(corners.begin(), corners.end());
        size_t ci = 0, oi = 0;
        while ((int)cells.size() < n) { if ((U01(rng) < pchan && ci < chan.size()) || oi >= other.size()) cells.push_back(chan[ci++]); else cells.push_back(other[oi++]); }
        if ((int)cells.size() > n) cells.resize(n);   // n < 2^d: drop corners
        for (int i = 0; i < n; i++) {
            int x = cells[i], ones = 0; for (int k = 0; k < d; k++) { S.c[i*d+k] = 0.5 + (x % 3) + 0.02*N01(rng); if (x % 3 == 1) ones++; x /= 3; }
            D* R = &S.R[i*d*d]; for (int q = 0; q < d*d; q++) R[q] = (q % (d+1) == 0) ? 1 : 0;
            if (ones == 0) { for (int q = 0; q < d*d; q++) R[q] += 0.02*N01(rng); }
            else {
                // rotate by 20-45 deg in a random 2-plane, plus jitter
                vector<D> a(d, 0.0), b(d, 0.0);
                vector<int> mid; { int y = cells[i]; for (int k = 0; k < d; k++) { if (y % 3 == 1) mid.push_back(k); y /= 3; } }
                if (mid.size() >= 2 && U01(rng) < 0.8) {   // rotate within the plane of two "middle" axes (about the face normal in 3D)
                    shuffle(mid.begin(), mid.end(), rng); a[mid[0]] = 1; b[mid[1]] = 1;
                } else {
                    for (int k = 0; k < d; k++) { a[k] = N01(rng); b[k] = N01(rng); }
                    D na = 0; for (D v : a) na += v*v; na = sqrt(na); for (D& v : a) v /= na;
                    D ab = 0; for (int k = 0; k < d; k++) ab += a[k]*b[k]; for (int k = 0; k < d; k++) b[k] -= ab*a[k];
                    D nb = 0; for (D v : b) nb += v*v; nb = sqrt(nb); for (D& v : b) v /= nb;
                }
                D tmin = getenv("TMIN") ? atof(getenv("TMIN")) : 20, tmax = getenv("TMAX") ? atof(getenv("TMAX")) : 45;
                D th = (tmin + (tmax-tmin)*U01(rng)) * M_PI/180 * (U01(rng) < 0.5 ? -1 : 1);
                for (int k = 0; k < d; k++) for (int m = 0; m < d; m++)
                    R[k*d+m] += (cos(th)-1)*(a[k]*a[m]+b[k]*b[m]) + sin(th)*(b[k]*a[m]-a[k]*b[m]) + JIT*N01(rng);
            }
            orthonormalize(R);
        }
        for (int p = 0; p < P; p++) { D nr = 0; for (int k = 0; k < d; k++) { D w = S.c[PJ[p]*d+k]-S.c[PI[p]*d+k]; S.u[p*d+k] = w; nr += w*w; } nr = sqrt(nr)+1e-12; for (int k = 0; k < d; k++) S.u[p*d+k] /= nr; }
        s_end = getenv("S0") ? atof(getenv("S0")) : 3.0;
    }
    Grad g; g.c.resize(S.c.size()); g.R.resize(S.R.size()); g.u.resize(S.u.size());
    Opt opt; opt.reset(S);
    for (int it = 0; it < (templ ? 0 : steps); it++) {
        D f = (D)it/(steps-1);
        D t = min(1.0, f/0.8), s = s_start + (s_end - s_start)*min(1.0, f/0.9);
        D lr = 0.02*(1 - 0.9*f);
        energy(S, &g, s, t); opt.step(S, g, lr);
        if (SIGMA > 0) { D sg = SIGMA*(1 - min(1.0, f/0.9)); if (sg > 0) {
            for (auto& x : S.c) x += sg*N01(rng)*0.3;
            for (int i = 0; i < n; i++) { for (int q = 0; q < d*d; q++) S.R[i*d*d+q] += sg*N01(rng); orthonormalize(&S.R[i*d*d]); } } }
    }
    // rigid shrink
    Result best; D s = s_end, delta = 0.02; State bestS; bool have = false; D bestSval = 1e18;
    opt.reset(S);
    for (int round = 0; round < 600 && delta > 2e-8; round++) {
        D lr = min(3e-3, max(2e-7, delta*0.3));
        D E = 1;
        for (int it = 0; it < 1500; it++) { E = energy(S, &g, s, 1.0); if (E < 1e-14) break; opt.step(S, g, lr); }
        E = energy(S, nullptr, s, 1.0);
        if (E < 1e-14) { bestS = S; bestSval = s; have = true; s -= delta; }
        else if (!have) { s += (getenv("GROW") ? atof(getenv("GROW")) : 0.05); opt.reset(S); }
        else { S = bestS; opt.reset(S); delta /= 2; s = bestSval - delta; }
    }
    if (have) {
        // joint refinement: minimise s + mu*E over (state, s), then repair to exact feasibility
        State Q = bestS; D sv = bestSval; D mu = 1e2; Opt o2; o2.reset(Q);
        for (int stage = 0; stage < (getenv("REF")?7:0); stage++, mu *= 10) {
            D lr = 2e-3/pow(3.0, stage); D ms = 0, vs = 0; long ks = 0;
            for (int it = 0; it < 6000; it++) {
                D E = energy(Q, &g, sv, 1.0); (void)E;
                for (auto& x : g.c) x *= mu; for (auto& x : g.R) x *= mu; for (auto& x : g.u) x *= mu;
                // dE/ds
                D dEds = 0;
                for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(Q.R[i*d*d+k*d+m]); D v2 = Q.c[i*d+k] + 0.5*l1 - sv; if (v2 > 0) dEds += -2*v2; }
                D gs = 1 + mu*dEds; ks++;
                ms = 0.9*ms + 0.1*gs; vs = 0.999*vs + 0.001*gs*gs;
                sv -= lr*(ms/(1-pow(0.9,ks)))/(sqrt(vs/(1-pow(0.999,ks)))+1e-12);
                o2.step(Q, g, lr);
            }
        }
        // repair: scale centres about origin so every pair has clearance >= 0, then set s exactly
        D clr, wall; measure(Q, sv, clr, wall);
        D lam = 1.0; if (clr < 0) lam = 1.0 + (-clr)*2.0 + 1e-12;
        for (auto& x : Q.c) x *= lam;
        D sn = 0; for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(Q.R[i*d*d+k*d+m]); sn = max(sn, Q.c[i*d+k] + 0.5*l1); }
        // lower walls: shift so min extent is 0
        D mn = 1e18; for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(Q.R[i*d*d+k*d+m]); mn = min(mn, Q.c[i*d+k] - 0.5*l1); }
        (void)mn; sn = 0; for (int k = 0; k < d; k++) { D lo = 1e18, hi = -1e18; for (int i = 0; i < n; i++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(Q.R[i*d*d+k*d+m]); lo = min(lo, Q.c[i*d+k] - 0.5*l1); hi = max(hi, Q.c[i*d+k] + 0.5*l1); } for (int i = 0; i < n; i++) Q.c[i*d+k] -= lo; sn = max(sn, hi - lo); }
        best.s = sn; best.S = Q; measure(Q, sn, best.clr, best.wall);
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: pack d n starts soft_steps seed out.json [threads]\n"); return 1; }
    d = atoi(argv[1]); n = atoi(argv[2]); int starts = atoi(argv[3]); int steps = atoi(argv[4]);
    unsigned long seed0 = strtoul(argv[5], 0, 10); string out = argv[6];
    int T = argc > 7 ? atoi(argv[7]) : (int)thread::hardware_concurrency();
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) { PI.push_back(i); PJ.push_back(j); }
    P = PI.size(); if (getenv("SIGMA")) SIGMA = atof(getenv("SIGMA")); if (getenv("JIT")) JIT = atof(getenv("JIT"));
    vector<Result> res(starts); atomic<int> next(0); mutex mu; D gbest = 1e18; auto t0 = chrono::steady_clock::now();
    vector<thread> th;
    for (int w = 0; w < T; w++) th.emplace_back([&]() {
        for (;;) { int k = next++; if (k >= starts) break; res[k] = run(seed0*1000003UL + k, steps);
            lock_guard<mutex> lk(mu);
            if (res[k].s < gbest) { gbest = res[k].s; double el = chrono::duration<double>(chrono::steady_clock::now()-t0).count();
                fprintf(stderr, "start %d: s=%.9f clr=%.1e wall=%.1e (%.0fs)\n", k, res[k].s, res[k].clr, res[k].wall, el); } }
    });
    for (auto& x : th) x.join();
    vector<int> idx(starts); iota(idx.begin(), idx.end(), 0); sort(idx.begin(), idx.end(), [&](int a, int b) { return res[a].s < res[b].s; });
    printf("d=%d n=%d trivial=%d vol_bound=%.4f\nbest sides:", d, n, (int)ceil(pow((D)n,1.0/d)-1e-9), pow((D)n,1.0/d));
    for (int q = 0; q < min(starts, 10); q++) printf(" %.7f", res[idx[q]].s); printf("\n");
    FILE* f = fopen(out.c_str(), "w"); fprintf(f, "{\"d\":%d,\"n\":%d,\"results\":[", d, n);
    for (int q = 0; q < min(starts, 30); q++) { const Result& r = res[idx[q]]; if (r.s > 1e17) continue; if (q) fprintf(f, ",");
        fprintf(f, "{\"s\":%.12f,\"clearance\":%.3e,\"wall\":%.3e,\"centers\":[", r.s, r.clr, r.wall);
        for (size_t z = 0; z < r.S.c.size(); z++) fprintf(f, "%s%.15g", z?",":"", r.S.c[z]);
        fprintf(f, "],\"R\":["); for (size_t z = 0; z < r.S.R.size(); z++) fprintf(f, "%s%.15g", z?",":"", r.S.R[z]);
        fprintf(f, "],\"u\":["); for (size_t z = 0; z < r.S.u.size(); z++) fprintf(f, "%s%.15g", z?",":"", r.S.u[z]); fprintf(f, "]}"); }
    fprintf(f, "]}\n"); fclose(f);
    return 0;
}
