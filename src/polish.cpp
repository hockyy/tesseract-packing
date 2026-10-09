// Constrained polish: augmented Lagrangian + L-BFGS, hand gradients.
// min s  s.t.  per pair  sigma*u.(cj-ci) - h_i(u) - h_j(u) >= 0,  walls,  R^T R = I.
// build: g++ -O3 -march=native polish.cpp -o polish.exe
// run:   polish.exe in.json out.json [topk]
#include <bits/stdc++.h>
using namespace std;
typedef double D;
static int d, n;
static bool FIXS = false;

static vector<D> getArr(const string& t, size_t from, const string& key, size_t& endpos) {
    size_t p = t.find("\"" + key + "\":[", from); vector<D> v; if (p == string::npos) { endpos = string::npos; return v; }
    p = t.find('[', p) + 1; size_t e = t.find(']', p); const char* s = t.c_str() + p; char* q;
    while (s < t.c_str() + e) { D x = strtod(s, &q); if (q == s) break; v.push_back(x); s = q; while (*s == ',' || *s == ' ') s++; }
    endpos = e; return v;
}
static inline D sgn(D x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

struct Prob {
    vector<int> pi, pj; vector<D> sg;           // active pairs and fixed signs
    vector<D> lp, lw, lo; D mu = 10;            // multipliers: pair, wall(2nd), orth
    int nd, ndd, A, N;                          // N = total vars
    void setup(int active) { A = active; nd = n * d; ndd = n * d * d; N = nd + ndd + A * d + 1; }
    // L and gradient. If upd, also writes constraint values for multiplier update.
    D eval(const vector<D>& x, vector<D>* g, vector<D>* gp = nullptr, vector<D>* gw = nullptr, vector<D>* eo = nullptr) {
        D L = FIXS ? 0.0 : x[N - 1]; if (g) { fill(g->begin(), g->end(), 0.0); (*g)[N - 1] = FIXS ? 0.0 : 1; }
        const D* c = &x[0]; const D* R = &x[nd]; const D* U = &x[nd + ndd]; D s = x[N - 1];
        const D margin = 1e-12;
        for (int p = 0; p < A; p++) {
            int i = pi[p], j = pj[p]; const D* up = U + p*d; D nr = 0; for (int k = 0; k < d; k++) nr += up[k]*up[k]; nr = sqrt(nr);
            D uh[4]; for (int k = 0; k < d; k++) uh[k] = up[k]/nr;
            const D* Ri = R + i*d*d; const D* Rj = R + j*d*d;
            D ai[4], aj[4], hs = 0, uw = 0;
            for (int m = 0; m < d; m++) { D a = 0, b = 0; for (int k = 0; k < d; k++) { a += Ri[k*d+m]*uh[k]; b += Rj[k*d+m]*uh[k]; } ai[m] = a; aj[m] = b; hs += 0.5*(fabs(a)+fabs(b)); }
            for (int k = 0; k < d; k++) uw += uh[k]*(c[j*d+k]-c[i*d+k]);
            D gv = sg[p]*uw - hs - margin;
            if (gp) (*gp)[p] = gv;
            D w = -max(0.0, lp[p] - mu*gv);
            L += (w*w - lp[p]*lp[p])/(2*mu);   // (max(0,lam-mu g)^2 - lam^2)/(2mu)
            if (g && w != 0) {
                D gh[4];
                for (int k = 0; k < d; k++) { D v = sg[p]*(c[j*d+k]-c[i*d+k]); for (int m = 0; m < d; m++) v -= 0.5*(sgn(ai[m])*Ri[k*d+m] + sgn(aj[m])*Rj[k*d+m]); gh[k] = v; }
                D dot = 0; for (int k = 0; k < d; k++) dot += gh[k]*uh[k];
                for (int k = 0; k < d; k++) {
                    (*g)[nd+ndd+p*d+k] += w*(gh[k]-dot*uh[k])/nr;
                    (*g)[j*d+k] += w*sg[p]*uh[k]; (*g)[i*d+k] -= w*sg[p]*uh[k];
                    for (int m = 0; m < d; m++) { (*g)[nd+i*d*d+k*d+m] += w*(-0.5*sgn(ai[m])*uh[k]); (*g)[nd+j*d*d+k*d+m] += w*(-0.5*sgn(aj[m])*uh[k]); }
                }
            }
        }
        for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) {
            const D* Rr = R + i*d*d; D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(Rr[k*d+m]);
            D hw = 0.5*l1; D xx = c[i*d+k];
            for (int side = 0; side < 2; side++) {
                int id = (i*d+k)*2 + side; D gv = (side == 0 ? xx - hw : s - xx - hw) - margin;
                if (gw) (*gw)[id] = gv;
                D w = -max(0.0, lw[id] - mu*gv); L += (w*w - lw[id]*lw[id])/(2*mu);
                if (g && w != 0) {
                    (*g)[i*d+k] += w*(side == 0 ? 1 : -1); if (side == 1) (*g)[N-1] += w;
                    for (int m = 0; m < d; m++) (*g)[nd+i*d*d+k*d+m] += w*(-0.5*sgn(Rr[k*d+m]));
                }
            }
        }
        int oid = 0;
        for (int i = 0; i < n; i++) { const D* Rr = R + i*d*d;
            for (int m = 0; m < d; m++) for (int m2 = m; m2 < d; m2++, oid++) {
                D e = -(m == m2 ? 1.0 : 0.0); for (int k = 0; k < d; k++) e += Rr[k*d+m]*Rr[k*d+m2];
                if (eo) (*eo)[oid] = e;
                D w = lo[oid] + mu*e; L += lo[oid]*e + 0.5*mu*e*e;
                if (g) for (int k = 0; k < d; k++) {
                    if (m == m2) (*g)[nd+i*d*d+k*d+m] += 2*w*Rr[k*d+m];
                    else { (*g)[nd+i*d*d+k*d+m] += w*Rr[k*d+m2]; (*g)[nd+i*d*d+k*d+m2] += w*Rr[k*d+m]; }
                }
            } }
        if (FIXS && g) (*g)[N - 1] = 0;
        return L;
    }
};

static D dotv(const vector<D>& a, const vector<D>& b) { D s = 0; for (size_t i = 0; i < a.size(); i++) s += a[i]*b[i]; return s; }

static void lbfgs(Prob& pb, vector<D>& x, int maxit) {
    int N = pb.N, M = 25; vector<D> g(N), gn(N), xn(N), dir(N);
    vector<vector<D>> S, Y; vector<D> rho;
    D f = pb.eval(x, &g);
    for (int it = 0; it < maxit; it++) {
        D gnorm = 0; for (D v : g) gnorm = max(gnorm, fabs(v)); if (gnorm < 1e-11) break;
        dir = g; int k = S.size(); vector<D> al(k);
        for (int q = k-1; q >= 0; q--) { al[q] = rho[q]*dotv(S[q], dir); for (int i = 0; i < N; i++) dir[i] -= al[q]*Y[q][i]; }
        if (k) { D gam = dotv(S[k-1], Y[k-1])/dotv(Y[k-1], Y[k-1]); for (auto& v : dir) v *= gam; } else { D nr = sqrt(dotv(g, g)); for (auto& v : dir) v *= 1e-3/max(nr, 1e-12)*1.0; }
        for (int q = 0; q < k; q++) { D be = rho[q]*dotv(Y[q], dir); for (int i = 0; i < N; i++) dir[i] += S[q][i]*(al[q]-be); }
        for (auto& v : dir) v = -v;
        D gd = dotv(g, dir); if (gd >= 0) { S.clear(); Y.clear(); rho.clear(); dir = g; for (auto& v : dir) v = -v*1e-3/max(sqrt(dotv(g,g)),1e-12); gd = dotv(g, dir); }
        D step = 1; bool ok = false; D fn = f;
        for (int ls = 0; ls < 40; ls++) {
            for (int i = 0; i < N; i++) xn[i] = x[i] + step*dir[i];
            fn = pb.eval(xn, &gn);
            if (fn <= f + 1e-4*step*gd) { ok = true; break; }
            step *= 0.5;
        }
        if (!ok) { if (S.empty()) break; S.clear(); Y.clear(); rho.clear(); continue; }
        vector<D> sv(N), yv(N); for (int i = 0; i < N; i++) { sv[i] = xn[i]-x[i]; yv[i] = gn[i]-g[i]; }
        D sy = dotv(sv, yv);
        if (sy > 1e-14) { S.push_back(sv); Y.push_back(yv); rho.push_back(1/sy); if ((int)S.size() > M) { S.erase(S.begin()); Y.erase(Y.begin()); rho.erase(rho.begin()); } }
        x = xn; g = gn; f = fn;
    }
}

static void gs(D* R) {
    for (int m = 0; m < d; m++) {
        for (int p = 0; p < m; p++) { D dot = 0; for (int k = 0; k < d; k++) dot += R[k*d+m]*R[k*d+p]; for (int k = 0; k < d; k++) R[k*d+m] -= dot*R[k*d+p]; }
        D nr = 0; for (int k = 0; k < d; k++) nr += R[k*d+m]*R[k*d+m]; nr = sqrt(nr); for (int k = 0; k < d; k++) R[k*d+m] /= nr;
    }
}

struct Pack { vector<D> c, R, u; D s; };

static void clearances(const Pack& P, vector<D>& clr, D& wall) {
    int NP = n*(n-1)/2; clr.assign(NP, 0); int p = 0;
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) {
        D nr = 0; for (int k = 0; k < d; k++) nr += P.u[p*d+k]*P.u[p*d+k]; nr = sqrt(nr); D hs = 0, uw = 0;
        for (int m = 0; m < d; m++) { D a = 0, b = 0; for (int k = 0; k < d; k++) { a += P.R[i*d*d+k*d+m]*P.u[p*d+k]/nr; b += P.R[j*d*d+k*d+m]*P.u[p*d+k]/nr; } hs += 0.5*(fabs(a)+fabs(b)); }
        for (int k = 0; k < d; k++) uw += P.u[p*d+k]/nr*(P.c[j*d+k]-P.c[i*d+k]);
        clr[p] = fabs(uw) - hs;
    }
    wall = 1e18;
    for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(P.R[i*d*d+k*d+m]); wall = min(wall, min(P.c[i*d+k]-0.5*l1, P.s-P.c[i*d+k]-0.5*l1)); }
}

static D fitBox(Pack& P) { // shift and set s exactly to the extent
    D sn = 0;
    for (int k = 0; k < d; k++) { D lo = 1e18, hi = -1e18;
        for (int i = 0; i < n; i++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(P.R[i*d*d+k*d+m]); lo = min(lo, P.c[i*d+k]-0.5*l1); hi = max(hi, P.c[i*d+k]+0.5*l1); }
        for (int i = 0; i < n; i++) P.c[i*d+k] -= lo; sn = max(sn, hi-lo); }
    P.s = sn; return sn;
}

static Pack polish(Pack P, int verbose) {
    int NP = n*(n-1)/2;
    for (int round = 0; round < 6; round++) {
        vector<D> clr; D wall; clearances(P, clr, wall);
        Prob pb; int p = 0; vector<int> act;
        for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) if (clr[p] < 0.6) { pb.pi.push_back(i); pb.pj.push_back(j); act.push_back(p); }
        pb.setup(act.size());
        pb.sg.resize(pb.A); for (int a = 0; a < pb.A; a++) { int q = act[a]; D uw = 0; for (int k = 0; k < d; k++) uw += P.u[q*d+k]*(P.c[pb.pj[a]*d+k]-P.c[pb.pi[a]*d+k]); pb.sg[a] = uw >= 0 ? 1 : -1; }
        pb.lp.assign(pb.A, 0); pb.lw.assign(n*d*2, 0); pb.lo.assign(n*d*(d+1)/2, 0); pb.mu = 10;
        vector<D> x(pb.N); copy(P.c.begin(), P.c.end(), x.begin()); copy(P.R.begin(), P.R.end(), x.begin()+pb.nd);
        for (int a = 0; a < pb.A; a++) for (int k = 0; k < d; k++) x[pb.nd+pb.ndd+a*d+k] = P.u[act[a]*d+k]; x[pb.N-1] = P.s;
        D prevViol = 1e18;
        for (int outer = 0; outer < 40; outer++) {
            lbfgs(pb, x, 3000);
            vector<D> gp(pb.A), gw(n*d*2), eo(pb.lo.size());
            pb.eval(x, nullptr, &gp, &gw, &eo);
            D viol = 0;
            for (int a = 0; a < pb.A; a++) { viol = max(viol, -gp[a]); pb.lp[a] = max(0.0, pb.lp[a]-pb.mu*gp[a]); }
            for (size_t a = 0; a < gw.size(); a++) { viol = max(viol, -gw[a]); pb.lw[a] = max(0.0, pb.lw[a]-pb.mu*gw[a]); }
            for (size_t a = 0; a < eo.size(); a++) { viol = max(viol, fabs(eo[a])); pb.lo[a] += pb.mu*eo[a]; }
            if (verbose) fprintf(stderr, "  r%d outer %d s=%.12f viol=%.2e mu=%.0e\n", round, outer, x[pb.N-1], viol, pb.mu);
            if (viol < 1e-13) break;
            if (viol > 0.25*prevViol) pb.mu = min(pb.mu*5, 1e9);
            prevViol = viol;
        }
        // write back (A active pairs updated; others keep their old axes)
        copy(x.begin(), x.begin()+pb.nd, P.c.begin()); copy(x.begin()+pb.nd, x.begin()+pb.nd+pb.ndd, P.R.begin());
        for (int a = 0; a < pb.A; a++) for (int k = 0; k < d; k++) P.u[act[a]*d+k] = x[pb.nd+pb.ndd+a*d+k];
        P.s = x[pb.N-1];
        for (int i = 0; i < n; i++) gs(&P.R[i*d*d]);
        // all pairs OK?  (excluded pairs may need their axis re-chosen: reselect best of centre-difference)
        clearances(P, clr, wall); int bad = 0; for (int q = 0; q < NP; q++) if (clr[q] < -1e-7) bad++;
        if (!bad) break;
        // reseed axes of bad pairs along centre difference and iterate with them active
        p = 0; for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) if (clr[p] < -1e-7) { D nr = 0; for (int k = 0; k < d; k++) { D w = P.c[j*d+k]-P.c[i*d+k]; P.u[p*d+k] = w; nr += w*w; } nr = sqrt(nr)+1e-12; for (int k = 0; k < d; k++) P.u[p*d+k] /= nr; }
        P.s += 0.05;
    }
    // repair: scale centres so all pair clearances >= 0, refit box
    vector<D> clr; D wall; clearances(P, clr, wall); D mn = *min_element(clr.begin(), clr.end());
    if (mn < 0) { D lam = 1 + 2*(-mn) + 1e-13; for (auto& v : P.c) v *= lam; }
    for (int i = 0; i < n; i++) gs(&P.R[i*d*d]);
    fitBox(P);
    return P;
}

static void writePacks(const string& fn, const vector<Pack>& outs) {
    FILE* f = fopen(fn.c_str(), "w"); fprintf(f, "{\"d\":%d,\"n\":%d,\"results\":[", d, n);
    for (size_t q = 0; q < outs.size(); q++) { if (q) fprintf(f, ","); const Pack& P = outs[q];
        fprintf(f, "{\"s\":%.15g,\"centers\":[", P.s); for (size_t z = 0; z < P.c.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.c[z]);
        fprintf(f, "],\"R\":["); for (size_t z = 0; z < P.R.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.R[z]);
        fprintf(f, "],\"u\":["); for (size_t z = 0; z < P.u.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.u[z]); fprintf(f, "]}"); }
    fprintf(f, "]}\n"); fclose(f);
}

// random rotation by angle th in a random 2-plane, applied on the left of R (d x d)
static void rotate(D* R, D th, mt19937_64& rng) {
    normal_distribution<D> N01(0, 1);
    vector<D> a(d), b(d); for (int k = 0; k < d; k++) { a[k] = N01(rng); b[k] = N01(rng); }
    D na = 0; for (D v : a) na += v*v; na = sqrt(na); for (D& v : a) v /= na;
    D ab = 0; for (int k = 0; k < d; k++) ab += a[k]*b[k]; for (int k = 0; k < d; k++) b[k] -= ab*a[k];
    D nb = 0; for (D v : b) nb += v*v; nb = sqrt(nb); for (D& v : b) v /= nb;
    vector<D> G(d*d); for (int k = 0; k < d; k++) for (int m = 0; m < d; m++) G[k*d+m] = (k == m) + (cos(th)-1)*(a[k]*a[m]+b[k]*b[m]) + sin(th)*(b[k]*a[m]-a[k]*b[m]);
    vector<D> T(d*d, 0); for (int k = 0; k < d; k++) for (int m = 0; m < d; m++) for (int q = 0; q < d; q++) T[k*d+m] += G[k*d+q]*R[q*d+m];
    copy(T.begin(), T.end(), R); gs(R);
}

static Pack perturb(const Pack& P0, mt19937_64& rng, int& kind) {
    Pack P = P0; uniform_real_distribution<D> U(0, 1); normal_distribution<D> N01(0, 1);
    vector<int> moved; kind = rng() % 4;
    if (kind == 0 || kind == 1) {          // rotate + nudge 1 (or 2) cubes
        int cnt = kind == 0 ? 1 : 2;
        for (int q = 0; q < cnt; q++) { int i = rng() % n; moved.push_back(i);
            rotate(&P.R[i*d*d], (5 + 40*U(rng))*M_PI/180*(U(rng) < .5 ? -1 : 1), rng);
            for (int k = 0; k < d; k++) P.c[i*d+k] += 0.1*N01(rng); }
    } else if (kind == 2) {                // relocate one cube to a random spot, random orientation
        int i = rng() % n; moved.push_back(i);
        for (int k = 0; k < d; k++) P.c[i*d+k] = 0.5 + (P.s-1)*U(rng);
        for (int q = 0; q < d*d; q++) P.R[i*d*d+q] = N01(rng); gs(&P.R[i*d*d]);
    } else {                               // swap two cubes' positions
        int i = rng() % n, j = rng() % n; if (i == j) j = (j+1) % n; moved.push_back(i); moved.push_back(j);
        for (int k = 0; k < d; k++) swap(P.c[i*d+k], P.c[j*d+k]);
    }
    // global small shake of everyone (helps escape) + slight box inflation
    for (auto& v : P.c) v += 0.01*N01(rng);
    P.s *= 1.0;
    int p = 0;
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) {
        if (find(moved.begin(), moved.end(), i) == moved.end() && find(moved.begin(), moved.end(), j) == moved.end()) continue;
        D nr = 0; for (int k = 0; k < d; k++) { D w = P.c[j*d+k]-P.c[i*d+k]; P.u[p*d+k] = w; nr += w*w; } nr = sqrt(nr)+1e-12; for (int k = 0; k < d; k++) P.u[p*d+k] /= nr;
    }
    return P;
}

static int basinHop(vector<Pack> starts, const string& outfn, double seconds, int T) {
    mutex mu; Pack gbest = starts[0]; for (auto& p : starts) if (p.s < gbest.s) gbest = p;
    fprintf(stderr, "BH start: best s=%.12f, %zu starting layouts, %d threads, %.0fs\n", gbest.s, starts.size(), T, seconds);
    auto t0 = chrono::steady_clock::now(); atomic<long> hops(0), acc(0);
    vector<thread> th;
    for (int w = 0; w < T; w++) th.emplace_back([&, w]() {
        mt19937_64 rng(1234567ULL*(w+1) + (unsigned long long)chrono::steady_clock::now().time_since_epoch().count());
        Pack cur = starts[w % starts.size()];
        int stale = 0;
        while (chrono::duration<double>(chrono::steady_clock::now()-t0).count() < seconds) {
            int kind; Pack Q = polish(perturb(cur, rng, kind), 0); hops++;
            vector<D> clr; D wall; clearances(Q, clr, wall);
            D mn = *min_element(clr.begin(), clr.end());
            if (mn < -1e-9 || wall < -1e-9) continue;
            if (Q.s < cur.s - 1e-9) { cur = Q; stale = 0; acc++;
                lock_guard<mutex> lk(mu);
                if (Q.s < gbest.s - 1e-10) { gbest = Q; writePacks(outfn, {gbest});
                    fprintf(stderr, "[%5.0fs] thread %d move %d: NEW BEST s=%.12f (hops %ld)\n", chrono::duration<double>(chrono::steady_clock::now()-t0).count(), w, kind, Q.s, (long)hops); }
            } else if (++stale > 60) {   // restart chain from global best or a random start
                lock_guard<mutex> lk(mu); cur = (rng() % 2) ? gbest : starts[rng() % starts.size()]; stale = 0;
            }
        }
    });
    for (auto& x : th) x.join();
    fprintf(stderr, "BH done: %ld hops, %ld accepted, best s=%.12f\n", (long)hops, (long)acc, gbest.s);
    writePacks(outfn, {gbest});
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: polish in.json out.json [topk]\n"); return 1; }
    ifstream in(argv[1]); stringstream ss; ss << in.rdbuf(); string t = ss.str();
    size_t e; auto dv = getArr(t, 0, "d", e); (void)dv;
    { size_t p = t.find("\"d\":"); d = atoi(t.c_str()+p+4); p = t.find("\"n\":"); n = atoi(t.c_str()+p+4); }
    int topk = argc > 3 ? atoi(argv[3]) : 3;
    FIXS = getenv("FIXS") != nullptr; size_t pos = t.find("\"results\"");
    vector<Pack> outs;
    if (getenv("BH")) {
        vector<Pack> starts;
        for (int r = 0; r < topk; r++) { Pack P; size_t e1, e2, e3; P.c = getArr(t, pos, "centers", e1); if (e1 == string::npos) break;
            P.R = getArr(t, e1, "R", e2); P.u = getArr(t, e2, "u", e3); pos = e3; size_t sp = t.rfind("\"s\":", e1); P.s = atof(t.c_str()+sp+4);
            starts.push_back(polish(P, 0)); }
        int T = getenv("THREADS") ? atoi(getenv("THREADS")) : (int)thread::hardware_concurrency();
        return basinHop(starts, argv[2], atof(getenv("BH")), T);
    }
    for (int r = 0; r < topk; r++) {
        Pack P; size_t e1, e2, e3, e4; P.c = getArr(t, pos, "centers", e1); if (e1 == string::npos) break;
        P.R = getArr(t, e1, "R", e2); P.u = getArr(t, e2, "u", e3); pos = e3;
        size_t sp = t.rfind("\"s\":", e1); P.s = atof(t.c_str()+sp+4); (void)e4;
        Pack Q = polish(P, argc > 4);
        vector<D> clr; D wall; clearances(Q, clr, wall);
        fprintf(stderr, "raw s=%.9f -> polished s=%.12f  min pair clearance=%.2e wall=%.2e\n", P.s, Q.s, *min_element(clr.begin(), clr.end()), wall);
        outs.push_back(Q);
    }
    FILE* f = fopen(argv[2], "w"); fprintf(f, "{\"d\":%d,\"n\":%d,\"results\":[", d, n);
    for (size_t q = 0; q < outs.size(); q++) { if (q) fprintf(f, ","); const Pack& P = outs[q];
        fprintf(f, "{\"s\":%.15g,\"centers\":[", P.s); for (size_t z = 0; z < P.c.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.c[z]);
        fprintf(f, "],\"R\":["); for (size_t z = 0; z < P.R.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.R[z]);
        fprintf(f, "],\"u\":["); for (size_t z = 0; z < P.u.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.u[z]); fprintf(f, "]}"); }
    fprintf(f, "]}\n"); fclose(f);
}
