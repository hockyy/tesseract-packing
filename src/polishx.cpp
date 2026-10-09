// polishx: polish.cpp core (augmented Lagrangian + L-BFGS) plus
//   * multithreaded polishing of many seeds (THREADS)
//   * per-cube stress (sum of final contact multipliers) kept with each packing
//   * hole-based insertion ladder (INSERT=k: n-packing -> (n+k) seeds)
//   * basin hopping with extra moves (MOVES bitmask, HOPS budget):
//       0 rotate+nudge 1 cube   1 rotate+nudge 2 cubes   2 relocate random   3 swap two
//       4 remove most-stressed cube(s) and reinsert at best hole
//       5 remove random cube and reinsert at best hole
//       6 pinwheel: rotate a cube and its contact neighbours in a common 2-plane
//       7 rattler shuffle: move a zero-stress cube into the best hole + nudge a stressed one
// build: g++ -O3 -march=native -pthread polishx.cpp -o polishx.exe
// run:   polishx.exe in.json out.json [topk]
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
    vector<int> pi, pj; vector<D> sg;
    vector<D> lp, lw, lo; D mu = 10;
    int nd, ndd, A, N;
    void setup(int active) { A = active; nd = n * d; ndd = n * d * d; N = nd + ndd + A * d + 1; }
    D eval(const vector<D>& x, vector<D>* g, vector<D>* gp = nullptr, vector<D>* gw = nullptr, vector<D>* eo = nullptr) {
        D L = FIXS ? 0.0 : x[N - 1]; if (g) { fill(g->begin(), g->end(), 0.0); (*g)[N - 1] = FIXS ? 0.0 : 1; }
        const D* c = &x[0]; const D* R = &x[nd]; const D* U = &x[nd + ndd]; D s = x[N - 1];
        const D margin = 1e-12;
        for (int p = 0; p < A; p++) {
            int i = pi[p], j = pj[p]; const D* up = U + p*d; D nr = 0; for (int k = 0; k < d; k++) nr += up[k]*up[k]; nr = sqrt(nr);
            D uh[8]; for (int k = 0; k < d; k++) uh[k] = up[k]/nr;
            const D* Ri = R + i*d*d; const D* Rj = R + j*d*d;
            D ai[8], aj[8], hs = 0, uw = 0;
            for (int m = 0; m < d; m++) { D a = 0, b = 0; for (int k = 0; k < d; k++) { a += Ri[k*d+m]*uh[k]; b += Rj[k*d+m]*uh[k]; } ai[m] = a; aj[m] = b; hs += 0.5*(fabs(a)+fabs(b)); }
            for (int k = 0; k < d; k++) uw += uh[k]*(c[j*d+k]-c[i*d+k]);
            D gv = sg[p]*uw - hs - margin;
            if (gp) (*gp)[p] = gv;
            D w = -max(0.0, lp[p] - mu*gv);
            L += (w*w - lp[p]*lp[p])/(2*mu);
            if (g && w != 0) {
                D gh[8];
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

struct Pack { vector<D> c, R, u; D s; vector<D> stress; };

static inline int pidx(int i, int j) { if (i > j) swap(i, j); return i*n - i*(i+1)/2 + (j - i - 1); }

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

static D fitBox(Pack& P) {
    D sn = 0;
    for (int k = 0; k < d; k++) { D lo = 1e18, hi = -1e18;
        for (int i = 0; i < n; i++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(P.R[i*d*d+k*d+m]); lo = min(lo, P.c[i*d+k]-0.5*l1); hi = max(hi, P.c[i*d+k]+0.5*l1); }
        for (int i = 0; i < n; i++) P.c[i*d+k] -= lo; sn = max(sn, hi-lo); }
    P.s = sn; return sn;
}

static Pack polish(Pack P, int verbose) {
    int NP = n*(n-1)/2;
    P.stress.assign(n, 0);
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
        copy(x.begin(), x.begin()+pb.nd, P.c.begin()); copy(x.begin()+pb.nd, x.begin()+pb.nd+pb.ndd, P.R.begin());
        for (int a = 0; a < pb.A; a++) for (int k = 0; k < d; k++) P.u[act[a]*d+k] = x[pb.nd+pb.ndd+a*d+k];
        P.s = x[pb.N-1];
        for (int i = 0; i < n; i++) gs(&P.R[i*d*d]);
        // stress = sum of contact multipliers (pairs + walls) of the last round
        fill(P.stress.begin(), P.stress.end(), 0.0);
        for (int a = 0; a < pb.A; a++) { P.stress[pb.pi[a]] += pb.lp[a]; P.stress[pb.pj[a]] += pb.lp[a]; }
        for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) P.stress[i] += pb.lw[(i*d+k)*2+1];   // upper walls only (lower ones are translation gauge)
        clearances(P, clr, wall); int bad = 0; for (int q = 0; q < NP; q++) if (clr[q] < -1e-7) bad++;
        if (!bad) break;
        p = 0; for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) if (clr[p] < -1e-7) { D nr = 0; for (int k = 0; k < d; k++) { D w = P.c[j*d+k]-P.c[i*d+k]; P.u[p*d+k] = w; nr += w*w; } nr = sqrt(nr)+1e-12; for (int k = 0; k < d; k++) P.u[p*d+k] /= nr; }
        P.s += 0.05;
    }
    vector<D> clr; D wall; clearances(P, clr, wall); D mn = *min_element(clr.begin(), clr.end());
    if (mn < 0) { D lam = 1 + 2*(-mn) + 1e-13; for (auto& v : P.c) v *= lam; }
    for (int i = 0; i < n; i++) gs(&P.R[i*d*d]);
    fitBox(P);
    return P;
}

static void writePacks(const string& fn, const vector<Pack>& outs) {
    string tmp = fn + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w"); fprintf(f, "{\"d\":%d,\"n\":%d,\"results\":[", d, n);
    for (size_t q = 0; q < outs.size(); q++) { if (q) fprintf(f, ","); const Pack& P = outs[q];
        fprintf(f, "{\"s\":%.15g,\"centers\":[", P.s); for (size_t z = 0; z < P.c.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.c[z]);
        fprintf(f, "],\"R\":["); for (size_t z = 0; z < P.R.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.R[z]);
        fprintf(f, "],\"u\":["); for (size_t z = 0; z < P.u.size(); z++) fprintf(f, "%s%.17g", z?",":"", P.u[z]); fprintf(f, "]}"); }
    fprintf(f, "]}\n"); fclose(f);
    remove(fn.c_str()); rename(tmp.c_str(), fn.c_str());
}

// rotation by th in the 2-plane spanned by orthonormal a,b (left multiplication)
static void rotPlane(D* R, const D* a, const D* b, D th) {
    vector<D> G(d*d); for (int k = 0; k < d; k++) for (int m = 0; m < d; m++) G[k*d+m] = (k == m) + (cos(th)-1)*(a[k]*a[m]+b[k]*b[m]) + sin(th)*(b[k]*a[m]-a[k]*b[m]);
    vector<D> T(d*d, 0); for (int k = 0; k < d; k++) for (int m = 0; m < d; m++) for (int q = 0; q < d; q++) T[k*d+m] += G[k*d+q]*R[q*d+m];
    copy(T.begin(), T.end(), R); gs(R);
}
static void randPlane(vector<D>& a, vector<D>& b, mt19937_64& rng) {
    normal_distribution<D> N01(0, 1); a.assign(d, 0); b.assign(d, 0);
    for (int k = 0; k < d; k++) { a[k] = N01(rng); b[k] = N01(rng); }
    D na = 0; for (D v : a) na += v*v; na = sqrt(na); for (D& v : a) v /= na;
    D ab = 0; for (int k = 0; k < d; k++) ab += a[k]*b[k]; for (int k = 0; k < d; k++) b[k] -= ab*a[k];
    D nb = 0; for (D v : b) nb += v*v; nb = sqrt(nb); for (D& v : b) v /= nb;
}
static void rotate(D* R, D th, mt19937_64& rng) { vector<D> a, b; randPlane(a, b, rng); rotPlane(R, a.data(), b.data(), th); }

// ---------- hole finding (others fixed) ----------
// approximate separation of a test cube (centre p, rotation Q) from cube j using the 2d face normals
static D faceSep(const D* p, const D* Q, const D* cj, const D* Rj) {
    D best = -1e18;
    for (int side = 0; side < 2; side++) for (int m = 0; m < d; m++) {
        const D* M = side ? Rj : Q; D a[8]; for (int k = 0; k < d; k++) a[k] = M[k*d+m];
        D uw = 0, h = 0; for (int k = 0; k < d; k++) uw += a[k]*(cj[k]-p[k]);
        for (int q = 0; q < d; q++) { D x = 0, y = 0; for (int k = 0; k < d; k++) { x += Q[k*d+q]*a[k]; y += Rj[k*d+q]*a[k]; } h += 0.5*(fabs(x)+fabs(y)); }
        best = max(best, fabs(uw) - h);
    }
    return best;
}
// penalty of placing a cube at (p,Q) in packing P, ignoring cube skip
static D placePen(const Pack& P, int skip, const D* p, const D* Q, D s) {
    D pen = 0;
    for (int j = 0; j < n; j++) { if (j == skip) continue;
        D dist = 0; for (int k = 0; k < d; k++) dist = max(dist, fabs(p[k]-P.c[j*d+k])); if (dist > 2.1) continue;
        D sp = faceSep(p, Q, &P.c[j*d], &P.R[j*d*d]); if (sp < 0) pen += sp*sp; }
    for (int k = 0; k < d; k++) { D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(Q[k*d+m]); D hw = 0.5*l1; D v1 = hw - p[k], v2 = p[k] + hw - s; if (v1 > 0) pen += v1*v1; if (v2 > 0) pen += v2*v2; }
    return pen;
}
// cube-norm hole value of a point
static D holeVal(const Pack& P, int skip, const D* p, D s) {
    D v = 1e18;
    for (int k = 0; k < d; k++) v = min(v, min(p[k], s - p[k]));
    for (int j = 0; j < n; j++) { if (j == skip) continue; D mx = 0;
        for (int m = 0; m < d; m++) { D a = 0; for (int k = 0; k < d; k++) a += P.R[j*d*d+k*d+m]*(p[k]-P.c[j*d+k]); mx = max(mx, fabs(a)); }
        v = min(v, mx - 0.5); }
    return v;
}
// find a good spot + orientation for cube `skip` (or for a new cube if skip<0 and the packing arrays already hold n cubes with the new one last)
static void bestSpot(const Pack& P, int skip, mt19937_64& rng, vector<D>& pbest, vector<D>& Qbest, int effort = 1) {
    uniform_real_distribution<D> U(0, 1); normal_distribution<D> N01(0, 1);
    D s = P.s; int ns = 3000*effort; vector<pair<D, vector<D>>> cand;
    vector<D> p(d);
    for (int t = 0; t < ns; t++) { for (int k = 0; k < d; k++) p[k] = U(rng)*s; D h = holeVal(P, skip, p.data(), s); cand.push_back({h, p}); }
    sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
    cand.resize(min((size_t)40, cand.size()));
    for (auto& cp : cand) { D step = 0.08; for (int it = 0; it < 120; it++) { vector<D> q = cp.second; for (int k = 0; k < d; k++) q[k] += step*N01(rng); D h = holeVal(P, skip, q.data(), s); if (h > cp.first) { cp.first = h; cp.second = q; } if (it % 40 == 39) step *= 0.5; } }
    sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
    // pick one of the top distinct holes at random (bias to the best)
    vector<int> distinct;
    for (int a = 0; a < (int)cand.size() && distinct.size() < 6; a++) { bool ok = true; for (int b : distinct) { D dm = 0; for (int k = 0; k < d; k++) dm = max(dm, fabs(cand[a].second[k]-cand[b].second[k])); if (dm < 0.3) ok = false; } if (ok) distinct.push_back(a); }
    int pick = distinct[min((int)distinct.size()-1, (int)(U(rng)*U(rng)*distinct.size()))];
    vector<D> p0 = cand[pick].second;
    // orientation candidates: identity, copies of others' orientations, random; local refine of (p,Q) on face-separation penalty
    D bestPen = 1e18;
    vector<D> Q(d*d);
    int nOri = 12;
    for (int o = 0; o < nOri; o++) {
        if (o == 0) { for (int q = 0; q < d*d; q++) Q[q] = (q % (d+1) == 0); }
        else if (o < 7) { int j = rng() % n; if (j == skip) j = (j+1) % n; copy(P.R.begin()+j*d*d, P.R.begin()+(j+1)*d*d, Q.begin()); }
        else { for (auto& v : Q) v = N01(rng); gs(Q.data()); }
        vector<D> pp = p0; D pen = placePen(P, skip, pp.data(), Q.data(), s);
        D st = 0.05, sr = 0.15;
        for (int it = 0; it < 150; it++) {
            vector<D> p2 = pp, Q2 = Q; for (int k = 0; k < d; k++) p2[k] += st*N01(rng);
            if (U(rng) < 0.5) rotate(Q2.data(), sr*N01(rng), rng);
            D pe = placePen(P, skip, p2.data(), Q2.data(), s); if (pe < pen) { pen = pe; pp = p2; Q = Q2; }
            if (it % 50 == 49) { st *= 0.5; sr *= 0.5; }
        }
        if (pen < bestPen) { bestPen = pen; pbest = pp; Qbest = Q; }
    }
}


// ---------- soft growth at fixed side: penalty energy with per-cube scale a_i, Adam ----------
static D softEnergy(const Pack& P, const vector<D>& a, D s, vector<D>* gc, vector<D>* gR, vector<D>* gu) {
    D E = 0; if (gc) { fill(gc->begin(), gc->end(), 0); fill(gR->begin(), gR->end(), 0); fill(gu->begin(), gu->end(), 0); }
    int p = 0; D ai[8], aj[8];
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) {
        const D* ci = &P.c[i*d]; const D* cj = &P.c[j*d];
        D dm = 0; for (int k = 0; k < d; k++) dm = max(dm, fabs(cj[k]-ci[k])); if (dm > 0.87*(a[i]+a[j]) + 0.05) continue;
        const D* u = &P.u[p*d]; const D* Ri = &P.R[i*d*d]; const D* Rj = &P.R[j*d*d];
        D si = 0, sj = 0, uw = 0;
        for (int m = 0; m < d; m++) { D x = 0, y = 0; for (int k = 0; k < d; k++) { x += Ri[k*d+m]*u[k]; y += Rj[k*d+m]*u[k]; } ai[m] = x; aj[m] = y; si += fabs(x); sj += fabs(y); }
        for (int k = 0; k < d; k++) uw += u[k]*(cj[k]-ci[k]);
        D sg = uw >= 0 ? 1.0 : -1.0;
        D viol = 0.5*(a[i]*si + a[j]*sj) - sg*uw;
        if (viol > 0) { E += viol*viol;
            if (gc) { D f = 2*viol;
                for (int k = 0; k < d; k++) { D g = -sg*(cj[k]-ci[k]);
                    for (int m = 0; m < d; m++) { g += 0.5*(a[i]*sgn(ai[m])*Ri[k*d+m] + a[j]*sgn(aj[m])*Rj[k*d+m]);
                        (*gR)[i*d*d+k*d+m] += f*0.5*a[i]*sgn(ai[m])*u[k]; (*gR)[j*d*d+k*d+m] += f*0.5*a[j]*sgn(aj[m])*u[k]; }
                    (*gu)[p*d+k] += f*g; (*gc)[j*d+k] += -f*sg*u[k]; (*gc)[i*d+k] += f*sg*u[k]; } } }
    }
    for (int i = 0; i < n; i++) for (int k = 0; k < d; k++) {
        const D* R = &P.R[i*d*d]; D l1 = 0; for (int m = 0; m < d; m++) l1 += fabs(R[k*d+m]); D hw = 0.5*a[i]*l1, x = P.c[i*d+k];
        D v1 = hw - x, v2 = x + hw - s;
        if (v1 > 0) { E += v1*v1; if (gc) { (*gc)[i*d+k] -= 2*v1; for (int m = 0; m < d; m++) (*gR)[i*d*d+k*d+m] += 2*v1*0.5*a[i]*sgn(R[k*d+m]); } }
        if (v2 > 0) { E += v2*v2; if (gc) { (*gc)[i*d+k] += 2*v2; for (int m = 0; m < d; m++) (*gR)[i*d*d+k*d+m] += 2*v2*0.5*a[i]*sgn(R[k*d+m]); } }
    }
    return E;
}
struct AdamV { vector<D> m, v; long k = 0; void reset(size_t N) { m.assign(N, 0); v.assign(N, 0); k = 0; }
    void step(vector<D>& x, const vector<D>& g, D lr) { k++; D c1 = 1 - pow(0.9, k), c2 = 1 - pow(0.999, k);
        for (size_t q = 0; q < x.size(); q++) { m[q] = 0.9*m[q] + 0.1*g[q]; v[q] = 0.999*v[q] + 0.001*g[q]*g[q]; x[q] -= lr*(m[q]/c1)/(sqrt(v[q]/c2) + 1e-12); } } };
// grow cubes in `grow` from scale a0 to 1 inside fixed side s; returns final energy
static D softGrow(Pack& P, const vector<int>& growIdx, D a0, D s, int K) {
    vector<D> a(n, 1.0); for (int i : growIdx) a[i] = a0;
    vector<D> gc(P.c.size()), gR(P.R.size()), gu(P.u.size()); AdamV Ac, AR, Au; Ac.reset(gc.size()); AR.reset(gR.size()); Au.reset(gu.size());
    D E = 0; int NP = n*(n-1)/2;
    for (int it = 0; it < K; it++) {
        D f = (D)it/K; D t = min(1.0, f/0.6); for (int i : growIdx) a[i] = a0 + (1-a0)*t;
        D lr = 0.004*(1 - 0.95*f);
        E = softEnergy(P, a, s, &gc, &gR, &gu);
        if (t >= 1 && E < 1e-20) break;
        Ac.step(P.c, gc, lr); AR.step(P.R, gR, lr); Au.step(P.u, gu, lr);
        for (int i = 0; i < n; i++) gs(&P.R[i*d*d]);
        for (int q = 0; q < NP; q++) { D nr = 0; for (int k = 0; k < d; k++) nr += P.u[q*d+k]*P.u[q*d+k]; nr = sqrt(nr); if (nr < 1e-12) { P.u[q*d] = 1; nr = 1; } for (int k = 0; k < d; k++) P.u[q*d+k] /= nr; }
    }
    for (int i : growIdx) a[i] = 1; P.s = s;
    return softEnergy(P, a, s, nullptr, nullptr, nullptr);
}

static void resetAxes(Pack& P, const vector<int>& moved) {
    int p = 0;
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) {
        if (find(moved.begin(), moved.end(), i) == moved.end() && find(moved.begin(), moved.end(), j) == moved.end()) continue;
        D nr = 0; for (int k = 0; k < d; k++) { D w = P.c[j*d+k]-P.c[i*d+k]; P.u[p*d+k] = w; nr += w*w; } nr = sqrt(nr)+1e-12; for (int k = 0; k < d; k++) P.u[p*d+k] /= nr;
    }
}

static int MOVEMASK = 0x0F;
static Pack perturb(const Pack& P0, mt19937_64& rng, int& kind) {
    Pack P = P0; uniform_real_distribution<D> U(0, 1); normal_distribution<D> N01(0, 1);
    vector<int> moved;
    do kind = rng() % 10; while (!((MOVEMASK >> kind) & 1));
    if (kind == 0 || kind == 1) {
        int cnt = kind == 0 ? 1 : 2;
        for (int q = 0; q < cnt; q++) { int i = rng() % n; moved.push_back(i);
            rotate(&P.R[i*d*d], (5 + 40*U(rng))*M_PI/180*(U(rng) < .5 ? -1 : 1), rng);
            for (int k = 0; k < d; k++) P.c[i*d+k] += 0.1*N01(rng); }
    } else if (kind == 2) {
        int i = rng() % n; moved.push_back(i);
        for (int k = 0; k < d; k++) P.c[i*d+k] = 0.5 + (P.s-1)*U(rng);
        for (int q = 0; q < d*d; q++) P.R[i*d*d+q] = N01(rng); gs(&P.R[i*d*d]);
    } else if (kind == 3) {
        int i = rng() % n, j = rng() % n; if (i == j) j = (j+1) % n; moved.push_back(i); moved.push_back(j);
        for (int k = 0; k < d; k++) swap(P.c[i*d+k], P.c[j*d+k]);
    } else if (kind == 4 || kind == 5) {
        int i;
        if (kind == 4 && !P.stress.empty()) {   // roulette on stress
            D tot = 0; for (D v : P.stress) tot += v; D r = U(rng)*tot; i = 0; for (; i < n-1; i++) { r -= P.stress[i]; if (r <= 0) break; }
        } else i = rng() % n;
        moved.push_back(i);
        vector<D> p, Q; bestSpot(P, i, rng, p, Q);
        for (int k = 0; k < d; k++) P.c[i*d+k] = p[k]; copy(Q.begin(), Q.end(), P.R.begin()+i*d*d);
    } else if (kind == 6) {   // pinwheel: cube i and neighbours rotate in the same plane about their joint centre
        int i = rng() % n; vector<D> a, b; randPlane(a, b, rng);
        if (U(rng) < 0.5) { a.assign(d, 0); b.assign(d, 0); int x = rng() % d, y = rng() % d; if (x == y) y = (y+1) % d; a[x] = 1; b[y] = 1; }
        D th = (3 + 12*U(rng))*M_PI/180*(U(rng) < .5 ? -1 : 1);
        vector<int> grp{i}; for (int j = 0; j < n; j++) if (j != i) { D dm = 0; for (int k = 0; k < d; k++) dm = max(dm, fabs(P.c[j*d+k]-P.c[i*d+k])); if (dm < 1.3) grp.push_back(j); }
        vector<D> cen(d, 0); for (int j : grp) for (int k = 0; k < d; k++) cen[k] += P.c[j*d+k]/grp.size();
        for (int j : grp) { moved.push_back(j); D sgn = (j == i) ? 1 : -1;   // alternate sense like gears
            rotPlane(&P.R[j*d*d], a.data(), b.data(), sgn*th);
            vector<D> v(d); for (int k = 0; k < d; k++) v[k] = P.c[j*d+k]-cen[k];
            D va = 0, vb = 0; for (int k = 0; k < d; k++) { va += v[k]*a[k]; vb += v[k]*b[k]; }
            D c2 = cos(0.3*th), s2 = sin(0.3*th);
            for (int k = 0; k < d; k++) P.c[j*d+k] = cen[k] + v[k] + (va*(c2-1) - vb*s2)*a[k] + (va*s2 + vb*(c2-1))*b[k];
        }
    } else if (kind == 8 || kind == 9) {   // remove a cube (8: stress roulette, 9: random), re-grow it at the best hole at fixed side
        int i;
        if (kind == 8 && !P.stress.empty()) { D tot = 0; for (D v : P.stress) tot += v; D r = U(rng)*tot; i = 0; for (; i < n-1; i++) { r -= P.stress[i]; if (r <= 0) break; } }
        else i = rng() % n;
        vector<D> p, Q; bestSpot(P, i, rng, p, Q);
        for (int k = 0; k < d; k++) P.c[i*d+k] = p[k]; copy(Q.begin(), Q.end(), P.R.begin()+i*d*d);
        resetAxes(P, {i});
        D starget = P.s*(1 - 0.002*U(rng));
        softGrow(P, {i}, 0.3, starget, 6000);
        for (auto& v : P.c) v += 1e-4*N01(rng);
        return P;
    } else {   // kind 7: rattler shuffle — a zero-stress cube goes to the best hole, a stressed cube gets nudged into its old place's direction
        vector<int> rat; D mx = 0; for (D v : P.stress) mx = max(mx, v);
        for (int i = 0; i < n; i++) if (P.stress.empty() || P.stress[i] < 1e-3*mx) rat.push_back(i);
        int i = rat.empty() ? (int)(rng() % n) : rat[rng() % rat.size()]; moved.push_back(i);
        vector<D> oldc(P.c.begin()+i*d, P.c.begin()+(i+1)*d);
        vector<D> p, Q; bestSpot(P, i, rng, p, Q);
        for (int k = 0; k < d; k++) P.c[i*d+k] = p[k]; copy(Q.begin(), Q.end(), P.R.begin()+i*d*d);
        // nearest stressed cube to the vacated spot moves part-way into it
        int jb = -1; D bd = 1e18; for (int j = 0; j < n; j++) if (j != i && !P.stress.empty() && P.stress[j] >= 1e-3*mx) { D dd = 0; for (int k = 0; k < d; k++) dd += pow(P.c[j*d+k]-oldc[k], 2); if (dd < bd) { bd = dd; jb = j; } }
        if (jb >= 0) { moved.push_back(jb); D f = 0.3 + 0.5*U(rng); for (int k = 0; k < d; k++) P.c[jb*d+k] += f*(oldc[k]-P.c[jb*d+k]); rotate(&P.R[jb*d*d], (5 + 20*U(rng))*M_PI/180, rng); }
    }
    for (auto& v : P.c) v += 0.01*N01(rng);
    resetAxes(P, moved);
    return P;
}

static vector<Pack> readPacks(const string& t, int topk) {
    vector<Pack> v; size_t pos = t.find("\"results\"");
    for (int r = 0; r < topk; r++) { Pack P; size_t e1, e2, e3; P.c = getArr(t, pos, "centers", e1); if (e1 == string::npos) break;
        P.R = getArr(t, e1, "R", e2); P.u = getArr(t, e2, "u", e3); pos = e3; size_t sp = t.rfind("\"s\":", e1); P.s = atof(t.c_str()+sp+4); v.push_back(P); }
    return v;
}

static vector<Pack> polishAll(const vector<Pack>& in, int T) {
    vector<Pack> out(in.size()); atomic<int> next(0); mutex mu; vector<thread> th;
    for (int w = 0; w < T; w++) th.emplace_back([&]() { for (;;) { int k = next++; if (k >= (int)in.size()) break; out[k] = polish(in[k], 0);
        vector<D> clr; D wall; clearances(out[k], clr, wall); lock_guard<mutex> lk(mu);
        fprintf(stderr, "seed %d: raw s=%.6f -> polished s=%.12f  clr=%.1e wall=%.1e\n", k, in[k].s, out[k].s, *min_element(clr.begin(), clr.end()), wall); } });
    for (auto& x : th) x.join();
    sort(out.begin(), out.end(), [](const Pack& a, const Pack& b) { return a.s < b.s; });
    return out;
}

// add k cubes to packing P0 (n currently counts the NEW total)
static Pack insertCubes(const Pack& P0, int k, mt19937_64& rng) {
    int n0 = n - k; Pack P = P0; normal_distribution<D> N01(0, 1);
    P.c.resize(n*d); P.R.resize(n*d*d);
    // u for the new pair list: rebuild index mapping
    vector<D> u(n*(n-1)/2*d);
    { int p = 0; for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++, p++) {
        if (j < n0) { int q = i*n0 - i*(i+1)/2 + (j - i - 1); for (int z = 0; z < d; z++) u[p*d+z] = P0.u[q*d+z]; } } }
    P.u = u;
    for (int a = 0; a < k; a++) {
        int idx = n0 + a; int saveN = n; n = idx;   // hole search among first idx cubes
        Pack T = P; T.c.resize(idx*d); T.R.resize(idx*d*d);
        vector<D> p, Q; bestSpot(T, -1, rng, p, Q, 2);
        n = saveN;
        for (int z = 0; z < d; z++) P.c[idx*d+z] = p[z]; copy(Q.begin(), Q.end(), P.R.begin()+idx*d*d);
    }
    vector<int> moved; for (int a = 0; a < k; a++) moved.push_back(n0 + a);
    for (auto& v : P.c) v += 0.003*N01(rng);
    resetAxes(P, moved);
    return P;
}

static int basinHop(vector<Pack> starts, const string& outfn, double seconds, long maxHops, int T) {
    mutex mu; Pack gbest = starts[0]; for (auto& p : starts) if (p.s < gbest.s) gbest = p;
    fprintf(stderr, "BH start: best s=%.12f, %zu starting layouts, %d threads, %.0fs, hops<=%ld, moves=0x%x\n", gbest.s, starts.size(), T, seconds, maxHops, MOVEMASK);
    auto t0 = chrono::steady_clock::now(); atomic<long> hops(0), acc(0);
    vector<long> tried(10, 0), improved(10, 0);
    vector<thread> th;
    for (int w = 0; w < T; w++) th.emplace_back([&, w]() {
        mt19937_64 rng(1234567ULL*(w+1) + (getenv("SEED") ? atoll(getenv("SEED")) : (unsigned long long)chrono::steady_clock::now().time_since_epoch().count()));
        Pack cur = starts[w % starts.size()];
        int stale = 0;
        while (chrono::duration<double>(chrono::steady_clock::now()-t0).count() < seconds && hops < maxHops) {
            int kind; Pack Q = polish(perturb(cur, rng, kind), 0); long h = ++hops;
            vector<D> clr; D wall; clearances(Q, clr, wall);
            D mn = *min_element(clr.begin(), clr.end());
            { lock_guard<mutex> lk(mu); tried[kind]++; }
            if (mn < -1e-9 || wall < -1e-9) continue;
            if (Q.s < cur.s - 1e-9) { cur = Q; stale = 0; acc++;
                lock_guard<mutex> lk(mu); improved[kind]++;
                if (Q.s < gbest.s - 1e-10) { gbest = Q; writePacks(outfn, {gbest});
                    fprintf(stderr, "[%5.0fs] thread %d move %d: NEW BEST s=%.12f (hops %ld)\n", chrono::duration<double>(chrono::steady_clock::now()-t0).count(), w, kind, Q.s, h); }
            } else if (++stale > 60) {
                lock_guard<mutex> lk(mu); cur = (rng() % 2) ? gbest : starts[rng() % starts.size()]; stale = 0;
            }
        }
    });
    for (auto& x : th) x.join();
    fprintf(stderr, "BH done: %ld hops, %ld accepted, best s=%.12f  [%.0fs]\n", (long)hops, (long)acc, gbest.s, chrono::duration<double>(chrono::steady_clock::now()-t0).count());
    for (int k = 0; k < 10; k++) if (tried[k]) fprintf(stderr, "  move %d: tried %ld improved %ld\n", k, tried[k], improved[k]);
    writePacks(outfn, {gbest});
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: polishx in.json out.json [topk]\n"); return 1; }
    ifstream in(argv[1]); stringstream ss; ss << in.rdbuf(); string t = ss.str();
    { size_t p = t.find("\"d\":"); d = atoi(t.c_str()+p+4); p = t.find("\"n\":"); n = atoi(t.c_str()+p+4); }
    int topk = argc > 3 ? atoi(argv[3]) : 3;
    FIXS = getenv("FIXS") != nullptr;
    if (getenv("MOVES")) MOVEMASK = strtol(getenv("MOVES"), 0, 0);
    int T = getenv("THREADS") ? atoi(getenv("THREADS")) : 8; T = min(T, 8);
    vector<Pack> base = readPacks(t, topk);
    vector<Pack> starts;
    if (getenv("INSERT")) {     // ladder: n -> n+k
        int k = atoi(getenv("INSERT")); int nseed = getenv("NSEED") ? atoi(getenv("NSEED")) : 16;
        mt19937_64 rng(getenv("SEED") ? atoll(getenv("SEED")) : 1);
        int n0 = n; n = n0 + k;
        vector<Pack> seeds;
        D f0 = getenv("INFLATE") ? atof(getenv("INFLATE")) : 1.0, f1 = getenv("INFLATE2") ? atof(getenv("INFLATE2")) : f0;
        for (int q = 0; q < nseed; q++) { n = n0; Pack B = base[q % base.size()]; D f = f0 + (f1-f0)*q/max(1, nseed-1);
            for (auto& v : B.c) v *= f; B.s *= f; n = n0 + k; Pack S = insertCubes(B, k, rng);
            if (getenv("GROWIN")) { vector<int> gi; for (int a = 0; a < k; a++) gi.push_back(n0+a);
                D E = softGrow(S, gi, 0.3, B.s*(getenv("GSLACK") ? atof(getenv("GSLACK")) : 1.0), getenv("GSTEPS") ? atoi(getenv("GSTEPS")) : 30000);
                fprintf(stderr, "grow seed %d: s=%.6f E=%.3e\n", q, S.s, E); }
            seeds.push_back(S); }
        starts = polishAll(seeds, T);
    } else if (getenv("NOPOLISH")) { starts = base; for (auto& P : starts) P.stress.assign(n, 1.0); }
    else starts = polishAll(base, T);
    if (getenv("BH")) {
        long maxHops = getenv("HOPS") ? atol(getenv("HOPS")) : LONG_MAX;
        int keep = getenv("KEEP") ? atoi(getenv("KEEP")) : (int)starts.size(); if ((int)starts.size() > keep) starts.resize(keep);
        return basinHop(starts, argv[2], atof(getenv("BH")), maxHops, T);
    }
    writePacks(argv[2], starts);
    return 0;
}
