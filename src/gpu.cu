// GPU soft-to-rigid packing search: one thread block == one random start, all in double precision.
// build (from a VS dev prompt or with cl on PATH):  nvcc -O3 -arch=sm_89 gpu.cu -o gpu.exe
// run:   gpu.exe d n total_starts soft_steps seed out.json [batch]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
typedef double D;
#define BS 128
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

struct Params { int d, n, P, steps; D sstart, send; unsigned long long seed; };

__device__ __forceinline__ D sgnf(D x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

__device__ void zero_vec(D* g, int K) { for (int q = threadIdx.x; q < K; q += BS) g[q] = 0; }

// energy (+ gradient into g if wantg). Returns total energy in *sE after the final sync.
__device__ void energy_grad(const D* x, D* g, int wantg, D s, D t, const Params& pr, const int* PI, const int* PJ, D* sE) {
    int d = pr.d, n = pr.n, P = pr.P, nd = n * d, ndd = n * d * d;
    const D* c = x; const D* R = x + nd; const D* U = x + nd + ndd;
    D* gc = g; D* gR = g + nd; D* gu = g + nd + ndd;
    if (wantg) zero_vec(g, nd + ndd + P * d);
    if (threadIdx.x == 0) *sE = 0;
    __syncthreads();
    D local = 0;
    for (int p = threadIdx.x; p < P; p += BS) {
        int i = PI[p], j = PJ[p];
        const D* u = U + p * d; const D* Ri = R + i * d * d; const D* Rj = R + j * d * d;
        const D* ci = c + i * d; const D* cj = c + j * d;
        D ai[4], aj[4], si = 0, sj = 0, uw = 0;
        for (int m = 0; m < d; m++) {
            D a = 0, b = 0; for (int k = 0; k < d; k++) { a += Ri[k * d + m] * u[k]; b += Rj[k * d + m] * u[k]; }
            ai[m] = a; aj[m] = b; si += fabs(a); sj += fabs(b);
        }
        for (int k = 0; k < d; k++) uw += u[k] * (cj[k] - ci[k]);
        D sg = uw >= 0 ? 1.0 : -1.0;
        D viol = (1 - t) + 0.5 * t * (si + sj) - sg * uw;
        if (viol > 0) {
            local += viol * viol;
            if (wantg) {
                D f = 2 * viol;
                for (int k = 0; k < d; k++) {
                    D gk = -sg * (cj[k] - ci[k]);
                    for (int m = 0; m < d; m++) {
                        gk += 0.5 * t * (sgnf(ai[m]) * Ri[k * d + m] + sgnf(aj[m]) * Rj[k * d + m]);
                        atomicAdd(&gR[i * d * d + k * d + m], f * 0.5 * t * sgnf(ai[m]) * u[k]);
                        atomicAdd(&gR[j * d * d + k * d + m], f * 0.5 * t * sgnf(aj[m]) * u[k]);
                    }
                    gu[p * d + k] = f * gk;
                    atomicAdd(&gc[j * d + k], f * (-sg * u[k]));
                    atomicAdd(&gc[i * d + k], f * (sg * u[k]));
                }
            }
        }
    }
    for (int q = threadIdx.x; q < n * d; q += BS) {
        int i = q / d, k = q % d; const D* Rr = R + i * d * d; D l1 = 0;
        for (int m = 0; m < d; m++) l1 += fabs(Rr[k * d + m]);
        D hw = 0.5 * ((1 - t) + t * l1), xx = c[q];
        D v1 = hw - xx, v2 = xx + hw - s;
        if (v1 > 0) { local += v1 * v1; if (wantg) { atomicAdd(&gc[q], -2 * v1); for (int m = 0; m < d; m++) atomicAdd(&gR[i * d * d + k * d + m], 2 * v1 * 0.5 * t * sgnf(Rr[k * d + m])); } }
        if (v2 > 0) { local += v2 * v2; if (wantg) { atomicAdd(&gc[q], 2 * v2); for (int m = 0; m < d; m++) atomicAdd(&gR[i * d * d + k * d + m], 2 * v2 * 0.5 * t * sgnf(Rr[k * d + m])); } }
    }
    atomicAdd(sE, local);
    __syncthreads();
}

__device__ void adam_step(D* x, const D* g, D* m, D* v, long k, D lr, const Params& pr) {
    int d = pr.d, n = pr.n, P = pr.P, nd = n * d, ndd = n * d * d, K = nd + ndd + P * d;
    D c1 = 1 - pow(0.9, (D)k), c2 = 1 - pow(0.999, (D)k);
    for (int q = threadIdx.x; q < K; q += BS) {
        m[q] = 0.9 * m[q] + 0.1 * g[q]; v[q] = 0.999 * v[q] + 0.001 * g[q] * g[q];
        x[q] -= lr * (m[q] / c1) / (sqrt(v[q] / c2) + 1e-12);
    }
    __syncthreads();
    if (threadIdx.x < n) {   // Gram-Schmidt on columns of R_i
        D* R = x + nd + threadIdx.x * d * d;
        for (int mm = 0; mm < d; mm++) {
            for (int p = 0; p < mm; p++) { D dot = 0; for (int k2 = 0; k2 < d; k2++) dot += R[k2 * d + mm] * R[k2 * d + p]; for (int k2 = 0; k2 < d; k2++) R[k2 * d + mm] -= dot * R[k2 * d + p]; }
            D nr = 0; for (int k2 = 0; k2 < d; k2++) nr += R[k2 * d + mm] * R[k2 * d + mm]; nr = sqrt(nr); for (int k2 = 0; k2 < d; k2++) R[k2 * d + mm] /= nr;
        }
    }
    for (int p = threadIdx.x; p < P; p += BS) {
        D* u = x + nd + ndd + p * d; D nr = 0; for (int k2 = 0; k2 < d; k2++) nr += u[k2] * u[k2]; nr = sqrt(nr);
        if (nr < 1e-12) { u[0] = 1; nr = 1; } for (int k2 = 0; k2 < d; k2++) u[k2] /= nr;
    }
    __syncthreads();
}

__device__ unsigned long long xs(unsigned long long& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
__device__ D urand(unsigned long long& s) { return (xs(s) >> 11) * (1.0 / 9007199254740992.0); }
__device__ D nrand(unsigned long long& s) { D a = urand(s) + 1e-300, b = urand(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

__global__ void search(Params pr, const int* PI, const int* PJ, D* W, D* outS, int startBase) {
    int b = blockIdx.x;
    int d = pr.d, n = pr.n, P = pr.P, nd = n * d, ndd = n * d * d, K = nd + ndd + P * d;
    D* x = W + (size_t)b * 5 * K; D* g = x + K; D* am = g + K; D* av = am + K; D* best = av + K;
    __shared__ D sE; __shared__ D ctl[4]; // ctl: 0 s, 1 delta, 2 bestS, 3 have
    __shared__ long adamk;
    if (threadIdx.x == 0) {
        unsigned long long rs = pr.seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)(startBase + b) * 0xD1B54A32D192ED03ULL + 88172645463325252ULL;
        for (int q = 0; q < 8; q++) xs(rs);
        for (int q = 0; q < nd; q++) x[q] = urand(rs) * pr.sstart;
        for (int q = 0; q < ndd; q++) x[nd + q] = nrand(rs);
        for (int i = 0; i < n; i++) {   // orthonormalise
            D* R = x + nd + i * d * d;
            for (int mm = 0; mm < d; mm++) {
                for (int p = 0; p < mm; p++) { D dot = 0; for (int k2 = 0; k2 < d; k2++) dot += R[k2 * d + mm] * R[k2 * d + p]; for (int k2 = 0; k2 < d; k2++) R[k2 * d + mm] -= dot * R[k2 * d + p]; }
                D nr = 0; for (int k2 = 0; k2 < d; k2++) nr += R[k2 * d + mm] * R[k2 * d + mm]; nr = sqrt(nr); for (int k2 = 0; k2 < d; k2++) R[k2 * d + mm] /= nr;
            }
        }
        for (int p = 0; p < P; p++) { D nr = 0; for (int k2 = 0; k2 < d; k2++) { D w = x[PJ[p] * d + k2] - x[PI[p] * d + k2]; x[nd + ndd + p * d + k2] = w; nr += w * w; } nr = sqrt(nr) + 1e-12; for (int k2 = 0; k2 < d; k2++) x[nd + ndd + p * d + k2] /= nr; }
        adamk = 0;
    }
    zero_vec(am, K); zero_vec(av, K);
    __syncthreads();
    for (int it = 0; it < pr.steps; it++) {
        D f = (D)it / (pr.steps - 1);
        D t = fmin(1.0, f / 0.8), s = pr.sstart + (pr.send - pr.sstart) * fmin(1.0, f / 0.9);
        D lr = 0.02 * (1 - 0.9 * f);
        energy_grad(x, g, 1, s, t, pr, PI, PJ, &sE);
        if (threadIdx.x == 0) adamk++;
        __syncthreads();
        adam_step(x, g, am, av, adamk, lr, pr);
    }
    // rigid shrink
    if (threadIdx.x == 0) { ctl[0] = pr.send; ctl[1] = 0.02; ctl[2] = 1e18; ctl[3] = 0; adamk = 0; }
    zero_vec(am, K); zero_vec(av, K);
    __syncthreads();
    for (int round = 0; round < 600; round++) {
        D delta = ctl[1];
        if (delta <= 2e-8) break;
        D s = ctl[0];
        D lr = fmin(3e-3, fmax(2e-7, delta * 0.3));
        D E = 1;
        for (int it = 0; it < 1500; it++) {
            energy_grad(x, g, 1, s, 1.0, pr, PI, PJ, &sE);
            E = sE;
            if (E < 1e-14) break;
            if (threadIdx.x == 0) adamk++;
            __syncthreads();
            adam_step(x, g, am, av, adamk, lr, pr);
        }
        energy_grad(x, g, 0, s, 1.0, pr, PI, PJ, &sE);
        E = sE; __syncthreads();
        if (E < 1e-14) {
            for (int q = threadIdx.x; q < K; q += BS) best[q] = x[q];
            __syncthreads();
            if (threadIdx.x == 0) { ctl[2] = s; ctl[3] = 1; ctl[0] = s - delta; }
        } else if (ctl[3] == 0) {
            if (threadIdx.x == 0) { ctl[0] = s + 0.05; adamk = 0; }
            zero_vec(am, K); zero_vec(av, K);
        } else {
            for (int q = threadIdx.x; q < K; q += BS) x[q] = best[q];
            zero_vec(am, K); zero_vec(av, K);
            __syncthreads();
            if (threadIdx.x == 0) { adamk = 0; ctl[1] = delta / 2; ctl[0] = ctl[2] - delta / 2; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) outS[b] = ctl[3] > 0 ? ctl[2] : 1e18;
}

struct Cand { D s; std::vector<D> st; };

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: gpu d n total_starts soft_steps seed out.json [batch]\n"); return 1; }
    int d = atoi(argv[1]), n = atoi(argv[2]); long total = atol(argv[3]); int steps = atoi(argv[4]);
    unsigned long long seed = strtoull(argv[5], 0, 10); std::string out = argv[6];
    int batch = argc > 7 ? atoi(argv[7]) : 1024;
    std::vector<int> hPI, hPJ; for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) { hPI.push_back(i); hPJ.push_back(j); }
    int P = hPI.size(), nd = n * d, ndd = n * d * d, K = nd + ndd + P * d;
    D vol = pow((D)n, 1.0 / d); int triv = (int)ceil(vol - 1e-9);
    Params pr{d, n, P, steps, 1.35 * vol + 0.3, std::max(vol * 1.02, (D)triv), seed};
    int *dPI, *dPJ; D *dW, *dS;
    CK(cudaMalloc(&dPI, P * sizeof(int))); CK(cudaMalloc(&dPJ, P * sizeof(int)));
    CK(cudaMemcpy(dPI, hPI.data(), P * sizeof(int), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dPJ, hPJ.data(), P * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dW, (size_t)batch * 5 * K * sizeof(D))); CK(cudaMalloc(&dS, batch * sizeof(D)));
    std::vector<Cand> top; auto t0 = std::chrono::steady_clock::now(); D gbest = 1e18;
    for (long base = 0; base < total; base += batch) {
        search<<<batch, BS>>>(pr, dPI, dPJ, dW, dS, (int)base);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<D> hS(batch); CK(cudaMemcpy(hS.data(), dS, batch * sizeof(D), cudaMemcpyDeviceToHost));
        std::vector<int> idx(batch); std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return hS[a] < hS[b]; });
        for (int q = 0; q < 5 && q < batch; q++) {
            int b = idx[q]; if (hS[b] > 1e17) break;
            Cand c; c.s = hS[b]; c.st.resize(K);
            CK(cudaMemcpy(c.st.data(), dW + (size_t)b * 5 * K + 4 * (size_t)K, K * sizeof(D), cudaMemcpyDeviceToHost));
            top.push_back(c);
        }
        std::sort(top.begin(), top.end(), [](const Cand& a, const Cand& b) { return a.s < b.s; }); if (top.size() > 5) top.resize(5);
        if (!top.empty()) gbest = top[0].s;
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "starts %ld/%ld best s=%.9f (%.0fs)\n", base + batch, total, gbest, el);
    }
    printf("d=%d n=%d trivial=%d vol_bound=%.4f\nbest sides:", d, n, triv, vol);
    for (auto& c : top) printf(" %.7f", c.s); printf("\n");
    FILE* f = fopen(out.c_str(), "w"); fprintf(f, "{\"d\":%d,\"n\":%d,\"results\":[", d, n);
    for (size_t q = 0; q < top.size(); q++) {
        if (q) fprintf(f, ",");
        fprintf(f, "{\"s\":%.12f,\"centers\":[", top[q].s);
        for (int z = 0; z < nd; z++) fprintf(f, "%s%.15g", z ? "," : "", top[q].st[z]);
        fprintf(f, "],\"R\":["); for (int z = 0; z < ndd; z++) fprintf(f, "%s%.15g", z ? "," : "", top[q].st[nd + z]);
        fprintf(f, "],\"u\":["); for (int z = 0; z < P * d; z++) fprintf(f, "%s%.15g", z ? "," : "", top[q].st[nd + ndd + z]); fprintf(f, "]}");
    }
    fprintf(f, "]}\n"); fclose(f);
    return 0;
}
