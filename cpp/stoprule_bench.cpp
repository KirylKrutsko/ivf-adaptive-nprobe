// Wall-clock benchmark of the 3-feature stopping rule implemented inside the scan loop, in C++,
// against plain Faiss on the same IVF4096,PQ16x8 index (SIFT1M test queries).
//
// Methods (all re-rank the 200 best PQ candidates exactly, as in the Python experiments):
//   plain        index->search(nprobe)                                   -- the baseline
//   loop         our scan loop over the same nprobe cells, in one go     -- must equal plain
//   loop-rounds  same cells, scanned in the stopping rule's rounds, rule evaluated but ignored
//   stop         stopping rule: after each round, continue iff
//                  d_next/d1 < K * (r10/d1)^p * c^q
//                quantizer asks for the 256 nearest centroids up front (as in Python)
//   stop-lazyN   same rule; the quantizer asks for the N nearest centroids (N = 32, 64); queries
//                that get past them are suspended and resumed after one batched search for the
//                256 nearest, run for those queries only
// The scan loop is what IndexIVF::search_preassigned does internally: one InvertedListScanner
// per query, one heap kept alive across all cells, scan_codes per list.
//
// Inputs from cpp/export_inputs.py (data/cpp/). Single thread; median of REPS timed runs after
// a warm-up. Build and run: cpp/run.sh

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using faiss::idx_t;
using Clock = std::chrono::steady_clock;

static const int SCHED[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28,
                            32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256};
static const int NS = sizeof(SCHED) / sizeof(int);
static const int KR = 200, K10 = 10, CMAX = 256, REPS = 5;

template <class T>
static std::vector<T> read_raw(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T))) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        exit(1);
    }
    return v;
}

static std::vector<float> read_fvecs(const std::string& path, int d, size_t& n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t bytes = f.tellg();
    n = bytes / ((d + 1) * 4);
    f.seekg(0);
    std::vector<float> out(n * d);
    std::vector<float> row(d + 1);
    for (size_t i = 0; i < n; i++) {
        f.read(reinterpret_cast<char*>(row.data()), (d + 1) * 4);
        std::copy(row.begin() + 1, row.end(), out.begin() + i * d);
    }
    return out;
}

struct Run {
    std::vector<idx_t> I10;  // nq x 10 after re-ranking
    std::vector<int> cells;  // cells scanned per query
    double ndis = 0;         // total vectors scanned
    double t_quant = 0, t_scan = 0, t_rerank = 0;  // seconds
    double share_extended = 0;
};

struct Ctx {
    faiss::IndexIVF* ivf;
    const faiss::InvertedLists* il;
    const float* xq;
    const float* xb;
    int nq, d, nlist;
};

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static void rerank(const Ctx& c, const std::vector<idx_t>& I200, std::vector<idx_t>& I10) {
    std::vector<std::pair<float, idx_t>> cand(KR);
    for (int i = 0; i < c.nq; i++) {
        const float* q = c.xq + (size_t)i * c.d;
        int m = 0;
        for (int j = 0; j < KR; j++) {
            idx_t id = I200[(size_t)i * KR + j];
            if (id < 0) continue;
            cand[m++] = {faiss::fvec_L2sqr(q, c.xb + (size_t)id * c.d, c.d), id};
        }
        int top = std::min(m, K10);
        std::partial_sort(cand.begin(), cand.begin() + top, cand.begin() + m);
        for (int j = 0; j < K10; j++) I10[(size_t)i * K10 + j] = j < top ? cand[j].second : -1;
    }
}

static Run run_plain(const Ctx& c, int nprobe) {
    Run r;
    std::vector<float> D((size_t)c.nq * KR);
    std::vector<idx_t> I((size_t)c.nq * KR);
    r.I10.resize((size_t)c.nq * K10);
    c.ivf->nprobe = nprobe;
    faiss::indexIVF_stats.reset();
    auto t0 = Clock::now();
    c.ivf->search(c.nq, c.xq, KR, D.data(), I.data());
    auto t1 = Clock::now();
    rerank(c, I, r.I10);
    auto t2 = Clock::now();
    r.t_scan = secs(t0, t1);  // quantizer + scan inside Faiss
    r.t_rerank = secs(t1, t2);
    r.ndis = (double)faiss::indexIVF_stats.ndis;
    r.cells.assign(c.nq, nprobe);
    return r;
}

enum Mode { LOOP = 0, LOOP_ROUNDS = 1, STOP = 2, STOP_LAZY = 3 };

// Per-query scan state, so a query can be suspended (needs more centroids) and resumed.
struct QState {
    int s = 0, c_prev = 0, ncells = 0;
    float r10sq = std::numeric_limits<float>::infinity();
    double d1 = 0;
};

// lazy0: STOP_LAZY asks the quantizer for the lazy0 nearest centroids for every query; queries that
// get past them are suspended, one batched search for the 256 nearest runs for just those queries,
// and they resume (cells already scanned keep their order and are not scanned again).
static Run run_loop(const Ctx& c, Mode mode, int nprobe, double K, double P, double Q, int lazy0 = 64) {
    Run r;
    const int nq = c.nq;
    r.cells.assign(nq, 0);
    r.I10.resize((size_t)nq * K10);
    std::vector<idx_t> I200((size_t)nq * KR);
    const double logK = std::log(K);
    volatile double sink = 0;  // keeps the ignored rule evaluation of LOOP_ROUNDS alive
    const bool fixed = (mode == LOOP || mode == LOOP_ROUNDS);
    const int limit = fixed ? nprobe : CMAX;

    // --- coarse quantizer
    auto t0 = Clock::now();
    const int kq = fixed ? nprobe : (mode == STOP_LAZY ? lazy0 : CMAX);
    std::vector<float> Dq((size_t)nq * kq);
    std::vector<idx_t> Iq((size_t)nq * kq);
    c.ivf->quantizer->search(nq, c.xq, kq, Dq.data(), Iq.data());
    auto t1 = Clock::now();

    std::unique_ptr<faiss::InvertedListScanner> sc(c.ivf->get_InvertedListScanner(false));
    std::vector<float> Hd((size_t)nq * KR), buf(KR);  // one heap per query, kept across rounds
    std::vector<idx_t> Hi((size_t)nq * KR);
    std::vector<QState> st(nq);
    size_t ndis = 0;

    // run query i from its saved state; false = suspended because it needs more than `avail` centroids
    auto process = [&](int i, const float* dq, const idx_t* iq, int avail) -> bool {
        QState& S = st[i];
        float* simi = Hd.data() + (size_t)i * KR;
        idx_t* idxi = Hi.data() + (size_t)i * KR;
        sc->set_query(c.xq + (size_t)i * c.d);
        S.ncells = limit;
        while (S.s < NS) {
            const int cend = (mode == LOOP) ? limit : std::min(SCHED[S.s], limit);
            if (cend > avail) return false;
            for (int j = S.c_prev; j < cend; j++) {
                idx_t list = iq[j];
                if (list < 0) continue;
                size_t n = c.il->list_size(list);
                if (n == 0) continue;
                sc->set_list(list, dq[j]);
                faiss::InvertedLists::ScopedCodes codes(c.il, list);
                faiss::InvertedLists::ScopedIds ids(c.il, list);
                sc->scan_codes(n, codes.get(), ids.get(), simi, idxi, KR);
                ndis += n;
            }
            S.c_prev = cend;
            if (mode == LOOP || cend >= limit || S.s == NS - 1) break;
            if (cend >= avail) return false;  // the rule needs the next centroid's distance
            // --- the rule, between rounds: 10th-best PQ distance, next centroid, cells so far.
            // The 10th-best distance never grows as more cells are scanned, so only heap entries
            // <= the previous one can be in the new top 10: collect them branch-free, then select.
            int m = 0;
            for (int j = 0; j < KR; j++) { float v = simi[j]; buf[m] = v; m += (v <= S.r10sq); }
            float best[K10];
            if (m > 2 * K10 + 8) {  // first rounds: threshold still loose, insertion over the candidates
                std::fill(best, best + K10, std::numeric_limits<float>::infinity());
                for (int j = m - 1; j >= 0; j--) {
                    float v = buf[j];
                    if (v >= best[K10 - 1]) continue;
                    int p = K10 - 1;
                    while (p > 0 && best[p - 1] > v) { best[p] = best[p - 1]; p--; }
                    best[p] = v;
                }
            } else {
                std::nth_element(buf.begin(), buf.begin() + (K10 - 1), buf.begin() + m);
                best[K10 - 1] = buf[K10 - 1];
            }
            S.r10sq = best[K10 - 1];
            const double r10 = std::sqrt(std::max((double)S.r10sq, 1e-12));
            const double dn = std::sqrt(std::max((double)dq[cend], 1e-12));
            const bool go = std::log(dn / S.d1) < logK + P * std::log(r10 / S.d1) + Q * std::log((double)cend);
            S.s++;
            if (mode == LOOP_ROUNDS) { sink = sink + (go ? 1 : 0); continue; }
            if (!go) { S.ncells = cend; break; }
        }
        faiss::heap_reorder<faiss::CMax<float, idx_t>>(KR, simi, idxi);
        std::copy(idxi, idxi + KR, I200.begin() + (size_t)i * KR);
        r.cells[i] = S.ncells;
        return true;
    };

    std::vector<int> pending;
    for (int i = 0; i < nq; i++) {
        faiss::heap_heapify<faiss::CMax<float, idx_t>>(KR, Hd.data() + (size_t)i * KR, Hi.data() + (size_t)i * KR);
        st[i].d1 = std::sqrt(std::max((double)Dq[(size_t)i * kq], 1e-12));
        if (!process(i, Dq.data() + (size_t)i * kq, Iq.data() + (size_t)i * kq, kq)) pending.push_back(i);
    }
    double t_ext = 0;
    if (!pending.empty()) {  // only STOP_LAZY gets here
        const int np = pending.size();
        std::vector<float> xs((size_t)np * c.d), Dx((size_t)np * CMAX), dl(CMAX);
        std::vector<idx_t> Ix((size_t)np * CMAX), il(CMAX);
        for (int t = 0; t < np; t++)
            std::copy(c.xq + (size_t)pending[t] * c.d, c.xq + (size_t)(pending[t] + 1) * c.d, xs.begin() + (size_t)t * c.d);
        auto te0 = Clock::now();
        c.ivf->quantizer->search(np, xs.data(), CMAX, Dx.data(), Ix.data());
        t_ext = secs(te0, Clock::now());
        for (int t = 0; t < np; t++) {
            const int i = pending[t];
            const float* od = Dq.data() + (size_t)i * kq;
            const idx_t* oi = Iq.data() + (size_t)i * kq;
            std::copy(od, od + kq, dl.begin());
            std::copy(oi, oi + kq, il.begin());
            int m = kq;
            for (int j = 0; j < CMAX && m < CMAX; j++) {
                idx_t id = Ix[(size_t)t * CMAX + j];
                if (std::find(oi, oi + kq, id) != oi + kq) continue;
                il[m] = id; dl[m] = Dx[(size_t)t * CMAX + j]; m++;
            }
            process(i, dl.data(), il.data(), m);
        }
    }
    auto t2 = Clock::now();
    rerank(c, I200, r.I10);
    auto t3 = Clock::now();
    r.t_quant = secs(t0, t1) + t_ext;
    r.t_scan = secs(t1, t2) - t_ext;
    r.t_rerank = secs(t2, t3);
    r.ndis = (double)ndis;
    r.share_extended = (double)pending.size() / nq;
    return r;
}


// ---- static fitted rule: decide the number of cells before scanning, then scan in one go ----
struct Fitted {
    std::vector<int> ranks;
    std::vector<double> mu, sd, w;  // w: one weight per feature + intercept (last)
    double gamma = 1;
};

static Fitted read_fitted(const std::string& path) {
    Fitted f;
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    std::string line;
    auto nums = [&](auto& v) { std::getline(in, line); std::istringstream ss(line); typename std::decay_t<decltype(v)>::value_type x; while (ss >> x) v.push_back(x); };
    nums(f.ranks); nums(f.mu); nums(f.sd); nums(f.w);
    in >> f.gamma;
    return f;
}

// kq: nearest centroids requested for every query (>= the largest feature rank). Queries whose
// predicted cell count exceeds kq get one batched search for the 256 nearest.
static Run run_fitted(const Ctx& c, const Fitted& f, int kq) {
    Run r;
    const int nq = c.nq;
    r.cells.assign(nq, 0);
    r.I10.resize((size_t)nq * K10);
    std::vector<idx_t> I200((size_t)nq * KR);
    const int nf = f.mu.size();

    auto t0 = Clock::now();
    std::vector<float> Dq((size_t)nq * kq);
    std::vector<idx_t> Iq((size_t)nq * kq);
    c.ivf->quantizer->search(nq, c.xq, kq, Dq.data(), Iq.data());
    std::vector<double> x(nf);
    std::vector<int> pending;
    for (int i = 0; i < nq; i++) {
        const float* dq = Dq.data() + (size_t)i * kq;
        const idx_t* iq = Iq.data() + (size_t)i * kq;
        const double d1 = std::sqrt(std::max((double)dq[0], 1e-12));
        int k = 0;
        for (int rk : f.ranks) x[k++] = std::log(std::sqrt(std::max((double)dq[rk - 1], 1e-12)) / d1);
        x[k++] = std::log(d1);
        double ls = 0;
        for (int j = 0; j < 8; j++) ls += c.il->list_size(iq[j]);
        x[k++] = std::log(ls / 8);
        double z = f.w[nf];
        for (int j = 0; j < nf; j++) z += (x[j] - f.mu[j]) / f.sd[j] * f.w[j];
        int n = (int)std::nearbyint(f.gamma * std::exp(z));
        n = std::max(1, std::min(n, CMAX));
        r.cells[i] = n;
        if (n > kq) pending.push_back(i);
    }
    std::vector<float> Dx;
    std::vector<idx_t> Ix;
    std::vector<int> slot(nq, -1);
    if (!pending.empty()) {
        const int np = pending.size();
        std::vector<float> xs((size_t)np * c.d);
        for (int t = 0; t < np; t++) {
            std::copy(c.xq + (size_t)pending[t] * c.d, c.xq + (size_t)(pending[t] + 1) * c.d, xs.begin() + (size_t)t * c.d);
            slot[pending[t]] = t;
        }
        Dx.resize((size_t)np * CMAX);
        Ix.resize((size_t)np * CMAX);
        c.ivf->quantizer->search(np, xs.data(), CMAX, Dx.data(), Ix.data());
    }
    auto t1 = Clock::now();

    std::unique_ptr<faiss::InvertedListScanner> sc(c.ivf->get_InvertedListScanner(false));
    std::vector<float> simi(KR);
    std::vector<idx_t> idxi(KR);
    size_t ndis = 0;
    for (int i = 0; i < nq; i++) {
        const float* dq = slot[i] < 0 ? Dq.data() + (size_t)i * kq : Dx.data() + (size_t)slot[i] * CMAX;
        const idx_t* iq = slot[i] < 0 ? Iq.data() + (size_t)i * kq : Ix.data() + (size_t)slot[i] * CMAX;
        sc->set_query(c.xq + (size_t)i * c.d);
        faiss::heap_heapify<faiss::CMax<float, idx_t>>(KR, simi.data(), idxi.data());
        for (int j = 0; j < r.cells[i]; j++) {
            idx_t list = iq[j];
            if (list < 0) continue;
            size_t n = c.il->list_size(list);
            if (n == 0) continue;
            sc->set_list(list, dq[j]);
            faiss::InvertedLists::ScopedCodes codes(c.il, list);
            faiss::InvertedLists::ScopedIds ids(c.il, list);
            sc->scan_codes(n, codes.get(), ids.get(), simi.data(), idxi.data(), KR);
            ndis += n;
        }
        faiss::heap_reorder<faiss::CMax<float, idx_t>>(KR, simi.data(), idxi.data());
        std::copy(idxi.begin(), idxi.end(), I200.begin() + (size_t)i * KR);
    }
    auto t2 = Clock::now();
    rerank(c, I200, r.I10);
    auto t3 = Clock::now();
    r.t_quant = secs(t0, t1);  // quantizer + features + prediction (+ batched extension)
    r.t_scan = secs(t1, t2);
    r.t_rerank = secs(t2, t3);
    r.ndis = (double)ndis;
    r.share_extended = (double)pending.size() / nq;
    return r;
}

static double recall10(const Run& r, const std::vector<int64_t>& gt, int nq) {
    double hits = 0;
    for (int i = 0; i < nq; i++)
        for (int a = 0; a < K10; a++)
            for (int b = 0; b < K10; b++)
                if (r.I10[(size_t)i * K10 + a] == gt[(size_t)i * K10 + b]) { hits++; break; }
    return hits / (nq * (double)K10);
}

template <class F>
static Run timed(F f) {
    f();  // warm-up
    std::vector<Run> runs;
    for (int k = 0; k < REPS; k++) runs.push_back(f());
    std::vector<double> tot;
    for (auto& x : runs) tot.push_back(x.t_quant + x.t_scan + x.t_rerank);
    std::vector<size_t> o(REPS);
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return tot[a] < tot[b]; });
    return runs[o[REPS / 2]];
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data/cpp";
    std::string out_csv = argc > 2 ? argv[2] : "results/cpp_stoprule_bench.csv";
    const bool only_fitted = argc > 3 && std::string(argv[3]) == "fitted";
    omp_set_num_threads(1);

    std::ifstream meta(dir + "/meta.txt");
    int nq, d, T;
    std::string index_path, base_path;
    meta >> nq >> d >> index_path >> base_path >> T;
    struct Target { double R; int nprobe; double K, P, Q; };
    std::vector<Target> targets(T);
    for (auto& t : targets) meta >> t.R >> t.nprobe >> t.K >> t.P >> t.Q;

    auto xq = read_raw<float>(dir + "/queries.f32", (size_t)nq * d);
    auto gt = read_raw<int64_t>(dir + "/gt10.i64", (size_t)nq * K10);
    size_t nb;
    auto xb = read_fvecs(base_path, d, nb);
    std::unique_ptr<faiss::Index> idx(faiss::read_index(index_path.c_str()));
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx.get());
    auto* flat = dynamic_cast<faiss::IndexFlat*>(ivf->quantizer);
    if (!ivf || !flat) { fprintf(stderr, "expected IndexIVF with a flat quantizer\n"); return 1; }
    Ctx c{ivf, ivf->invlists, xq.data(), xb.data(), nq, d, (int)ivf->nlist};
    printf("index %s: nlist %zu, ntotal %lld; %d queries, base %zu; omp threads %d\n", index_path.c_str(),
           ivf->nlist, (long long)ivf->ntotal, nq, nb, omp_get_max_threads());

    FILE* fo = fopen(out_csv.c_str(), "w");
    fprintf(fo, "target,method,nprobe,recall,ndis_mean,work_mean,cells_mean,quantizer_ms,scan_ms,rerank_ms,"
                "total_ms,speedup_vs_plain,cells_ne_python,top10_ne_plain,share_extended\n");
    printf("\n%-5s %-18s %7s %8s %9s %7s %7s %7s %7s %8s %8s %9s %9s\n", "R", "method", "recall", "work",
           "cells", "quant", "scan", "rerank", "total", "speedup", "vs-py", "vs-plain", "extended");
    for (auto& t : targets) {
        char rb[32];
        snprintf(rb, sizeof rb, "%g", t.R);
        auto py = read_raw<int32_t>(dir + "/cells_R" + rb + ".i32", nq);
        std::map<std::string, std::vector<int32_t>> py_fitted;
        std::deque<std::string> names;  // stable storage for method names
        Run plain = timed([&] { return run_plain(c, t.nprobe); });
        const double t_plain = plain.t_quant + plain.t_scan + plain.t_rerank;
        struct M { const char* name; Run r; bool rule; std::string variant; };
        std::vector<M> ms;
        ms.push_back({"plain", plain, false, ""});
        if (only_fitted) {  // variants listed by cpp/export_fitted.py: "<variant> <kq>" per line
            std::ifstream vf(dir + "/fitted_variants.txt");
            std::string vname;
            int vkq;
            while (vf >> vname >> vkq) {
                Fitted fm = read_fitted(dir + "/fitted_" + vname + "_R" + rb + ".txt");
                py_fitted[vname] = read_raw<int32_t>(dir + "/cells_fitted_" + vname + "_R" + rb + ".i32", nq);
                names.push_back("fitted-" + vname + "-k" + std::to_string(vkq));
                ms.push_back({names.back().c_str(), timed([&] { return run_fitted(c, fm, vkq); }), true});
                ms.back().variant = vname;
            }
        } else {
            ms.push_back({"loop", timed([&] { return run_loop(c, LOOP, t.nprobe, t.K, t.P, t.Q); }), false});
            ms.push_back({"loop-rounds", timed([&] { return run_loop(c, LOOP_ROUNDS, t.nprobe, t.K, t.P, t.Q); }), false});
            ms.push_back({"stop", timed([&] { return run_loop(c, STOP, t.nprobe, t.K, t.P, t.Q); }), true});
            ms.push_back({"stop-lazy32", timed([&] { return run_loop(c, STOP_LAZY, t.nprobe, t.K, t.P, t.Q, 32); }), true});
            ms.push_back({"stop-lazy64", timed([&] { return run_loop(c, STOP_LAZY, t.nprobe, t.K, t.P, t.Q, 64); }), true});
        }
        for (auto& m : ms) {
            const Run& r = m.r;
            double tot = r.t_quant + r.t_scan + r.t_rerank;
            double cm = std::accumulate(r.cells.begin(), r.cells.end(), 0.0) / nq;
            int ne_py = -1, ne_plain = 0;
            if (m.rule) {
                const auto& ref = m.variant.empty() ? py : py_fitted[m.variant];
                ne_py = 0;
                for (int i = 0; i < nq; i++) ne_py += r.cells[i] != ref[i];
            }
            for (int i = 0; i < nq; i++)
                if (!std::equal(r.I10.begin() + (size_t)i * K10, r.I10.begin() + (size_t)(i + 1) * K10,
                                plain.I10.begin() + (size_t)i * K10)) ne_plain++;
            double rec = recall10(r, gt, nq);
            double ms_q = 1e3 / nq;
            printf("%-5g %-18s %7.4f %8.0f %9.1f %7.4f %7.4f %7.4f %7.4f %8.2fx %8d %9d %8.1f%%\n", t.R, m.name, rec,
                   r.ndis / nq + KR, cm, r.t_quant * ms_q, r.t_scan * ms_q, r.t_rerank * ms_q, tot * ms_q,
                   t_plain / tot, ne_py, ne_plain, 100 * r.share_extended);
            fprintf(fo, "%g,%s,%d,%.5f,%.1f,%.1f,%.2f,%.5f,%.5f,%.5f,%.5f,%.4f,%d,%d,%.4f\n", t.R, m.name, t.nprobe, rec,
                    r.ndis / nq, r.ndis / nq + KR, cm, r.t_quant * ms_q, r.t_scan * ms_q, r.t_rerank * ms_q, tot * ms_q,
                    t_plain / tot, ne_py, ne_plain, r.share_extended);
        }
    }
    fclose(fo);
    printf("\nwrote %s\n", out_csv.c_str());
    return 0;
}
