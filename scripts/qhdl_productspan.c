/* qhdl_productspan.c -- inner loop of the product-span (MinRank-style) AND minimiser.
 *
 * Problem (functions on N <= 64*NW points, packed as NW x uint64):
 *   U = span of D basis functions (includes the constant). Fixed factors L_1..L_{k-1} (functions in U) are given.
 *   Q = span(U, L_j * u_i).  For a candidate last factor L = XOR of the basis functions selected by a D-bit mask,
 *   the products L*u_i modulo Q are the columns of a linear map; a nonzero combination of the r target functions
 *   lies in U + sum_j L_j U  iff its image (modulo Q) lies in the column span.  We enumerate masks in Gray-code
 *   order (one basis flip per step) and test with GF(2) elimination.
 * Evidence tier of anything produced: static construction; the Python wrapper re-checks every hit exactly.
 * Build: cc -O3 -march=native -shared -fPIC -o libqhdl_productspan.so qhdl_productspan.c
 */
#include <stdint.h>
#include <string.h>

#ifndef NW
#define NW 4            /* words per function vector; points N <= 64*NW.  Python builds NW=4 (N<=256) and NW=16 (N<=1024) */
#endif
#define NB (64 * NW)
typedef struct { uint64_t w[NW]; } V4;

static inline int hib(const V4 *v, int nw) {
    for (int i = nw - 1; i >= 0; i--) if (v->w[i]) return i * 64 + 63 - __builtin_clzll(v->w[i]);
    return -1;
}
static inline void xr(V4 *a, const V4 *b) { for (int i = 0; i < NW; i++) a->w[i] ^= b->w[i]; }

/* simple full reduce: for each set bit from high to low, if pivot exists xor.  Returns nothing. */
static void red(V4 *v, const V4 *basis, const unsigned char *has) {
    for (int i = NW - 1; i >= 0; i--) {
        for (int b = 63; b >= 0; b--) {
            if ((v->w[i] >> b) & 1) {
                int p = i * 64 + b;
                if (has[p]) xr(v, &basis[p]);
            }
        }
    }
}

static int addvec(V4 *v, V4 *basis, unsigned char *has) {
    red(v, basis, has);
    int h = hib(v, NW);
    if (h < 0) return 0;
    basis[h] = *v; has[h] = 1; return 1;
}

/* returns 1 and fills *out_mask on success; *tests = number of masks tested */
/* minj: only a combination whose highest-index member is >= minj counts as a hit (minj = r-1 forces the last target
 * into the combination: used by the in-place model, where the pivot wire must be fixed before the search). */
int ps_search2(int D, const uint64_t *Uf, int r, const uint64_t *Tf, int nfix, const uint64_t *Lf,
               uint64_t start_mask, uint64_t max_iter, uint64_t *out_mask, uint64_t *tests_done, int *qdim_out, int minj)
{
    V4 U[64], T[64], L[64];
    if (D > 64 || r > 64 || nfix > 64) return -1;
    memcpy(U, Uf, D * sizeof(V4)); memcpy(T, Tf, r * sizeof(V4)); memcpy(L, Lf, nfix * sizeof(V4));
    V4 Qb[NB]; unsigned char Qh[NB]; memset(Qh, 0, sizeof Qh);
    for (int i = 0; i < D; i++) { V4 v = U[i]; addvec(&v, Qb, Qh); }
    for (int j = 0; j < nfix; j++)
        for (int i = 0; i < D; i++) {
            V4 v; for (int t = 0; t < NW; t++) v.w[t] = L[j].w[t] & U[i].w[t];
            addvec(&v, Qb, Qh);
        }
    int qd = 0; for (int p = 0; p < NB; p++) qd += Qh[p];
    *qdim_out = qd;
    /* fully reduce basis so pivot columns vanish elsewhere (RREF) */
    for (int p = 0; p < NB; p++) if (Qh[p]) {
        for (int q = p + 1; q < NB; q++) if (Qh[q] && ((Qb[q].w[p >> 6] >> (p & 63)) & 1)) xr(&Qb[q], &Qb[p]);
    }
    /* nonpivot positions: after reduction every vector is zero on pivots, so gather the other positions into the low
       bits (lossless) and run the tests on nw words */
    int pos[NB], m = 0;
    for (int p = 0; p < NB; p++) if (!Qh[p]) pos[m++] = p;
    int nw = (m + 63) / 64; if (nw < 1) nw = 1;
    #define COMPACT(v) do { V4 o_; memset(&o_, 0, sizeof o_); \
        for (int j_ = 0; j_ < m; j_++) if (((v).w[pos[j_] >> 6] >> (pos[j_] & 63)) & 1) o_.w[j_ >> 6] |= 1ULL << (j_ & 63); \
        (v) = o_; } while (0)
    static V4 R[64][64];
    for (int b = 0; b < D; b++) for (int i = 0; i < D; i++) {
        V4 v; for (int t = 0; t < NW; t++) v.w[t] = U[b].w[t] & U[i].w[t];
        red(&v, Qb, Qh); COMPACT(v); R[b][i] = v;
    }
    V4 Tr[64];
    for (int j = 0; j < r; j++) { Tr[j] = T[j]; red(&Tr[j], Qb, Qh); COMPACT(Tr[j]); }
    V4 C[64];
    memset(C, 0, sizeof C);
    uint64_t mask = start_mask;
    for (int b = 0; b < D; b++) if ((mask >> b) & 1) for (int i = 0; i < D; i++) xr(&C[i], &R[b][i]);
    uint64_t it = 0;
    for (;;) {
        V4 eb[NB]; unsigned char eh[NB]; memset(eh, 0, NB);
        for (int i = 0; i < D; i++) { V4 v = C[i]; int h;
            for (;;) { h = hib(&v, nw); if (h < 0 || !eh[h]) break; xr(&v, &eb[h]); }
            if (h >= 0) { eb[h] = v; eh[h] = 1; } }
        int ok = 0;
        for (int j = 0; j < r; j++) {
            V4 v = Tr[j]; int h;
            for (;;) { h = hib(&v, nw); if (h < 0 || !eh[h]) break; xr(&v, &eb[h]); }
            if (h < 0) { if (j >= minj) { ok = 1; break; } continue; }
            eb[h] = v; eh[h] = 1;
        }
        it++;
        if (ok) { *out_mask = mask; *tests_done = it; return 1; }
        if (it >= max_iter) { *tests_done = it; return 0; }
        int b = __builtin_ctzll(it);
        if (b >= D) { *tests_done = it; return 0; }
        mask ^= (1ULL << b);
        for (int i = 0; i < D; i++) xr(&C[i], &R[b][i]);
    }
}

int ps_search(int D, const uint64_t *Uf, int r, const uint64_t *Tf, int nfix, const uint64_t *Lf,
              uint64_t start_mask, uint64_t max_iter, uint64_t *out_mask, uint64_t *tests_done, int *qdim_out)
{
    return ps_search2(D, Uf, r, Tf, nfix, Lf, start_mask, max_iter, out_mask, tests_done, qdim_out, 0);
}
