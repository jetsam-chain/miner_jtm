// jetsam_pow_hybrid.cuh — TowerHash arithmetic in the HYBRID basis
//     GF(2^128) = GF(2^64)[y'] / (y'^2 + y' + tau''),   GF(2^64) = GF(2)[x] / (x^64 + x^4 + x^3 + x + 1)
//
// Why this basis: the state never changes basis inside the permutation. A GF(2^128)
// multiply is three carry-less 64x64 products (CLMAD.LO/HI, Karatsuba over y') plus
// 5-term shift reductions; a squaring is a bit-spread (PRMT) plus reductions, zero
// CLMAD; the MDS constants are GF(2^64)-subfield elements, so a constant multiply is
// two 64-bit products. The tower<->hybrid change of basis (a fixed GF(2)-linear map)
// happens once per hash on input (nonce: 8 table lookups) and once on output (digest),
// never per round. Jetsam's own basis would pay a conversion in every single round.
//
// Where the constants come from: tools/gen_hybrid.py derives the whole representation
// from the consensus reference — it finds phi as a root of the tower generator's
// minimal polynomial (Cantor-Zassenhaus), picks the Frobenius conjugate that makes
// tau' sparsest, and emits jetsam_pow_hybrid_constants.cuh. Re-run it and you get
// those files back byte for byte; nothing here is a magic number.
//
// Bit-exactness: gen_hybrid.py checks the isomorphism and the whole hybrid sponge
// against the golden vectors in Python; the CUDA gate (--selftest golden.txt,
// 12000/12000) checks this code on the card you are going to mine with.
#pragma once
#include <cstdint>
typedef unsigned long long u64;
struct F128 { u64 lo, hi; };          // hybrid element: lo = coefficient of 1, hi = coefficient of y'

#include "jetsam_pow_hybrid_constants.cuh"
#if HYB_TAU_NTERMS != 1
#error "this header expects a monomial tau''"
#endif

__device__ __forceinline__ F128 fx(F128 a, F128 b) { F128 r; r.lo = a.lo ^ b.lo; r.hi = a.hi ^ b.hi; return r; }

// Define JETSAM_FORCE_SOFT_CLMUL to compile the software path on any card. It
// exists so the sub-sm_80 arithmetic can be put through the bit-exact gate on
// hardware that does have clmad — otherwise it could only be tested on the very
// GPUs that are hardest to get hold of.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(JETSAM_FORCE_SOFT_CLMUL)
// Ampere and later carry a hardware carry-less multiply-add.
__device__ __forceinline__ u64 clmad_lo(u64 a, u64 b, u64 c) { u64 d; asm("clmad.lo.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ u64 clmad_hi(u64 a, u64 b, u64 c) { u64 d; asm("clmad.hi.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }

#elif defined(__CUDA_ARCH__)
// Below sm_80 there is no clmad — ptxas refuses it outright:
//   "Feature 'clmad.lo' requires .target sm_80 or higher"
// so the product is built from ordinary integer multiplies, by the standard
// spaced-bits method. Split each operand by the residue of its bit positions
// mod 4: a factor then holds bits only every 4th position, so in the integer
// product each occupied column accumulates at most 32/4 = 8 terms — three bits,
// which fits under the next occupied column. No carry ever crosses, and masking
// the result to its lane leaves exactly the carry-less product.
//
// Four such splits per operand give sixteen 32x32 products; Karatsuba then
// builds 64x64 from three of those instead of four. Checked against the naive
// shift-xor reference on 4,194,304 random pairs: zero disagreements.
__device__ __forceinline__ u64 clmul32(unsigned a, unsigned b) {
    u64 a0 = a & 0x11111111u, a1 = a & 0x22222222u, a2 = a & 0x44444444u, a3 = a & 0x88888888u;
    u64 b0 = b & 0x11111111u, b1 = b & 0x22222222u, b2 = b & 0x44444444u, b3 = b & 0x88888888u;
    u64 p0 = (a0 * b0) ^ (a1 * b3) ^ (a2 * b2) ^ (a3 * b1);
    u64 p1 = (a0 * b1) ^ (a1 * b0) ^ (a2 * b3) ^ (a3 * b2);
    u64 p2 = (a0 * b2) ^ (a1 * b1) ^ (a2 * b0) ^ (a3 * b3);
    u64 p3 = (a0 * b3) ^ (a1 * b2) ^ (a2 * b1) ^ (a3 * b0);
    return (p0 & 0x1111111111111111ULL) | (p1 & 0x2222222222222222ULL)
         | (p2 & 0x4444444444444444ULL) | (p3 & 0x8888888888888888ULL);
}
__device__ __forceinline__ void clmul64(u64 a, u64 b, u64& lo, u64& hi) {
    unsigned xlo = (unsigned)a, xhi = (unsigned)(a >> 32);
    unsigned ylo = (unsigned)b, yhi = (unsigned)(b >> 32);
    u64 z0 = clmul32(xlo, ylo), z2 = clmul32(xhi, yhi);
    u64 z1 = clmul32(xlo ^ xhi, ylo ^ yhi) ^ z0 ^ z2;
    lo = z0 ^ (z1 << 32);
    hi = z2 ^ (z1 >> 32);
}
__device__ __forceinline__ u64 clmad_lo(u64 a, u64 b, u64 c) { u64 lo, hi; clmul64(a, b, lo, hi); return lo ^ c; }
__device__ __forceinline__ u64 clmad_hi(u64 a, u64 b, u64 c) { u64 lo, hi; clmul64(a, b, lo, hi); return hi ^ c; }

#else
// host pass: software model (never executed on the host in the miner; keeps the device code compilable)
static inline void soft_clmul64(u64 a, u64 b, u64& lo, u64& hi) { lo = 0; hi = 0; for (int i = 0; i < 64; ++i) if ((b >> i) & 1ULL) { lo ^= a << i; if (i) hi ^= a >> (64 - i); } }
static inline u64 clmad_lo(u64 a, u64 b, u64 c) { u64 lo, hi; soft_clmul64(a, b, lo, hi); return lo ^ c; }
static inline u64 clmad_hi(u64 a, u64 b, u64 c) { u64 lo, hi; soft_clmul64(a, b, lo, hi); return hi ^ c; }
static inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) { u64 v = ((u64)y << 32) | x; unsigned r = 0; for (int i = 0; i < 4; ++i) { unsigned sel = (s >> (4 * i)) & 7; r |= ((unsigned)((v >> (8 * sel)) & 0xFF)) << (8 * i); } return r; }
#endif

// ------------------------------------------------------------- GF(2^64), poly basis
/// Reduce a 128-bit carry-less product hi:lo modulo x^64 + x^4 + x^3 + x + 1.
// PRECONDITION: hi < 2^63 (bit 63 clear). This holds at EVERY call site in this
// file, and it holds structurally, not by luck:
//   - p64_mul / p64_sqr : hi = clmad_hi of a carry-less 64x64 product. That product
//     has degree <= 126, so the high word has degree <= 62: bit 63 is clear.
//   - p64_mul_tau       : hi = v >> (64 - HYB_TAU_TERMS) < 2^HYB_TAU_TERMS.
//   - p64_sqr (spread)  : hi = spread32(...) puts bit b at 2b, highest is 62.
//   - hmul              : m0h/m1h/m2h all come from clmad_hi.
// If hi >= 2^63, `t` overflows and the fold loses a bit: ~50 % of results wrong.
// The guard below only exists in a verification build (HYB_CHECK_REDUCE).
#ifdef HYB_CHECK_REDUCE
__device__ unsigned long long g_reduce_violations[1];
#endif
__device__ __forceinline__ u64 p64_reduce(u64 hi, u64 lo) {
#ifdef HYB_CHECK_REDUCE
    if (hi >> 63) atomicAdd((unsigned long long*)g_reduce_violations, 1ULL);
#endif
    // The pentanomial factors: 1+x+x^3+x^4 = (1+x)(1+x^3), one shift fewer than
    // the expanded form.
    u64 t  = hi ^ (hi << 1);                                   // hi * (1+x)
    u64 t3 = t << 3;                                           // hi * (1+x) * x^3
    // Spill past x^64. Since t < 2^64 (precondition) it fits in 3 bits, which is
    // what lets it be folded back through an 8-byte table.
    unsigned sp = (unsigned)(t >> 61);
    // sp * (1+x+x^3+x^4) for sp in [0,8) = {0,1b,36,2d,6c,77,5a,41}, which fits in
    // two immediates and is read by a single PRMT. Entry 0 being zero, the unused
    // selector nibbles produce zero bytes: nothing to mask off.
    unsigned f = __byte_perm(0x2d361b00u, 0x415a776cu, sp);
    return lo ^ t ^ t3 ^ (u64)f;
}
__device__ __forceinline__ u64 p64_mul(u64 a, u64 b) {
    return p64_reduce(clmad_hi(a, b, 0ULL), clmad_lo(a, b, 0ULL));
}
/// Multiply by tau'' = x^HYB_TAU_TERMS.
__device__ __forceinline__ u64 p64_mul_tau(u64 v) {
    return p64_reduce(v >> (64 - HYB_TAU_TERMS), v << HYB_TAU_TERMS);
}
/// Interleave the 16 low bits of w with zeros (bit b -> bit 2b) with two byte permutes.
__device__ __forceinline__ unsigned spread16(unsigned w) {
    unsigned t = __byte_perm(0x05040100u, 0x15141110u, w & 0x7777u);   // low 3 bits of each nibble
#ifdef HYB_PRMT_SIGN
    // PRMT sign-replicate mode: a selector nibble with bit 3 set yields 0xFF (msb of 0x80), else 0x80.
    unsigned h = __byte_perm(0x80808080u, 0x80808080u, w) & 0x40404040u;
#else
    unsigned h = __byte_perm(0x00004000u, 0u, (w >> 3) & 0x1111u);      // bit 3 of each nibble -> bit 6 of the byte
#endif
    return t | h;
}
__device__ __forceinline__ u64 spread32(unsigned v) {
    return (u64)spread16(v & 0xFFFFu) | ((u64)spread16(v >> 16) << 32);
}
/// Squaring: in characteristic two, a^2 is a with zeros interleaved, then reduced.
__device__ __forceinline__ u64 p64_sqr(u64 a) {
#ifdef HYB_SQR_CLMAD
    return p64_reduce(clmad_hi(a, a, 0ULL), clmad_lo(a, a, 0ULL));
#else
    return p64_reduce(spread32((unsigned)(a >> 32)), spread32((unsigned)a));
#endif
}

// ------------------------------------------------------------- GF(2^128) hybrid
/// (a0 + a1 y')(b0 + b1 y') with y'^2 = y' + tau'' : Karatsuba, three 64x64 products.
__device__ __forceinline__ F128 hmul(F128 a, F128 b) {
    u64 m0l = clmad_lo(a.lo, b.lo, 0ULL), m0h = clmad_hi(a.lo, b.lo, 0ULL);
    u64 m1l = clmad_lo(a.hi, b.hi, 0ULL), m1h = clmad_hi(a.hi, b.hi, 0ULL);
    u64 sa = a.lo ^ a.hi, sb = b.lo ^ b.hi;
    u64 m2l = clmad_lo(sa, sb, m0l), m2h = clmad_hi(sa, sb, m0h);  // (m2 ^ m0) through the addend
    F128 r;
    r.hi = p64_reduce(m2h, m2l);                                   // a0 b1 + a1 b0 + a1 b1
    r.lo = p64_reduce(m0h, m0l) ^ p64_mul_tau(p64_reduce(m1h, m1l)); // a0 b0 + tau'' a1 b1
    return r;
}
/// (a0 + a1 y')^2 = (a0^2 + tau'' a1^2) + a1^2 y'
__device__ __forceinline__ F128 hsqr(F128 a) {
    F128 r; r.hi = p64_sqr(a.hi); r.lo = p64_sqr(a.lo) ^ p64_mul_tau(r.hi); return r;
}
/// Multiply by a GF(2^64)-subfield constant.
__device__ __forceinline__ F128 hmulc(u64 c, F128 a) {
    F128 r; r.lo = p64_mul(c, a.lo); r.hi = p64_mul(c, a.hi); return r;
}
__device__ __forceinline__ F128 sbox_hyb(F128 x) {
    F128 x2 = hsqr(x), x4 = hsqr(x2), x6 = hmul(x, x2);
    return hmul(x6, x4);
}
/// MDS_FULL = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]] from the products by 2 and 4.
// FOUR constant products instead of six: the matrix factors over u0 = a+b and
// u1 = c+d, because C4 never appears alone on a or on c without also appearing on
// its neighbour. Expanded and checked row by row:
//   y1 = 4u0 + 2b + u1            = 4a + 6b +  c +  d
//   y0 = y1 + u0 + 2d             = 5a + 7b +  c + 3d
//   y3 = u0 + 4u1 + 2d            =  a +  b + 4c + 6d
//   y2 = y3 + u1 + 2b             =  a + 3b + 5c + 7d
__device__ __forceinline__ void mds_full_hyb(F128* s) {
    F128 a = s[0], b = s[1], c = s[2], d = s[3];
    F128 u0 = fx(a, b), u1 = fx(c, d);
    F128 v = hmulc(HYB_C2, b), w = hmulc(HYB_C2, d);
    F128 p = hmulc(HYB_C4, u0), q = hmulc(HYB_C4, u1);
    F128 y1 = fx(fx(p, v), u1);
    F128 y0 = fx(fx(y1, u0), w);
    F128 y3 = fx(fx(u0, q), w);
    F128 y2 = fx(fx(y3, u1), v);
    s[0] = y0; s[1] = y1; s[2] = y2; s[3] = y3;
}
/// MDS_PARTIAL = ones + diag(c_i): out_i = c_i s_i + (S + s_i), S = XOR of the state.
#ifdef HYB_DIAG_TABLE
// c_i * v as a 64->64 linear map in byte tables (shared, 8 x 256 u64 per constant): 8 LDS.64, 0 CLMAD.
#define HYB_DIAG_WORDS (4 * 8 * 256)
__device__ __forceinline__ u64 p64_mul_tbl(const u64* t, u64 v) {
    u64 r = 0ULL;
#pragma unroll
    for (int k = 0; k < 8; ++k) { r ^= t[k * 256 + (int)(v & 255ULL)]; v >>= 8; }
    return r;
}
__device__ __forceinline__ void mds_partial_hyb(F128* s, const u64* tbl) {
    F128 S = fx(fx(s[0], s[1]), fx(s[2], s[3]));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const u64* t = tbl + i * 2048;
        F128 m; m.lo = p64_mul_tbl(t, s[i].lo); m.hi = p64_mul_tbl(t, s[i].hi);
        s[i] = fx(m, fx(S, s[i]));
    }
}
#define MDS_PARTIAL_HYB(s) mds_partial_hyb(s, sh_diag)
#else
#define HYB_DIAG_WORDS 0
__device__ __forceinline__ void mds_partial_hyb(F128* s) {
    F128 S = fx(fx(s[0], s[1]), fx(s[2], s[3]));
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = fx(hmulc(HYB_DIAG[i], s[i]), fx(S, s[i]));
}
#define MDS_PARTIAL_HYB(s) mds_partial_hyb(s)
#endif
#define TH_N_ROUNDS 66
#define TH_F_ROUNDS 8
#define TH_P_ROUNDS 58
#ifdef HYB_DIAG_TABLE
#define PERM_ARGS , const u64* sh_diag
#else
#define PERM_ARGS
#endif
__device__ __forceinline__ void permute_hyb(F128* s PERM_ARGS) {
    mds_full_hyb(s);
#pragma unroll 1
    for (int r = 0; r < TH_N_ROUNDS; ++r) {
        const bool partial = (r >= TH_F_ROUNDS / 2) && (r < TH_F_ROUNDS / 2 + TH_P_ROUNDS);
        if (!partial) {
#pragma unroll
            for (int i = 0; i < 4; ++i) s[i] = sbox_hyb(fx(s[i], RC_HYB[i][r]));
            mds_full_hyb(s);
        } else {
            s[0] = sbox_hyb(fx(s[0], RC_HYB[0][r]));
            MDS_PARTIAL_HYB(s);
        }
    }
}
/// Two independent states in lockstep (ILP x2): same arithmetic, interleaved chains.
__device__ __forceinline__ void permute_hyb2(F128* s, F128* u PERM_ARGS) {
    mds_full_hyb(s); mds_full_hyb(u);
#pragma unroll 1
    for (int r = 0; r < TH_N_ROUNDS; ++r) {
        const bool partial = (r >= TH_F_ROUNDS / 2) && (r < TH_F_ROUNDS / 2 + TH_P_ROUNDS);
        if (!partial) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { s[i] = sbox_hyb(fx(s[i], RC_HYB[i][r])); u[i] = sbox_hyb(fx(u[i], RC_HYB[i][r])); }
            mds_full_hyb(s); mds_full_hyb(u);
        } else {
            s[0] = sbox_hyb(fx(s[0], RC_HYB[0][r])); u[0] = sbox_hyb(fx(u[0], RC_HYB[0][r]));
            MDS_PARTIAL_HYB(s); MDS_PARTIAL_HYB(u);
        }
    }
}

// ------------------------------------------------------------- basis change tables (shared)
// Four 64->64 GF(2)-linear maps as nibble tables [16 chunks][16]: PHI, BPHI (hi -> beta*phi(hi)),
// PHI_INV, PHI_INV_B (hi' -> phi^-1(beta*hi')).  tower (lo,hi) -> hybrid (phi(lo)^BPHI(hi), phi(hi)).
#define HYB_SH_WORDS 1024
__device__ __forceinline__ void load_hyb_tables(u64* sh) {
    const u64* src[4] = { &PHI_NIB[0][0], &BPHI_NIB[0][0], &PHI_INV_NIB[0][0], &PHI_INV_B_NIB[0][0] };
    for (int i = threadIdx.x; i < HYB_SH_WORDS; i += blockDim.x) sh[i] = src[i >> 8][i & 255];
}
#ifdef HYB_DIAG_TABLE
#include "jetsam_pow_hybrid_tables.cuh"
__device__ __forceinline__ void load_diag_tables(u64* sh) {
    const u64* src = &HYB_DIAG_BYTE[0][0][0];
    for (int i = threadIdx.x; i < HYB_DIAG_WORDS; i += blockDim.x) sh[i] = src[i];
}
#endif
__device__ __forceinline__ u64 lin64(const u64* t, u64 v) {
    u64 r = 0ULL;
#pragma unroll
    for (int c = 0; c < 16; ++c) { r ^= t[c * 16 + (int)(v & 15ULL)]; v >>= 4; }
    return r;
}
__device__ __forceinline__ u64 lin32(const u64* t, unsigned v) {
    u64 r = 0ULL;
#pragma unroll
    for (int c = 0; c < 8; ++c) { r ^= t[c * 16 + (int)(v & 15u)]; v >>= 4; }
    return r;
}
__device__ __forceinline__ F128 tower_to_hyb(const u64* sh, F128 v) {
    F128 r; r.hi = lin64(sh, v.hi); r.lo = lin64(sh, v.lo) ^ lin64(sh + 256, v.hi); return r;
}
__device__ __forceinline__ F128 hyb_to_tower(const u64* sh, F128 h) {
    F128 r; r.hi = lin64(sh + 512, h.hi); r.lo = lin64(sh + 512, h.lo) ^ lin64(sh + 768, h.hi); return r;
}
// host-side versions (template fields, nonce base)
static inline u64 h_lin64(const u64 t[16][16], u64 v) { u64 r = 0; for (int c = 0; c < 16; ++c) { r ^= t[c][v & 15]; v >>= 4; } return r; }
static inline F128 h_tower_to_hyb(F128 v) { F128 r; r.hi = h_lin64(H_PHI_NIB, v.hi); r.lo = h_lin64(H_PHI_NIB, v.lo) ^ h_lin64(H_BPHI_NIB, v.hi); return r; }
static inline F128 h_hyb_to_tower(F128 h) { F128 r; r.hi = h_lin64(H_PHI_INV_NIB, h.hi); r.lo = h_lin64(H_PHI_INV_NIB, h.lo) ^ h_lin64(H_PHI_INV_B_NIB, h.hi); return r; }

// ------------------------------------------------------------- target test (tower-basis digest)
__host__ __device__ __forceinline__ bool le256_lt(F128 d0, F128 d1, F128 t0, F128 t1) {
    if (d1.hi != t1.hi) return d1.hi < t1.hi;
    if (d1.lo != t1.lo) return d1.lo < t1.lo;
    if (d0.hi != t0.hi) return d0.hi < t0.hi;
    return d0.lo < t0.lo;
}
