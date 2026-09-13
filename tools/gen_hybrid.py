#!/usr/bin/env python3
"""gen_hybrid.py — build the HYBRID representation of Jetsam's GF(2^128) and emit
CUDA constants for it.

Hybrid basis:
    GF(2^128) = GF(2^64)[y] / (y^2 + y + tau'),   GF(2^64) = GF(2)[x] / (P),
    P = x^64 + x^4 + x^3 + x + 1.
The top level is the SAME as Jetsam's tower (lo64 + hi64*y4); only the GF(2^64)
coordinate changes basis through a field isomorphism  phi : GF(2^64)_tower -> GF(2^64)_poly.
So a tower element (lo, hi) becomes (phi(lo), phi(hi)) and every field operation is
carry-less 64x64 multiplies + a 5-term reduction — no tower<->flat conversion ever.

Everything is anchored on the Rust reference: tower arithmetic is computed THROUGH
the flat basis (TOWER_TO_FLAT / FLAT_TO_TOWER matrices parsed from hardware.rs and
the GCM multiply), exactly like the reference does.  The script:
  1. finds phi (root of the minimal polynomial of the tower generator y3 in GF(2^64)_poly,
     Cantor–Zassenhaus), picks the Frobenius conjugate that makes tau' sparsest,
  2. verifies phi is a field isomorphism on random samples,
  3. re-implements the permutation in the hybrid basis and checks golden vectors,
  4. emits jetsam_pow_hybrid_constants.cuh (+ tables header).
"""
import os, re, sys, random, json

# Point SRC at a checkout of the Jetsam node sources; everything else is derived.
#   JETSAM_SRC=/path/to/jetsam/src python3 tools/gen_hybrid.py
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("JETSAM_SRC", os.path.join(HERE, "..", "..", "jetsam", "src"))
GOLDEN = os.environ.get("JETSAM_GOLDEN", os.path.join(HERE, "..", "test", "golden.txt"))
OUTDIR = os.environ.get("JETSAM_OUTDIR", HERE)
if not os.path.isdir(SRC):
    sys.exit(f"node sources not found at {SRC} - set JETSAM_SRC to a checkout of the Jetsam node")
random.seed(20260905)

def read(p):
    with open(p) as f: return f.read()

def parse_u128_array(text, name, count):
    m = re.search(r"const\s+" + name + r"\s*:\s*\[u128;\s*\d+\]\s*=\s*\[(.*?)\];", text, re.S)
    if not m: sys.exit(f"cannot find {name}")
    vals = [int(x.replace("_", ""), 16) for x in re.findall(r"0x[0-9a-fA-F_]+", m.group(1))]
    assert len(vals) == count, name
    return vals

M128 = (1 << 128) - 1
M64 = (1 << 64) - 1

def apply_matrix(matrix, v):
    r = 0; k = 0
    while v:
        if v & 1: r ^= matrix[k]
        v >>= 1; k += 1
    return r

hw = read(f"{SRC}/jetsam_core/src/hardware.rs")
T2F = parse_u128_array(hw, "TOWER_TO_FLAT", 128)
F2T = parse_u128_array(hw, "FLAT_TO_TOWER", 128)
t2f = lambda v: apply_matrix(T2F, v)
f2t = lambda v: apply_matrix(F2T, v)

# ---------------------------------------------------------------- flat (GCM) field
def clmul(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        a <<= 1; b >>= 1
    return r

def gcm_reduce(x):                          # mod x^128 + x^7 + x^2 + x + 1
    hi = x >> 128; lo = x & M128
    v = clmul(hi, 0x87)                     # degree <= 134
    lo ^= v & M128
    lo ^= clmul(v >> 128, 0x87)
    return lo

def flat_mul(a, b): return gcm_reduce(clmul(a, b))
def tower_mul(a, b): return f2t(flat_mul(t2f(a), t2f(b)))   # Jetsam's field, through the reference matrices

# sanity: 1 is the identity, tower 2 (=x in GF(2^8)) squared is 4
assert tower_mul(1, 0x1234) == 0x1234 and tower_mul(2, 2) == 4 and tower_mul(0x20, 0x20) == 0x6C, (hex(tower_mul(1,0x1234)), hex(tower_mul(2,2)), hex(tower_mul(0x20,0x20)))

# ---------------------------------------------------------------- GF(2^64) poly field
P64 = (1 << 64) | (1 << 4) | (1 << 3) | (1 << 1) | 1

def pred(x):                                 # reduce a < 2^128 poly mod P64
    hi = x >> 64; lo = x & M64
    q = hi ^ (hi << 1) ^ (hi << 3) ^ (hi << 4)          # hi * (x^4+x^3+x+1), <= 68 bits
    lo ^= q & M64
    sp = q >> 64                                        # <= 4 bits
    lo ^= sp ^ (sp << 1) ^ (sp << 3) ^ (sp << 4)
    return lo

def pmul(a, b): return pred(clmul(a, b))
def psqr(a): return pred(clmul(a, a))
def ppow(a, e):
    r = 1
    while e:
        if e & 1: r = pmul(r, a)
        a = psqr(a); e >>= 1
    return r
def pinv(a):
    assert a
    return ppow(a, (1 << 64) - 2)

# P64 irreducible? x^(2^64) == x mod P and x^(2^32) != x
xx = 2
for _ in range(64): xx = psqr(xx)
assert xx == 2, "x^(2^64) != x : P64 not a field"
xx = 2
for _ in range(32): xx = psqr(xx)
assert xx != 2

# ---------------------------------------------------------------- GF(2) linear algebra on 64-bit columns
def solve_basis(cols):
    """cols: 64 ints (64-bit column vectors). Return inverse as function or None if singular."""
    n = len(cols)
    # Gaussian elimination: represent matrix rows; we want for a vector v the coefficients c with sum c_i cols[i] = v.
    # Build augmented system: row-reduce the 64x64 matrix whose columns are cols.
    rows = [0] * 64                          # rows[bit] = bitmask over columns having that bit
    for i, c in enumerate(cols):
        for bit in range(64):
            if (c >> bit) & 1: rows[bit] |= 1 << i
    # rows: 64 equations (one per output bit): sum_{i} rows[bit]_i * c_i = v_bit
    # Solve by elimination to get c_i as linear function of v: track a transform.
    piv_col = {}
    R = rows[:]; T = [1 << b for b in range(64)]   # T[bit] = combination of original rows
    used = [False] * 64
    for col in range(n):
        p = None
        for r in range(64):
            if not used[r] and (R[r] >> col) & 1: p = r; break
        if p is None: return None
        used[p] = True; piv_col[col] = p
        for r in range(64):
            if r != p and (R[r] >> col) & 1:
                R[r] ^= R[p]; T[r] ^= T[p]
    def inv(v):
        c = 0
        for col in range(n):
            p = piv_col[col]
            if bin(T[p] & v).count("1") & 1: c |= 1 << col
        return c
    return inv

# tower generator g = y3 (the GF(2^64) top variable): tower coordinate bit 32
g = 1 << 32
gp = [1]
for i in range(1, 65): gp.append(tower_mul(gp[-1], g))
assert all(v >> 64 == 0 for v in gp), "powers of y3 must stay in the GF(2^64) subfield"
Ainv = solve_basis(gp[:64])
assert Ainv is not None, "y3 does not generate GF(2^64)"
mg_low = Ainv(gp[64])                        # g^64 = sum c_i g^i  ->  m_g(x) = x^64 + sum c_i x^i
mg = [(mg_low >> i) & 1 for i in range(64)] + [1]    # coefficients over GF(2), degree 64

# ---------------------------------------------------------------- polynomials over GF(2^64)_poly
def ptrim(a):
    while a and a[-1] == 0: a.pop()
    return a
def padd(a, b):
    n = max(len(a), len(b)); r = [0] * n
    for i, v in enumerate(a): r[i] ^= v
    for i, v in enumerate(b): r[i] ^= v
    return ptrim(r)
def pmulp(a, b):
    if not a or not b: return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y: r[i + j] ^= pmul(x, y)
    return ptrim(r)
def pmod(a, m):
    a = a[:]; dm = len(m) - 1; inv_lead = pinv(m[-1])
    while len(a) - 1 >= dm and a:
        if a[-1]:
            coef = pmul(a[-1], inv_lead); shift = len(a) - 1 - dm
            for i, y in enumerate(m):
                if y: a[shift + i] ^= pmul(coef, y)
        ptrim(a)
    return a
def pgcd(a, b):
    a, b = a[:], b[:]
    while b:
        a, b = b, pmod(a, b)
    if a:
        il = pinv(a[-1]); a = [pmul(v, il) for v in a]
    return a
def pmulmod(a, b, m): return pmod(pmulp(a, b), m)

def find_root(mpoly):
    """A root in GF(2^64)_poly of mpoly (coeffs in GF(2^64)_poly, splits completely)."""
    f = mpoly[:]
    while len(f) - 1 > 1:
        d = len(f) - 1
        # x^(2^i) mod f for i = 0..63
        xp = [[0, 1]]
        for i in range(1, 64):
            prev = xp[-1]
            sq = [0] * (2 * len(prev) - 1)
            for j, v in enumerate(prev):
                if v: sq[2 * j] = psqr(v)
            xp.append(pmod(ptrim(sq), f))
        while True:
            delta = random.getrandbits(64)
            # T = sum_i (delta x)^(2^i) = sum_i delta^(2^i) * x^(2^i)
            T = []; dk = delta
            for i in range(64):
                T = padd(T, [pmul(dk, v) for v in xp[i]])
                dk = psqr(dk)
            gcd = pgcd(f, T)
            if 0 < len(gcd) - 1 < d:
                f = gcd if len(gcd) - 1 <= d // 2 or True else f
                break
        # keep the smaller factor to converge faster
    assert len(f) == 2
    return pmul(f[0], pinv(f[1]))            # root of a x + b is b/a ... f = [b, a] -> root = b/a

mg_poly = [c for c in mg]                    # GF(2) coefficients are valid GF(2^64) elements 0/1
rho = find_root(mg_poly)
# check: m_g(rho) == 0
acc = 0; pw = 1
for c in mg:
    if c: acc ^= pw
    pw = pmul(pw, rho)
assert acc == 0, "rho is not a root"

def build_phi(rho):
    rp = [1]
    for i in range(1, 64): rp.append(pmul(rp[-1], rho))
    # phi(v) = B (A^-1 v):  v (tower) -> coeffs c over g^i -> sum c_i rho^i
    def phi(v):
        c = Ainv(v); r = 0
        for i in range(64):
            if (c >> i) & 1: r ^= rp[i]
        return r
    cols = [phi(1 << k) for k in range(64)]
    inv = solve_basis(cols); assert inv is not None
    return cols, [inv(1 << k) for k in range(64)]

TAU4 = 0x2000_0000_0000_0000                 # Block64::TAU — the y4 extension constant, a GF(2^64) tower element
# choose the Frobenius conjugate of rho whose tau' is sparsest (cheapest constant multiply)
best = None
for k in range(64):
    cols, inv = build_phi(rho)
    tau_h = apply_matrix(cols, TAU4)
    w = bin(tau_h).count("1")
    if best is None or w < best[0]: best = (w, k, rho, cols, inv, tau_h)
    rho = psqr(rho)
w, k, rho, PHI, PHI_INV, TAU_H = best
phi = lambda v: apply_matrix(PHI, v)
phi_inv = lambda v: apply_matrix(PHI_INV, v)

# ---- make the top-level constant sparse: y' = y + beta  =>  y'^2 = y' + (tau' + beta + beta^2).
# beta^2 + beta ranges over the trace-0 hyperplane, so tau'' can be any trace-1 element: pick x^k.
def trace64(a):
    t = 0; v = a
    for _ in range(64): t ^= v; v = psqr(v)
    assert t in (0, 1); return t
assert trace64(TAU_H) == 1, "y^2+y+tau' must be irreducible (trace 1)"
# sparsest trace-1 element: a monomial if any, else a binomial x^i + x^j (both are cheap shift-and-fold multiplies)
mono = [k for k in range(64) if trace64(1 << k) == 1]
if mono:
    TAU_TERMS = [mono[0]]
else:
    TAU_TERMS = next(([i, j] for i in range(64) for j in range(i + 1, 64) if trace64((1 << i) | (1 << j)) == 1), None)
assert TAU_TERMS, "no sparse trace-1 element"
TAU_NEW = sum(1 << k for k in TAU_TERMS)
TAU_K = TAU_TERMS[0]
print("trace-1 monomials:", mono, " chosen tau'' terms:", TAU_TERMS)
L_cols = [psqr(1 << i) ^ (1 << i) for i in range(64)]          # columns of beta -> beta^2 + beta
def solve_linear(cols, target):
    # find c with sum c_i cols[i] = target (GF(2)); cols may be rank-deficient (rank 63 here)
    rows = [0] * 64
    for i, c in enumerate(cols):
        for bit in range(64):
            if (c >> bit) & 1: rows[bit] |= 1 << i
    aug = [(rows[b], (target >> b) & 1) for b in range(64)]
    piv = []; r = 0
    for col in range(64):
        pr = next((i for i in range(r, 64) if (aug[i][0] >> col) & 1), None)
        if pr is None: continue
        aug[r], aug[pr] = aug[pr], aug[r]
        for i in range(64):
            if i != r and (aug[i][0] >> col) & 1:
                aug[i] = (aug[i][0] ^ aug[r][0], aug[i][1] ^ aug[r][1])
        piv.append((r, col)); r += 1
    for i in range(r, 64):
        assert aug[i][1] == 0, "inconsistent system"
    c = 0
    for ri, col in piv:
        if aug[ri][1]: c |= 1 << col
    return c
BETA = solve_linear(L_cols, TAU_H ^ TAU_NEW)
assert psqr(BETA) ^ BETA == TAU_H ^ TAU_NEW
print(f"top level: y' = y + beta, beta = 0x{BETA:016x}, tau'' = 0x{TAU_NEW:x} (terms {TAU_TERMS})")
print(f"phi chosen: conjugate #{k}, tau' = 0x{TAU_H:016x} (popcount {w})")

# ---------------------------------------------------------------- verify phi is a field isomorphism
for _ in range(300):
    a = random.getrandbits(64); b = random.getrandbits(64)
    assert phi(tower_mul(a, b)) == pmul(phi(a), phi(b))
    assert phi(a ^ b) == phi(a) ^ phi(b) and phi_inv(phi(a)) == a
assert phi(1) == 1
print("phi: field isomorphism GF(2^64)_tower -> GF(2^64)_poly verified (300 random products)")

# ---------------------------------------------------------------- hybrid GF(2^128) arithmetic
def mulxk(v): return pmul(v, TAU_NEW)
def to_hyb(v):                                                      # tower u128 -> (lo', hi') hybrid
    lo, hi = phi(v & M64), phi(v >> 64)
    return (lo ^ pmul(BETA, hi), hi)
def from_hyb(h):
    hi = phi_inv(h[1]); lo = phi_inv(h[0] ^ pmul(BETA, h[1]))
    return lo | (hi << 64)
def hmul(a, b):
    a0, a1 = a; b0, b1 = b
    m0 = pmul(a0, b0); m1 = pmul(a1, b1); m2 = pmul(a0 ^ a1, b0 ^ b1)
    return (m0 ^ mulxk(m1), m2 ^ m0)
def hsqr(a):
    a0, a1 = a; s0 = psqr(a0); s1 = psqr(a1)
    return (s0 ^ mulxk(s1), s1)
def hxor(a, b): return (a[0] ^ b[0], a[1] ^ b[1])
for _ in range(300):
    a = random.getrandbits(128); b = random.getrandbits(128)
    assert from_hyb(hmul(to_hyb(a), to_hyb(b))) == tower_mul(a, b), "hybrid multiply != tower multiply"
    assert from_hyb(hsqr(to_hyb(a))) == tower_mul(a, a)
print("hybrid GF(2^128) multiply/square == Jetsam tower multiply (300 random pairs)")

# constant multiplies by GF(2^64)-subfield constants (MDS): c * (a0 + a1 y) = (c a0) + (c a1) y
def hmulc(c, a): return (pmul(c, a[0]), pmul(c, a[1]))
C2_H, C4_H = phi(2), phi(4)
DIAG_T = [0x20, 0x2000, 0x200, 0x800]
DIAG_H = [phi(c) for c in DIAG_T]
for c in [2, 4] + DIAG_T:
    for _ in range(50):
        a = random.getrandbits(128)
        assert from_hyb(hmulc(phi(c), to_hyb(a))) == tower_mul(c, a)

# ---------------------------------------------------------------- permutation in the hybrid basis vs golden
perm = read(f"{SRC}/jetsam_poseidon2b/src/native/permutation.rs")
m = re.search(r"pub const ROUND_CONSTANTS:\s*\[\[u128;\s*N_ROUNDS\];\s*STATE_SIZE\]\s*=\s*\[(.*?)\n\];", perm, re.S)
rows = re.findall(r"\[\s*((?:0x[0-9a-fA-F_]+\s*,?\s*)+)\]", m.group(1), re.S)
RC_TOWER = [[int(x.replace("_", ""), 16) for x in re.findall(r"0x[0-9a-fA-F_]+", row)] for row in rows]
assert len(RC_TOWER) == 4 and all(len(r) == 66 for r in RC_TOWER)
# Capacity IV, straight from the domain tag, exactly as native/domain.rs does it:
#   capacity_iv(tag) = [Block128::from(label << 64), Block128::from(label)]
# with label = the eight ASCII bytes of the tag read big-endian. Block128 is already
# a tower-basis element, so these two words ARE the tower IV — no conversion needed.
dom = read(f"{SRC}/jetsam_poseidon2b/src/native/domain.rs")
md = re.search(r'TAG_POWHDR:\s*DomainTag\s*=\s*DomainTag::new\(b"(.{8})"\)', dom)
assert md, "TAG_POWHDR not found in domain.rs"
POW_TAG = md.group(1).encode()
label = int.from_bytes(POW_TAG, "big")
IV_TOWER = [label << 64, label]
print(f"domain tag: {POW_TAG.decode()} -> label 0x{label:016x}")
RC_H = [[to_hyb(v) for v in row] for row in RC_TOWER]
IV_H = [to_hyb(v) for v in IV_TOWER]
N_ROUNDS, F_ROUNDS, P_ROUNDS = 66, 8, 58

def mds_full_h(s):
    a, b, c, d = s
    a4 = hmulc(C4_H, a); b2 = hmulc(C2_H, b); b4 = hmulc(C4_H, b); c4 = hmulc(C4_H, c); d2 = hmulc(C2_H, d); d4 = hmulc(C4_H, d)
    X = hxor
    return [X(X(X(a4, a), X(X(b4, b2), b)), X(c, X(d2, d))),
            X(X(a4, X(b4, b2)), X(c, d)),
            X(X(a, X(b2, b)), X(X(c4, c), X(X(d4, d2), d))),
            X(X(a, b), X(c4, X(d4, d2)))]
def mds_partial_h(s):
    S = hxor(hxor(s[0], s[1]), hxor(s[2], s[3]))
    return [hxor(hmulc(DIAG_H[i], s[i]), hxor(S, s[i])) for i in range(4)]
def sbox_h(x):
    x2 = hsqr(x); x4 = hsqr(x2); x6 = hmul(x, x2); return hmul(x6, x4)
def permute_h(s):
    s = mds_full_h(s)
    for r in range(N_ROUNDS):
        partial = F_ROUNDS // 2 <= r < F_ROUNDS // 2 + P_ROUNDS
        if not partial:
            s = [sbox_h(hxor(s[i], RC_H[i][r])) for i in range(4)]
            s = mds_full_h(s)
        else:
            s[0] = sbox_h(hxor(s[0], RC_H[0][r]))
            s = mds_partial_h(s)
    return s

def golden_vectors(n):
    out = []
    with open(GOLDEN) as f:
        for line in f:
            if line[0] != 'V': continue
            it = line[1:].split()
            fields = [int(x, 16) for x in it[:16]]           # big-endian u128 hex per field (tower)
            digest = bytes.fromhex(it[21])
            out.append((fields, digest))
            if len(out) >= n: break
    return out

def digest_bytes(s0, s1):
    return (s0 & M64).to_bytes(8, 'little') + (s0 >> 64).to_bytes(8, 'little') + (s1 & M64).to_bytes(8, 'little') + (s1 >> 64).to_bytes(8, 'little')

ok = 0
vecs = golden_vectors(12)
for fields, digest in vecs:
    s = [(0, 0), (0, 0), IV_H[0], IV_H[1]]
    for k in range(8):
        s[0] = hxor(s[0], to_hyb(fields[2 * k])); s[1] = hxor(s[1], to_hyb(fields[2 * k + 1]))
        s = permute_h(s)
    d = digest_bytes(from_hyb(s[0]), from_hyb(s[1]))
    ok += (d == digest)
print(f"hybrid sponge vs golden: {ok} / {len(vecs)} vectors match")
assert ok == len(vecs)

# ---------------------------------------------------------------- MDS-partial diag constant tables (64->64 linear maps)
# byte tables: T[c][k][v] = c * (v << 8k), 8 x 256 u64 per constant (16 KB each)
def f128(lo, hi): return f"{{0x{lo:016x}ULL,0x{hi:016x}ULL}}"
out = os.path.join(OUTDIR, "jetsam_pow_hybrid_constants.cuh")
with open(out, "w") as o:
    o.write("// GENERATED by gen_hybrid.py — hybrid basis GF(2^64)[y]/(y^2+y+TAU_H), GF(2^64)=GF(2)[x]/(x^64+x^4+x^3+x+1).\n")
    o.write("// Anchored on the Rust reference (TOWER_TO_FLAT/FLAT_TO_TOWER + GCM multiply); golden-checked in Python.\n#pragma once\n\n")
    o.write(f"#define HYB_TAU 0x{TAU_NEW:016x}ULL   // y'^2 = y' + tau'', terms x^{TAU_TERMS}\n")
    o.write("#define HYB_TAU_TERMS " + ",".join(str(k) for k in TAU_TERMS) + "\n#define HYB_TAU_NTERMS " + str(len(TAU_TERMS)) + "\n")
    o.write(f"#define HYB_C2 0x{C2_H:016x}ULL\n#define HYB_C4 0x{C4_H:016x}ULL\n")
    o.write("__device__ __constant__ u64 HYB_DIAG[4] = {" + ", ".join(f"0x{v:016x}ULL" for v in DIAG_H) + "};\n")
    o.write("__device__ __constant__ F128 IV_HYB[2] = {" + ", ".join(f128(*v) for v in IV_H) + "};\n")
    o.write("__device__ __constant__ F128 RC_HYB[4][66] = {\n")
    for i in range(4):
        o.write("  {\n")
        for r in range(0, 66, 3):
            o.write("    " + ", ".join(f128(*RC_H[i][kk]) for kk in range(r, min(r + 3, 66))) + ",\n")
        o.write("  },\n")
    o.write("};\n")
    # nibble tables for phi and phi^-1 on 64-bit values: [16 chunks][16 values] u64
    o.write("// phi (tower->hybrid) and phi^-1 as nibble tables on 64-bit halves: PHI_NIB[chunk][nibble]\n")
    bphi = lambda h: pmul(BETA, phi(h))                 # hi -> beta*phi(hi)      (lo' = phi(lo) ^ bphi(hi))
    phiinv_b = lambda h: phi_inv(pmul(BETA, h))          # hi' -> phi^-1(beta*hi') (lo = phiinv(lo') ^ phiinv_b(hi'))
    for name, fn in (("PHI_NIB", phi), ("BPHI_NIB", bphi), ("PHI_INV_NIB", phi_inv), ("PHI_INV_B_NIB", phiinv_b)):
        o.write(f"__device__ __constant__ u64 {name}[16][16] = {{\n")
        for c in range(16):
            o.write("  {" + ", ".join(f"0x{fn(v << (4 * c)):016x}ULL" for v in range(16)) + "},\n")
        o.write("};\n")
    # host-side copies (plain arrays) for template conversion
    for name, fn in (("H_PHI_NIB", phi), ("H_BPHI_NIB", bphi), ("H_PHI_INV_NIB", phi_inv), ("H_PHI_INV_B_NIB", phiinv_b)):
        o.write(f"static const u64 {name}[16][16] = {{\n")
        for c in range(16):
            o.write("  {" + ", ".join(f"0x{fn(v << (4 * c)):016x}ULL" for v in range(16)) + "},\n")
        o.write("};\n")
print("wrote", out)

out2 = os.path.join(OUTDIR, "jetsam_pow_hybrid_tables.cuh")
with open(out2, "w") as o:
    o.write("// GENERATED by gen_hybrid.py — byte tables for the MDS_PARTIAL diagonal constants (64->64 linear maps).\n")
    o.write("// HYB_DIAG_BYTE[i][k][v] = DIAG_H[i] * (v << 8k) in GF(2^64)_poly.  4 x 16 KB.\n#pragma once\n")
    o.write("__device__ u64 HYB_DIAG_BYTE[4][8][256] = {\n")
    for i in range(4):
        o.write(" {\n")
        for kk in range(8):
            o.write("  {" + ",".join(f"0x{pmul(DIAG_H[i], v << (8 * kk)):016x}ULL" for v in range(256)) + "},\n")
        o.write(" },\n")
    o.write("};\n")
print("wrote", out2)
json.dump({"tau_h": TAU_H, "tau_terms": TAU_TERMS, "beta": BETA, "c2": C2_H, "c4": C4_H, "diag": DIAG_H, "phi_cols": PHI, "phi_inv_cols": PHI_INV,
           "conjugate": k}, open(os.path.join(OUTDIR, "hybrid.json"), "w"))
print("TAU_H=0x%016x  C2=0x%016x  C4=0x%016x  DIAG=%s" % (TAU_H, C2_H, C4_H, ["0x%016x" % v for v in DIAG_H]))
