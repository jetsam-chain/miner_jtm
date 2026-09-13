# Jetsam GPU miner

A CUDA miner for **TowerHash**, the Jetsam proof-of-work. It talks straight to a
Jetsam node over JSON-RPC, so you can mine solo against your own node, or point it
at a pool that speaks the same two calls.

No telemetry, no bundled address, no phone-home. The only host it ever contacts is
the one you pass to `--rpc`.

## Requirements

- An NVIDIA GPU, **Turing (sm_75) or newer**.
- **CUDA 13.3 or newer** for the fast path. The round function uses `clmad`, a
  carry-less multiply-add that older toolkits do not know — including CUDA 13.0, which
  is recent enough to look fine and is not. Older toolkits still build: the script
  detects it and compiles a software multiply instead, bit-for-bit identical and
  about a third of the speed. Watch for the warning it prints.
- Linux is the tested platform. The source builds for Windows (Winsock2 under
  `_WIN32`), but no Windows binary is published and that path has not been run.

## Build

```sh
./build.sh
```

It builds for every architecture your CUDA knows, from Turing up, and embeds PTX so
that cards newer than your toolkit still run. If nvcc rejects your host compiler:

```sh
CCBIN=/usr/bin/g++-12 ./build.sh
```

## Run the gate first

```sh
./build/jetsam-miner --selftest test/golden.txt --device 0
```

It must print `12000 / 12000`. Those vectors come from the node's own reference
implementation: if a single one disagrees, this binary would mine invalid blocks on
your electricity. Run it after every build, on every card you intend to mine with —
a kernel can be correct on one architecture and wrong on another.

## Mine

```sh
./build/jetsam-miner --rpc http://127.0.0.1:9701 --coinbase <your-address>
```

Add `--key TOKEN` if your node or pool requires one. One instance per card:

```sh
./build/jetsam-miner --rpc http://127.0.0.1:9701 --coinbase <addr> --device 0
./build/jetsam-miner --rpc http://127.0.0.1:9701 --coinbase <addr> --device 1
```

| Option | Default | |
|---|---|---|
| `--rpc URL` | `http://127.0.0.1:9701` | node JSON-RPC endpoint (http only) |
| `--coinbase ADDR` | — | where the reward goes; required for solo mining |
| `--key TOKEN` | — | bearer token, if the node asks for one |
| `--device N` | `0` | which GPU |
| `--poll-ms MS` | `400` | how often to ask for a new template |
| `--blocks N` | one per SM | CUDA blocks |
| `--threads N` | `1024` | threads per block |
| `--batch N` | `32` | nonces per thread per launch |

To measure the card without a node, `JETSAM_BENCH_SECONDS=30 ./build/jetsam-miner`
runs the kernel against an impossible target and prints the rate. Measured figures
for several cards are in [BENCHMARK.md](BENCHMARK.md), along with what they do and
do not tell you.

The miner keeps hashing across template changes — it never stops the GPU to talk to
the node — and reports its measured rate in an `X-Jetsam-Hashrate` header so a pool
can attribute work to it. Mining solo, nothing reads that header.

## Where the constants come from

`src/jetsam_pow_hybrid_constants.cuh` and `..._tables.cuh` hold round constants, an
IV, MDS constants and basis-change tables. **They are generated, not transcribed.**

The miner does not work in Jetsam's own field representation. It works in a hybrid
basis, `GF(2^128) = GF(2^64)[y]/(y^2+y+tau')` over `GF(2^64) = GF(2)[x]/(x^64+x^4+x^3+x+1)`,
because in that basis a multiply is three carry-less products and a shift reduction,
and the state never needs converting inside the permutation — Jetsam's own basis
would pay a conversion in every round.

`tools/gen_hybrid.py` derives that representation from the node sources: it finds
the isomorphism as a root of the tower generator's minimal polynomial
(Cantor–Zassenhaus), picks the Frobenius conjugate that makes `tau'` sparsest,
verifies it is a field isomorphism on random products, re-implements the sponge in
the new basis, checks it against the golden vectors, and writes both headers.

```sh
JETSAM_SRC=/path/to/jetsam/src python3 tools/gen_hybrid.py
```

That reproduces the shipped headers byte for byte. Nothing in this repository is a
magic number you have to take on trust.

## Turing and the software multiply

The hybrid basis leans on `clmad`, a carry-less multiply-add that exists on Ampere
and later. Turing has no such instruction, and neither does any AMD GPU. For those,
`jetsam_pow_hybrid.cuh` builds the product from ordinary integer multiplies by the
spaced-bits method: split each operand by the residue of its bit positions mod 4, so
each column of the integer product accumulates at most eight terms and no carry ever
crosses between occupied columns; mask the result back and you have the carry-less
product. Karatsuba then builds 64×64 from three 32×32 products instead of four.

It was checked against a naive shift-xor reference on 4,194,304 random pairs with
zero disagreements, and it passes the same 12000-vector gate. Measured cost on one
card: **×2.3** — a Turing card mines at about 43 % of what the same silicon would do
with `clmad`. Slower, but real mining.

`-DJETSAM_FORCE_SOFT_CLMUL` forces that path on any card, which is how it gets
tested on hardware that does have `clmad`.

## Licence

Apache 2.0 — see `LICENSE`, and `NOTICE` for attribution.
