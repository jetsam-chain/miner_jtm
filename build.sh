#!/usr/bin/env bash
# Build the Jetsam GPU miner (TowerHash, CUDA).
#
#   ./build.sh                 # every architecture your CUDA knows, from Turing up
#   ./build.sh -gencode arch=compute_86,code=sm_86      # one architecture only
#
# Requires CUDA 11.x or newer and a host compiler nvcc accepts. nvcc rejects very
# recent g++; set CCBIN if your default is too new:
#   CCBIN=/usr/bin/g++-12 ./build.sh
#
# Blackwell (sm_100/120) needs CUDA 12.8+; Hopper (sm_90) needs 11.8+. Rather than
# fail on a toolkit that has never heard of your card's architecture, this script
# asks nvcc what it supports and builds for the intersection. Anything newer than
# the toolkit still runs, through the PTX embedded below.
set -eu
NVCC=${NVCC:-nvcc}
CCBIN=${CCBIN:-}
OUT=${OUT:-jetsam-miner}

# Ampere and later (sm_80+) get the hardware carry-less multiply-add, clmad. Turing
# (sm_75) has no such instruction — ptxas rejects it outright — so it compiles the
# software multiply in jetsam_pow_hybrid.cuh instead. Same results, bit for bit; about
# half the throughput. Both paths must pass --selftest before you mine with them.
WANT="75 80 86 89 90 100 120"
if [ $# -gt 0 ]; then
    GEN="$*"
else
    HAVE=$($NVCC --list-gpu-arch 2>/dev/null | sed 's/compute_//' | tr '\n' ' ')
    [ -n "$HAVE" ] || HAVE="$WANT"          # very old nvcc without --list-gpu-arch
    GEN=""; NEWEST=""
    for a in $WANT; do
        case " $HAVE " in *" $a "*) GEN="$GEN -gencode arch=compute_$a,code=sm_$a"; NEWEST=$a;; esac
    done
    [ -n "$GEN" ] || { echo "this CUDA supports no architecture from Turing up" >&2; exit 1; }
    # PTX for anything newer than this toolkit: the driver JIT-compiles it at load.
    GEN="$GEN -gencode arch=compute_75,code=compute_75 -gencode arch=compute_$NEWEST,code=compute_$NEWEST"
    echo "building for sm_$(echo $GEN | grep -oE 'code=sm_[0-9]+' | sed 's/code=sm_//' | tr '\n' ' ')+ PTX"
fi

cd "$(dirname "$0")"
mkdir -p build

# clmad needs a recent PTX ISA as well as recent silicon: older ptxas answers
# "Not a name of any known instruction: 'clmad.lo'" and the build dies. Ask it
# once, on a two-line program, rather than let the user meet that error.
SOFT=""
cat > build/.clmad_probe.cu <<'EOF'
__global__ void k(unsigned long long* o) {
    unsigned long long d;
    asm("clmad.lo.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(o[0]), "l"(o[1]), "l"(0ULL));
    o[0] = d;
}
EOF
if ! $NVCC ${CCBIN:+-ccbin $CCBIN} -arch=sm_80 -ptx -o /dev/null build/.clmad_probe.cu >/dev/null 2>&1 \
   || ! $NVCC ${CCBIN:+-ccbin $CCBIN} -arch=sm_80 -cubin -o /dev/null build/.clmad_probe.cu >/dev/null 2>&1; then
    SOFT="-DJETSAM_FORCE_SOFT_CLMUL"
    echo "!! this CUDA toolkit does not know the clmad instruction."
    echo "!! building the software multiply instead - correct, but roughly half the"
    echo "!! throughput. CUDA 12.8 or newer gets you the fast path on sm_80+ cards."
fi
rm -f build/.clmad_probe.cu

# shellcheck disable=SC2086
$NVCC ${CCBIN:+-ccbin $CCBIN} -O3 -std=c++17 $GEN $SOFT -cudart static \
      -o "build/$OUT" src/jetsam_miner.cu
echo "built build/$OUT  ($(stat -c%s "build/$OUT") bytes)"
echo
echo "Now run the gate before you mine with it:"
echo "  ./build/$OUT --selftest test/golden.txt --device 0     # must print 12000 / 12000"
