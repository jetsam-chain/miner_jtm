// jetsam_miner.cu — CUDA GPU miner for TowerHash, the Jetsam proof-of-work.
// The sponge state lives in the HYBRID basis GF(2^64)[y]/(y^2+y+tau) with
// GF(2^64) in polynomial basis (see jetsam_pow_hybrid.cuh): no basis conversion in
// the round loop, S-box on CLMAD 64x64, MDS_PARTIAL diagonal from byte tables in
// shared memory. Template fields are converted once on the host; the digest once per
// hash. Geometry: one 1024-thread block per SM (73 KB of shared memory per block).
//
//   ./jetsam-miner --selftest test/golden.txt          bit-exact gate (run it first)
//   ./jetsam-miner --rpc http://127.0.0.1:9701 [--key TOKEN] [--coinbase ADDR] [--device 0]
//
// Protocol: jetsam_getBlockTemplate("") -> {template_id, pow_fields_hex (16x16 B LE),
// nonce_field_index, difficulty_target_hex (32 B LE)}; jetsam_submitBlock(id, nonce_hex).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#ifdef _WIN32
// Winsock is the same BSD sockets API with different spellings. Everything the
// miner needs is covered by these few aliases, so the code below stays single-source.
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef int socklen_t;
#  define close(fd)            closesocket(fd)
#  define SOCK_T               SOCKET
#  define BAD_SOCK             INVALID_SOCKET
   // setsockopt takes char* on Windows, and SO_RCVTIMEO is a DWORD of milliseconds
   // rather than a struct timeval — hence the small wrapper used for the timeouts.
   static inline void set_sock_timeout(SOCK_T fd, int which, int ms) {
       DWORD v = (DWORD)ms; setsockopt(fd, SOL_SOCKET, which, (const char*)&v, sizeof v);
   }
   static inline void set_nodelay(SOCK_T fd) {
       BOOL one = TRUE; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
   }
#else
#  include <unistd.h>
#  include <netdb.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  define SOCK_T               int
#  define BAD_SOCK             (-1)
   static inline void set_sock_timeout(SOCK_T fd, int which, int ms) {
       struct timeval tv{ ms / 1000, (ms % 1000) * 1000 };
       setsockopt(fd, SOL_SOCKET, which, &tv, sizeof tv);
   }
   static inline void set_nodelay(SOCK_T fd) {
       int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
   }
#endif
#include <cuda_runtime.h>
#define HYB_DIAG_TABLE 1
#include "jetsam_pow_hybrid.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA ERR %s @%d : %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const char* ts() {
    static char b[32];
    time_t t = time(nullptr); struct tm g;
#ifdef _WIN32
    gmtime_s(&g, &t);
#else
    gmtime_r(&t, &g);
#endif
    snprintf(b, sizeof b, "%02d:%02d:%02d", g.tm_hour, g.tm_min, g.tm_sec);
    return b;
}
#define LOG(...) do{ printf("[%s] ", ts()); printf(__VA_ARGS__); printf("\n"); fflush(stdout);}while(0)

#ifndef MINE_LAUNCH_BOUNDS
#define MINE_LAUNCH_BOUNDS
#endif

// =========================================================== kernels
#define SH_BYTES ((HYB_SH_WORDS + HYB_DIAG_WORDS) * 8)
#define SH_SETUP  extern __shared__ __align__(16) u64 sh[]; load_hyb_tables(sh); u64* sh_diag = sh + HYB_SH_WORDS; load_diag_tables(sh_diag); __syncthreads();

// Gate and re-validation: raw tower-basis fields in, tower-basis digest out.
__global__ void k_selftest(const F128* __restrict__ fields, F128* __restrict__ out, int n)
{
    SH_SETUP
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    F128 s[4];
    s[0].lo = 0; s[0].hi = 0; s[1] = s[0];
    s[2] = IV_HYB[0]; s[3] = IV_HYB[1];
#pragma unroll 1
    for (int k = 0; k < 8; ++k) {
        s[0] = fx(s[0], tower_to_hyb(sh, fields[i * 16 + 2 * k]));
        s[1] = fx(s[1], tower_to_hyb(sh, fields[i * 16 + 2 * k + 1]));
        permute_hyb(s, sh_diag);
    }
    out[2 * i] = hyb_to_tower(sh, s[0]);
    out[2 * i + 1] = hyb_to_tower(sh, s[1]);
}

// ff: the 16 template fields already in the hybrid basis (ff[0] is ignored); base is the
// hybrid image of the nonce with its low 32 bits zero, so nonce = base ^ PHI(idx).
// The nonce sits in lane 0 of the FIRST absorbed pair: all eight permutations per attempt.
__global__ void __launch_bounds__(1024, 1)      // one 1024-thread block per SM: ptxas must fit 64 regs
k_mine(const F128* __restrict__ ff,
       F128 base, F128 t0, F128 t1,
       unsigned start, unsigned per_thread,
       unsigned* __restrict__ found, unsigned* __restrict__ fidx)
{
    SH_SETUP
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
#pragma unroll 1
    for (unsigned it = 0; it < per_thread; ++it) {
        unsigned idx = start + tid + it * stride;
        F128 nonce; nonce.lo = base.lo ^ lin32(sh, idx); nonce.hi = base.hi;
        F128 s[4];
        s[0].lo = 0; s[0].hi = 0; s[1] = s[0];
        s[2] = IV_HYB[0]; s[3] = IV_HYB[1];
#pragma unroll 1
        for (int k = 0; k < 8; ++k) {
            F128 f0 = (k == 0) ? nonce : ff[2 * k];
            s[0] = fx(s[0], f0);
            s[1] = fx(s[1], ff[2 * k + 1]);
            permute_hyb(s, sh_diag);
        }
        F128 d1 = hyb_to_tower(sh, s[1]);
        if (d1.hi <= t1.hi) {
            F128 d0 = hyb_to_tower(sh, s[0]);
            if (le256_lt(d0, d1, t0, t1)) {
                if (atomicCAS(found, 0u, 1u) == 0u) *fidx = idx;
            }
        }
    }
}

// =========================================================== host helpers
static int hexval(char c){ return (c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:(c>='A'&&c<='F')?c-'A'+10:-1; }

// ---- minimal HTTP/1.1 client (plain JSON-RPC) ----
struct Endpoint { std::string host; int port = 9701; std::string path = "/"; };
static bool parse_url(const std::string& u, Endpoint& e) {
    std::string s = u;
    if (s.rfind("http://", 0) == 0) s = s.substr(7);
    else if (s.rfind("https://", 0) == 0) { fprintf(stderr, "TLS not supported: use http:// (or a local tunnel)\n"); return false; }
    size_t sl = s.find('/');
    if (sl != std::string::npos) { e.path = s.substr(sl); s = s.substr(0, sl); }
    size_t co = s.find(':');
    if (co != std::string::npos) { e.port = atoi(s.c_str() + co + 1); s = s.substr(0, co); }
    e.host = s;
    return !e.host.empty();
}

// Throughput over the last statistics window, reported to the node in the
// X-Jetsam-Hashrate header. Without it a pool can know NOTHING of what this card
// is worth: it only observes template requests, and a solution is far too rare to
// infer a rate from. Harmless when you mine solo against your own node.
static std::atomic<unsigned long long> g_measured_hs{0};

// Identity announced to the node. The device number is appended at start-up: a
// multi-GPU machine runs ONE instance per card, all behind the same address, and
// without that number they would collapse into a single line in the pool view.
static char g_version_header[48] = "gpu-tower-clmad/0.2";

// `status` receives the HTTP status code, or 0 if the node could not be reached at
// all. The caller needs the difference: an unreachable node and a node that refuses
// the mining key are two different problems with two different fixes.
static std::string http_post(const Endpoint& e, const std::string& body, const std::string& bearer, int tmo_ms,
                             int* status = nullptr)
{
    if (status) *status = 0;
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    char portstr[16]; snprintf(portstr, sizeof portstr, "%d", e.port);
    if (getaddrinfo(e.host.c_str(), portstr, &hints, &res) != 0) return "";
    SOCK_T fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == BAD_SOCK) { freeaddrinfo(res); return ""; }
    set_sock_timeout(fd, SO_RCVTIMEO, tmo_ms);
    set_sock_timeout(fd, SO_SNDTIMEO, tmo_ms);
    set_nodelay(fd);
    if (connect(fd, res->ai_addr, (socklen_t)res->ai_addrlen) != 0) { close(fd); freeaddrinfo(res); return ""; }
    freeaddrinfo(res);
    std::string req = "POST " + e.path + " HTTP/1.1\r\nHost: " + e.host + "\r\n"
                      "Content-Type: application/json\r\nConnection: close\r\n";
    if (!bearer.empty()) req += "Authorization: Bearer " + bearer + "\r\n";
    // The binary states what it is and what it is worth. A pool indexes its
    // workers by address: without this marker a GPU miner and a CPU miner on the
    // SAME machine collapse into one line and each overwrites the other's rate.
    // The `gpu` prefix is what lets them be counted separately.
    req += std::string("X-Jetsam-Version: ") + g_version_header + "\r\n";
    unsigned long long hs = g_measured_hs.load(std::memory_order_relaxed);
    if (hs > 0) req += "X-Jetsam-Hashrate: " + std::to_string(hs) + "\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t off = 0;
    // send/recv return int on Windows and ssize_t elsewhere; long long holds both.
    while (off < req.size()) { long long w = send(fd, req.data() + off, (int)(req.size() - off), 0); if (w <= 0) { close(fd); return ""; } off += (size_t)w; }
    std::string resp; char buf[8192]; long long r;
    while ((r = recv(fd, buf, (int)sizeof buf, 0)) > 0) resp.append(buf, (size_t)r);
    close(fd);
    if (status && resp.compare(0, 5, "HTTP/") == 0) {
        size_t sp = resp.find(' ');
        if (sp != std::string::npos) *status = atoi(resp.c_str() + sp + 1);
    }
    size_t hdr = resp.find("\r\n\r\n");
    return (hdr == std::string::npos) ? std::string() : resp.substr(hdr + 4);
}

// Rudimentary JSON extractors (the schema is fixed and known)
static std::string jstr(const std::string& j, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = j.find(k); if (p == std::string::npos) return "";
    p = j.find(':', p + k.size()); if (p == std::string::npos) return "";
    while (p < j.size() && (j[p] == ':' || j[p] == ' ')) ++p;
    if (p >= j.size() || j[p] != '"') return "";
    size_t q = j.find('"', p + 1); if (q == std::string::npos) return "";
    return j.substr(p + 1, q - p - 1);
}
static long long jnum(const std::string& j, const char* key, long long dflt) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = j.find(k); if (p == std::string::npos) return dflt;
    p = j.find(':', p + k.size()); if (p == std::string::npos) return dflt;
    ++p; while (p < j.size() && j[p] == ' ') ++p;
    return atoll(j.c_str() + p);
}

struct Tmpl {
    std::string id, fields_hex, target_hex;
    long long height = -1, nonce_idx = -1, expires = 0;
    bool ok() const { return !id.empty() && fields_hex.size() == 512 && target_hex.size() == 64 && nonce_idx == 0; }
};

static bool fetch_template(const Endpoint& e, const std::string& key, const std::string& coinbase, Tmpl& t,
                           int* status = nullptr)
{
    std::string body = std::string("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"jetsam_getBlockTemplate\","
                                   "\"params\":[\"") + coinbase + "\"]}";
    std::string r = http_post(e, body, key, 8000, status);
    if (r.empty()) return false;
    Tmpl n;
    n.id = jstr(r, "template_id");
    n.fields_hex = jstr(r, "pow_fields_hex");
    n.target_hex = jstr(r, "difficulty_target_hex");
    n.height = jnum(r, "height", -1);
    n.nonce_idx = jnum(r, "nonce_field_index", -1);
    n.expires = jnum(r, "expires_in_seconds", 0);
    if (!n.ok()) { static int once = 0; if (!once++) LOG("unexpected RPC response: %.200s", r.c_str()); return false; }
    t = n; return true;
}

// =========================================================== gate bit-exact
static F128 parse128_be(const char* p) {
    F128 r; r.hi = 0; r.lo = 0;
    for (int i = 0; i < 16; ++i) r.hi = (r.hi << 4) | (u64)hexval(p[i]);
    for (int i = 16; i < 32; ++i) r.lo = (r.lo << 4) | (u64)hexval(p[i]);
    return r;
}
static int run_selftest(const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) { LOG("golden file not found: %s", path); return 1; }
    static char line[4096]; int cap = 20000, n = 0;
    F128* hf = (F128*)malloc((size_t)cap * 16 * sizeof(F128));
    unsigned char* hd = (unsigned char*)malloc((size_t)cap * 32);
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'V') continue;
        const char* p = line + 1;
        for (int j = 0; j < 16; ++j) { while (*p == ' ') ++p; hf[(size_t)n*16+j] = parse128_be(p); p += 32; }
        for (int j = 0; j < 5; ++j)  { while (*p == ' ') ++p; p += 32; }
        while (*p == ' ') ++p;
        for (int b = 0; b < 32; ++b) hd[(size_t)n*32+b] = (unsigned char)((hexval(p[2*b])<<4)|hexval(p[2*b+1]));
        if (++n >= cap) break;
    }
    fclose(f);
    F128 *df, *dout; CK(cudaMalloc(&df,(size_t)n*16*sizeof(F128))); CK(cudaMalloc(&dout,(size_t)n*2*sizeof(F128)));
    CK(cudaMemcpy(df, hf, (size_t)n*16*sizeof(F128), cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(k_selftest, cudaFuncAttributeMaxDynamicSharedMemorySize, SH_BYTES));
    k_selftest<<<(n+63)/64,64,SH_BYTES>>>(df, dout, n); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    F128* ho = (F128*)malloc((size_t)n*2*sizeof(F128));
    CK(cudaMemcpy(ho, dout, (size_t)n*2*sizeof(F128), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        unsigned char g[32];
        for (int b=0;b<8;++b){ g[b]=(unsigned char)(ho[2*i].lo>>(8*b)); g[8+b]=(unsigned char)(ho[2*i].hi>>(8*b));
                               g[16+b]=(unsigned char)(ho[2*i+1].lo>>(8*b)); g[24+b]=(unsigned char)(ho[2*i+1].hi>>(8*b)); }
        if (memcmp(g, hd + (size_t)i*32, 32)) ++bad;
    }
    LOG("BIT-EXACT GATE: %d / %d vectors match the Rust reference%s", n - bad, n, bad ? "  ** FAILED **" : "");
    return bad ? 2 : 0;
}

// =========================================================== miner
struct Shared { std::mutex m; Tmpl t; bool fresh = false; };

int main(int argc, char** argv)
{
    std::string rpc = "http://127.0.0.1:9701", key, coinbase, golden;
    // Defaults: one 1024-thread block per SM (the 73 KB of shared memory allow a single
    // block per SM), 8 nonces per thread per launch. blocks=0 -> multiProcessorCount.
    int device = 0, poll_ms = 400, blocks = 0, thr = 1024;
    unsigned per_thread = 32;   // ~1M nonces per launch (~120 ms at 8 MH/s): amortises the host wake-up under CPU load
    bool selftest_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]{ return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--selftest")  { selftest_only = true; golden = nx(); }
        else if (a == "--rpc")       rpc = nx();
        else if (a == "--key")       key = nx();
        else if (a == "--coinbase")  coinbase = nx();
        else if (a == "--device")    device = atoi(nx());
        else if (a == "--poll-ms")   poll_ms = atoi(nx());
        else if (a == "--blocks")    blocks = atoi(nx());
        else if (a == "--threads")   thr = atoi(nx());
        else if (a == "--batch")     per_thread = (unsigned)atoi(nx());
        else { printf("usage: %s [--selftest golden.txt] [--rpc URL] [--key TOK] [--coinbase ADDR]\n"
                      "          [--device N] [--poll-ms MS] [--blocks N] [--threads N] [--batch N]\n", argv[0]); return 1; }
    }
    CK(cudaSetDevice(device));
    // Flags apply to the CURRENT device: set them after cudaSetDevice, otherwise they land
    // on device 0 (a spinning host thread and a phantom context on card 0 for every other card).
    if (getenv("JETSAM_BLOCKING_SYNC")) CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, device));
    if (blocks <= 0) blocks = prop.multiProcessorCount;
    CK(cudaFuncSetAttribute(k_mine, cudaFuncAttributeMaxDynamicSharedMemorySize, SH_BYTES));
    CK(cudaFuncSetAttribute(k_selftest, cudaFuncAttributeMaxDynamicSharedMemorySize, SH_BYTES));   // gate AND found-nonce re-validation
    snprintf(g_version_header, sizeof g_version_header, "gpu-tower-clmad/0.2.d%d", device);   // same family as 0.1: the dashboard's "behind" check stays meaningful
    LOG("GPU %d : %s (sm_%d%d, %d SM)  [hybrid-basis kernel, MDS tables in shared]", device, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    if (selftest_only) return run_selftest(golden.empty() ? "golden.txt" : golden.c_str());
    if (!golden.empty() && run_selftest(golden.c_str())) return 2;

    // --- bench mode: raw k_mine kernel throughput, no network, no template ---
    // Impossible target (0) -> no hit ever; the fields come from a fixed LCG,
    // so the measurement is deterministic and identical across versions.
    if (getenv("JETSAM_BENCH_SECONDS")) {
        double secs = atof(getenv("JETSAM_BENCH_SECONDS"));
        if (secs <= 0) secs = 10.0;
        F128 bp[16]; unsigned long long seed = 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < 16; ++i) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; bp[i].lo = seed;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; bp[i].hi = seed;
        }
        F128 bp0; seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        bp0.lo = seed << 32; bp0.hi = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
        F128 bph[16]; for (int i = 0; i < 16; ++i) bph[i] = h_tower_to_hyb(bp[i]);
        F128 bbase = h_tower_to_hyb(bp0);
        F128* bff; CK(cudaMalloc(&bff, 16*sizeof(F128)));
        CK(cudaMemcpy(bff, bph, 16*sizeof(F128), cudaMemcpyHostToDevice));
        F128 zt{0, 0};                       // target 0: never reached
        unsigned *bf, *bi; CK(cudaMalloc(&bf, 4)); CK(cudaMalloc(&bi, 4));
        CK(cudaMemset(bf, 0, 4));
        // warm-up
        k_mine<<<blocks, thr, SH_BYTES>>>(bff, bbase, zt, zt, 0, per_thread, bf, bi);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        double total = 0; int iters = 0;
        auto tb0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count() < secs) {
            k_mine<<<blocks, thr, SH_BYTES>>>(bff, bbase, zt, zt,
                                    (unsigned)iters * 1664525u, per_thread, bf, bi);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            total += (double)blocks * thr * per_thread; ++iters;
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count();
        LOG("BENCH %.2f MH/s  (%.0f hashes in %.2fs, %d launches, blocks=%d thr=%d batch=%u)",
            total / el / 1e6, total, el, iters, blocks, thr, per_thread);
        return 0;
    }

    Endpoint ep;
    if (!parse_url(rpc, ep)) return 1;
    LOG("RPC %s:%d%s", ep.host.c_str(), ep.port, ep.path.c_str());

    // --- reach the node once, before mining, and say plainly what is wrong ---
    // Without this the miner simply spins: no template, no error, no output. The
    // first run of a new user is exactly the run that must not be silent.
    {
        Tmpl probe; int st = 0;
        if (!fetch_template(ep, key, coinbase, probe, &st)) {
            if (st == 401 || st == 403)
                LOG("the node refused the mining key (HTTP %d) - %s", st,
                    key.empty() ? "this node requires one: pass --key TOKEN"
                                : "the key was rejected: check --key");
            else if (st == 0)
                LOG("no answer from %s:%d - is the node running, and is its RPC listening there?",
                    ep.host.c_str(), ep.port);
            else if (st == 404)
                LOG("HTTP 404 from %s:%d%s - wrong RPC path, or this is not a Jetsam node",
                    ep.host.c_str(), ep.port, ep.path.c_str());
            else
                LOG("the node answered HTTP %d but gave no usable block template", st);
            return 1;
        }
        LOG("node reachable, mining on top of height %lld", probe.height);
    }

    // --- watch thread: fetches templates WITHOUT ever stopping the GPU ---
    Shared sh; std::atomic<bool> stop{false};
    std::thread poller([&]{
        int fails = 0;
        while (!stop.load()) {
            Tmpl t; int st = 0;
            if (fetch_template(ep, key, coinbase, t, &st)) {
                if (fails) { LOG("node reachable again"); fails = 0; }
                std::lock_guard<std::mutex> g(sh.m);
                if (t.height != sh.t.height || t.id != sh.t.id) { sh.t = t; sh.fresh = true; }
            } else if (++fails == 1 || fails % 50 == 0) {
                // The template already in hand stays valid for a while, so a blip is
                // not fatal and must not stop the GPU - but it must be visible.
                if (st) LOG("no template: node answered HTTP %d (%d tries)", st, fails);
                else    LOG("no template: node unreachable (%d tries)", fails);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        }
    });

    F128 *dff; CK(cudaMalloc(&dff, 16*sizeof(F128)));   // 16 raw fields (index 0 = nonce, replaced in-kernel)
    F128 *dvf, *dvo; CK(cudaMalloc(&dvf, 16*sizeof(F128))); CK(cudaMalloc(&dvo, 2*sizeof(F128)));
    unsigned *dfound, *dfidx; CK(cudaMalloc(&dfound,4)); CK(cudaMalloc(&dfidx,4));
    unsigned *hfound, *hfidx;
    CK(cudaHostAlloc(&hfound, 4, cudaHostAllocDefault)); CK(cudaHostAlloc(&hfidx, 4, cudaHostAllocDefault));
    cudaStream_t st; CK(cudaStreamCreate(&st));

    Tmpl cur; F128 t0{0,0}, t1{0,0}, p0base{0,0}, base_h{0,0};
    unsigned long long nonce_hi = 0;   // high 96 bits of the nonce
    unsigned idx = 0;
    bool have = false;
    double hashes = 0; auto tstat = std::chrono::steady_clock::now();
    auto t_start = std::chrono::steady_clock::now();
    auto idle_since = t_start;         // start of the current wait (have==false)
    auto tpl_at = t_start;             // when the current template was installed
    double idle_ms = 0;
    unsigned long long solved = 0, accepted = 0;
    // Each miner instance must start on a different slice of the nonce space:
    // two cards drawing the same high 96 bits would search the same nonces.
    // random_device + the clock covers it, and is portable (getpid/random are not).
    std::mt19937_64 rng((std::random_device{}() ^ (unsigned long long)
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    while (true) {
        // ---- template switch (without breaking the cadence) ----
        bool swap = false; Tmpl nt;
        { std::lock_guard<std::mutex> g(sh.m); if (sh.fresh) { nt = sh.t; sh.fresh = false; swap = true; } }
        if (swap) {
            F128 hf[16];
            for (int j = 0; j < 16; ++j) {                       // 16 LE bytes per field
                F128 v; v.lo = 0; v.hi = 0;
                const char* p = nt.fields_hex.c_str() + j*32;
                for (int b = 0; b < 8; ++b)  v.lo |= (u64)((hexval(p[2*b])<<4)|hexval(p[2*b+1])) << (8*b);
                for (int b = 0; b < 8; ++b)  v.hi |= (u64)((hexval(p[2*(8+b)])<<4)|hexval(p[2*(8+b)+1])) << (8*b);
                hf[j] = v;
            }
            const char* tp = nt.target_hex.c_str();
            t0.lo=t0.hi=t1.lo=t1.hi=0;
            for (int b=0;b<8;++b)  t0.lo |= (u64)((hexval(tp[2*b])<<4)|hexval(tp[2*b+1])) << (8*b);
            for (int b=0;b<8;++b)  t0.hi |= (u64)((hexval(tp[2*(8+b)])<<4)|hexval(tp[2*(8+b)+1])) << (8*b);
            for (int b=0;b<8;++b)  t1.lo |= (u64)((hexval(tp[2*(16+b)])<<4)|hexval(tp[2*(16+b)+1])) << (8*b);
            for (int b=0;b<8;++b)  t1.hi |= (u64)((hexval(tp[2*(24+b)])<<4)|hexval(tp[2*(24+b)+1])) << (8*b);
            // Hybrid basis: convert the 16 template fields once per template.
            F128 hfh[16]; for (int j = 0; j < 16; ++j) hfh[j] = h_tower_to_hyb(hf[j]);
            CK(cudaMemcpy(dff, hfh, 16*sizeof(F128), cudaMemcpyHostToDevice));
            nonce_hi = rng();
            p0base.lo = nonce_hi << 32; p0base.hi = nonce_hi >> 32;   // high 96 bits, low 32 bits zero
            base_h = h_tower_to_hyb(p0base);                          // nonce = base_h ^ PHI(idx) on the device
            auto now_sw = std::chrono::steady_clock::now();
            if (!have)                 // leaving a real wait: that is what we count
                idle_ms += std::chrono::duration<double,std::milli>(now_sw - idle_since).count();
            idx = 0; cur = nt; have = true; tpl_at = now_sw;
            int zb = 0; for (int b = 31; b >= 0 && !((hexval(tp[2*b])<<4)|hexval(tp[2*b+1])); --b) zb += 8;
            LOG("template h=%lld  target ~%d bits  expires in %llds", nt.height, zb, (long long)nt.expires);
        }
        if (have) {
            double tpl_age = std::chrono::duration<double>(std::chrono::steady_clock::now() - tpl_at).count();
            double ttl = (cur.expires > 0 ? (double)cur.expires : 30.0) + 3.0;   // re-fetch margin
            if (tpl_age > ttl) {
                LOG("template h=%lld expired locally (%.0fs > %.0fs) — pausing until fresh work arrives",
                    cur.height, tpl_age, ttl);
                have = false;
                idle_since = std::chrono::steady_clock::now();
                { std::lock_guard<std::mutex> g(sh.m); sh.t.id.clear(); sh.t.height = -1; }
            }
        }
        if (!have) { std::this_thread::sleep_for(std::chrono::milliseconds(150)); continue; }

        // ---- one batch of nonces ----
        CK(cudaMemsetAsync(dfound, 0, 4, st));
        k_mine<<<blocks, thr, SH_BYTES, st>>>(dff, base_h, t0, t1,
                                       idx, per_thread, dfound, dfidx);
        CK(cudaMemcpyAsync(hfound, dfound, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(hfidx,  dfidx,  4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        unsigned span = (unsigned)blocks * (unsigned)thr * per_thread;
        hashes += span;
        unsigned previdx = idx; idx += span;

        if (*hfound) {
            unsigned fi = *hfidx;
            F128 nonce; nonce.lo = (nonce_hi << 32) | (u64)fi; nonce.hi = nonce_hi >> 32;
            // --- re-validate through the reference path (full sponge) before submitting ---
            F128 vf[16];
            for (int j = 0; j < 16; ++j) {
                F128 v; v.lo = 0; v.hi = 0;
                const char* p = cur.fields_hex.c_str() + j*32;
                for (int b = 0; b < 8; ++b) v.lo |= (u64)((hexval(p[2*b])<<4)|hexval(p[2*b+1])) << (8*b);
                for (int b = 0; b < 8; ++b) v.hi |= (u64)((hexval(p[2*(8+b)])<<4)|hexval(p[2*(8+b)+1])) << (8*b);
                vf[j] = v;
            }
            vf[0] = nonce;
            CK(cudaMemcpy(dvf, vf, 16*sizeof(F128), cudaMemcpyHostToDevice));
            k_selftest<<<1,1,SH_BYTES>>>(dvf, dvo, 1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            F128 dg[2]; CK(cudaMemcpy(dg, dvo, 2*sizeof(F128), cudaMemcpyDeviceToHost));
            bool okref = le256_lt(dg[0], dg[1], t0, t1);
            char nh[33]; for (int b = 0; b < 8; ++b) snprintf(nh+2*b, 3, "%02x", (unsigned)((nonce.lo>>(8*b))&0xff));
            for (int b = 0; b < 8; ++b) snprintf(nh+16+2*b, 3, "%02x", (unsigned)((nonce.hi>>(8*b))&0xff));
            if (!okref) { LOG("!! nonce rejected by the reference re-validation (idx=%u) -- NOT submitted", fi); }
            else {
                ++solved;
                LOG("NONCE FOUND h=%lld idx=%u nonce=%s", cur.height, fi, nh);
                std::string body = std::string("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"jetsam_submitBlock\","
                                               "\"params\":[\"") + cur.id + "\",\"" + nh + "\"]}";
                std::string r = http_post(ep, body, key, 15000);
                bool err = r.find("\"error\"") != std::string::npos;
                if (!err) ++accepted;
                LOG("submit h=%lld : %s%.200s", cur.height, err ? "REFUSED " : "OK ", r.c_str());
                idle_since = std::chrono::steady_clock::now();
                Tmpl imm;
                if (fetch_template(ep, key, coinbase, imm) && imm.id != cur.id) {
                    std::lock_guard<std::mutex> g(sh.m); sh.t = imm; sh.fresh = true;
                } else {
                    std::lock_guard<std::mutex> g(sh.m); sh.t.id.clear(); sh.t.height = -1;
                }
                have = false;
            }
            idx = previdx + span;
        }
        if (idx < previdx) {                      // 32-bit wrap: pick a new high part
            nonce_hi = rng();
            p0base.lo = nonce_hi << 32; p0base.hi = nonce_hi >> 32;
            base_h = h_tower_to_hyb(p0base);
            idx = 0;
        }
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - tstat).count();
        if (el >= 10.0) {
            double idle_cur = idle_ms + (have ? 0.0
                : std::chrono::duration<double,std::milli>(now - idle_since).count());
            double wall = std::chrono::duration<double>(now - t_start).count();
            double duty = wall > 0 ? 100.0 * (1.0 - (idle_cur / 1000.0) / wall) : 0.0;
            LOG("%.2f MH/s   h=%lld age=%.0fs   found=%llu accepted=%llu   duty=%.1f%%  (idle %.1fs / %.0fs)",
                hashes/el/1e6, cur.height,
                std::chrono::duration<double>(now - tpl_at).count(),
                solved, accepted, duty, idle_cur/1000.0, wall);
            // The same figure that was just logged goes out on the next request:
            // what is displayed and what is declared cannot drift apart.
            g_measured_hs.store((unsigned long long)(hashes / el), std::memory_order_relaxed);
            hashes = 0; tstat = now;
        }
    }
    stop = true; poller.join();
    return 0;
}
