# Benchmark

Every figure below was produced by this miner, on a card that had just passed
`--selftest test/golden.txt` (12000 / 12000). A number from a kernel that has not
passed the gate is worthless, so none is reported here.

## Method

- `JETSAM_BENCH_SECONDS=120 ./jetsam-miner --device 0` — the kernel against an
  impossible target: no network, no template, no idle time. This measures the card,
  not the node it is talking to.
- One rented instance per model, **verified to have no other tenant on the GPU**
  (`nvidia-smi --query-compute-apps`), destroyed immediately after the run.
- Only cards running at their full power limit are listed. A host that caps a card
  below its rating produces a number that says more about the host than about the
  GPU, so those runs are discarded rather than published.

## Results

| GPU | Arch | SMs | MH/s | Power | Temp |
|---|---|---|---|---|---|
| RTX 5090 | Blackwell, sm_120 | 170 | **75.99** | 600 W drawn of 600 W | 72 °C |
| RTX 4090 | Ada, sm_89 | 128 | **56.80** | 450 W available | — |

Both figures: 120 s, CUDA 13.3 build, `clmad` fast path, exclusive GPU.

## What this does not tell you

- **One card is one sample.** Hosts differ: five RTX 5090s rented on the same day
  elsewhere spanned 53 to 62 MH/s — a 17 % spread on the same model. Treat a figure
  as an order of magnitude, not a specification. Measure your own card:
  `JETSAM_BENCH_SECONDS=120 ./jetsam-miner --device 0`.
- **A shared GPU gives a meaningless number.** A card that also hosts an LLM produced
  7.5 and 11.3 MH/s for the *same binary* twenty minutes apart. Check what else is on
  the GPU before believing any measurement, including your own.
- **Turing and older run the software multiply** (no `clmad` instruction), at roughly
  a third of these figures. Correct, bit for bit, just slower.
