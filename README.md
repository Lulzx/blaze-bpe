# blaze-bpe

A fast, tiktoken-compatible BPE tokenizer in Rust. Produces **byte-identical
output** to OpenAI's tiktoken for the `cl100k_base` (GPT-3.5/4) and
`o200k_base` (GPT-4o) encodings, verified against tiktoken-rs on edge cases,
6000 randomized fuzz strings, and full corpora.

## Benchmarks

Apple Silicon (M-series), single thread, vs tiktoken-rs 0.6:

| corpus (cl100k_base)  | blaze MB/s | tiktoken-rs MB/s | speedup |
|-----------------------|-----------:|-----------------:|--------:|
| english (shakespeare) |      543.5 |             14.6 |  37.1x  |
| source code           |      724.5 |             12.1 |  60.0x  |
| multilingual          |      368.9 |             19.9 |  18.5x  |
| json                  |      675.3 |             11.7 |  57.9x  |

o200k_base: 21.9x–28.0x. Batch encoding with rayon (545 × 2KB docs):
**~3.0 GB/s**. Chunked-parallel encoding of a *single* long document
(`encode` auto-splits at piece-safe boundaries above 256KB): **2.4 GB/s**
on an 8.9 MB document (same-session best-of-5; the isolated peak is ~3.9 GB/s),
4.3x over sequential, byte-identical output. On multi-GB corpora blaze is joined
— and at 1 GB overtaken — by [gigatoken](https://github.com/marcelroed/gigatoken),
a whole-file bulk tokenizer benchmarked alongside it below.

### Cross-implementation showdown

Same machine, same corpus (shakespeare, 1.12 MB, cl100k_base), identical
token counts across all implementations (301,829). Rust rivals via
`cargo run --release --example showdown`, Python via tiktoken 0.12 /
tokenizers 0.22 / rs-bpe 0.1 (built from source, native target), Go via
omnitoken 0.1.4 / tiktoken-go 0.1.8 on go1.26:

| implementation           | single-thread MB/s | multicore batch MB/s |
|--------------------------|-------------------:|---------------------:|
| **blaze-bpe**            |          **617.1** |           **2960.6** |
| blaze-bpe (cold cache)   |              271.3 |                      |
| fastokens (Crusoe/NVIDIA)|              174.0 |                147.9 |
| omnitoken (pure Go)      |               60.3 |                      |
| bpe-openai (rust-gems)   |               53.5 |                127.6 |
| rs-bpe (Python bindings) |               46.7 |                      |
| gigatoken (whole-file)   |               44.6 |                      |
| tiktoken (Python, native core) |         21.8 |     89.3 (threads)   |
| tiktoken-rs              |               14.4 |                118.9 |
| tiktoken-go              |                9.2 |                      |
| HF tokenizers (Python)   |                4.7 |                 10.0 |

Chunked-parallel single document: 1330 MB/s (1.1 MB), 3877 MB/s (8.9 MB).
Counting without materializing tokens (`count`, same 1.1 MB doc,
chunked-parallel): 1230 MB/s.

gigatoken's 44.6 MB/s here is not a single-thread number — it's its whole-file
`bench` on this 1.1 MB doc, where its fixed per-call setup (~20 ms) dominates and
its parallelism has nothing to bite on. That overhead only amortizes on
gigabyte-scale input; the [1 GB benchmark](#encoding-1-gb-of-text-cl100k_base)
shows it climbing from last place at 1 MB to first at 1 GB.

Decode (ids -> text, same corpus): omnitoken 545 MB/s, blaze 480 MB/s,
rs-bpe 306 MB/s, tiktoken-go 194 MB/s, tiktoken (Python) 188 MB/s.

Rival marketing claims, re-measured here on identical input:

- **rs-bpe** advertises "up to 15.1x vs tiktoken on small texts"; measured:
  2.1x on the full document, 1.7x on 2000 unique ~46 B lines, 2.7x at best
  on a single repeated line where its result cache fully applies. blaze-bpe
  is ~13x rs-bpe single-threaded on the identical input.
- **omnitoken** advertises "15.84x CountTokens / 13.09x EncodeOrdinary vs
  tiktoken-go, 2.29x Decode"; measured: 6.4x / 6.3x / 2.8x — and the
  headline baseline (tiktoken-go, 9.2 MB/s) is a pure-Go port several times
  slower than tiktoken's real Rust core. blaze-bpe is ~10x omnitoken
  single-threaded on the identical input.

### vs gigatoken on real-world text (OpenWebText)

[gigatoken](https://github.com/marcelroed/gigatoken) is a whole-file tokenizer
built for raw throughput on huge corpora: it swallows an entire file, finds
document and piece boundaries itself, and saturates every core. Its published
benchmark runs on `owt_train.txt` (OpenWebText, 11.9 GB). Measured here on the
same corpus and machine (M4 Pro, 8P+4E, 24 GB), both cl100k_base — gigatoken via
a HuggingFace cl100k tokenizer (`Xenova/gpt-4`), blaze via `encode` (whole-file
chunked-parallel):

| corpus (cl100k)    |  size | blaze parallel | blaze 1-thread | gigatoken |
|--------------------|------:|---------------:|---------------:|----------:|
| shakespeare        | 1.1 MB|     0.87 GB/s  |     0.57 GB/s  | 0.05 GB/s |
| big.txt (English)  |  32 MB|     3.53 GB/s  |     0.55 GB/s  | 0.83 GB/s |
| OpenWebText sample |   1 GB|     2.05 GB/s  |     0.26 GB/s  | 3.75 GB/s |

The two tools cross over with input size. blaze wins by ~13x at 1 MB and ~4x at
32 MB: its per-thread piece cache and near-zero fixed overhead make it fast the
instant it's called, so short-to-medium inputs — the common serving/RAG case —
are where it dominates. gigatoken pays a large fixed setup cost per call (a
~20 ms floor, visible as the 48 MB/s on 1 MB) that only amortizes on
gigabyte-scale files, and there it pulls ahead: 3.75 GB/s on 1 GB of OpenWebText
vs blaze's 2.05.

Two honest caveats. blaze's throughput is content-sensitive — big.txt (English
prose, cache-friendly) runs at 3.53 GB/s but diverse real web text at 1 GB drops
to 2.05, whereas gigatoken is nearly content-agnostic. And gigatoken through the
HF cl100k tokenizer emits 214.6M tokens on the 1 GB sample vs blaze's 215.7M:
blaze is byte-identical to tiktoken, while HF's cl100k pre-tokenizer differs by
~0.5%. gigatoken also OOM-crashed above ~1 GB on this 24 GB machine (its
published multi-GB/s runs used far larger boxes), so the comparison is capped at
1 GB here.

Reproduce (streams a 1 GB slice of OpenWebText; the 4.6 GB gzip is never stored):

```sh
url=$(curl -sI https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz \
      | grep -i '^location:' | tr -d '\r' | awk '{print $2}')
curl -sL "$url" | gunzip | head -c 1000000000 > data/owt_1g.txt
cargo run --release --example owt -- data/owt_1g.txt cl100k
uvx --with tokenizers --from gigatoken gigatoken bench 'Xenova/gpt-4' data/owt_1g.txt
```

### Encoding 1 GB of text (cl100k_base)

One 1.0 GB document (shakespeare x900, 271,646,100 tokens), identical token
output across all implementations. **First-run measurement**: every timed pass
is a fresh process, so cache population is included in the timing (no warmup) —
the honest number for a one-shot bulk job, and apples-to-apples with gigatoken's
fresh-process `bench`. Best of 5, one thermal session, M4 Pro.
[gigatoken](https://github.com/marcelroed/gigatoken) and blaze (parallel) both
auto-parallelize across all 12 cores; quicktok/rs-bpe/tiktoken are single-thread
(Python bindings, FFI fully amortized at this size). At 1 GB gigatoken's
whole-file design pulls ahead of blaze's chunked-parallel encode — the same
crossover the [OpenWebText](#vs-gigatoken-on-real-world-text-openwebtext) section
traces on real web text:

| implementation           | tokens/s   | wall time for 1 GB | vs tiktoken |
|--------------------------|-----------:|-------------------:|------------:|
| **gigatoken**            |  **1278M** |         **0.21 s** |    **224x** |
| **blaze-bpe (parallel)** |   **863M** |         **0.32 s** |    **151x** |
| blaze-bpe (1 thread)     |       130M |              2.1 s |         23x |
| quicktok 0.4 (C++)       |      37.1M |              7.3 s |        6.5x |
| rs-bpe 0.1               |      12.3M |             22.1 s |        2.2x |
| tiktoken 0.12            |       5.7M |             47.7 s |          1x |

![Encoding 1 GB of text, cl100k_base](assets/blaze_1gb_graph.png)

(HF tokenizers omitted at this size: its per-token `Encoding` metadata needs
~16-22 GB of RAM. At 32 MB it measures 1.0M tokens/s. These are all-core
first-run figures — each timed pass a fresh process with a cold tokenizer cache;
a warm pre-populated loop measures somewhat higher but is not what a one-shot
bulk tokenization does.)

Same 1 GB document with **o200k_base** (the encoding used by GPT-4o/4.1/5,
o1, o3 and o4-mini; 267,845,400 tokens, again byte-identical everywhere),
same first-run methodology: gigatoken 1085M tokens/s (0.25 s, **139x tiktoken**),
blaze parallel 573M (0.47 s, 73x), blaze single-thread 81M (10x), quicktok 31.4M
(4.0x), rs-bpe 10.5M (1.3x), tiktoken 7.8M (34.3 s). gigatoken's throughput is
vocab-agnostic, so on the bigger o200k vocab — which costs blaze and the others
MB/s — its lead over blaze widens to 1.9x.

![Encoding 1 GB of text, o200k_base](assets/blaze_1gb_o200k_graph.png)

### vs Python tiktoken on one long document (32 MB, cl100k)

Interleaved measurement (`python3 scripts/ratio.py`), identical token output,
M4 Pro (8P+4E): tiktoken ~21 MB/s single-thread; blaze chunked-parallel
**~4.2 GB/s (~198x)**, warm sequential ~575 MB/s. Battery power throttles
all-core clocks; measure plugged in. This is the warm, pre-populated *isolated*
peak (nothing else running); the first-run all-core figures in the 1 GB charts
and tables above run lower, since they include cache population and no warmup.
Per-regime single-thread numbers
(`cargo run --release --example regimes`): cold/novel ~290 MB/s, warm/repeat
~580 MB/s, one unbroken novel CJK run ~100 MB/s (repeated CJK sentences hit
the long-piece cache and run at ~370 MB/s). The cold/novel figure includes the
vocab-seeded piece cache (every single-token piece is pre-inserted at load, so a
first-touch miss skips the BPE merge); with seeding disabled
(`BLAZE_NO_VOCAB_SEED=1 cargo run --release --example regimes`) a genuinely
empty-cache cold pass runs ~1.5x slower (~190 → ~125 MB/s idle, gap widening
under throttle since the merge work dominates at low clocks). Warm/steady-state
throughput is identical either way — seeding only affects the first touch of each
distinct piece, which is why it matters for a single cold shot but not for a
long-running encode loop.

#### End-to-end (loading included)

The ratios above time the *encode loop only* — both harnesses hoist vocab
loading and warmup out of the timer, so it's a fair apples-to-apples encode
comparison, not a cold single-shot. Construction is a one-time tax: blaze's
vocab load + pair-table build is ~28 ms, tiktoken's `get_encoding` + lazy core
init is ~78 ms. Counting load plus one cold encode pass (M4 Pro):

| doc     | blaze e2e | tiktoken e2e | e2e ratio | hot-loop ratio |
|---------|----------:|-------------:|----------:|---------------:|
| 1.1 MB  |     29 ms |       131 ms |     4.5x  |          ~40x  |
| 8.9 MB  |     31 ms |       521 ms |      17x  |         ~130x  |
| 32.3 MB |     37 ms |      1687 ms |      45x  |         ~175x  |

blaze is faster end-to-end at every size, but the multiple collapses on small
inputs where its own ~28 ms load dominates a sub-millisecond encode — the
warm-loop numbers are the ones that matter for a long-running process
(serving tokenizer, training pipeline) that loads once and encodes forever.

Reproduce with:

```sh
cargo test --release                     # byte-identical correctness vs tiktoken-rs
cargo run --release --example compare    # throughput comparison table
cargo run --release --example firstrun -- FILE  # first-run (cold-cache) throughput on a file
cargo bench                              # criterion benchmarks
```

The 1 GB first-run bars are reproduced with `scripts/bench_gigatoken.sh` (fresh
gigatoken invocations) alongside `cargo run --release --example firstrun` for
blaze; `python3 scripts/make_charts.py` renders the charts from those numbers.

## How it works

BPE tokenization is a two-stage pipeline, and blaze-bpe reimplements both
stages from scratch while reproducing tiktoken's semantics exactly:

1. **Pre-tokenization.** The input is split into *pieces* — words,
   contractions, number runs, punctuation clusters, whitespace runs —
   according to the cl100k/o200k splitting pattern. tiktoken does this with a
   backtracking regex engine; blaze-bpe implements the same splitting rules
   as a hand-written scanner (`src/scan.rs`), SIMD-accelerated on aarch64.
   Pieces never span this boundary, which is also what makes safe parallel
   chunking possible.

2. **BPE merge.** Each piece starts as a sequence of single bytes, and the
   lowest-ranked adjacent pair (per the vocab's merge ranks) is repeatedly
   merged until no adjacent pair exists in the vocab. Ranks equal token ids,
   so the final sequence of merged spans *is* the token ids. Before merging,
   each piece is looked up in per-thread caches — most words in real text
   repeat, so the merge loop only runs for genuinely novel pieces.

Special tokens (`encode_with_specials`) are found with an Aho-Corasick pass
that splits the text around them; everything between is encoded normally.
Decoding is a table lookup: each id maps back to its byte sequence, which are
concatenated and UTF-8-validated (`decode_bytes` skips validation).

Inputs above ~384KB are transparently split at provably piece-safe boundaries
and encoded on all cores (`encode`); `encode_batch` parallelizes over
documents instead. Both produce byte-identical output to sequential encoding.

## Why it's fast

1. **Bitmask pre-tokenization scanner** (`src/scan.rs`). tiktoken spends
   most of its time in `fancy_regex`, a backtracking regex engine. We
   implement the exact splitting semantics of the cl100k/o200k patterns
   (ordered alternation, possessive quantifiers, greedy backtracking)
   directly. On aarch64, cl100k classifies 64 bytes at a time with NEON into
   1-bit-per-byte class masks (letter/digit/newline/whitespace/non-ASCII);
   piece boundaries then come from shift + count-trailing-zeros instead of
   per-byte branches. A "pure lane" emits `word (space word)*` runs — the
   dominant shape in real text — at one ctz per piece with zero dispatch
   branches. Non-ASCII, contractions and window boundaries fall back to an
   exact scalar scanner with table-driven character classes.

2. **Id-pair merge table.** At load time we precompute
   `(left_id, right_id) -> merged_id` for every token splittable into two
   existing tokens. The merge loop then hashes a single `u64` per candidate
   pair instead of re-hashing byte slices like tiktoken's
   `ranks.get(&piece[start..end])`. Merge priority equals the merged token id,
   so greedy lowest-rank merging reproduces tiktoken exactly. Pieces longer
   than 128 bytes (unbroken CJK) switch from the O(n²) scan-for-min loop to a
   heap-based O(n log n) merge — ~11x on CJK.

3. **Direct-mapped piece caches.** Real text repeats words heavily. Pieces up
   to 16 bytes are packed into two words by a single unaligned 16-byte load
   plus branchless masking; a cache hit is one indexed load, three integer
   compares and a fixed 24-byte copy — no probing, no allocation. Each thread
   has an adaptive direct-mapped cache (2^18 × 48B slots), seeded on miss
   from a shared immutable table prebuilt from the vocab (every vocab token
   ≤ 16 bytes is a guaranteed single-token piece), so threads start warm and
   single-token pieces never merge. A second direct-mapped cache covers long
   pieces of 17–96 bytes (unbroken CJK letter-runs, long Cyrillic words) with
   exact byte verification per slot — repeated CJK sentences skip the merge
   loop entirely, taking multilingual text from ~90 to ~370 MB/s.

4. **Rayon batch encoding** (`encode_batch`), with per-thread caches and no
   shared mutable state.

5. **Chunked-parallel long documents.** `encode` transparently splits inputs
   above ~384KB into ~192KB chunks, encodes them in parallel, and
   parallel-copies the parts into the output at precomputed offsets. Cuts are
   only made at positions where a pre-tokenization piece provably ends
   regardless of scan alignment (after `\n` followed by non-whitespace, or
   between an ASCII letter and a following space/digit/punct other than `'`),
   so the result is byte-identical to sequential encoding;
   `encode_sequential` bypasses this. Inputs with no safe cut (e.g. unbroken
   CJK) fall back to sequential automatically.

Merging itself uses tiktoken's neighbor-update algorithm (recompute only the
two ranks adjacent to a merge) over `SmallVec` stack buffers — pieces are
short, so no heap allocation in the steady state.

## Usage

```rust
let tok = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken")?;
let ids = tok.encode("hello world");             // auto-parallel above 256KB
let ids = tok.encode_sequential(&long_text);     // force single-threaded
let ids = tok.encode_with_specials("<|endoftext|>hi");
let n   = tok.count("how many tokens?");
let all = tok.encode_batch(&docs);               // rayon-parallel
let s   = tok.decode(&ids)?;
```

Custom vocabularies: `Tokenizer::from_tiktoken_data(data, pattern, specials)`
accepts any `.tiktoken`-format vocab (base64 token + rank per line).

## Layout

- `src/scan.rs` — pre-tokenization scanners for both patterns
- `src/lib.rs` — vocab loading, pair-table construction, BPE merge, cache,
  encode/decode API
- `tests/correctness.rs` — byte-identical comparison vs tiktoken-rs
- `examples/compare.rs` — throughput comparison harness
- `data/` — vocab files (downloaded from OpenAI's public blob storage) and
  benchmark corpus
