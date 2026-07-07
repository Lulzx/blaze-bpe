# blaze-bpe

A fast, tiktoken-compatible BPE tokenizer in Rust. Produces **byte-identical
output** to OpenAI's tiktoken for the `cl100k_base` (GPT-3.5/4) and
`o200k_base` (GPT-4o) encodings, verified against tiktoken-rs on edge cases,
6000 randomized fuzz strings, and full corpora.

## Benchmarks

Apple Silicon (M-series), single thread, vs tiktoken-rs 0.6:

| corpus (cl100k_base)  | blaze MB/s | tiktoken-rs MB/s | speedup |
|-----------------------|-----------:|-----------------:|--------:|
| english (shakespeare) |      248.8 |             14.2 |  17.5x  |
| source code           |      332.8 |             11.2 |  29.6x  |
| multilingual          |      593.0 |             19.7 |  30.1x  |
| json                  |      309.2 |             11.6 |  26.7x  |

o200k_base: 14.7x–22.3x. Batch encoding with rayon (545 × 2KB docs):
**~1.5 GB/s**. Chunked-parallel encoding of a *single* long document
(`encode` auto-splits at piece-safe boundaries above 256KB): **1.83 GB/s**
on a 9 MB document, 7.9x over sequential, byte-identical output.

### Cross-implementation showdown

Same machine, same corpus (shakespeare, 1.12 MB, cl100k_base), identical
token counts across all implementations. Rust rivals via
`cargo run --release --example showdown`, Python via tiktoken 0.12 /
tokenizers 0.22:

| implementation           | single-thread MB/s | multicore batch MB/s |
|--------------------------|-------------------:|---------------------:|
| **blaze-bpe**            |          **240.1** |           **1667.7** |
| blaze-bpe (cold cache)   |              158.7 |                      |
| fastokens (Crusoe/NVIDIA)|              205.2 |                215.2 |
| bpe-openai (rust-gems)   |               53.5 |                193.6 |
| tiktoken (Python, native core) |         21.0 |     89.3 (threads)   |
| tiktoken-rs              |               14.4 |                118.3 |
| HF tokenizers (Python)   |                4.7 |                 10.0 |

Chunked-parallel single document: 1180 MB/s (1.1 MB), ~2000 MB/s (32 MB).

### vs Python tiktoken on one long document (32 MB, cl100k)

Interleaved measurement (`python3 scripts/ratio.py`), identical token output,
M4 Pro (8P+4E): tiktoken ~21 MB/s single-thread; blaze chunked-parallel
**~4.2 GB/s (~198x)**, warm sequential ~575 MB/s. Battery power throttles
all-core clocks; measure plugged in. Per-regime single-thread numbers
(`cargo run --release --example regimes`): cold/novel ~215 MB/s, warm/repeat
~575 MB/s, unbroken CJK ~90 MB/s.

Reproduce with:

```sh
cargo test --release                     # byte-identical correctness vs tiktoken-rs
cargo run --release --example compare    # throughput comparison table
cargo bench                              # criterion benchmarks
```

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
   single-token pieces never merge.

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
