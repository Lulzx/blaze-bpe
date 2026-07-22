//! First-run throughput: a single cold `encode()` pass, no warmup, so cache
//! population is included in the timing — the honest number for a one-shot bulk
//! tokenization (and apples-to-apples with gigatoken's fresh-process `bench`).
//!
//! Each process run is itself a first run (empty thread-local caches). Run the
//! process several times and keep the best to match how the gigatoken figures
//! are taken (best of N fresh invocations).
//!
//! Usage: cargo run --release --example firstrun -- file [file ...]
//!        BLAZE_ENC=o200k cargo run --release --example firstrun -- file

use std::time::Instant;

fn main() {
    let tok = match std::env::var("BLAZE_ENC").as_deref() {
        Ok("o200k") => blaze_bpe::Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap(),
        _ => blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap(),
    };
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        // No warmup: this is the first and only encode of the process, so every
        // per-thread cache starts empty and fills during the timed pass.
        let seq = std::env::var_os("BLAZE_SEQ").is_some();
        let t = Instant::now();
        let n = if seq { tok.encode_sequential(&text).len() } else { tok.encode(&text).len() };
        let secs = t.elapsed().as_secs_f64();
        let tag = if seq { "blaze_firstrun_seq" } else { "blaze_firstrun" };
        println!("{tag}\t{path}\t{n}\t{:.1}\t{:.3}s", n as f64 / secs, secs);
    }
}
