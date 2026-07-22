//! Gigatoken-style whole-file throughput benchmark.
//!
//! Mirrors `gigatoken bench <model> owt_train.txt`: read the entire corpus into
//! memory and encode it, letting the tokenizer do its own boundary detection
//! and parallelization. Reports GB/s over the full file.
//!
//! Run with:
//!   cargo run --release --example owt -- data/owt_sample.txt cl100k
//!
//! NOTE: gigatoken benchmarks the GPT-2 vocab; blaze-bpe ships cl100k_base /
//! o200k_base, so absolute token counts differ from GPT-2. Throughput (GB/s)
//! on the same bytes is the comparable figure.

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "data/owt_sample.txt".into());
    let enc = args.next().unwrap_or_else(|| "cl100k".into());

    let t = Instant::now();
    let text = std::fs::read_to_string(&path).expect("read corpus");
    let read_secs = t.elapsed().as_secs_f64();
    let bytes = text.len();
    let gb = bytes as f64 / 1e9;

    let tok = match enc.as_str() {
        "o200k" => blaze_bpe::Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap(),
        _ => blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap(),
    };

    println!(
        "file: {}  ({:.3} GB, read in {:.2}s = {:.2} GB/s)\nencoding: {}",
        path, gb, read_secs, gb / read_secs, enc
    );

    // Warmup + timed runs of the full-file chunked-parallel encode.
    let mut ntok = tok.encode(&text).len();
    let iters = 3;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        ntok = tok.encode(&text).len();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "\n[chunked-parallel encode]  {} tokens  ({:.3} bytes/token)",
        ntok,
        bytes as f64 / ntok as f64
    );
    println!(
        "  best {:.3}s  =>  {:.2} GB/s  ({:.1} M tokens/s)",
        best,
        gb / best,
        ntok as f64 / best / 1e6
    );

    // Single-thread reference on a 1 GB slice (aligned to a char boundary).
    let mut cut = bytes.min(1_000_000_000);
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let slice = &text[..cut];
    let sgb = cut as f64 / 1e9;
    let mut best_seq = f64::INFINITY;
    let mut stok = tok.encode_sequential(slice).len();
    for _ in 0..iters {
        let t = Instant::now();
        stok = tok.encode_sequential(slice).len();
        best_seq = best_seq.min(t.elapsed().as_secs_f64());
    }
    println!(
        "\n[single-thread encode, {:.3} GB slice]  {} tokens",
        sgb, stok
    );
    println!("  best {:.3}s  =>  {:.2} GB/s", best_seq, sgb / best_seq);
}
