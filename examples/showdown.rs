//! Full Rust-side comparison: blaze-bpe vs tiktoken-rs vs bpe-openai
//! (rust-gems linear-time BPE) vs fastokens (Crusoe/NVIDIA), cl100k_base.
//!
//! Run with: cargo run --release --example showdown

use std::time::Instant;

fn time_best<F: FnMut() -> usize>(iters: usize, mut f: F) -> (f64, usize) {
    let mut tokens = f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        tokens = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, tokens)
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / 1e6 / secs
}

fn main() {
    let text = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    // A "long prompt": one 1.1MB document. Batch: 2KB documents.
    let docs: Vec<&str> = text
        .as_bytes()
        .chunks(2048)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    let total = text.len();
    let iters = 10;

    let blaze = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let tk = tiktoken_rs::cl100k_base().unwrap();
    let gems = bpe_openai::cl100k_base();
    let ftk = fastokens::Tokenizer::from_file(std::path::Path::new("data/cl100k_fastokens.json"))
        .unwrap();

    // Sanity: identical token counts on the corpus.
    let want = tk.encode_ordinary(&text).len();
    assert_eq!(blaze.encode(&text).len(), want, "blaze count mismatch");
    println!("corpus: shakespeare, {:.2} MB, {} tokens (tiktoken)", total as f64 / 1e6, want);
    println!("bpe-openai count: {}", gems.encode(&text).len());
    println!("fastokens count:  {}", ftk.encode(&text).unwrap().len());

    println!("\n--- single-threaded, one long document ---");
    let (t, _) = time_best(iters, || blaze.encode_sequential(&text).len());
    println!("{:<24}{:>10.1} MB/s", "blaze-bpe", mbps(total, t));
    let (t, _) = time_best(iters, || tk.encode_ordinary(&text).len());
    println!("{:<24}{:>10.1} MB/s", "tiktoken-rs", mbps(total, t));
    let (t, _) = time_best(iters, || gems.encode(&text).len());
    println!("{:<24}{:>10.1} MB/s", "bpe-openai (rust-gems)", mbps(total, t));
    let (t, _) = time_best(iters, || ftk.encode(&text).unwrap().len());
    println!("{:<24}{:>10.1} MB/s", "fastokens", mbps(total, t));
    // Cold-cache blaze: encoding with a second tokenizer instance between
    // iterations evicts the thread-local piece cache (owner mismatch), so
    // every timed run starts cold.
    let poison = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let (t, _) = time_best(iters, || {
        poison.encode_sequential("evict");
        blaze.encode_sequential(&text).len()
    });
    println!("{:<24}{:>10.1} MB/s", "blaze-bpe (cold cache)", mbps(total, t));

    println!("\n--- multicore, one long document (chunked-parallel) ---");
    let (t, _) = time_best(iters, || blaze.encode(&text).len());
    println!("{:<24}{:>10.1} MB/s", "blaze-bpe (chunked)", mbps(total, t));
    let big = text.repeat(8); // ~9 MB single document
    let (t, _) = time_best(iters, || blaze.encode(&big).len());
    println!("{:<24}{:>10.1} MB/s  ({:.1} MB doc)", "blaze-bpe (chunked)", mbps(big.len(), t), big.len() as f64 / 1e6);
    let (t, _) = time_best(iters, || blaze.encode_sequential(&big).len());
    println!("{:<24}{:>10.1} MB/s  ({:.1} MB doc)", "blaze-bpe (sequential)", mbps(big.len(), t), big.len() as f64 / 1e6);

    println!("\n--- multicore batch ({} x 2KB docs) ---", docs.len());
    let (t, _) = time_best(iters, || blaze.encode_batch(&docs).len());
    println!("{:<24}{:>10.1} MB/s", "blaze-bpe (rayon)", mbps(total, t));
    let (t, _) = time_best(iters, || {
        ftk.encode_batch(&docs, false).unwrap().len()
    });
    println!("{:<24}{:>10.1} MB/s", "fastokens (batch)", mbps(total, t));
    let (t, _) = time_best(iters, || {
        use rayon::prelude::*;
        docs.par_iter().map(|d| gems.encode(d).len()).sum::<usize>()
    });
    println!("{:<24}{:>10.1} MB/s", "bpe-openai (rayon)", mbps(total, t));
    let (t, _) = time_best(iters, || {
        use rayon::prelude::*;
        docs.par_iter()
            .map(|d| tk.encode_ordinary(d).len())
            .sum::<usize>()
    });
    println!("{:<24}{:>10.1} MB/s", "tiktoken-rs (rayon)", mbps(total, t));
}
