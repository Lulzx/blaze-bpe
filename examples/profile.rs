//! Rough phase breakdown: scanner-only vs full encode.

use std::time::Instant;

fn main() {
    let text = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    let tok = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();

    let mut best_scan = f64::INFINITY;
    let mut pieces = 0usize;
    for _ in 0..12 {
        let t = Instant::now();
        pieces = blaze_bpe::count_pieces(&text, blaze_bpe::Pattern::Cl100k);
        best_scan = best_scan.min(t.elapsed().as_secs_f64());
    }
    let mut best_enc = f64::INFINITY;
    let mut ntok = 0;
    for _ in 0..12 {
        let t = Instant::now();
        ntok = tok.encode_sequential(&text).len();
        best_enc = best_enc.min(t.elapsed().as_secs_f64());
    }
    let mb = text.len() as f64 / 1e6;
    println!("pieces: {pieces}, tokens: {ntok}");
    println!("scan only:   {:>8.1} MB/s", mb / best_scan);
    println!("full encode: {:>8.1} MB/s (warm)", mb / best_enc);
    println!("scanner share of encode time: {:.0}%", best_scan / best_enc * 100.0);
}
