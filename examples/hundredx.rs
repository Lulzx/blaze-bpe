//! Long-document throughput for the 100x-vs-tiktoken target.
//! Pass a path to a large text file; prints chunked-parallel MB/s.

use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: hundredx <file>");
    let text = std::fs::read_to_string(&path).unwrap();
    let tok = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let mb = text.len() as f64 / 1e6;

    let mut best = f64::INFINITY;
    let mut ntok = 0;
    for _ in 0..12 {
        let t = Instant::now();
        ntok = tok.encode(&text).len();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("doc: {mb:.1} MB, tokens: {ntok}");
    println!("blaze chunked-parallel: {:.0} MB/s ({:.1} ms)", mb / best, best * 1e3);

    let mut best_seq = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        std::hint::black_box(tok.encode_sequential(&text));
        best_seq = best_seq.min(t.elapsed().as_secs_f64());
    }
    println!("blaze sequential:       {:.0} MB/s", mb / best_seq);
    println!("parallel scaling:       {:.2}x", best_seq / best);
}
