//! Tokens/sec on large documents, for the cross-implementation comparison graph.
//! Prints both single-thread (encode_sequential) and auto-chunked-parallel (encode).
//! Usage: cargo run --release --example speedgraph -- file1 file2 ...

use std::time::Instant;

fn best_of<F: FnMut() -> usize>(rounds: usize, mut f: F) -> (f64, usize) {
    let mut ntok = f(); // warmup
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        ntok = std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, ntok)
}

fn main() {
    let tok = match std::env::var("BLAZE_ENC").as_deref() {
        Ok("o200k") => blaze_bpe::Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap(),
        _ => blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap(),
    };
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        let (t_seq, n) = best_of(5, || tok.encode_sequential(&text).len());
        let (t_par, n2) = best_of(5, || tok.encode(&text).len());
        assert_eq!(n, n2);
        println!("blaze_seq\t{path}\t{n}\t{:.1}", n as f64 / t_seq);
        println!("blaze_par\t{path}\t{n}\t{:.1}", n as f64 / t_par);
    }
}
