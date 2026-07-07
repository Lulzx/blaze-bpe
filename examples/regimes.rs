//! Per-regime single-thread baselines: cold cache, warm cache, CJK runs.

use std::time::Instant;

fn best<F: FnMut() -> usize>(iters: usize, mut f: F) -> (f64, usize) {
    let mut b = f64::INFINITY;
    let mut n = 0;
    for _ in 0..iters {
        let t = Instant::now();
        n = f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    (b, n)
}

fn main() {
    let text = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    let tok = std::sync::Arc::new(
        blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap(),
    );
    let mb = text.len() as f64 / 1e6;

    // Cold: fresh thread each iteration -> empty thread-local piece cache.
    let (b, _) = best(8, || {
        let tok = tok.clone();
        let t = text.clone();
        std::thread::spawn(move || tok.encode_sequential(&t).len())
            .join()
            .unwrap()
    });
    println!("cold  (novel text):  {:>7.1} MB/s", mb / b);

    // Warm: same thread, cache populated by prior iterations.
    let (b, _) = best(12, || tok.encode_sequential(&text).len());
    println!("warm  (repeat text): {:>7.1} MB/s", mb / b);

    // CJK: unbroken Han runs (one giant letter piece per line).
    let line = "天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲騰致雨露結為霜金生麗水玉出崑岡"
        .repeat(8);
    let cjk = format!("{line}\n").repeat(300);
    let cjk_mb = cjk.len() as f64 / 1e6;
    let (b, ntok) = best(4, || tok.encode_sequential(&cjk).len());
    println!("cjk   ({:.1} MB seq): {:>7.1} MB/s ({} tokens)", cjk_mb, mb / b * cjk_mb / mb, ntok);

    // CJK parallel: does chunking find cuts?
    let (b, _) = best(4, || tok.encode(&cjk).len());
    println!("cjk   (encode auto): {:>7.1} MB/s", cjk_mb / b);
}
