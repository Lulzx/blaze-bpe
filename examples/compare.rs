//! Head-to-head throughput comparison: blaze-bpe vs tiktoken-rs.
//!
//! Run with: cargo run --release --example compare

use std::time::Instant;

fn mbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / 1e6 / secs
}

fn time_best<F: FnMut() -> usize>(iters: usize, mut f: F) -> (f64, usize) {
    // Warmup
    let mut total_tokens = f();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        total_tokens = f();
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    (best, total_tokens)
}

fn make_corpora() -> Vec<(&'static str, String)> {
    let shakespeare = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    let code = r#"
fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
    if let Some(&id) = self.token_map.get(piece) { out.push(id); return; }
    let mut ids: SmallVec<[u32; 64]> = piece.iter().map(|&b| self.byte_token[b as usize]).collect();
    for (i, &r) in ranks.iter().enumerate() { if r < best { best = r; best_i = i; } }
}
const CACHE_CAP: usize = 1 << 16; // entries per thread
let response = client.post(url).json(&serde_json::json!({"model": "gpt-4o", "temperature": 0.7}))?;
"#
    .repeat(400);
    let multilingual = "日本語のテキストを処理する速度を測定します。中文分词的速度也很重要。\
Скорость токенизации русского текста. सटीक और तेज़ टोकनाइज़ेशन। \
naïve café résumé — currency: €42, £3.50, ¥1000. Emoji: 😀🚀👍🏽 "
        .repeat(600);
    let json = r#"{"id": 12345, "name": "example", "tags": ["fast", "bpe", "rust"], "nested": {"score": 0.987, "ok": true}, "text": "The quick brown fox jumps over the lazy dog."}"#
        .repeat(500);
    vec![
        ("english (shakespeare)", shakespeare),
        ("source code", code),
        ("multilingual", multilingual),
        ("json", json),
    ]
}

fn main() {
    let blaze_cl = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let blaze_o2 = blaze_bpe::Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap();
    let tk_cl = tiktoken_rs::cl100k_base().unwrap();
    let tk_o2 = tiktoken_rs::o200k_base().unwrap();

    let corpora = make_corpora();
    let iters = 8;

    for (vocab, blaze, tk) in [
        ("cl100k_base", &blaze_cl, &tk_cl),
        ("o200k_base", &blaze_o2, &tk_o2),
    ] {
        println!("\n=== {vocab} (single-threaded) ===");
        println!(
            "{:<24}{:>10}{:>14}{:>14}{:>10}",
            "corpus", "MB", "blaze MB/s", "tiktoken MB/s", "speedup"
        );
        for (name, text) in &corpora {
            let (bt, btok) = time_best(iters, || blaze.encode_sequential(text).len());
            let (tt, ttok) = time_best(iters, || tk.encode_ordinary(text).len());
            assert_eq!(btok, ttok, "token count mismatch on {name}");
            println!(
                "{:<24}{:>10.2}{:>14.1}{:>14.1}{:>9.2}x",
                name,
                text.len() as f64 / 1e6,
                mbps(text.len(), bt),
                mbps(text.len(), tt),
                tt / bt
            );
        }
    }

    // Batch encoding: split shakespeare into ~2k-char documents.
    let text = &corpora[0].1;
    let docs: Vec<&str> = text
        .as_bytes()
        .chunks(2048)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .collect();
    let total: usize = docs.iter().map(|d| d.len()).sum();
    println!("\n=== cl100k_base batch ({} docs, rayon) ===", docs.len());
    let (bt, _) = time_best(8, || blaze_cl.encode_batch(&docs).len());
    let (tt, _) = time_best(8, || {
        docs.iter().map(|d| tk_cl.encode_ordinary(d).len()).sum()
    });
    println!(
        "blaze batch: {:>8.1} MB/s   tiktoken single: {:>8.1} MB/s   speedup: {:.2}x",
        mbps(total, bt),
        mbps(total, tt),
        tt / bt
    );
}
