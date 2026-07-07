use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_encode(c: &mut Criterion) {
    let text = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    let blaze = blaze_bpe::Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let tiktoken = tiktoken_rs::cl100k_base().unwrap();

    let mut group = c.benchmark_group("encode_cl100k");
    for &size in &[10_000usize, 100_000, 1_000_000] {
        let sample: String = text.chars().take(size).collect();
        group.throughput(Throughput::Bytes(sample.len() as u64));
        group.bench_with_input(BenchmarkId::new("blaze", size), &sample, |b, s| {
            b.iter(|| blaze.encode(s))
        });
        group.bench_with_input(BenchmarkId::new("tiktoken-rs", size), &sample, |b, s| {
            b.iter(|| tiktoken.encode_ordinary(s))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
