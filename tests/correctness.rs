//! Byte-identical comparison against tiktoken-rs (a faithful port of
//! OpenAI's tiktoken) for both cl100k_base and o200k_base.

use blaze_bpe::Tokenizer;

fn edge_cases() -> Vec<String> {
    let fixed: &[&str] = &[
        "",
        " ",
        "  ",
        "   ",
        "hello world",
        "Hello, world!",
        "  hello   world  ",
        "don't can't won't it's I'll we've they're you'd he'S SHE'LL WE'VE THEY'RE",
        "'s 't 'll 've 're 'd 'm 'x 'stop 'them 'lemon",
        "1 12 123 1234 12345 123456789012",
        "a1b2c3 42x 3.14159 -17e-4",
        "\n", "\r", "\r\n", "\n\n\n", "  \n", "\n  ", "  \n  \n  ",
        "hello\nworld", "hello \n world", "hello  \n\n  world",
        "tab\there", "\tabc", " \t\n abc",
        "!!!", " !!!", "?!?!\n\n", "...hello", "))) (((",
        "foo.bar.baz", "a-b-c", "snake_case_name camelCaseName",
        "fn main() { println!(\"hi\"); }",
        "x += 1; // comment\nlet y = &mut *ptr;",
        "https://example.com/path?q=1&r=2#frag",
        "email@test.co.nz, \"quoted\" and 'single'",
        "日本語のテキスト",
        "中文文本处理速度",
        "Привет мир",
        "مرحبا بالعالم",
        "हिन्दी में नमस्ते",
        "한국어 텍스트",
        "naïve café résumé Zürich",
        "e\u{301}toile de\u{301}ja\u{300} vu",
        "emoji 😀😃🚀 and 👍🏽 flags 🇳🇿🇺🇸",
        "mixed 英語と日本語 and русский текст",
        "½ ⅓ Ⅳ ① ² ₃",
        "Ⅻ34Ⅻ 12³45",
        "\u{85}\u{a0}\u{2028}\u{2003}word",
        "word\u{a0}word \u{a0}\u{a0}x",
        "ALLCAPS Mixed lowerUPPER XMLHttpRequest HTTPServer",
        "HELLOworld xHELLO ᵃbc ˢtuff",
        "  'twas the night", "o'clock O'CLOCK o'Clock",
        "1'2 x'y '' ''' ' ' '",
        "\u{0}\u{1}\u{2} control \u{7f} chars",
        "trailing space ",
        "trailing spaces   ",
        "trailing newline\n",
        "trailing newlines\n\n\n",
        " \t \t ",
        "🙂",
        "a",
        "Z",
        "9",
        ".",
        "\u{10FFFF} max codepoint",
    ];
    let mut v: Vec<String> = fixed.iter().map(|s| s.to_string()).collect();
    // A long piece to exceed the cacheable length and stress the merge loop.
    v.push("a".repeat(200));
    v.push("ab".repeat(150));
    v.push("supercalifragilisticexpialidociousantidisestablishmentarianism".repeat(3));
    v.push(" ".repeat(100) + "x");
    v.push("\n".repeat(50));
    v.push("=".repeat(80) + "\n\n");
    v
}

/// Deterministic xorshift PRNG for fuzz strings.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next() % xs.len() as u64) as usize]
    }
}

fn fuzz_cases(n: usize) -> Vec<String> {
    let atoms: Vec<&str> = vec![
        "hello", "World", "THE", "a", "I", "don", "'t", "'S", "'ll", "'", "''",
        " ", "  ", "\t", "\n", "\r\n", "\n\n", " \n",
        "1", "42", "007", "12345", "3.14",
        ",", ".", "!", "?!", "...", "-", "_", "()", "{}", ";", ":", "\"",
        "日本", "語", "中文", "мир", "café", "ü", "ß", "é",
        "\u{301}", "\u{a0}", "\u{2028}", "😀", "🚀", "🇳🇿",
        "XMLHttp", "Request", "iPhone", "eBay", "McDonald",
        "ᵃ", "ˢ", "Ⅻ", "½", "①",
        "<|endoftext|>", "foo", "bar_baz", "qux/quux", "a=b", "x+=1",
    ];
    let mut rng = Rng(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            let len = (rng.next() % 30 + 1) as usize;
            let mut s = String::new();
            for _ in 0..len {
                s.push_str(rng.pick(&atoms).to_owned());
            }
            s
        })
        .collect()
}

fn compare(ours: &Tokenizer, reference: &tiktoken_rs::CoreBPE, cases: &[String]) {
    for case in cases {
        let expected = reference.encode_ordinary(case);
        let got = ours.encode(case);
        assert_eq!(
            got, expected,
            "mismatch on {:?}\n ours: {:?}\n tiktoken: {:?}",
            case, got, expected
        );
    }
}

#[test]
fn cl100k_matches_tiktoken() {
    let ours = Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let reference = tiktoken_rs::cl100k_base().unwrap();
    compare(&ours, &reference, &edge_cases());
    compare(&ours, &reference, &fuzz_cases(3000));
}

#[test]
fn o200k_matches_tiktoken() {
    let ours = Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap();
    let reference = tiktoken_rs::o200k_base().unwrap();
    compare(&ours, &reference, &edge_cases());
    compare(&ours, &reference, &fuzz_cases(3000));
}

#[test]
fn cl100k_matches_on_shakespeare() {
    let text = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    let ours = Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let reference = tiktoken_rs::cl100k_base().unwrap();
    assert_eq!(ours.encode(&text), reference.encode_ordinary(&text));
}

#[test]
fn o200k_matches_on_shakespeare() {
    let text = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    let ours = Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap();
    let reference = tiktoken_rs::o200k_base().unwrap();
    assert_eq!(ours.encode(&text), reference.encode_ordinary(&text));
}

#[test]
fn chunked_parallel_matches_sequential() {
    let shakespeare = std::fs::read_to_string("data/shakespeare.txt").unwrap();
    // Long docs of different shapes: normal prose, one giant line with no
    // newlines, CJK with no spaces (few/no safe cuts), and mixed content.
    let one_line = "{\"key\": \"value\", \"n\": 42, \"list\": [1, 2, 3]} ".repeat(20_000);
    let cjk = "日本語のテキスト、中文文本。改行なしで長い行を作る。".repeat(10_000);
    let mixed = format!("{shakespeare}\n\n{one_line}\n{cjk}\nend 'twas  \n it's");
    for (tok, reference) in [
        (
            Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap(),
            tiktoken_rs::cl100k_base().unwrap(),
        ),
        (
            Tokenizer::o200k_base("data/o200k_base.tiktoken").unwrap(),
            tiktoken_rs::o200k_base().unwrap(),
        ),
    ] {
        for text in [&shakespeare, &one_line, &cjk, &mixed] {
            assert!(text.len() > 512 * 1024 || text == &shakespeare);
            let par = tok.encode(text);
            assert_eq!(par, tok.encode_sequential(text));
            assert_eq!(par, reference.encode_ordinary(text));
        }
    }
}

#[test]
fn decode_roundtrip() {
    let ours = Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    for case in edge_cases() {
        let ids = ours.encode(&case);
        assert_eq!(ours.decode(&ids).unwrap(), case);
    }
}

#[test]
fn specials_and_batch() {
    let ours = Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let ids = ours.encode_with_specials("hi<|endoftext|>there");
    assert!(ids.contains(&100257));
    assert_eq!(ours.decode(&ids).unwrap(), "hi<|endoftext|>there");
    // encode() must treat the special as plain text.
    assert!(!ours.encode("hi<|endoftext|>there").contains(&100257));

    let texts = vec!["hello world"; 64];
    let batch = ours.encode_batch(&texts);
    assert!(batch.iter().all(|ids| *ids == batch[0]));
    assert_eq!(ours.decode(&batch[0]).unwrap(), "hello world");
}

#[test]
fn long_piece_cache_matches_tiktoken() {
    let ours = Tokenizer::cl100k_base("data/cl100k_base.tiktoken").unwrap();
    let reference = tiktoken_rs::cl100k_base().unwrap();
    let mut cases = Vec::new();
    // Pieces of 17..=96 bytes hit the long-piece cache; repeat each text so
    // both the cold (fill) and hot (verified hit) paths run, and step CJK run
    // lengths across the 96-byte cache limit and the 128-byte heap threshold.
    for reps in 1..=16 {
        let cjk = "统一码万国码单一".repeat(reps);
        cases.push(format!("{cjk} and {cjk}, {cjk}."));
    }
    cases.push(
        "日本語のテキストを処理する速度を測定します。中文分词的速度也很重要。\
         Скорость токенизации русского текста. सटीक और तेज़ टोकनाइज़ेशन। "
            .repeat(50),
    );
    // Same slot reused by different pieces: byte verification must reject
    // collisions and stale entries.
    for w in ["Скорость", "токенизации", "русского"] {
        cases.push(w.repeat(3).repeat(4));
    }
    compare(&ours, &reference, &cases);
}
