//! blaze-bpe: a fast, tiktoken-compatible BPE tokenizer.
//!
//! Produces byte-identical output to tiktoken for the cl100k_base and
//! o200k_base encodings. Speed comes from four things:
//!
//! 1. A hand-rolled pre-tokenization scanner ([`scan`]) instead of a
//!    backtracking regex engine.
//! 2. An id-pair merge table: `(left_id, right_id) -> merged_id` is
//!    precomputed at load time, so the merge loop hashes a single `u64`
//!    instead of hashing byte slices.
//! 3. A thread-local piece cache: repeated words (the common case in real
//!    text) skip BPE merging entirely.
//! 4. Rayon-parallel batch encoding.

mod scan;

pub use scan::Pattern;

use aho_corasick::{AhoCorasick, MatchKind};
use base64::Engine as _;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

const RANK_NONE: u32 = u32::MAX;
/// Target chunk size for parallel encoding of a single long document.
const CHUNK_TARGET: usize = 192 * 1024;
/// `encode` switches to chunked-parallel encoding at this input size.
const PARALLEL_THRESHOLD: usize = 2 * CHUNK_TARGET;

static NEXT_TOKENIZER_ID: AtomicU64 = AtomicU64::new(1);


const MAX_CACHED_PIECE: usize = 16;
const CACHE_CAP: usize = 1 << 16;

type TokenIds = SmallVec<[u32; 6]>;

/// A piece of <= 16 bytes packed into two words plus its length, so hashing
/// is two fixed multiply rounds and equality is two integer compares --
/// no pointer chase, no memcmp, no per-insert allocation. `len` disambiguates
/// zero-padding from literal NUL bytes in a piece.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PieceKey {
    w0: u64,
    w1: u64,
    len: u8,
}

impl std::hash::Hash for PieceKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.w0 ^ (self.len as u64).rotate_left(57));
        state.write_u64(self.w1);
    }
}

impl PieceKey {
    #[inline]
    fn new(piece: &[u8]) -> Self {
        debug_assert!(piece.len() <= 16);
        let (a, b) = piece.split_at(piece.len().min(8));
        PieceKey {
            w0: read_u64_padded(a),
            w1: read_u64_padded(b),
            len: piece.len() as u8,
        }
    }
}

#[inline]
fn read_u64_padded(b: &[u8]) -> u64 {
    debug_assert!(b.len() <= 8);
    let mut buf = [0u8; 8];
    buf[..b.len()].copy_from_slice(b);
    u64::from_le_bytes(buf)
}

struct PieceCache {
    owner: u64,
    map: FxHashMap<PieceKey, TokenIds>,
}

thread_local! {
    static PIECE_CACHE: RefCell<PieceCache> = RefCell::new(PieceCache {
        owner: 0,
        map: FxHashMap::default(),
    });
}

#[derive(Debug)]
pub enum Error {
    InvalidVocabLine(usize),
    MissingByteToken(u8),
    InvalidUtf8(std::string::FromUtf8Error),
    UnknownTokenId(u32),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidVocabLine(n) => write!(f, "invalid vocab file line {n}"),
            Error::MissingByteToken(b) => write!(f, "vocab has no token for byte 0x{b:02x}"),
            Error::InvalidUtf8(e) => write!(f, "decoded bytes are not valid UTF-8: {e}"),
            Error::UnknownTokenId(id) => write!(f, "unknown token id {id}"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

pub struct Tokenizer {
    id: u64,
    pattern: Pattern,
    /// token id -> bytes (includes special tokens; gaps are empty slices)
    token_bytes: Vec<Box<[u8]>>,
    /// token bytes -> id (regular tokens only)
    token_map: FxHashMap<Box<[u8]>, u32>,
    /// single byte -> token id
    byte_token: [u32; 256],
    /// (left_id << 32 | right_id) -> merged token id (merge priority == id)
    pair_table: FxHashMap<u64, u32>,
    specials: FxHashMap<String, u32>,
    special_ac: Option<AhoCorasick>,
}

impl Tokenizer {
    /// Build from the contents of a `.tiktoken` vocab file
    /// (`<base64 token> <rank>` per line) plus special tokens.
    pub fn from_tiktoken_data(
        data: &str,
        pattern: Pattern,
        special_tokens: &[(&str, u32)],
    ) -> Result<Self, Error> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut entries: Vec<(Box<[u8]>, u32)> = Vec::with_capacity(220_000);
        for (ln, line) in data.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (tok, rank) = line.split_once(' ').ok_or(Error::InvalidVocabLine(ln))?;
            let bytes = b64.decode(tok).map_err(|_| Error::InvalidVocabLine(ln))?;
            let rank: u32 = rank.parse().map_err(|_| Error::InvalidVocabLine(ln))?;
            entries.push((bytes.into_boxed_slice(), rank));
        }

        let max_id = entries
            .iter()
            .map(|&(_, r)| r)
            .chain(special_tokens.iter().map(|&(_, r)| r))
            .max()
            .unwrap_or(0);

        let mut token_bytes = vec![Box::<[u8]>::from(&[][..]); max_id as usize + 1];
        let mut token_map =
            FxHashMap::with_capacity_and_hasher(entries.len(), Default::default());
        for (bytes, rank) in entries {
            token_bytes[rank as usize] = bytes.clone();
            token_map.insert(bytes, rank);
        }

        let mut byte_token = [RANK_NONE; 256];
        for b in 0..=255u8 {
            match token_map.get(&[b][..]) {
                Some(&id) => byte_token[b as usize] = id,
                None => return Err(Error::MissingByteToken(b)),
            }
        }

        // Pair merge table: for every token, every split into two existing
        // tokens is a possible merge with priority equal to the merged id.
        let mut pair_table =
            FxHashMap::with_capacity_and_hasher(token_map.len() * 3, Default::default());
        for (bytes, &id) in &token_map {
            if bytes.len() < 2 {
                continue;
            }
            for k in 1..bytes.len() {
                if let (Some(&l), Some(&r)) =
                    (token_map.get(&bytes[..k]), token_map.get(&bytes[k..]))
                {
                    pair_table.insert(pair_key(l, r), id);
                }
            }
        }

        let mut specials = FxHashMap::default();
        for &(s, id) in special_tokens {
            token_bytes[id as usize] = s.as_bytes().into();
            specials.insert(s.to_string(), id);
        }
        let special_ac = if specials.is_empty() {
            None
        } else {
            Some(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(specials.keys())
                    .expect("failed to build special-token matcher"),
            )
        };

        Ok(Tokenizer {
            id: NEXT_TOKENIZER_ID.fetch_add(1, Ordering::Relaxed),
            pattern,
            token_bytes,
            token_map,
            byte_token,
            pair_table,
            specials,
            special_ac,
        })
    }

    /// Load a cl100k_base vocab (GPT-3.5/GPT-4) from a `.tiktoken` file.
    pub fn cl100k_base(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let data = std::fs::read_to_string(path).map_err(Error::Io)?;
        Self::from_tiktoken_data(
            &data,
            Pattern::Cl100k,
            &[
                ("<|endoftext|>", 100257),
                ("<|fim_prefix|>", 100258),
                ("<|fim_middle|>", 100259),
                ("<|fim_suffix|>", 100260),
                ("<|endofprompt|>", 100276),
            ],
        )
    }

    /// Load an o200k_base vocab (GPT-4o) from a `.tiktoken` file.
    pub fn o200k_base(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let data = std::fs::read_to_string(path).map_err(Error::Io)?;
        Self::from_tiktoken_data(
            &data,
            Pattern::O200k,
            &[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)],
        )
    }

    pub fn vocab_len(&self) -> usize {
        self.token_map.len()
    }

    /// Encode text, treating special-token strings as ordinary text.
    ///
    /// Documents larger than ~256KB are split at piece-safe boundaries and
    /// encoded chunk-parallel with rayon; output is identical to sequential
    /// encoding.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.len() >= PARALLEL_THRESHOLD {
            return self.encode_parallel(text);
        }
        self.encode_sequential(text)
    }

    /// Encode on the calling thread only, regardless of input size.
    pub fn encode_sequential(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 16);
        self.encode_ordinary_into(text, &mut out);
        out
    }

    fn encode_parallel(&self, text: &str) -> Vec<u32> {
        use rayon::prelude::*;
        let mut cuts: Vec<usize> = vec![0];
        let mut target = CHUNK_TARGET;
        while target < text.len() {
            match find_safe_cut(text, target) {
                Some(c) => {
                    cuts.push(c);
                    target = c + CHUNK_TARGET;
                }
                None => break,
            }
        }
        if cuts.len() == 1 {
            return self.encode_sequential(text);
        }
        cuts.push(text.len());
        let parts: Vec<Vec<u32>> = cuts
            .par_windows(2)
            .map(|w| self.encode_sequential(&text[w[0]..w[1]]))
            .collect();
        // Parallel concat: copy each part into its precomputed offset of the
        // output buffer. Regions are disjoint by construction.
        let total: usize = parts.iter().map(Vec::len).sum();
        let mut offsets = Vec::with_capacity(parts.len());
        let mut acc = 0usize;
        for p in &parts {
            offsets.push(acc);
            acc += p.len();
        }
        let mut out: Vec<u32> = Vec::with_capacity(total);
        let base = out.as_mut_ptr() as usize;
        parts.par_iter().zip(offsets).for_each(|(p, off)| unsafe {
            std::ptr::copy_nonoverlapping(p.as_ptr(), (base as *mut u32).add(off), p.len());
        });
        unsafe { out.set_len(total) };
        out
    }

    /// Encode text, mapping registered special-token strings to their ids.
    pub fn encode_with_specials(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 16);
        let Some(ac) = &self.special_ac else {
            self.encode_ordinary_into(text, &mut out);
            return out;
        };
        let mut pos = 0;
        for m in ac.find_iter(text) {
            if m.start() > pos {
                self.encode_ordinary_into(&text[pos..m.start()], &mut out);
            }
            out.push(self.specials[&text[m.start()..m.end()]]);
            pos = m.end();
        }
        if pos < text.len() {
            self.encode_ordinary_into(&text[pos..], &mut out);
        }
        out
    }

    /// Number of tokens `encode` would produce.
    pub fn count(&self, text: &str) -> usize {
        self.encode(text).len()
    }

    /// Encode many texts in parallel with rayon.
    pub fn encode_batch<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode(t.as_ref())).collect()
    }

    /// Decode token ids to bytes.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let bytes = self
                .token_bytes
                .get(id as usize)
                .ok_or(Error::UnknownTokenId(id))?;
            if bytes.is_empty() {
                return Err(Error::UnknownTokenId(id));
            }
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }

    /// Decode token ids to a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        String::from_utf8(self.decode_bytes(ids)?).map_err(Error::InvalidUtf8)
    }

    pub fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>) {
        PIECE_CACHE.with_borrow_mut(|cache| {
            if cache.owner != self.id {
                cache.map.clear();
                cache.owner = self.id;
            }
            let bytes = text.as_bytes();
            scan::for_each_piece(text, self.pattern, |start, end| {
                let piece = &bytes[start..end];
                if piece.len() == 1 {
                    out.push(self.byte_token[piece[0] as usize]);
                    return;
                }
                if piece.len() <= MAX_CACHED_PIECE {
                    let key = PieceKey::new(piece);
                    if let Some(ids) = cache.map.get(&key) {
                        out.extend_from_slice(ids);
                        return;
                    }
                    let from = out.len();
                    self.encode_piece(piece, out);
                    if cache.map.len() >= CACHE_CAP {
                        cache.map.clear();
                    }
                    if out.len() - from <= 6 {
                        cache.map.insert(key, TokenIds::from_slice(&out[from..]));
                    }
                } else {
                    self.encode_piece(piece, out);
                }
            });
        });
    }

    /// BPE-encode a single pre-tokenization piece (>= 2 bytes).
    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if let Some(&id) = self.token_map.get(piece) {
            out.push(id);
            return;
        }
        let mut ids: SmallVec<[u32; 64]> = piece
            .iter()
            .map(|&b| self.byte_token[b as usize])
            .collect();
        // ranks[i] = merged id of (ids[i], ids[i+1]), RANK_NONE if not mergeable.
        let mut ranks: SmallVec<[u32; 64]> = (0..ids.len() - 1)
            .map(|i| self.pair_rank(ids[i], ids[i + 1]))
            .collect();
        loop {
            let mut best = RANK_NONE;
            let mut best_i = 0usize;
            for (i, &r) in ranks.iter().enumerate() {
                if r < best {
                    best = r;
                    best_i = i;
                }
            }
            if best == RANK_NONE {
                break;
            }
            ids[best_i] = best;
            ids.remove(best_i + 1);
            ranks.remove(best_i);
            if best_i < ranks.len() {
                ranks[best_i] = self.pair_rank(ids[best_i], ids[best_i + 1]);
            }
            if best_i > 0 {
                ranks[best_i - 1] = self.pair_rank(ids[best_i - 1], ids[best_i]);
            }
        }
        out.extend_from_slice(&ids);
    }

    #[inline]
    fn pair_rank(&self, l: u32, r: u32) -> u32 {
        match self.pair_table.get(&pair_key(l, r)) {
            Some(&id) => id,
            None => RANK_NONE,
        }
    }
}

/// Count pre-tokenization pieces (scanner-only pass, for profiling).
pub fn count_pieces(text: &str, pattern: Pattern) -> usize {
    let mut n = 0;
    scan::for_each_piece(text, pattern, |_, _| n += 1);
    n
}

#[inline]
fn pair_key(l: u32, r: u32) -> u64 {
    ((l as u64) << 32) | r as u64
}

/// Find a byte offset >= `from` where a pre-tokenization piece provably ends,
/// no matter how the scan is aligned, so the text can be cut there and the
/// two halves encoded independently with identical output.
///
/// Two conditions qualify (for both cl100k and o200k patterns):
///
/// 1. After `\n`, when the next char is not whitespace. Word pieces can never
///    contain `\n` (their optional prefix class excludes it); any piece
///    covering the `\n` is a punct piece's `[\r\n]*` tail or a whitespace
///    piece ending at the run's last newline, and both end here because the
///    next char is neither newline nor whitespace.
///
/// 2. Between an ASCII letter and a following space/digit/punct. Letters only
///    occur in word pieces (`\p{L}+` / `A*B+` bodies, contraction suffixes),
///    all of which terminate when a non-letter follows. `'` is excluded
///    because a contraction suffix could extend the piece across the cut.
fn find_safe_cut(text: &str, from: usize) -> Option<usize> {
    let b = text.as_bytes();
    let n = text.len();
    let mut i = from.max(1);
    while i < n {
        let prev = b[i - 1];
        let cur = b[i];
        if prev == b'\n' {
            let ws = if cur < 0x80 {
                matches!(cur, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
            } else {
                // `i` is a char boundary: the previous byte is ASCII.
                text[i..].chars().next().unwrap().is_whitespace()
            };
            if !ws {
                return Some(i);
            }
        } else if prev.is_ascii_alphabetic()
            && cur != b'\''
            && (cur == b' ' || cur.is_ascii_digit() || cur.is_ascii_punctuation())
        {
            return Some(i);
        }
        i += 1;
    }
    None
}
